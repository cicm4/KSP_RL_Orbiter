"""Finite-history policies for matched simulator architecture experiments.

History is part of the policy input, including executed previous actions and a
valid-token flag. PPO therefore recomputes the *same* bounded history at update
time; it never shuffles recurrent states from different spacecraft together.
The Transformer sees only past/current tokens, and emits only the last token.
"""
from __future__ import annotations

import math
import torch
from torch import nn

ARCHITECTURES = ("mlp256", "mlp512", "mlp1024", "gru", "lstm", "transformer")


class History:
    """Trailing [worlds, length, observation+action+valid] window.

    ``append`` builds a *new* contiguous window each step and never writes
    into one it returned earlier, so callers may keep previous ``data``
    tensors (the PPO rollout stores them without cloning). The values are
    identical to the historical in-place shift: reset rows are zeroed, the
    window moves one token left and the newest token holds the observation,
    the executed previous action (zero after a reset) and a valid flag.
    No boolean-mask indexing is used, so a reset causes no host sync.
    """

    def __init__(self, count, length, observations, actions, device):
        self.observations, self.actions = observations, actions
        self.data = torch.zeros(count, length, observations + actions + 1, device=device)

    @torch.no_grad()
    def append(self, observation, previous_action=None, reset=None):
        data = self.data
        if observation.shape != (data.shape[0], self.observations):
            raise ValueError("history observation shape mismatch")
        count = data.shape[0]
        if previous_action is None:
            action = data.new_zeros(count, self.actions)
        else:
            action = torch.as_tensor(previous_action).to(
                device=data.device, dtype=data.dtype).expand(count, self.actions)
        valid = data.new_ones(count, 1)
        token = torch.cat((observation.to(device=data.device, dtype=data.dtype), action, valid), -1)
        if reset is not None:
            reset = torch.as_tensor(reset, device=data.device)
            if reset.dtype != torch.bool:
                # Integer world indices, as accepted by the historical
                # ``data[reset] = 0`` assignment.
                indices, reset = reset, torch.zeros(count, dtype=torch.bool, device=data.device)
                reset[indices] = True
            if reset.shape != (count,):
                raise ValueError("history reset mask shape mismatch")
            # Only the executed-action slots are cleared in the new token.
            token[:, self.observations:-1].masked_fill_(reset[:, None], 0)
        older = data[:, 1:]
        if reset is not None:
            older = older.masked_fill(reset[:, None, None], 0)
        self.data = torch.cat((older, token[:, None]), 1)
        return self.data


class Core(nn.Module):
    def __init__(self, architecture, features, output, context):
        super().__init__()
        self.architecture, self.context = architecture, context
        if architecture.startswith("mlp"):
            width = int(architecture[3:])
            self.network = nn.Sequential(nn.Linear(features, width), nn.Tanh(),
                                         nn.Linear(width, width), nn.Tanh())
        elif architecture in ("gru", "lstm"):
            width = 256
            self.encoder = nn.Sequential(nn.Linear(features, width), nn.Tanh())
            self.memory = (nn.GRU if architecture == "gru" else nn.LSTM)(
                width, width, batch_first=True)
        elif architecture == "transformer":
            width = 128
            self.encoder = nn.Linear(features, width)
            self.position = nn.Parameter(torch.randn(1, context, width) * .01)
            # Construct layers independently (TransformerEncoder clones can
            # otherwise begin with identical weights). Dropout=0 makes PPO's
            # old/new likelihoods comparable for a fixed policy/history.
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(width, 4, dim_feedforward=512,
                                           dropout=0., activation="gelu",
                                           batch_first=True, norm_first=True)
                for _ in range(2)])
            self.norm = nn.LayerNorm(width)
        else:
            raise ValueError(f"unknown architecture {architecture}")
        self.output = nn.Linear(width, output)

    def forward(self, history):
        if history.ndim != 3 or history.shape[1] != self.context:
            raise ValueError("policy requires its declared fixed history window")
        if self.architecture.startswith("mlp"):
            x = self.network(history[:, -1])
        elif self.architecture in ("gru", "lstm"):
            x, _ = self.memory(self.encoder(history))
            x = x[:, -1]
        else:
            x = self.encoder(history) + self.position
            padding = history[:, :, -1] == 0
            # All tokens in this window precede the decision being evaluated.
            # Attention among these tokens does not expose a future action.
            for layer in self.layers:
                x = layer(x, src_key_padding_mask=padding)
            x = self.norm(x[:, -1])
        return self.output(x)


ACTION_TRANSFORMS = ("tanh", "scaled_tanh")


class SequencePolicy(nn.Module):
    """Tanh-Gaussian actor with an optional saturation-free action mapping.

    ``tanh`` is the original v1 mapping: action = tanh(z). Teacher labels at
    exactly +/-1 (full throttle, vertical pitch) then drive z without bound,
    and a small Normal std cannot explore those dimensions at all.
    ``scaled_tanh`` executes clamp(action_scale * tanh(z), -1, 1), so the
    bounds are reached at finite z. An optional norm deadzone maps short
    action vectors to exactly zero (orbital coasting under exploration noise).
    PPO ratios use the stored latent z, so this deterministic mapping does
    not bias the likelihood ratio.
    """

    def __init__(self, architecture, observations, actions, context=16, initial_std=.15,
                 action_transform="tanh", action_scale=1., deadzone=0.):
        super().__init__()
        if (architecture not in ARCHITECTURES or context < 1 or not 0 < initial_std <= 2
                or action_transform not in ACTION_TRANSFORMS):
            raise ValueError("invalid sequence policy configuration")
        if action_transform == "tanh" and (action_scale != 1 or deadzone != 0):
            raise ValueError("the original tanh mapping has no scale or deadzone")
        if not 1 <= action_scale <= 1.5 or not 0 <= deadzone < .5:
            raise ValueError("invalid action scale or deadzone")
        self.spec = dict(architecture=architecture, observations=observations,
                         actions=actions, context=context, initial_std=initial_std)
        if action_transform != "tanh":
            # v1 checkpoints keep their original spec (and loader) unchanged.
            self.spec.update(action_transform=action_transform,
                             action_scale=action_scale, deadzone=deadzone)
        self.action_scale, self.deadzone = action_scale, deadzone
        features = observations + actions + 1
        self.actor = Core(architecture, features, actions, context)
        self.critic = Core(architecture, features, 1, context)
        self.log_std = nn.Parameter(torch.full((actions,), math.log(initial_std)))

    def distribution(self, history):
        return torch.distributions.Normal(self.actor(history), self.log_std.exp())

    def execute(self, squashed):
        """Map tanh(z) in (-1, 1) to the executed environment action."""
        action = (self.action_scale * squashed).clamp(-1, 1)
        if self.deadzone:
            norm = action.norm(dim=-1, keepdim=True)
            gain = ((norm - self.deadzone) / (1 - self.deadzone)).clamp(min=0)
            action = action / norm.clamp(min=1e-12) * gain
        return action

    def squashed_target(self, action):
        """Inverse of ``execute`` for teacher labels, strictly inside (-1, 1)."""
        action = action.clamp(-1, 1)
        if self.deadzone:
            norm = action.norm(dim=-1, keepdim=True)
            scale = torch.where(norm > 0, (self.deadzone + (1 - self.deadzone) * norm)
                                / norm.clamp(min=1e-12), 0)
            action = action * scale
        return action / self.action_scale

    def forward(self, history):
        return self.execute(self.actor(history).tanh())

    @staticmethod
    def log_probability(distribution, latent):
        # Stable log(1-tanh(z)^2), including saturated samples.
        correction = 2 * (math.log(2) - latent - nn.functional.softplus(-2 * latent))
        return (distribution.log_prob(latent) - correction).sum(-1)

    def sample(self, history):
        distribution = self.distribution(history)
        latent = distribution.sample()
        return self.execute(latent.tanh()), latent, self.log_probability(distribution, latent)
