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
    def __init__(self, count, length, observations, actions, device):
        self.observations, self.actions = observations, actions
        self.data = torch.zeros(count, length, observations + actions + 1, device=device)

    @torch.no_grad()
    def append(self, observation, previous_action=None, reset=None):
        if observation.shape != (self.data.shape[0], self.observations):
            raise ValueError("history observation shape mismatch")
        if reset is not None:
            self.data[reset] = 0
        self.data[:, :-1] = self.data[:, 1:].clone()
        self.data[:, -1] = 0
        self.data[:, -1, :self.observations] = observation
        if previous_action is not None:
            self.data[:, -1, self.observations:-1] = previous_action
            if reset is not None:
                self.data[reset, -1, self.observations:-1] = 0
        self.data[:, -1, -1] = 1
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


class SequencePolicy(nn.Module):
    def __init__(self, architecture, observations, actions, context=16, initial_std=.15):
        super().__init__()
        if architecture not in ARCHITECTURES or context < 1 or not 0 < initial_std <= 2:
            raise ValueError("invalid sequence policy configuration")
        self.spec = dict(architecture=architecture, observations=observations,
                         actions=actions, context=context, initial_std=initial_std)
        features = observations + actions + 1
        self.actor = Core(architecture, features, actions, context)
        self.critic = Core(architecture, features, 1, context)
        self.log_std = nn.Parameter(torch.full((actions,), math.log(initial_std)))

    def distribution(self, history):
        return torch.distributions.Normal(self.actor(history), self.log_std.exp())

    def forward(self, history):
        return self.actor(history).tanh()

    @staticmethod
    def log_probability(distribution, latent):
        # Stable log(1-tanh(z)^2), including saturated samples.
        correction = 2 * (math.log(2) - latent - nn.functional.softplus(-2 * latent))
        return (distribution.log_prob(latent) - correction).sum(-1)

    def sample(self, history):
        distribution = self.distribution(history)
        latent = distribution.sample()
        return latent.tanh(), latent, self.log_probability(distribution, latent)
