"""GPU-native BC/DAgger/PPO experiment for morphology-conditioned ascent.

All simulator worlds, replay samples, rollouts and neural updates remain on
the requested device. Checkpoints are experimental .pt files under a new task
contract; they are deliberately incompatible with the approved KSP policies.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import platform
import signal
import time

import numpy as np
import torch
from torch import nn

from .generalized_env import (
    BatchedAscent,
    FEATURE_SCHEMA,
    OBSERVATIONS,
    TASK,
    VehicleEnvelope,
    cpu_observation,
    normalization_audit,
)
from .env import EnvConfig, HeadlessOrbiterEnv
from .guidance import expert_action

_STOP_SIGNAL = None
_STOP_CHECKPOINT = None
_PHASE_PROGRESS = {}
DEADLINE_TASK = "vehicle_conditioned_ascent_deadline_v5"
AERO_DEADLINE_TASK = "vehicle_conditioned_aero_ascent_deadline_v5"


def _request_stop(signum, _frame):
    global _STOP_SIGNAL
    _STOP_SIGNAL = signum


def _check_stop():
    if _STOP_SIGNAL is not None and _STOP_CHECKPOINT is not None:
        _STOP_CHECKPOINT()


class VehiclePolicy(nn.Module):
    def __init__(self, hidden=256, *, observations=OBSERVATIONS):
        super().__init__()
        self.hidden = hidden
        self.observations = observations
        self.actor = nn.Sequential(
            nn.Linear(observations, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )
        self.critic = nn.Sequential(
            nn.Linear(observations, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.log_std = nn.Parameter(torch.full((2,), -3.0))

    def forward(self, observation):
        return self.actor(observation).tanh()

    def distribution(self, observation):
        return torch.distributions.Normal(self.actor(observation), self.log_std.exp())

    def sample(self, observation):
        distribution = self.distribution(observation)
        latent = distribution.sample()
        action = latent.tanh()
        logp = (distribution.log_prob(latent) - (1 - action.square() + 1e-6).log()).sum(
            -1
        )
        return action, latent, logp, self.critic(observation).squeeze(-1)


def clean_json(value):
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(clean_json(data), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_policy(path, model, contract, ancestry):
    path = Path(path)
    temporary = path.with_suffix(".tmp.pt")
    torch.save(
        dict(
            task=contract.get("task", TASK),
            hidden=model.hidden,
            state_dict=model.state_dict(),
            contract=contract,
            ancestry=ancestry,
        ),
        temporary,
    )
    temporary.replace(path)
    write_json(
        path.with_suffix(".metadata.json"),
        dict(
            task=contract.get("task", TASK),
            contract=contract,
            ancestry=ancestry,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            saved_at_unix=time.time(),
            live_ksp_validated=False,
        ),
    )


def load_policy(path, *, device="cpu"):
    record = torch.load(path, map_location=device, weights_only=True)
    contract = record.get("contract", {})
    supported = {(TASK, OBSERVATIONS, FEATURE_SCHEMA)}
    from .generalized_aero_env import (
        TASK as AERO_TASK,
        OBSERVATIONS as AERO_OBSERVATIONS,
        FEATURE_SCHEMA as AERO_FEATURE_SCHEMA,
    )

    supported.add((AERO_TASK, AERO_OBSERVATIONS, AERO_FEATURE_SCHEMA))
    supported.add((DEADLINE_TASK, OBSERVATIONS, FEATURE_SCHEMA))
    supported.add((AERO_DEADLINE_TASK, AERO_OBSERVATIONS, AERO_FEATURE_SCHEMA))
    if (
        record.get("task"),
        contract.get("observations"),
        contract.get("feature_schema"),
    ) not in supported:
        raise ValueError("not a vehicle-conditioned ascent checkpoint")
    if contract.get("task", record["task"]) != record["task"]:
        raise ValueError("checkpoint task and contract disagree")
    if record["task"] in (DEADLINE_TASK, AERO_DEADLINE_TASK):
        physics = "aero" if record["task"] == AERO_DEADLINE_TASK else "constant-cda"
        objective = contract.get("objective", {})
        try:
            _, _, expected = experiment_contract(
                contract["curriculum"],
                physics,
                deadline_mode=objective["deadline_mode"],
                timeout_penalty=objective["timeout_penalty"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid finite-horizon objective contract") from error
        if contract != expected:
            raise ValueError("finite-horizon task/environment contract differs")
    model = VehiclePolicy(record["hidden"], observations=contract["observations"]).to(
        device
    )
    model.load_state_dict(record["state_dict"])
    return model, record


def transfer_aero_policy(path, *, device="cpu"):
    """Preserve a v3 actor exactly while exposing two new aero inputs to PPO.

    This is a new experiment initialized from existing weights, never a claim
    of resumed optimizer state or a pre-existing aero-trained checkpoint.
    """
    from .generalized_aero_env import OBSERVATIONS as AERO_OBSERVATIONS

    source, record = load_policy(path, device=device)
    if record["task"] != TASK or source.observations != OBSERVATIONS:
        raise ValueError("aero transfer requires an original v3 ascent checkpoint")
    target = VehiclePolicy(source.hidden, observations=AERO_OBSERVATIONS).to(device)
    state = target.state_dict()
    for key, value in source.state_dict().items():
        if key in ("actor.0.weight", "critic.0.weight"):
            state[key].zero_()
            state[key][:, :OBSERVATIONS].copy_(value)
        else:
            state[key].copy_(value)
    target.load_state_dict(state)
    return target, record


def initialize_actor_policy(path, contract, *, seed, device="cpu"):
    """Explicit actor-only initialization across objectives, with a fresh critic.

    Every ablation arm starts from the same parent actor and the same seeded
    critic. Neither the parent's critic nor optimizer state crosses this boundary.
    """
    source, record = load_policy(path, device=device)
    from .generalized_aero_env import TASK as AERO_TASK

    if record["task"] not in (TASK, AERO_TASK):
        raise ValueError("actor initialization requires an original v3/v4 checkpoint")
    source_physics = "aero" if record["task"] == AERO_TASK else "constant-cda"
    _, _, expected = experiment_contract(contract["curriculum"], source_physics)
    if record["contract"] != expected:
        raise ValueError("actor source must match its original curriculum contract")
    if source.observations > contract["observations"]:
        raise ValueError("actor initialization cannot discard source input features")
    # Linear layers initialize on CPU before the explicit device copy. Isolate
    # that RNG without touching generators belonging to any other CUDA device.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        target = VehiclePolicy(
            source.hidden, observations=contract["observations"]
        ).to(device)
    state = target.state_dict()
    for key, value in source.state_dict().items():
        if key == "actor.0.weight":
            state[key].zero_()
            state[key][:, : source.observations].copy_(value)
        elif key.startswith("actor.") or key == "log_std":
            state[key].copy_(value)
    target.load_state_dict(state)
    return target, record


def make_environment(count, *, envelope, **kwargs):
    from .generalized_aero_env import AeroVehicleEnvelope, BatchedAeroAscent

    cls = (
        BatchedAeroAscent
        if isinstance(envelope, AeroVehicleEnvelope)
        else BatchedAscent
    )
    return cls(count, envelope=envelope, **kwargs)


def scalar_observation(env, *, observations=OBSERVATIONS):
    if observations == OBSERVATIONS:
        return cpu_observation(env)
    from .generalized_aero_env import cpu_observation as aero_observation

    return aero_observation(env)


def wilson(success, count):
    if not count:
        return [0, 1]
    z, p = 1.959963984540054, success / count
    divisor = 1 + z * z / count
    center = (p + z * z / (2 * count)) / divisor
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count**2)) / divisor
    return [max(0, center - half), min(1, center + half)]


@torch.no_grad()
def evaluate(
    model,
    *,
    count,
    device,
    seed,
    envelope,
    kind="policy",
    physics_dt=0.25,
    export_crafts=False,
    deadline_mode="truncate",
    timeout_penalty=0.0,
):
    env = make_environment(
        count,
        device=device,
        seed=seed,
        envelope=envelope,
        physics_dt=physics_dt,
        deadline_mode=deadline_mode,
        timeout_penalty=timeout_penalty,
    )
    observation = env.observation()
    done = torch.zeros(count, dtype=torch.bool, device=device)
    results = [None] * count
    crafts = (
        [asdict(env.export_config(index)) for index in range(count)]
        if export_crafts
        else None
    )
    start = time.monotonic()
    for step in range(math.ceil(env.max_time / env.control_dt) + 1):
        _PHASE_PROGRESS.update(phase="validation_or_test", control_steps=step)
        _check_stop()
        action = env.teacher() if kind == "teacher" else model(observation)
        observation, _, terminated, truncated, info = env.step(action)
        finished = (terminated | truncated) & ~done
        indices = torch.nonzero(finished).flatten().cpu().tolist()
        if indices:
            values = {key: val[finished].cpu().tolist() for key, val in info.items()}
            for local, index in enumerate(indices):
                results[index] = dict(
                    world=index,
                    seed=seed,
                    **{key: values[key][local] for key in values},
                )
                if crafts:
                    results[index]["craft"] = crafts[index]
                    results[index]["craft_sha256"] = hashlib.sha256(
                        json.dumps(crafts[index], sort_keys=True).encode()
                    ).hexdigest()
        done |= terminated | truncated
        if bool(done.all()):
            break
        # Completed worlds are kept inert while other worlds finish; no reset,
        # so evaluation records exactly one independent craft per world.
        # step() needs a horizon guard; finished worlds have zero time remaining.
        env.time[done] = env.max_time
    if any(row is None for row in results):
        raise RuntimeError("evaluation failed to finish all worlds")
    success = sum(bool(row["success"]) for row in results)
    return dict(
        kind=kind,
        seed=seed,
        episodes=count,
        successes=success,
        success_rate=success / count,
        success_ci95=wilson(success, count),
        failure_counts=dict(
            Counter(str(row["reason"]) for row in results if not row["success"])
        ),
        objective=dict(deadline_mode=deadline_mode, timeout_penalty=timeout_penalty),
        mean_discounted_task_return=float(
            np.mean([row["discounted_task_return"] for row in results])
        )
        if "discounted_task_return" in results[0]
        else None,
        mean_elapsed=float(np.mean([row["time"] for row in results]))
        if "time" in results[0]
        else None,
        elapsed_s=time.monotonic() - start,
        results=results,
        scope="planar sequential-liquid-stage surrogate; no actual KSP evidence",
    )


def fit_actor(model, observations, targets, *, updates, batch_size, optimizer):
    size = observations.shape[0]
    total_loss = torch.zeros((), device=observations.device)
    model.train()
    for update in range(updates):
        _PHASE_PROGRESS.update(phase="supervised_fit", gradient_updates=update)
        _check_stop()
        selected = torch.randint(
            size, (min(batch_size, size),), device=observations.device
        )
        loss = nn.functional.mse_loss(model(observations[selected]), targets[selected])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.detach()
    model.eval()
    return float(total_loss / updates)


def collect_labels(model, env, *, steps, expert_mixing):
    observations, actions = [], []
    observation = env.observation()
    completed = torch.zeros((), dtype=torch.long, device=env.device)
    successes = torch.zeros_like(completed)
    start = time.monotonic()
    for step in range(steps):
        _PHASE_PROGRESS.update(
            phase="label_collection",
            control_steps=step,
            transitions=step * env.count,
            expert_mixing=expert_mixing,
        )
        _check_stop()
        with torch.no_grad():
            teacher = env.teacher()
            observations.append(observation)
            actions.append(teacher)
            if expert_mixing >= 1:
                command = teacher
            else:
                mask = env.uniform((env.count, 1), 0, 1) < expert_mixing
                command = torch.where(mask, teacher, model(observation))
            observation, _, terminated, truncated, info = env.step(command)
            done = terminated | truncated
            completed += done.sum()
            successes += (done & info["success"]).sum()
            if bool(done.any()):
                observation = env.reset(done)
    seconds = time.monotonic() - start
    return (
        torch.cat(observations),
        torch.cat(actions),
        dict(
            transitions=steps * env.count,
            elapsed_s=seconds,
            environment_steps_per_second=steps * env.count / seconds,
            completed_episodes=int(completed),
            completed_successes=int(successes),
            expert_mixing=expert_mixing,
        ),
    )


def complete_return_targets(rewards, terminated, truncated, gamma):
    """Monte Carlo targets only where a true terminal is observed ahead.

    Time limits and the unobserved end of a rollout retain an unknown bootstrap;
    they must never silently become zero-value terminal training examples.
    """
    returns = torch.zeros_like(rewards)
    valid = torch.zeros_like(terminated)
    value = torch.zeros_like(rewards[0])
    known = torch.zeros_like(terminated[0])
    for step in reversed(range(len(rewards))):
        value = rewards[step] + gamma * torch.where(terminated[step], 0, value)
        known = terminated[step] | (known & ~truncated[step])
        returns[step], valid[step] = value, known
    return returns, valid


def warm_value(model, env, *, steps, updates, batch_size):
    """Fit only the critic to this frozen actor's complete on-policy returns."""
    observations, rewards, terminals, time_limits = [], [], [], []
    observation = env.reset()
    initial_rollout_sha256 = hashlib.sha256(
        observation.detach().cpu().numpy().tobytes()
    ).hexdigest()
    start = time.monotonic()
    for step in range(steps):
        _PHASE_PROGRESS.update(
            phase="value_rollout", control_steps=step, transitions=step * env.count
        )
        _check_stop()
        with torch.no_grad():
            observations.append(observation)
            next_observation, reward, terminated, truncated, _ = env.step(
                model(observation)
            )
            rewards.append(reward)
            terminals.append(terminated)
            time_limits.append(truncated)
            done = terminated | truncated
            observation = env.reset(done) if bool(done.any()) else next_observation
    rollout_fingerprints = dict(
        initial_observation_sha256=initial_rollout_sha256,
        final_observation_sha256=hashlib.sha256(
            observation.detach().cpu().numpy().tobytes()
        ).hexdigest(),
        deterministic_frozen_actor=True,
    )
    returns, valid = complete_return_targets(
        torch.stack(rewards),
        torch.stack(terminals),
        torch.stack(time_limits),
        env.gamma,
    )
    obs = torch.stack(observations)[valid]
    targets = returns[valid]
    if not len(targets):
        return dict(
            complete_value_targets=0,
            updates=0,
            reason="no observed terminal returns",
            **rollout_fingerprints,
        )
    optimizer = torch.optim.Adam(model.critic.parameters(), lr=3e-4)
    total_loss = torch.zeros((), device=env.device)
    for update in range(updates):
        _PHASE_PROGRESS.update(phase="value_fit", gradient_updates=update)
        _check_stop()
        indices = torch.randint(
            len(targets), (min(batch_size, len(targets)),), device=env.device
        )
        prediction = model.critic(obs[indices]).squeeze(-1)
        loss = nn.functional.mse_loss(prediction, targets[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.critic.parameters(), 1)
        optimizer.step()
        total_loss += loss.detach()
    return dict(
        complete_value_targets=len(targets),
        updates=updates,
        mean_value_mse=float(total_loss / updates),
        elapsed_s=time.monotonic() - start,
        actor_updated=False,
        rollout_transitions=steps * env.count,
        **rollout_fingerprints,
    )


def ppo_train(
    model,
    env,
    *,
    updates,
    horizon,
    batch_size,
    learning_rate,
    log,
    selected_callback,
    target_kl=None,
    validation_interval=5,
):
    """On-device PPO with proper terminal and time-limit bootstrap separation."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    actor_parameters = [*model.actor.parameters(), model.log_std]
    critic_parameters = list(model.critic.parameters())

    def gradient_norm(parameters):
        return torch.linalg.vector_norm(
            torch.stack(
                [
                    torch.linalg.vector_norm(parameter.grad.detach())
                    for parameter in parameters
                    if parameter.grad is not None
                ]
            )
        )
    observation = env.observation()
    transitions = 0
    optimizer_steps = 0
    for update in range(updates):
        _PHASE_PROGRESS.update(
            phase="ppo", completed_updates=update, ppo_transitions=transitions
        )
        _check_stop()
        buffers = {
            key: []
            for key in (
                "obs",
                "latent",
                "logp",
                "values",
                "rewards",
                "done",
                "next_values",
            )
        }
        episode_count = torch.zeros((), dtype=torch.long, device=env.device)
        successes = torch.zeros_like(episode_count)
        start = time.monotonic()
        for _ in range(horizon):
            with torch.no_grad():
                action, latent, logp, value = model.sample(observation)
                next_observation, reward, terminated, truncated, info = env.step(action)
                next_value = model.critic(next_observation).squeeze(-1)
                next_value = torch.where(terminated, 0, next_value)
                for key, val in (
                    ("obs", observation),
                    ("latent", latent),
                    ("logp", logp),
                    ("values", value),
                    ("rewards", reward),
                    ("done", terminated | truncated),
                    ("next_values", next_value),
                ):
                    buffers[key].append(val)
                done = terminated | truncated
                episode_count += done.sum()
                successes += (done & info["success"]).sum()
                observation = env.reset(done) if bool(done.any()) else next_observation
        b = {key: torch.stack(val) for key, val in buffers.items()}
        advantage = torch.zeros_like(b["rewards"])
        gae = torch.zeros(env.count, device=env.device)
        for step in reversed(range(horizon)):
            delta = (
                b["rewards"][step]
                + env.gamma * b["next_values"][step]
                - b["values"][step]
            )
            gae = delta + env.gamma * 0.95 * (~b["done"][step]).float() * gae
            advantage[step] = gae
        returns = advantage + b["values"]
        td_error = b["rewards"] + env.gamma * b["next_values"] - b["values"]
        return_variance = returns.var(unbiased=False)
        explained_variance = (
            1 - (returns - b["values"]).var(unbiased=False) / return_variance
            if bool(return_variance > 0)
            else torch.full((), float("nan"), device=env.device)
        )
        action_saturation = (b["latent"].tanh().abs() >= 0.98).float().mean()
        flat = {key: val.flatten(0, 1) for key, val in b.items()}
        advantage = advantage.flatten()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        returns = returns.flatten()
        total_loss = torch.zeros((), device=env.device)
        total_kl = torch.zeros_like(total_loss)
        total_clipped = torch.zeros_like(total_loss)
        maximum_kl = torch.zeros_like(total_loss)
        total_actor_loss = torch.zeros_like(total_loss)
        total_critic_loss = torch.zeros_like(total_loss)
        total_grad_norm = torch.zeros_like(total_loss)
        total_actor_grad_norm = torch.zeros_like(total_loss)
        total_critic_grad_norm = torch.zeros_like(total_loss)
        checked_batches = 0
        loss_count = 0
        early_stopped = False
        for epoch in range(4):
            permutation = torch.randperm(len(returns), device=env.device)
            for selected in permutation.split(batch_size):
                distribution = model.distribution(flat["obs"][selected])
                action = flat["latent"][selected].tanh()
                logp = (
                    distribution.log_prob(flat["latent"][selected])
                    - (1 - action.square() + 1e-6).log()
                ).sum(-1)
                log_ratio = logp - flat["logp"][selected]
                ratio = log_ratio.exp()
                approximate_kl = ((ratio - 1) - log_ratio).detach().mean()
                total_kl += approximate_kl
                maximum_kl = torch.maximum(maximum_kl, approximate_kl)
                total_clipped += ((ratio - 1).abs() > 0.2).float().mean().detach()
                checked_batches += 1
                # Check before applying another optimizer step. Waiting for a
                # complete epoch permits 64 updates at the production batch size
                # and allowed the higher-rate pilot to overshoot substantially.
                if target_kl is not None:
                    measured_kl = float(approximate_kl)
                    if not math.isfinite(measured_kl):
                        raise FloatingPointError(
                            "PPO update produced a nonfinite KL divergence"
                        )
                    if measured_kl > target_kl:
                        early_stopped = True
                        break
                actor_loss = -torch.minimum(
                    ratio * advantage[selected],
                    ratio.clamp(0.8, 1.2) * advantage[selected],
                ).mean()
                value = model.critic(flat["obs"][selected]).squeeze(-1)
                value_loss = nn.functional.mse_loss(value, returns[selected])
                loss = actor_loss + 0.5 * value_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                actor_grad_norm = gradient_norm(actor_parameters)
                critic_grad_norm = gradient_norm(critic_parameters)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                total_loss += loss.detach()
                total_actor_loss += actor_loss.detach()
                total_critic_loss += value_loss.detach()
                total_grad_norm += grad_norm.detach()
                total_actor_grad_norm += actor_grad_norm
                total_critic_grad_norm += critic_grad_norm
                loss_count += 1
                optimizer_steps += 1
            if not math.isfinite(float(maximum_kl)):
                raise FloatingPointError(
                    "PPO update produced a nonfinite KL divergence"
                )
            if early_stopped:
                break
        if not loss_count:
            raise FloatingPointError(
                "PPO rejected the unchanged rollout policy before its first step"
            )
        transitions += horizon * env.count
        _PHASE_PROGRESS.update(
            phase="ppo",
            completed_updates=update + 1,
            ppo_transitions=transitions,
            cumulative_optimizer_steps=optimizer_steps,
        )
        _check_stop()
        log(
            "ppo",
            update=update + 1,
            transitions=transitions,
            completed_episodes=int(episode_count),
            completed_successes=int(successes),
            loss=float(total_loss / loss_count),
            actor_loss=float(total_actor_loss / loss_count),
            critic_loss=float(total_critic_loss / loss_count),
            gradient_norm_before_clip=float(total_grad_norm / loss_count),
            actor_gradient_norm_before_clip=float(total_actor_grad_norm / loss_count),
            critic_gradient_norm_before_clip=float(total_critic_grad_norm / loss_count),
            explained_variance=float(explained_variance),
            mean_td_error=float(td_error.mean()),
            mean_absolute_td_error=float(td_error.abs().mean()),
            action_tanh_saturation_fraction=float(action_saturation),
            cumulative_optimizer_steps=optimizer_steps,
            approximate_kl=float(total_kl / checked_batches),
            maximum_minibatch_kl=float(maximum_kl),
            clip_fraction=float(total_clipped / checked_batches),
            checked_minibatches=checked_batches,
            optimization_minibatches=loss_count,
            optimization_epochs=epoch + 1,
            kl_early_stop=early_stopped,
            kl_check="before_minibatch_optimizer_step",
            action_std=model.log_std.detach().exp().cpu().tolist(),
            elapsed_s=time.monotonic() - start,
        )
        if (update + 1) % validation_interval == 0 or update + 1 == updates:
            selected_callback(transitions)
    return transitions


def cpu_parity(
    model,
    *,
    envelope,
    seed,
    count=8,
    steps=120,
    physics_dt=0.25,
    device="cpu",
    deadline_mode="truncate",
    timeout_penalty=0.0,
):
    """Independent scalar RK4 implementation, same frozen craft/actions."""
    env = make_environment(
        count,
        seed=seed,
        envelope=envelope,
        physics_dt=physics_dt,
        device=device,
        deadline_mode=deadline_mode,
        timeout_penalty=timeout_penalty,
    )
    references = [env.cpu_reference(index) for index in range(count)]
    completed = [False] * count
    max_state_error, max_observation_error = 0.0, 0.0
    executed_steps = 0
    for _ in range(steps):
        active_before = [not done for done in completed]
        with torch.no_grad():
            action = model(env.observation().cpu()).cpu()
        for index, reference in enumerate(references):
            if not completed[index]:
                _, _, terminated, truncated, _ = reference.step(action[index].numpy())
                completed[index] = terminated or truncated
        env.step(action.to(device))
        executed_steps += 1
        for index, reference in enumerate(references):
            if not active_before[index]:
                continue
            expected = np.array([reference.x, reference.y, reference.vx, reference.vy])
            max_state_error = max(
                max_state_error,
                float(np.max(np.abs(env.state[index].cpu().numpy() - expected))),
            )
            max_observation_error = max(
                max_observation_error,
                float(
                    np.max(
                        np.abs(
                            env.observation()[index].cpu().numpy()
                            - scalar_observation(
                                reference, observations=model.observations
                            )
                        )
                    )
                ),
            )
        if all(completed):
            break
        # Stop completed GPU worlds at the exact terminal state; their scalar
        # references correctly reject stepping an already ended episode.
        for index, is_done in enumerate(completed):
            if is_done:
                env.time[index] = env.max_time
    return dict(
        seed=seed,
        crafts=count,
        control_steps=executed_steps,
        requested_control_steps=steps,
        backend_device=str(device),
        physics_dt=physics_dt,
        maximum_state_absolute_error_si=max_state_error,
        maximum_observation_absolute_error=max_observation_error,
        passed=max_state_error < 1e-5 and max_observation_error < 1e-5,
        scope="same actions and frozen craft, separate scalar RK4 implementation",
    )


def evaluate_captured_transfer(
    model, *, count=8, seed=3000000, deadline_mode="truncate", timeout_penalty=0.0
):
    """Real captured craft coefficients in the scalar surrogate, never KSP proof.

    Its fitted angle-dependent drag and lift are intentionally retained. They
    are outside the new batched model's constant-CdA physics and are an external
    realism stress, not a nominal score or a checkpoint-selection dataset.
    """
    config = EnvConfig.load(
        Path(__file__).with_name("configs") / "orbiter_rl_transfer.json"
    )
    config = replace(config, orbit_tolerance=20000, max_time=1600)
    results = {}
    for kind in ("policy", "teacher"):
        rows = []
        env = HeadlessOrbiterEnv(config)
        for episode in range(count):
            env.reset(seed=seed + episode)
            for _ in range(math.ceil(config.max_time / config.control_dt) + 1):
                _check_stop()
                with torch.no_grad():
                    action = (
                        expert_action(env)
                        if kind == "teacher"
                        else model(
                            torch.from_numpy(
                                scalar_observation(env, observations=model.observations)
                            )[None]
                        )[0].numpy()
                    )
                _, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    rows.append(
                        dict(
                            seed=seed + episode,
                            success=info["success"],
                            reason=info["termination_reason"],
                            periapsis=info["periapsis"],
                            apoapsis=info["apoapsis"],
                            time=info["time"],
                            # The scalar reference retains its original API;
                            # objective semantics are explicitly annotated.
                            deadline_mode=deadline_mode,
                            timeout_penalty=timeout_penalty,
                        )
                    )
                    break
        successes = sum(row["success"] for row in rows)
        results[kind] = dict(episodes=len(rows), successes=successes, results=rows)
    return dict(
        scope="captured Transfer craft in independent scalar surrogate only",
        retained_aerodynamic_coefficients=dict(
            drag_area=config.drag_area,
            drag_alpha_area=config.drag_alpha_area,
            lift_area=config.lift_area,
        ),
        outside_batched_constant_cda_physics=model.observations == OBSERVATIONS,
        objective_comparison="physical trajectory and terminal success only; scalar reward unchanged",
        checkpoint_selection=False,
        live_ksp_validated=False,
        **results,
    )


def experiment_contract(
    curriculum, physics="constant-cda", *, deadline_mode="truncate", timeout_penalty=0.0
):
    """Preserve v3/v4 defaults; finite mission deadlines have a distinct v5 task."""
    if deadline_mode not in ("truncate", "terminal"):
        raise ValueError("deadline mode must be truncate or terminal")
    if not math.isfinite(timeout_penalty) or timeout_penalty > 0:
        raise ValueError("timeout penalty must be finite and nonpositive")
    if deadline_mode == "truncate" and timeout_penalty != 0:
        raise ValueError("truncation cannot have a terminal timeout penalty")
    envelope = VehicleEnvelope(stages_min=2, stages_max=3)
    if curriculum == "stock":
        envelope = replace(
            envelope,
            stage_wet_to_upper_min=1.0,
            stage_wet_to_upper_max=6.0,
            stage_dry_fraction_min=0.08,
            stage_dry_fraction_max=0.30,
            initial_twr_min=1.15,
            initial_twr_max=2.5,
            upper_twr_min=0.6,
            upper_twr_max=3.0,
            isp_min=280,
            isp_max=380,
            sea_level_isp_ratio_min=0.2,
            sea_level_isp_ratio_max=1.0,
        )
    if physics == "aero":
        from .generalized_aero_env import (
            AeroVehicleEnvelope,
            TASK as AERO_TASK,
            OBSERVATIONS as AERO_OBSERVATIONS,
            FEATURE_SCHEMA as AERO_FEATURE_SCHEMA,
        )

        envelope = AeroVehicleEnvelope(**{**asdict(envelope), "stages_max": 4})
        task, observations, schema = AERO_TASK, AERO_OBSERVATIONS, AERO_FEATURE_SCHEMA
    elif physics == "constant-cda":
        task, observations, schema = TASK, OBSERVATIONS, FEATURE_SCHEMA
    else:
        raise ValueError("unknown ascent physics profile")
    four_stage = replace(envelope, stages_min=4, stages_max=4)
    contract = dict(
        task=task,
        observations=observations,
        feature_schema=schema,
        actions=2,
        train_envelope=asdict(envelope),
        heldout_stage_envelope=asdict(four_stage),
        physics_dt=0.25,
        control_dt=2.0,
        gamma=0.999,
        curriculum=curriculum,
        scope="Kerbin planar 2-3-stage liquid training; 4-stage heldout; not approved for KSP",
        limitations=[
            "No arbitrary staging graphs",
            "No SRB or airbreathing engines",
            "No crossfeed/asparagus",
            "No flex/aerodynamic stability simulation",
            "Attitude setpoint servo approximation",
            "Ideal delta-v screen is not finite-burn reachability proof",
        ],
    )
    if physics == "aero":
        contract.pop("heldout_stage_envelope")
        contract.update(
            four_stage_test_envelope=asdict(four_stage),
            four_stage_is_heldout=False,
            max_time=1600.0,
            orbit_tolerance=20000.0,
            mu=3.5316e12,
            radius=600000.0,
            rotation_period=21549.425,
            atmosphere_depth=70000.0,
            atmosphere=[list(row) for row in EnvConfig().atmosphere],
            physics="planar_angle_dependent_effective_areas",
            scope="Kerbin planar 2-4-stage liquid training; fresh-craft tests; not approved for KSP",
        )
        contract["limitations"].append(
            "No Mach-, part- or stage-dependent aerodynamic coefficients"
        )
    if deadline_mode == "terminal":
        contract.update(
            task=AERO_DEADLINE_TASK if physics == "aero" else DEADLINE_TASK,
            objective_version=1,
            objective=dict(
                deadline_mode=deadline_mode,
                timeout_penalty=float(timeout_penalty),
                max_time=1600.0,
                deadline_bootstrap=False,
                terminal_potential=0.0,
                task_reward="physical_terminal_plus_time_plus_deadline_penalty",
                shaping="gamma_phi_next_minus_phi_with_zero_terminal_potential",
            ),
        )
    return envelope, four_stage, contract


def main(argv=None):
    global _STOP_SIGNAL, _STOP_CHECKPOINT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--validation-seed", type=int)
    parser.add_argument("--test-seed", type=int)
    parser.add_argument("--worlds", type=int, default=1024)
    parser.add_argument("--label-steps", type=int, default=600)
    parser.add_argument("--bc-updates", type=int, default=2500)
    parser.add_argument("--dagger-rounds", type=int, default=5)
    parser.add_argument("--ppo-updates", type=int, default=50)
    parser.add_argument("--value-updates", type=int, default=5000)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--eval-worlds", type=int, default=128)
    parser.add_argument("--evaluate", type=Path)
    parser.add_argument(
        "--warm-start",
        "--resume",
        dest="warm_start",
        type=Path,
        help="Warm restart a matching checkpoint with fresh optimizers/worlds; not bitwise replay",
    )
    parser.add_argument(
        "--transfer-from",
        type=Path,
        help="Initialize v4 aero from v3 weights with zero new input columns",
    )
    parser.add_argument(
        "--actor-from",
        type=Path,
        help="Copy only a strict v3/v4 actor and log-std; initialize a fresh seeded critic",
    )
    parser.add_argument(
        "--deadline-mode", choices=("truncate", "terminal"), default="truncate"
    )
    parser.add_argument("--timeout-penalty", type=float, default=0.0)
    parser.add_argument(
        "--evaluation-suite", choices=("full", "ablation"), default="full"
    )
    parser.add_argument(
        "--ppo-only",
        action="store_true",
        help="Continue initialized weights directly with PPO; skip imitation",
    )
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--physics", choices=("constant-cda", "aero"), default="constant-cda"
    )
    parser.add_argument(
        "--curriculum", choices=("efficient", "stock"), default="efficient"
    )
    args = parser.parse_args(argv)
    if (
        min(
            args.worlds,
            args.label_steps,
            args.bc_updates,
            args.horizon,
            args.batch_size,
            args.hidden,
            args.eval_worlds,
            args.validation_interval,
        )
        < 1
    ):
        parser.error("counts must be positive")
    if args.dagger_rounds < 0 or args.ppo_updates < 0 or args.value_updates < 0:
        parser.error("training rounds cannot be negative")
    if not math.isfinite(args.target_kl) or args.target_kl <= 0:
        parser.error("target KL must be positive and finite")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning rate must be positive and finite")
    if (
        sum(
            bool(item)
            for item in (args.evaluate, args.warm_start, args.transfer_from, args.actor_from)
        )
        > 1
    ):
        parser.error(
            "evaluation, warm start, transfer and actor initialization are mutually exclusive"
        )
    if args.transfer_from and args.physics != "aero":
        parser.error("transfer initialization requires --physics aero")
    if args.actor_from and not args.ppo_only:
        parser.error("actor-only initialization requires --ppo-only")
    if args.transfer_from and args.deadline_mode != "truncate":
        parser.error("a new objective requires --actor-from with a fresh critic")
    if args.ppo_only and not (args.warm_start or args.transfer_from or args.actor_from):
        parser.error("PPO-only training requires initialized checkpoint weights")
    if args.ppo_only and not args.ppo_updates:
        parser.error("PPO-only training requires positive PPO updates")
    try:
        envelope, holdout, contract = experiment_contract(
            args.curriculum,
            args.physics,
            deadline_mode=args.deadline_mode,
            timeout_penalty=args.timeout_penalty,
        )
    except ValueError as error:
        parser.error(str(error))
    validation_seed = (
        args.validation_seed if args.validation_seed is not None else args.seed + 500000
    )
    test_seed = args.test_seed if args.test_seed is not None else args.seed + 1000000
    if validation_seed == test_seed:
        parser.error("checkpoint selection and final test seeds must differ")
    environment_objective = dict(
        deadline_mode=args.deadline_mode, timeout_penalty=args.timeout_penalty
    )
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        parser.error("run directory must be empty; warm restart into a new directory")
    initial_model = initial_record = None
    if args.evaluate:
        initial_model, initial_record = load_policy(args.evaluate, device=args.device)
        if initial_record["contract"] != contract:
            parser.error(
                "evaluation checkpoint contract differs from CLI curriculum; choose its original curriculum"
            )
    elif args.warm_start:
        initial_model, initial_record = load_policy(args.warm_start, device=args.device)
        if initial_record["contract"] != contract:
            parser.error("warm-start task/environment contract differs")
    elif args.transfer_from:
        initial_model, initial_record = transfer_aero_policy(
            args.transfer_from, device=args.device
        )
        _, _, source_contract = experiment_contract(args.curriculum)
        if initial_record["contract"] != source_contract:
            parser.error(
                "transfer source must match the original v3 curriculum contract"
            )
    elif args.actor_from:
        try:
            initial_model, initial_record = initialize_actor_policy(
                args.actor_from, contract, seed=args.seed, device=args.device
            )
        except ValueError as error:
            parser.error(str(error))
    args.run_dir.mkdir(parents=True, exist_ok=True)
    events = (args.run_dir / "events.jsonl").open("a", buffering=1)

    def log(event, **values):
        row = clean_json(dict(event=event, time_unix=time.time(), **values))
        events.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    write_json(args.run_dir / "contract.json", contract)
    audit_fn = normalization_audit
    if args.physics == "aero":
        from .generalized_aero_env import normalization_audit as audit_fn
    audit = audit_fn(envelope)
    if not audit["all_craft_and_actuator_features_below_clip"]:
        parser.error(
            "declared vehicle context can saturate its observation normalization"
        )
    write_json(args.run_dir / "normalization_audit.json", audit)
    write_json(
        args.run_dir / "runtime.json",
        dict(
            python=platform.python_version(),
            torch=torch.__version__,
            numpy=np.__version__,
            device=args.device,
            gpu=torch.cuda.get_device_name(args.device)
            if args.device.startswith("cuda")
            else None,
            seed=args.seed,
            args={
                key: str(val) if isinstance(val, Path) else val
                for key, val in vars(args).items()
            },
        ),
    )
    model = VehiclePolicy(args.hidden, observations=contract["observations"]).to(
        args.device
    )
    if initial_model is not None:
        model = initial_model
    ancestry = dict(
        algorithm="ppo_from_checkpoint" if args.ppo_only else "bc_dagger_then_ppo",
        bc_transitions=0,
        dagger_transitions=0,
        ppo_transitions=0,
        seed=args.seed,
        selected_phase="initial",
        selection_validation_seed=validation_seed,
        final_test_seed=test_seed,
        ppo_optimization=dict(
            learning_rate=args.learning_rate,
            target_kl=args.target_kl,
            kl_check="before_minibatch_optimizer_step",
            maximum_epochs=4,
            batch_size=args.batch_size,
            worlds=args.worlds,
            horizon=args.horizon,
        ),
    )

    def save_interrupted():
        progress = dict(_PHASE_PROGRESS)
        if progress.get("phase") == "ppo":
            ancestry["ppo_transitions"] = progress.get("ppo_transitions", 0)
        saved_ancestry = {
            **ancestry,
            "interrupted": True,
            "current_phase": progress.get("phase", "initial"),
            "phase_progress": progress,
            "resume_mode": "warm_restart_fresh_optimizers_and_worlds",
        }
        checkpoint = args.run_dir / "latest.pt"
        save_policy(checkpoint, model, contract, saved_ancestry)
        write_json(
            args.run_dir / "interrupted.json",
            dict(
                signal=_STOP_SIGNAL,
                checkpoint="latest.pt",
                best_checkpoint="best.pt",
                progress=progress,
                resume_mode="warm_restart_fresh_optimizers_and_worlds",
            ),
        )
        log(
            "interrupted_checkpoint",
            checkpoint="latest.pt",
            signal=_STOP_SIGNAL,
            **progress,
        )
        events.close()
        raise SystemExit(0)

    if not args.evaluate:
        _STOP_SIGNAL = None
        _STOP_CHECKPOINT = save_interrupted
        signal.signal(signal.SIGTERM, _request_stop)
        signal.signal(signal.SIGINT, _request_stop)
    if args.warm_start or args.transfer_from or args.actor_from:
        source = args.warm_start or args.transfer_from or args.actor_from
        ancestry["upstream"] = dict(
            sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            ancestry=initial_record["ancestry"],
            source_contract=initial_record["contract"],
            initialization=(
                "actor_only_fresh_seeded_critic"
                if args.actor_from
                else "zero_padded_v3_to_v4_inputs"
                if args.transfer_from
                else "matching_contract_warm_restart"
            ),
            optimizer_state_restored=False,
            critic_restored=not bool(args.actor_from),
        )
        if args.actor_from:
            ancestry["initialization"] = dict(
                actor_copied=True,
                log_std_copied=True,
                critic="fresh_seeded",
                critic_seed=args.seed,
                critic_sha256=hashlib.sha256(
                    b"".join(
                        value.detach().cpu().numpy().tobytes()
                        for value in model.critic.state_dict().values()
                    )
                ).hexdigest(),
                value_fit_seed=args.seed + 6000000,
                ppo_sampling_seed=args.seed + 7000000,
                optimizer_state_restored=False,
            )
    if args.evaluate:
        ancestry = initial_record["ancestry"]
    else:
        env = make_environment(
            args.worlds,
            device=args.device,
            seed=args.seed,
            envelope=envelope,
            **environment_objective,
        )
        optimizer = torch.optim.Adam(model.actor.parameters(), lr=3e-4)
        best_score = -1.0

        def select(phase, ppo_steps=0):
            nonlocal best_score
            ancestry["ppo_transitions"] = ppo_steps
            ancestry["ppo_optimizer_steps"] = (
                _PHASE_PROGRESS.get("cumulative_optimizer_steps", 0) if ppo_steps else 0
            )
            result = evaluate(
                model,
                count=args.eval_worlds,
                device=args.device,
                seed=validation_seed,
                envelope=envelope,
                **environment_objective,
            )
            write_json(args.run_dir / f"validation_{phase}.json", result)
            log(
                "validation",
                phase=phase,
                successes=result["successes"],
                episodes=result["episodes"],
            )
            is_selected = result["success_rate"] > best_score
            if is_selected:
                best_score = result["success_rate"]
                ancestry["selected_phase"] = phase
                save_policy(args.run_dir / "best.pt", model, contract, dict(ancestry))
            save_policy(
                args.run_dir / "latest.pt",
                model,
                contract,
                {**ancestry, "current_phase": phase, "selected": is_selected},
            )

        if args.warm_start or args.transfer_from or args.actor_from:
            select("warm_start")
        if not args.ppo_only:
            original_obs, original_actions, summary = collect_labels(
                model, env, steps=args.label_steps, expert_mixing=1.0
            )
            ancestry["bc_transitions"] = summary["transitions"]
            log("teacher_collection", **summary)
            loss = fit_actor(
                model,
                original_obs,
                original_actions,
                updates=args.bc_updates,
                batch_size=args.batch_size,
                optimizer=optimizer,
            )
            log("behavior_clone", updates=args.bc_updates, final_action_mse=loss)
            select("bc")
            for round_index in range(args.dagger_rounds):
                mixing = (
                    0.5 ** (round_index + 1)
                    if round_index < args.dagger_rounds - 1
                    else 0.0
                )
                env.reset()
                obs, actions, summary = collect_labels(
                    model, env, steps=args.label_steps, expert_mixing=mixing
                )
                ancestry["dagger_transitions"] += summary["transitions"]
                log("dagger_collection", round=round_index + 1, **summary)
                keep = torch.randperm(len(original_obs), device=args.device)[
                    : min(len(obs), len(original_obs))
                ]
                loss = fit_actor(
                    model,
                    torch.cat((original_obs[keep], obs)),
                    torch.cat((original_actions[keep], actions)),
                    updates=args.bc_updates,
                    batch_size=args.batch_size,
                    optimizer=optimizer,
                )
                log("dagger_fit", round=round_index + 1, action_mse=loss)
                select(f"dagger_{round_index + 1}")
        if args.ppo_updates:
            if not args.ppo_only:
                del original_obs, original_actions
                if args.dagger_rounds:
                    del obs, actions, keep
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
            model, record = load_policy(args.run_dir / "best.pt", device=args.device)
            ancestry = record["ancestry"]
            if args.value_updates:
                if args.actor_from:
                    # Paired arms use the same deterministic actor/world rollout
                    # and identically seeded critic minibatch sampling.
                    torch.manual_seed(args.seed + 6000000)
                value_summary = warm_value(
                    model,
                    env,
                    steps=args.label_steps,
                    updates=args.value_updates,
                    batch_size=args.batch_size,
                )
                ancestry["value_warm_start"] = value_summary
                log("value_warm_start", **value_summary)
            env.reset()
            if args.actor_from:
                torch.manual_seed(args.seed + 7000000)
            ppo_train(
                model,
                env,
                updates=args.ppo_updates,
                horizon=args.horizon,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                log=log,
                selected_callback=lambda steps: select(f"ppo_{steps}", steps),
                target_kl=args.target_kl,
                validation_interval=args.validation_interval,
            )
        model, record = load_policy(args.run_dir / "best.pt", device=args.device)
        ancestry = record["ancestry"]
    # Final evaluations have fresh seeds and cannot select a new checkpoint.
    final = {}
    cases = (
        ("nominal", envelope, args.eval_worlds * 2),
        (
            "four_stage_fresh_crafts"
            if args.physics == "aero"
            else "heldout_four_stage",
            holdout,
            args.eval_worlds * 2,
        ),
        (
            "heldout_single_stage",
            replace(
                envelope,
                stages_min=1,
                stages_max=1,
                stage_wet_to_upper_min=8.0,
                stage_wet_to_upper_max=20.0,
            ),
            args.eval_worlds,
        ),
        (
            "scale_invariance",
            replace(envelope, payload_min=15000, payload_max=60000),
            args.eval_worlds,
        ),
        (
            "stress_lower_twr",
            replace(
                envelope,
                initial_twr_min=1.15,
                initial_twr_max=1.35,
                upper_twr_min=0.65,
                upper_twr_max=0.9,
            ),
            args.eval_worlds,
        ),
        (
            "heldout_high_drag",
            replace(
                envelope, drag_area_per_tonne_min=0.18, drag_area_per_tonne_max=0.4
            ),
            args.eval_worlds,
        ),
        (
            "heldout_slow_attitude",
            replace(
                envelope,
                attitude_rate_min=2,
                attitude_rate_max=4,
                attitude_tau_min=1.5,
                attitude_tau_max=2.5,
            ),
            args.eval_worlds,
        ),
    )
    for label, selected_envelope, count in (
        cases[:2] if args.evaluation_suite == "ablation" else cases
    ):
        result = evaluate(
            model,
            count=count,
            device=args.device,
            seed=test_seed,
            envelope=selected_envelope,
            export_crafts=True,
            **environment_objective,
        )
        result["checkpoint_selection"] = False
        write_json(args.run_dir / f"test_{label}.json", result)
        final[label] = {
            key: result[key]
            for key in (
                "episodes",
                "successes",
                "success_rate",
                "success_ci95",
                "failure_counts",
            )
        }
        for key in ("mean_discounted_task_return", "mean_elapsed"):
            if key in result:
                final[label][key] = result[key]
        if args.evaluation_suite == "full":
            teacher = evaluate(
                model,
                count=count,
                device=args.device,
                seed=test_seed,
                envelope=selected_envelope,
                kind="teacher",
                **environment_objective,
            )
            write_json(args.run_dir / f"teacher_{label}.json", teacher)
            final[label]["teacher_successes"] = teacher["successes"]
        log("final_test", case=label, **final[label])
    if args.evaluation_suite == "ablation":
        final.update(
            selected_ancestry=ancestry,
            live_ksp_validated=False,
            evaluation_suite="ablation",
            selection_validation_seed=validation_seed,
            final_test_seed=test_seed,
            final_tests_select_checkpoint=False,
        )
        write_json(args.run_dir / "summary.json", final)
        log("complete", **final)
        events.close()
        return
    # Smaller physical integration timestep, same fresh craft/seed distribution.
    half = evaluate(
        model,
        count=args.eval_worlds,
        device=args.device,
        seed=test_seed,
        envelope=envelope,
        physics_dt=0.125,
        **environment_objective,
    )
    half["comparison"] = (
        "independent distribution robustness check, not paired same-craft numerical convergence"
    )
    write_json(args.run_dir / "test_half_timestep.json", half)
    final["half_timestep"] = {
        key: half[key] for key in ("episodes", "successes", "success_rate")
    }
    cpu_model, _ = load_policy(args.evaluate or args.run_dir / "best.pt", device="cpu")
    parity = cpu_parity(
        cpu_model,
        envelope=envelope,
        seed=test_seed + 1000000,
        **environment_objective,
    )
    write_json(args.run_dir / "cpu_parity.json", parity)
    final["cpu_parity"] = parity
    if args.device.startswith("cuda"):
        gpu_parity = cpu_parity(
            cpu_model,
            envelope=envelope,
            seed=test_seed + 1000001,
            device=args.device,
            **environment_objective,
        )
        write_json(args.run_dir / "gpu_scalar_parity.json", gpu_parity)
        final["gpu_scalar_parity"] = gpu_parity
    captured = evaluate_captured_transfer(
        cpu_model, seed=test_seed + 2000000, **environment_objective
    )
    write_json(args.run_dir / "test_captured_transfer.json", captured)
    final["captured_transfer"] = dict(
        policy_successes=captured["policy"]["successes"],
        teacher_successes=captured["teacher"]["successes"],
        episodes=captured["policy"]["episodes"],
        outside_batched_constant_cda_physics=args.physics == "constant-cda",
    )
    final["selected_ancestry"] = ancestry
    final["live_ksp_validated"] = False
    write_json(args.run_dir / "summary.json", final)
    log("complete", **final)
    events.close()


if __name__ == "__main__":
    main()
