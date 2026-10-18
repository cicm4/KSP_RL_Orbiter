"""Train/evaluate goal-conditioned local orbital capture across KSP bodies.

This skill starts on inbound approaches near a body. Its success rate does not
measure launch, interplanetary navigation, SOI acquisition, or actual KSP flights.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .capture_env import TASK, CaptureConfig, CaptureEnv, capture_teacher
from .system import SystemCatalog
from .train import EventLog, _jsonable, write_json


DEFAULT_TRAIN_BODIES = ("Kerbin", "Mun", "Duna", "Tylo", "Gilly", "Jool", "Moho", "Eve")
DEFAULT_HELDOUT_BODIES = (
    "Minmus",
    "Ike",
    "Pol",
    "Bop",
    "Vall",
    "Laythe",
    "Dres",
    "Eeloo",
)
CONTRACT_FIELDS = (
    "task",
    "algorithm",
    "environment",
    "body_catalog",
    "train_bodies",
    "heldout_bodies",
    "vehicle_label",
)
CONFIG_OVERRIDES = (
    "randomization",
    "max_time",
    "mass",
    "dry_mass",
    "thrust",
    "isp",
    "target_altitude_randomization",
    "arrival_vinf_fraction",
    "arrival_vinf_by_body",
)


def wilson_interval(successes, count, z=1.959963984540054):
    if count <= 0:
        return [0.0, 1.0]
    proportion = successes / count
    denominator = 1 + z * z / count
    center = (proportion + z * z / (2 * count)) / denominator
    half = (
        z
        * math.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count * count))
        / denominator
    )
    return [max(0.0, center - half), min(1.0, center + half)]


def body_sets(catalog, train_bodies=None, heldout_bodies=None):
    train = tuple(DEFAULT_TRAIN_BODIES if train_bodies is None else train_bodies)
    heldout = tuple(
        DEFAULT_HELDOUT_BODIES if heldout_bodies is None else heldout_bodies
    )
    if not train or len(set(train)) != len(train) or len(set(heldout)) != len(heldout):
        raise ValueError("Provide distinct body names and at least one training body")
    if set(train) & set(heldout):
        raise ValueError("Training and held-out body sets must not overlap")
    unknown = (set(train) | set(heldout)) - set(catalog)
    if unknown:
        raise ValueError(f"Body catalog is missing: {sorted(unknown)}")
    if any(catalog[name].get("parent") is None for name in train + heldout):
        raise ValueError(
            "Central stars are not inbound planetary-capture training tasks"
        )
    return train, heldout


def make_contract(
    config,
    catalog,
    train_bodies,
    heldout_bodies,
    vehicle_label="configured_surrogate_vehicle",
):
    return _jsonable(
        dict(
            task=TASK,
            algorithm="ppo",
            environment=asdict(config),
            body_catalog=catalog,
            train_bodies=list(train_bodies),
            heldout_bodies=list(heldout_bodies),
            vehicle_label=vehicle_label,
        )
    )


def save_checkpoint(model, path):
    path = Path(path)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.zip")
    model.save(temporary)
    os.replace(temporary, path)
    write_json(
        path.with_suffix(".metadata.json"),
        {
            **model.orbiter_capture_contract,
            "num_timesteps": model.num_timesteps,
            "saved_at_unix": time.time(),
        },
    )


def validate_contract(model, metadata):
    if metadata.get("task") != TASK or metadata.get("algorithm") != "ppo":
        raise ValueError(f"Checkpoint is not an {TASK} PPO policy")
    supplied = {key: metadata[key] for key in CONTRACT_FIELDS}
    if (
        not hasattr(model, "orbiter_capture_contract")
        or model.orbiter_capture_contract != supplied
    ):
        raise ValueError(
            "Capture checkpoint metadata differs from its embedded task/body/goal contract"
        )
    config = CaptureConfig.from_dict(metadata["environment"])
    if not math.isclose(float(model.gamma), config.gamma, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            "Capture learner gamma differs from its reward shaping discount"
        )
    train, heldout = body_sets(
        metadata["body_catalog"], metadata["train_bodies"], metadata["heldout_bodies"]
    )
    env = CaptureEnv(
        config, body_catalog={name: metadata["body_catalog"][name] for name in train}
    )
    try:
        for name in ("observation_space", "action_space"):
            actual, expected = getattr(model, name), getattr(env, name)
            if (
                type(actual) is not type(expected)
                or actual.shape != expected.shape
                or actual.dtype != expected.dtype
                or not np.array_equal(actual.low, expected.low)
                or not np.array_equal(actual.high, expected.high)
            ):
                raise ValueError(
                    f"Capture checkpoint {name} differs from the goal-conditioned contract"
                )
    finally:
        env.close()
    return config, metadata["body_catalog"], train, heldout


def load_capture_controller(path, *, env=None):
    path = Path(path)
    metadata = json.loads(path.with_suffix(".metadata.json").read_text())
    # Reject a launch checkpoint before deserializing it or opening output files.
    if metadata.get("task") != TASK or metadata.get("algorithm") != "ppo":
        raise ValueError(f"Checkpoint is not an {TASK} PPO policy")
    model = PPO.load(path, env=env, device="cpu")
    validate_contract(model, metadata)
    return model, metadata


def reference_capture_budget(env):
    """Diagnostic impulsive circularization budget, not a reachability proof."""
    if not hasattr(env, "telemetry"):
        return None
    data, config = env.telemetry(), env.config
    periapsis = data["periapsis_radius"]
    incoming_speed = math.sqrt(max(0.0, 2 * (data["energy"] + config.mu / periapsis)))
    circular_speed = math.sqrt(config.mu / periapsis)
    reference_delta_v = abs(incoming_speed - circular_speed)
    available = data["available_delta_v"]
    return dict(
        initial_available_delta_v=available,
        impulsive_circularization_delta_v=reference_delta_v,
        reference_budget_shortfall=available < reference_delta_v,
        note="Ideal impulse at incoming periapsis; finite burn, plane correction and target band alter requirements",
    )


def evaluate(
    model,
    config,
    catalog,
    bodies,
    seeds,
    *,
    kind="policy",
    target_clearance_multiplier=None,
):
    if kind not in {"policy", "teacher", "random"} or not bodies or not seeds:
        raise ValueError(
            "Evaluation needs a controller kind, bodies, and independent seeds"
        )
    per_body = {}
    for body in bodies:
        env = CaptureEnv(config, body_catalog={body: catalog[body]})
        episodes = []
        try:
            for seed in seeds:
                options = {"body": body}
                if target_clearance_multiplier is not None:
                    nominal = CaptureConfig.for_body(catalog[body])
                    # Scale the vacuum clearance above atmosphere, so a lower
                    # Jool/Eve stress remains a valid orbital goal.
                    options["target_altitude"] = (
                        nominal.atmosphere_depth
                        + (nominal.target_altitude - nominal.atmosphere_depth)
                        * target_clearance_multiplier
                    )
                observation, _ = env.reset(seed=seed, options=options)
                budget = reference_capture_budget(env)
                rng = np.random.default_rng(seed + 9187)
                total_return = 0.0
                for step in range(
                    math.ceil(env.config.max_time / env.config.control_dt) + 2
                ):
                    if kind == "teacher":
                        action = capture_teacher(env)
                    elif kind == "random":
                        action = rng.uniform(-1, 1, env.action_space.shape).astype(
                            np.float32
                        )
                    else:
                        action = model.predict(observation, deterministic=True)[0]
                    observation, reward, terminated, truncated, info = env.step(action)
                    total_return += float(reward)
                    if terminated or truncated:
                        break
                else:
                    raise RuntimeError(
                        f"Capture environment did not end within its time bound for {body}"
                    )
                episodes.append(
                    dict(
                        seed=seed,
                        success=bool(info.get("success", False)),
                        return_=total_return,
                        steps=step + 1,
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                        termination_reason=info.get("termination_reason", ""),
                        reference_capture_budget=budget,
                        final_info=info,
                    )
                )
        finally:
            env.close()
        successes = sum(item["success"] for item in episodes)
        failures = Counter(
            item["termination_reason"] or "unspecified"
            for item in episodes
            if not item["success"]
        )
        per_body[body] = dict(
            successes=successes,
            episodes=len(episodes),
            success_rate=successes / len(episodes),
            success_ci95=wilson_interval(successes, len(episodes)),
            failure_counts=dict(failures),
            reference_budget_shortfalls=sum(
                bool(
                    item["reference_capture_budget"]
                    and item["reference_capture_budget"]["reference_budget_shortfall"]
                )
                for item in episodes
            ),
            fuel_exhaustions=sum(
                item["termination_reason"] == "out_of_fuel" for item in episodes
            ),
            mean_return=float(np.mean([item["return_"] for item in episodes])),
            results=episodes,
        )
    count = sum(row["episodes"] for row in per_body.values())
    successes = sum(row["successes"] for row in per_body.values())
    return dict(
        task=TASK,
        scope="local inbound orbital capture surrogate only",
        kind=kind,
        successes=successes,
        episodes=count,
        success_rate=successes / count,
        macro_success_rate=float(
            np.mean([row["success_rate"] for row in per_body.values()])
        ),
        success_ci95=wilson_interval(successes, count),
        per_body=per_body,
        interval_note="Wilson 95% intervals describe sampled randomized episodes, not KSP transfer or end-to-end missions",
    )


def evaluate_stress(model, config, catalog, bodies, seeds):
    """Separately test new goals, faster arrivals, and changed craft parameters."""
    cases = {}
    for multiplier in (0.6, 1.4):
        cases[f"target_clearance_{multiplier:g}x"] = evaluate(
            model,
            config,
            catalog,
            bodies,
            seeds,
            target_clearance_multiplier=multiplier,
        )
    for fraction in (0.2, 0.7):
        cases[f"arrival_vinf_{fraction:g}_circular_speed"] = evaluate(
            model,
            replace(config, arrival_vinf_fraction=fraction, arrival_vinf_by_body={}),
            catalog,
            bodies,
            seeds,
        )
    for mass, thrust_factor, isp_factor in (
        (0.875 * config.mass, 1.0, 1.0),
        (1.03125 * config.mass, 0.8, 0.9),
    ):
        changed = replace(
            config,
            mass=mass,
            thrust=config.thrust * thrust_factor,
            isp=config.isp * isp_factor,
        )
        name = f"craft_mass_{mass:g}_thrust_{thrust_factor:g}_isp_{isp_factor:g}"
        if changed.mass * (1 - changed.randomization) <= changed.dry_mass:
            cases[name] = dict(
                status="infeasible_vehicle_randomization",
                reason="Randomized initial mass range includes values at or below dry mass",
                mass=changed.mass,
                minimum_randomized_mass=changed.mass * (1 - changed.randomization),
                dry_mass=changed.dry_mass,
            )
            continue
        cases[name] = evaluate(model, changed, catalog, bodies, seeds)
    return dict(
        task=TASK,
        scope="independent local capture distribution stresses; no end-to-end or KSP flight claim",
        cases=cases,
    )


def body_balanced_indices(labels, count, rng):
    """Keep long low-gravity captures from dominating multi-body imitation."""
    labels = np.asarray(labels)
    names = np.unique(labels)
    if not len(names):
        raise ValueError("Cannot sample an empty capture dataset")
    indices = []
    for index, body in enumerate(names):
        size = count // len(names) + (index < count % len(names))
        candidates = np.flatnonzero(labels == body)
        indices.extend(rng.choice(candidates, size=size, replace=True).tolist())
    return rng.permutation(indices)


def behavior_clone(
    model, config, catalog, train_bodies, *, episodes, seed, epochs, value_epochs, log
):
    """Stratified teacher data from training bodies only; no held-out gradients."""
    env = CaptureEnv(
        config, body_catalog={name: catalog[name] for name in train_bodies}
    )
    observations, actions, observation_bodies, value_observations, targets = (
        [],
        [],
        [],
        [],
        [],
    )
    teacher_results = Counter()
    try:
        for episode in range(episodes):
            body = train_bodies[episode % len(train_bodies)]
            observation, _ = env.reset(
                seed=seed + 10_000 + episode, options={"body": body}
            )
            trajectory = []
            for _ in range(math.ceil(env.config.max_time / env.config.control_dt) + 2):
                action = np.asarray(capture_teacher(env), dtype=np.float32)
                next_observation, reward, terminated, truncated, info = env.step(action)
                observations.append(observation.copy())
                actions.append(action.copy())
                observation_bodies.append(body)
                trajectory.append((observation.copy(), float(reward)))
                observation = next_observation
                if terminated or truncated:
                    teacher_results[(body, bool(info.get("success", False)))] += 1
                    # Truncations have an unknown bootstrap value. Fit only
                    # complete terminal Monte Carlo returns for PPO warmstart.
                    if terminated:
                        value = 0.0
                        for old_observation, reward in reversed(trajectory):
                            value = reward + config.gamma * value
                            value_observations.append(old_observation)
                            targets.append(value)
                    break
            else:
                raise RuntimeError(
                    "Capture teacher episode exceeded the environment bound"
                )
    finally:
        env.close()
    obs = torch.as_tensor(np.asarray(observations), device=model.device)
    target = torch.as_tensor(np.asarray(actions), device=model.device)
    actor_parameters = list(model.policy.mlp_extractor.policy_net.parameters()) + list(
        model.policy.action_net.parameters()
    )
    optimizer = torch.optim.Adam(actor_parameters, lr=1e-3)
    rng = np.random.default_rng(seed)
    final_loss = None
    model.policy.set_training_mode(True)
    for _ in range(epochs):
        for _ in range(max(1, math.ceil(len(obs) / 256))):
            selected = body_balanced_indices(
                observation_bodies, min(256, len(obs)), rng
            )
            prediction = model.policy.get_distribution(obs[selected]).distribution.mean
            loss = torch.nn.functional.mse_loss(prediction, target[selected])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_parameters, 1)
            optimizer.step()
            final_loss = float(loss.item())
    value_mse = None
    if value_epochs and targets:
        value_obs = torch.as_tensor(np.asarray(value_observations), device=model.device)
        returns = torch.as_tensor(
            np.asarray(targets), dtype=torch.float32, device=model.device
        )
        value_parameters = list(
            model.policy.mlp_extractor.value_net.parameters()
        ) + list(model.policy.value_net.parameters())
        value_optimizer = torch.optim.Adam(value_parameters, lr=1e-3)
        for _ in range(value_epochs):
            indices = rng.permutation(len(returns))
            for start in range(0, len(indices), 256):
                selected = indices[start : start + 256]
                loss = torch.nn.functional.mse_loss(
                    model.policy.predict_values(value_obs[selected]).flatten(),
                    returns[selected],
                )
                value_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(value_parameters, 1)
                value_optimizer.step()
        with torch.no_grad():
            value_mse = float(
                torch.nn.functional.mse_loss(
                    model.policy.predict_values(value_obs).flatten(), returns
                ).item()
            )
    model.policy.set_training_mode(False)
    results = {
        body: dict(
            successes=teacher_results[(body, True)],
            failures=teacher_results[(body, False)],
        )
        for body in train_bodies
    }
    log.write(
        "behavior_clone",
        bodies=list(train_bodies),
        transitions=len(observations),
        episodes=episodes,
        epochs=epochs,
        final_action_mse=final_loss,
        complete_value_targets=len(targets),
        value_epochs=value_epochs,
        final_value_mse=value_mse,
        body_balanced_batches=True,
        per_body=results,
    )
    fit_summary = (
        "labels collected without actor updates"
        if final_loss is None
        else f"MSE {final_loss:.5f}"
    )
    print(
        f"Capture BC: {len(observations):,} transitions; {fit_summary}; bodies {results}",
        flush=True,
    )
    return np.asarray(observations), np.asarray(actions), np.asarray(observation_bodies)


def dagger(
    model,
    config,
    catalog,
    train_bodies,
    demonstrations,
    *,
    rounds,
    episodes,
    epochs,
    beta,
    seed,
    validation_seeds,
    run_dir,
    log,
    minimum_success=1.0,
):
    """Label learner-visited states while retaining balanced original demonstrations.

    Expert mixing decays to zero; learner actions are stochastic so the state
    distribution includes the exploration that PPO will actually encounter.
    Held-out bodies are never queried for labels or checkpoint selection.
    """
    original_obs, original_actions = demonstrations[:2]
    original_bodies = (
        demonstrations[2]
        if len(demonstrations) > 2
        else np.full(len(original_obs), train_bodies[0])
    )
    if (
        not len(original_obs)
        or not 0 <= beta <= 1
        or min(rounds, episodes, epochs) <= 0
    ):
        raise ValueError("DAgger requires original demonstrations and beta in [0, 1]")
    device = model.device
    demo_obs = torch.as_tensor(original_obs, device=device)
    demo_actions = torch.as_tensor(original_actions, device=device)
    aggregated_obs, aggregated_actions, aggregated_bodies = [], [], []
    actor_parameters = list(model.policy.mlp_extractor.policy_net.parameters()) + list(
        model.policy.action_net.parameters()
    )
    optimizer = torch.optim.Adam(actor_parameters, lr=3e-4)
    rng = np.random.default_rng(seed + 7831)
    best_score = (-1.0, -math.inf)
    best_path = Path(run_dir) / "dagger_best.zip"
    env = CaptureEnv(
        config, body_catalog={name: catalog[name] for name in train_bodies}
    )
    try:
        for round_index in range(rounds):
            mixing = 0.0 if round_index == rounds - 1 else beta * (0.5**round_index)
            rollout_results = Counter()
            model.policy.set_training_mode(False)
            for episode in range(episodes):
                body = train_bodies[episode % len(train_bodies)]
                rollout_seed = seed + 30_000 + round_index * episodes + episode
                observation, _ = env.reset(seed=rollout_seed, options={"body": body})
                for _ in range(
                    math.ceil(env.config.max_time / env.config.control_dt) + 2
                ):
                    label = np.asarray(capture_teacher(env), dtype=np.float32)
                    aggregated_obs.append(observation.copy())
                    aggregated_actions.append(label.copy())
                    aggregated_bodies.append(body)
                    learner_action = model.predict(observation, deterministic=False)[0]
                    action = label if rng.random() < mixing else learner_action
                    observation, _, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        rollout_results[(body, bool(info.get("success", False)))] += 1
                        break
                else:
                    raise RuntimeError("DAgger rollout exceeded the capture time bound")
            recovery_obs = torch.as_tensor(np.asarray(aggregated_obs), device=device)
            recovery_actions = torch.as_tensor(
                np.asarray(aggregated_actions), device=device
            )
            model.policy.set_training_mode(True)
            final_loss = math.nan
            for _ in range(epochs):
                # Each minibatch has equal expert-trajectory and learner-state
                # weight even after recovery data greatly exceeds the demos.
                for _ in range(
                    max(1, math.ceil((len(demo_obs) + len(recovery_obs)) / 256))
                ):
                    left = body_balanced_indices(original_bodies, 128, rng)
                    right = body_balanced_indices(aggregated_bodies, 128, rng)
                    batch_obs = torch.cat((demo_obs[left], recovery_obs[right]))
                    target = torch.cat((demo_actions[left], recovery_actions[right]))
                    predicted = model.policy.get_distribution(
                        batch_obs
                    ).distribution.mean
                    loss = torch.nn.functional.mse_loss(predicted, target)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor_parameters, 1)
                    optimizer.step()
                    final_loss = float(loss.item())
            model.policy.set_training_mode(False)
            validation = evaluate(
                model, config, catalog, train_bodies, validation_seeds
            )
            score = (
                validation["macro_success_rate"],
                float(
                    np.mean(
                        [row["mean_return"] for row in validation["per_body"].values()]
                    )
                ),
            )
            if score > best_score:
                best_score = score
                save_checkpoint(model, best_path)
                write_json(Path(run_dir) / "dagger_evaluation_best.json", validation)
            rollout_summary = {
                body: dict(
                    successes=rollout_results[(body, True)],
                    failures=rollout_results[(body, False)],
                )
                for body in train_bodies
            }
            log.write(
                "dagger_round",
                round=round_index + 1,
                expert_mixing=mixing,
                original_demonstrations=len(demo_obs),
                aggregated_recovery_states=len(recovery_obs),
                final_action_mse=final_loss,
                body_balanced_batches=True,
                rollout_results=rollout_summary,
                validation=validation,
            )
            write_json(Path(run_dir) / "dagger_evaluation_latest.json", validation)
            print(
                f"DAgger round {round_index + 1}/{rounds} beta {mixing:.3f} | "
                f"recovery states {len(recovery_obs):,} | validation "
                f"{ {body: row['success_rate'] for body, row in validation['per_body'].items()} }",
                flush=True,
            )
    finally:
        env.close()
    # A later supervised round can regress too; restore the strongest validation
    # policy before PPO and keep its optimizer freshly initialized by PPO.
    best = PPO.load(best_path, device=device)
    model.policy.load_state_dict(best.policy.state_dict())
    validation = evaluate(model, config, catalog, train_bodies, validation_seeds)
    robust = all(
        row["success_rate"] >= minimum_success
        for row in validation["per_body"].values()
    )
    return dict(
        robust=robust, validation=validation, aggregated_states=len(aggregated_obs)
    )


def warmstart_current_values(
    model, config, catalog, train_bodies, *, episodes, epochs, seed, log
):
    """Fit PPO's critic to the initialized stochastic policy, holding actor fixed."""
    env = CaptureEnv(
        config, body_catalog={name: catalog[name] for name in train_bodies}
    )
    observations, targets = [], []
    successes = 0
    try:
        for episode in range(episodes):
            body = train_bodies[episode % len(train_bodies)]
            observation, _ = env.reset(
                seed=seed + 70_000 + episode, options={"body": body}
            )
            trajectory = []
            for _ in range(math.ceil(env.config.max_time / env.config.control_dt) + 2):
                action = model.predict(observation, deterministic=False)[0]
                following, reward, terminated, truncated, info = env.step(action)
                trajectory.append((observation.copy(), float(reward)))
                observation = following
                if terminated or truncated:
                    successes += bool(info.get("success", False))
                    if terminated:
                        value = 0.0
                        for state, reward in reversed(trajectory):
                            value = reward + config.gamma * value
                            observations.append(state)
                            targets.append(value)
                    break
            else:
                raise RuntimeError(
                    "Initialized policy exceeded the capture episode bound"
                )
    finally:
        env.close()
    if not observations or not epochs:
        return
    obs = torch.as_tensor(np.asarray(observations), device=model.device)
    returns = torch.as_tensor(
        np.asarray(targets), dtype=torch.float32, device=model.device
    )
    parameters = list(model.policy.mlp_extractor.value_net.parameters()) + list(
        model.policy.value_net.parameters()
    )
    optimizer = torch.optim.Adam(parameters, lr=1e-3)
    rng = np.random.default_rng(seed + 789)
    for _ in range(epochs):
        indices = rng.permutation(len(returns))
        for start in range(0, len(indices), 256):
            batch = indices[start : start + 256]
            loss = torch.nn.functional.mse_loss(
                model.policy.predict_values(obs[batch]).flatten(), returns[batch]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1)
            optimizer.step()
    with torch.no_grad():
        mse = float(
            torch.nn.functional.mse_loss(
                model.policy.predict_values(obs).flatten(), returns
            ).item()
        )
    log.write(
        "initialized_policy_value_warmstart",
        episodes=episodes,
        stochastic_successes=successes,
        terminal_transitions=len(returns),
        epochs=epochs,
        final_mse=mse,
    )
    print(
        f"Initialized stochastic policy: {successes}/{episodes} captures; critic MSE {mse:.3f}",
        flush=True,
    )


class CaptureCallback(BaseCallback):
    def __init__(
        self,
        run_dir,
        config,
        catalog,
        train_bodies,
        heldout_bodies,
        seeds,
        log,
        eval_freq,
        stopped,
    ):
        super().__init__()
        self.run_dir, self.config, self.catalog = run_dir, config, catalog
        self.train_bodies, self.heldout_bodies, self.seeds = (
            train_bodies,
            heldout_bodies,
            seeds,
        )
        self.log, self.eval_freq, self.stopped = log, eval_freq, stopped
        self.next_eval = 0
        self.best_score = (-1.0, -1.0)
        previous = run_dir / "evaluation_best.json"
        if previous.exists():
            record = json.loads(previous.read_text())["train_bodies"]
            self.best_score = (
                record["macro_success_rate"],
                float(
                    np.mean([row["mean_return"] for row in record["per_body"].values()])
                ),
            )

    def evaluate(self):
        results = {}
        for split, bodies in [
            ("train_bodies", self.train_bodies),
            ("heldout_bodies", self.heldout_bodies),
        ]:
            if bodies:
                results[split] = evaluate(
                    self.model, self.config, self.catalog, bodies, self.seeds
                )
        score = (
            results["train_bodies"]["macro_success_rate"],
            float(
                np.mean(
                    [
                        row["mean_return"]
                        for row in results["train_bodies"]["per_body"].values()
                    ]
                )
            ),
        )
        # Held-out body results are reported but never used to select weights.
        if score > self.best_score:
            self.best_score = score
            save_checkpoint(self.model, self.run_dir / "best.zip")
            write_json(
                self.run_dir / "evaluation_best.json",
                dict(train_step=self.num_timesteps, **results),
            )
        save_checkpoint(self.model, self.run_dir / "latest.zip")
        write_json(
            self.run_dir / "evaluation_latest.json",
            dict(train_step=self.num_timesteps, **results),
        )
        self.log.write("evaluation", train_step=self.num_timesteps, **results)
        train_rate = results["train_bodies"]["macro_success_rate"]
        heldout_rate = results.get("heldout_bodies", {}).get(
            "macro_success_rate", math.nan
        )
        print(
            f"Capture step {self.num_timesteps:,} | train bodies {train_rate:.0%} | held-out bodies {heldout_rate:.0%}",
            flush=True,
        )
        return results

    def _on_training_start(self):
        self.next_eval = self.num_timesteps + self.eval_freq

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.log.write(
                    "episode",
                    train_step=self.num_timesteps,
                    body=info.get("body_name", info.get("body")),
                    success=info.get("success", False),
                    termination_reason=info.get("termination_reason", ""),
                    **info["episode"],
                )
        if self.num_timesteps >= self.next_eval:
            self.evaluate()
            self.next_eval = self.num_timesteps + self.eval_freq
        return not self.stopped[0]


def comma_names(value):
    return tuple(item.strip() for item in value.split(",") if item.strip())


def body_fractions(value):
    fractions = {}
    try:
        for item in value.split(","):
            name, fraction = item.split("=")
            name, fraction = name.strip(), float(fraction)
            if (
                not name
                or name in fractions
                or not math.isfinite(fraction)
                or fraction < 0
            ):
                raise ValueError
            fractions[name] = fraction
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Use distinct Body=nonnegative_fraction pairs, for example Mun=.7,Minmus=1.9"
        ) from error
    return fractions


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["train", "evaluate"], nargs="?", default="train"
    )
    parser.add_argument(
        "--system",
        type=Path,
        help="KSP body catalog JSON; defaults to the packaged read-only stock-system capture",
    )
    parser.add_argument("--run-dir", type=Path, default=Path("runs/capture_ppo"))
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--resume", type=Path)
    source.add_argument(
        "--initialize",
        type=Path,
        help="Start a fresh experiment from compatible actor/critic weights; reset PPO optimizer and step count",
    )
    parser.add_argument("--train-bodies", type=comma_names)
    parser.add_argument("--heldout-bodies", type=comma_names)
    parser.add_argument("--steps", type=int, default=131_072)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=100_000)
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=4,
        help="Episodes per body; small values yield wide confidence intervals",
    )
    parser.add_argument("--eval-freq", type=int, default=16_384)
    parser.add_argument("--bc-episodes", type=int, default=32)
    parser.add_argument("--bc-epochs", type=int, default=60)
    parser.add_argument("--value-epochs", type=int, default=60)
    parser.add_argument("--dagger-rounds", type=int, default=0)
    parser.add_argument("--dagger-episodes", type=int, default=32)
    parser.add_argument("--dagger-epochs", type=int, default=30)
    parser.add_argument("--dagger-beta", type=float, default=0.8)
    parser.add_argument(
        "--min-initial-success",
        type=float,
        default=0.0,
        help="Per-training-body validation gate before PPO",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--log-std", type=float, default=-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--randomization", type=float)
    parser.add_argument("--max-time", type=float)
    parser.add_argument("--mass", type=float)
    parser.add_argument("--dry-mass", type=float)
    parser.add_argument("--thrust", type=float)
    parser.add_argument("--isp", type=float)
    parser.add_argument("--target-altitude-randomization", type=float)
    parser.add_argument("--arrival-vinf-fraction", type=float)
    parser.add_argument(
        "--arrival-vinf-by-body",
        type=body_fractions,
        help="Body-specific arrival speed / target circular speed, for example Mun=.7,Minmus=1.9",
    )
    parser.add_argument("--vehicle-label", default="configured_surrogate_vehicle")
    parser.add_argument(
        "--stress",
        action="store_true",
        help="Run separate altitude/arrival/craft tests after nominal evaluation",
    )
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args(argv)
    for name in [
        "steps",
        "n_envs",
        "rollout_steps",
        "batch_size",
        "eval_episodes",
        "eval_freq",
        "hidden_size",
    ]:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if (
        args.bc_episodes < 0
        or args.bc_epochs < 0
        or args.value_epochs < 0
        or args.learning_rate <= 0
    ):
        parser.error(
            "BC/value episode counts must be nonnegative and learning-rate positive"
        )
    if args.rollout_steps * args.n_envs < 2 or args.batch_size < 2:
        parser.error(
            "PPO rollout and batch sizes must contain at least two transitions"
        )
    if args.max_time is not None and args.max_time <= 0:
        parser.error("--max-time must be positive")
    if (
        args.dagger_rounds < 0
        or args.dagger_episodes <= 0
        or args.dagger_epochs <= 0
        or not 0 <= args.dagger_beta <= 1
        or not 0 <= args.min_initial_success <= 1
    ):
        parser.error(
            "Invalid DAgger counts, mixing coefficient or initial success threshold"
        )
    if args.dagger_rounds and (not args.bc_episodes or args.resume):
        parser.error(
            "DAgger requires a fresh run with original --bc-episodes demonstrations"
        )
    if args.command == "evaluate" and not args.resume:
        parser.error("evaluate requires --resume")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = None
    initializer = None
    initialization_record = None
    if args.resume or args.initialize:
        loaded, metadata = load_capture_controller(args.resume or args.initialize)
        config, catalog, train_bodies, heldout_bodies = validate_contract(
            loaded, metadata
        )
        if args.resume:
            model = loaded
        else:
            initializer = loaded
            initialization_record = dict(
                checkpoint=str(args.initialize),
                sha256=hashlib.sha256(args.initialize.read_bytes()).hexdigest(),
                source_num_timesteps=loaded.num_timesteps,
                source_train_bodies=list(train_bodies),
                source_heldout_bodies=list(heldout_bodies),
                optimizer_and_step_count_reset=True,
            )
        if args.resume and (
            args.train_bodies is not None
            and tuple(args.train_bodies) != train_bodies
            or args.heldout_bodies is not None
            and tuple(args.heldout_bodies) != heldout_bodies
        ):
            raise ValueError("Body sets differ from the saved capture checkpoint")
        if args.initialize:
            previous_training = set(train_bodies)
            train_bodies, heldout_bodies = body_sets(
                catalog,
                train_bodies if args.train_bodies is None else args.train_bodies,
                heldout_bodies if args.heldout_bodies is None else args.heldout_bodies,
            )
            if previous_training & set(heldout_bodies):
                raise ValueError(
                    "Previously trained bodies cannot become held-out bodies in a new experiment"
                )
            if not previous_training.issubset(train_bodies):
                raise ValueError(
                    "A new initialization must retain previously trained bodies in its training contract"
                )
        if (
            args.system is not None
            and json.loads(args.system.read_text())["bodies"] != catalog
        ):
            raise ValueError(
                "System catalog differs from the embedded capture checkpoint"
            )
        overrides = {
            key: getattr(args, key)
            for key in CONFIG_OVERRIDES
            if getattr(args, key) is not None
        }
        changed_config = replace(config, **overrides)
        if args.resume and asdict(changed_config) != asdict(config):
            raise ValueError(
                "Capture configuration differs from checkpoint; start a fresh experiment"
            )
        config = changed_config
    else:
        catalog = SystemCatalog.load(args.system).bodies
        train_bodies, heldout_bodies = body_sets(
            catalog, args.train_bodies, args.heldout_bodies
        )
        overrides = {
            key: getattr(args, key)
            for key in CONFIG_OVERRIDES
            if getattr(args, key) is not None
        }
        config = CaptureConfig.for_body(catalog[train_bodies[0]], **overrides)
    unknown_arrivals = set(getattr(config, "arrival_vinf_by_body", {})) - set(catalog)
    if unknown_arrivals:
        raise ValueError(
            f"Arrival fractions reference unknown bodies: {sorted(unknown_arrivals)}"
        )
    seeds = [args.eval_seed + i for i in range(args.eval_episodes)]
    # All input/contract validation happens before touching experiment outputs.
    if not args.resume and (args.run_dir / "config.json").exists():
        raise ValueError(
            "This run directory already contains an experiment; use --resume or a new --run-dir"
        )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    log = EventLog(args.run_dir / "progress.jsonl")
    vector = None
    callback = None
    stopped = [False]
    previous_handlers = {}

    def stop(signum, frame):
        stopped[0] = True
        print(f"Signal {signum}: saving at the next capture training step", flush=True)

    try:
        if args.command == "evaluate":
            results = {
                split: evaluate(model, config, catalog, bodies, seeds)
                for split, bodies in [
                    ("train_bodies", train_bodies),
                    ("heldout_bodies", heldout_bodies),
                ]
                if bodies
            }
            write_json(args.run_dir / "evaluation.json", results)
            log.write("standalone_evaluation", model=str(args.resume), **results)
            if args.stress:
                stresses = evaluate_stress(
                    model, config, catalog, train_bodies + heldout_bodies, seeds
                )
                write_json(args.run_dir / "evaluation_stress.json", stresses)
                log.write("stress_evaluation", **stresses)
            print(json.dumps(_jsonable(results), indent=2), flush=True)
            return results
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, stop)
        vehicle_label = args.vehicle_label
        if initializer is not None and vehicle_label == "configured_surrogate_vehicle":
            vehicle_label = initializer.orbiter_capture_contract["vehicle_label"]
        contract = (
            model.orbiter_capture_contract
            if model is not None
            else make_contract(
                config, catalog, train_bodies, heldout_bodies, vehicle_label
            )
        )
        write_json(
            args.run_dir / "config.json",
            dict(
                arguments=vars(args),
                contract=contract,
                initialization=initialization_record,
                evaluation_seeds=seeds,
            ),
        )
        if not args.skip_baselines:
            for kind in ("teacher", "random"):
                result = evaluate(
                    None,
                    config,
                    catalog,
                    train_bodies + heldout_bodies,
                    seeds,
                    kind=kind,
                )
                write_json(args.run_dir / f"baseline_{kind}.json", result)
                log.write("baseline", **result)
                print(
                    f"Capture {kind} baseline {result['success_rate']:.0%}", flush=True
                )
        train_catalog = {name: catalog[name] for name in train_bodies}
        vector = DummyVecEnv(
            [
                lambda: Monitor(CaptureEnv(config, body_catalog=train_catalog))
                for _ in range(args.n_envs)
            ]
        )
        vector.seed(args.seed)
        if model is not None:
            model.set_env(vector)
        else:
            policy_class = (
                "MlpPolicy" if initializer is None else initializer.policy_class
            )
            policy_kwargs = (
                dict(
                    log_std_init=args.log_std,
                    net_arch=dict(pi=[args.hidden_size] * 2, vf=[args.hidden_size] * 2),
                )
                if initializer is None
                else initializer.policy_kwargs
            )
            model = PPO(
                policy_class,
                vector,
                device="cpu",
                seed=args.seed,
                gamma=config.gamma,
                learning_rate=args.learning_rate,
                n_steps=args.rollout_steps,
                batch_size=min(args.batch_size, args.rollout_steps * args.n_envs),
                n_epochs=10,
                ent_coef=0,
                gae_lambda=1,
                target_kl=0.02,
                policy_kwargs=policy_kwargs,
            )
            if initializer is not None:
                model.policy.load_state_dict(initializer.policy.state_dict())
                log.write("imitation_initialization", **initialization_record)
            if args.bc_episodes:
                demonstrations = behavior_clone(
                    model,
                    config,
                    catalog,
                    train_bodies,
                    episodes=args.bc_episodes,
                    seed=args.seed,
                    epochs=args.bc_epochs,
                    value_epochs=args.value_epochs,
                    log=log,
                )
        model.orbiter_capture_contract = contract
        model.orbiter_training_status = "in_progress"
        if args.dagger_rounds:
            initialization = dagger(
                model,
                config,
                catalog,
                train_bodies,
                demonstrations,
                rounds=args.dagger_rounds,
                episodes=args.dagger_episodes,
                epochs=args.dagger_epochs,
                beta=args.dagger_beta,
                seed=args.seed,
                validation_seeds=seeds,
                run_dir=args.run_dir,
                log=log,
                minimum_success=args.min_initial_success,
            )
            if not initialization["robust"]:
                model.orbiter_training_status = "initial_validation_failed"
                save_checkpoint(model, args.run_dir / "latest.zip")
                log.write(
                    "training_blocked_initial_validation",
                    threshold=args.min_initial_success,
                    validation=initialization["validation"],
                )
                print(
                    "PPO was not started: the requested per-body initialization success gate failed.",
                    flush=True,
                )
                return model
            warmstart_current_values(
                model,
                config,
                catalog,
                train_bodies,
                episodes=max(args.bc_episodes, len(train_bodies)),
                epochs=args.value_epochs,
                seed=args.seed,
                log=log,
            )
        callback = CaptureCallback(
            args.run_dir,
            config,
            catalog,
            train_bodies,
            heldout_bodies,
            seeds,
            log,
            args.eval_freq,
            stopped,
        )
        callback.model = model
        callback.num_timesteps = model.num_timesteps
        initial_results = callback.evaluate()
        if args.min_initial_success and not all(
            row["success_rate"] >= args.min_initial_success
            for row in initial_results["train_bodies"]["per_body"].values()
        ):
            model.orbiter_training_status = "initial_validation_failed"
            log.write(
                "training_blocked_initial_validation",
                threshold=args.min_initial_success,
                validation=initial_results["train_bodies"],
            )
            print(
                "PPO was not started: the requested per-body initialization success gate failed.",
                flush=True,
            )
            return model
        log.write(
            "training_start",
            task=TASK,
            requested_steps=args.steps,
            train_bodies=train_bodies,
            heldout_bodies=heldout_bodies,
        )
        model.learn(
            args.steps, callback=callback, reset_num_timesteps=not bool(args.resume)
        )
        callback.evaluate()
        if args.stress and not stopped[0]:
            stresses = evaluate_stress(
                model, config, catalog, train_bodies + heldout_bodies, seeds
            )
            write_json(args.run_dir / "evaluation_stress.json", stresses)
            log.write("stress_evaluation", **stresses)
        model.orbiter_training_status = "stopped" if stopped[0] else "complete"
        log.write(
            "training_stopped" if stopped[0] else "training_complete",
            train_step=model.num_timesteps,
        )
        return model
    finally:
        if model is not None and callback is not None:
            save_checkpoint(model, args.run_dir / "latest.zip")
        if vector is not None:
            vector.close()
        log.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def cli_main(argv=None):
    """Console scripts need an exit status, while ``main`` returns results."""
    result = main(argv)
    return (
        2
        if getattr(result, "orbiter_training_status", None)
        == "initial_validation_failed"
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(cli_main())
