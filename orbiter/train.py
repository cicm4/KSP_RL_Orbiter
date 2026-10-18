"""Reproducible headless training and held-out evaluation.

Examples:
    python -m orbiter.train --algo tqc --steps 200000 --run-dir runs/tqc
    python -m orbiter.train evaluate --resume runs/tqc/best.zip --eval-episodes 20
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import random
import signal
import time
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import TQC

from .env import EnvConfig, HeadlessOrbiterEnv
from .guidance import expert_action
from .checkpoints import validate_checkpoint_contract


ALGORITHMS = {"sac": SAC, "tqc": TQC, "ppo": PPO}


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_jsonable(data), indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


class EventLog:
    def __init__(self, path: Path):
        self.handle = path.open("a", buffering=1)

    def write(self, event_type: str, **data: Any) -> None:
        row = {"event": event_type, "wall_time": time.time(), **data}
        self.handle.write(json.dumps(_jsonable(row), allow_nan=False) + "\n")

    def close(self) -> None:
        self.handle.close()


def evaluate_policy(model, config: EnvConfig, seeds: list[int], kind="policy") -> dict:
    """Use independent, fixed launch seeds; never sample training curriculum states."""
    evaluation_config = replace(config, curriculum=False)
    env = HeadlessOrbiterEnv(evaluation_config)
    episodes = []
    try:
        for seed in seeds:
            observation, info = env.reset(seed=seed)
            action_rng = np.random.default_rng(seed + 911)
            total_return = 0.0
            steps = 0
            highest_altitude = float(info.get("altitude", 0.0))
            while True:
                if kind == "random":
                    action = action_rng.uniform(-1.0, 1.0, env.action_space.shape).astype(np.float32)
                elif kind == "scripted":
                    action = np.zeros(env.action_space.shape, dtype=np.float32) if config.residual else expert_action(env)
                else:
                    action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                total_return += float(reward)
                steps += 1
                highest_altitude = max(highest_altitude, float(info.get("altitude", 0.0)))
                if terminated or truncated:
                    break
            episodes.append({
                "seed": seed,
                "success": bool(info.get("success", False)),
                "return": total_return,
                "steps": steps,
                "max_altitude": float(info.get("max_altitude", highest_altitude)),
                "apoapsis": info.get("apoapsis"),
                "periapsis": info.get("periapsis"),
                "termination_reason": info.get("termination_reason", info.get("reason", "")),
                "final_info": info,
            })
    finally:
        env.close()
    return {
        "kind": kind,
        "episodes": episodes,
        "success_rate": float(np.mean([episode["success"] for episode in episodes])),
        "mean_return": float(np.mean([episode["return"] for episode in episodes])),
        "mean_max_altitude": float(np.mean([episode["max_altitude"] for episode in episodes])),
    }


def atomic_save(model, path: Path) -> None:
    temporary = path.with_name("." + path.stem + ".tmp.zip")
    model.save(temporary)
    os.replace(temporary, path)
    if hasattr(model, "orbiter_config"):
        write_json(path.with_suffix(".metadata.json"), {
            "algorithm": model.orbiter_algorithm, "environment": model.orbiter_config,
            "num_timesteps": model.num_timesteps,
        })


def behavior_clone(model, config: EnvConfig, episodes: int, seed: int, log: EventLog,
                   epochs: int = 20, value_epochs: int = 100) -> None:
    """Imitate the exact actuator actions, and preserve correct terminal transitions."""
    env = HeadlessOrbiterEnv(replace(config, curriculum=False))
    observations, actions, transitions = [], [], []
    value_observations, value_targets = [], []
    successes = 0
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + 10000 + episode)
            episode_start = len(transitions)
            while True:
                action = np.zeros(env.action_space.shape, dtype=np.float32) if config.residual else np.asarray(expert_action(env), dtype=np.float32)
                next_observation, reward, terminated, truncated, info = env.step(action)
                observations.append(observation.copy())
                actions.append(action.copy())
                transitions.append((observation.copy(), next_observation.copy(), action.copy(),
                                    float(reward), bool(terminated or truncated),
                                    {**info, "TimeLimit.truncated": bool(truncated and not terminated)}))
                observation = next_observation
                if terminated or truncated:
                    successes += bool(info.get("success", False))
                    # A time limit is not an absorbing failure. Without a known
                    # bootstrap value it cannot supply a complete MC target.
                    if terminated:
                        discounted_return = 0.0
                        for transition in reversed(transitions[episode_start:]):
                            discounted_return = transition[3] + config.gamma * discounted_return
                            value_observations.append(transition[0])
                            value_targets.append(discounted_return)
                    break
    finally:
        env.close()
    obs = torch.as_tensor(np.asarray(observations), device=model.device)
    target = torch.as_tensor(np.asarray(actions), device=model.device)
    actor = getattr(model.policy, "actor", None)
    parameters = actor.parameters() if actor is not None else model.policy.parameters()
    optimizer = torch.optim.Adam(parameters, lr=1e-3)
    generator = np.random.default_rng(seed)
    final_loss = float("nan")
    model.policy.set_training_mode(True)
    for _ in range(epochs):
        indices = generator.permutation(len(observations))
        for start in range(0, len(indices), 256):
            batch_indices = indices[start:start + 256]
            batch_obs, batch_actions = obs[batch_indices], target[batch_indices]
            if actor is not None:
                predicted = actor(batch_obs, deterministic=True)
            else:
                predicted = model.policy.get_distribution(batch_obs).distribution.mean
            loss = torch.nn.functional.mse_loss(predicted, batch_actions)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
            optimizer.step()
            final_loss = float(loss.item())
    model.policy.set_training_mode(False)
    # Seed off-policy replay with the same actuator/observation contract used online.
    if getattr(model, "replay_buffer", None) is not None:
        n_envs = model.n_envs
        for observation, next_observation, action, reward, done, info in transitions:
            model.replay_buffer.add(
                np.repeat(observation[None], n_envs, axis=0),
                np.repeat(next_observation[None], n_envs, axis=0),
                np.repeat(action[None], n_envs, axis=0),
                np.full(n_envs, reward, dtype=np.float32),
                np.full(n_envs, done, dtype=bool),
                [dict(info) for _ in range(n_envs)],
            )
    log.write("behavior_clone", episodes=episodes, transitions=len(transitions),
              expert_successes=successes, epochs=epochs, final_mse=final_loss)
    print(f"Expert initialization: {successes}/{episodes} orbits, "
          f"{len(transitions)} transitions, final MSE {final_loss:.6f}", flush=True)
    if actor is None and value_epochs and value_observations:
        # BC has trained the policy while leaving V(s) near zero. Fit complete
        # discounted task returns before PPO forms advantages for this policy.
        # Actor and critic MLPs are separate, so this cannot move the BC actor.
        value_obs = torch.as_tensor(np.asarray(value_observations), device=model.device)
        returns = torch.as_tensor(np.asarray(value_targets), dtype=torch.float32, device=model.device)
        value_parameters = list(model.policy.mlp_extractor.value_net.parameters()) + list(model.policy.value_net.parameters())
        value_optimizer = torch.optim.Adam(value_parameters, lr=1e-3)
        with torch.no_grad():
            initial_mse = torch.nn.functional.mse_loss(model.policy.predict_values(value_obs).flatten(), returns).item()
        for _ in range(value_epochs):
            indices = generator.permutation(len(value_targets))
            for start in range(0, len(indices), 256):
                batch_indices = indices[start:start + 256]
                predicted = model.policy.predict_values(value_obs[batch_indices]).flatten()
                loss = torch.nn.functional.mse_loss(predicted, returns[batch_indices])
                value_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(value_parameters, 1.0)
                value_optimizer.step()
        with torch.no_grad():
            final_mse = torch.nn.functional.mse_loss(model.policy.predict_values(value_obs).flatten(), returns).item()
        log.write("value_warmstart", transitions=len(value_targets), epochs=value_epochs,
                  initial_mse=initial_mse, final_mse=final_mse,
                  mean_target=float(returns.mean().item()))
        print(f"Value warmstart: MSE {initial_mse:.2f} -> {final_mse:.2f}, "
              f"mean MC return {returns.mean().item():.2f}", flush=True)


class RunCallback(BaseCallback):
    def __init__(self, args, config, seeds, event_log, stop_requested, initial_best=None):
        super().__init__(verbose=0)
        self.args, self.config, self.seeds, self.log = args, config, seeds, event_log
        self.stop_requested = stop_requested
        self.best_score = tuple(initial_best) if initial_best else (-1.0, -float("inf"))
        self.next_evaluation = args.eval_freq
        self.started = time.monotonic()
        self.initial_steps = 0
        self.episode_count = 0

    def checkpoint(self, label="latest") -> None:
        atomic_save(self.model, self.args.run_dir / f"{label}.zip")
        if label == "latest" and self.args.save_replay and hasattr(self.model, "save_replay_buffer"):
            temporary = self.args.run_dir / ".replay.tmp.pkl"
            self.model.save_replay_buffer(temporary)
            os.replace(temporary, self.args.run_dir / "replay.pkl")
        write_json(self.args.run_dir / "state.json", {
            "num_timesteps": self.model.num_timesteps,
            "best_score": self.best_score,
            "updated_at": time.time(),
        })

    def evaluate(self) -> None:
        result = evaluate_policy(self.model, self.config, self.seeds)
        score = (result["success_rate"], result["mean_return"])
        training_metrics = {key: value for key, value in self.model.logger.name_to_value.items()
                            if key.startswith("train/")} if getattr(self.model, "_logger", None) is not None else {}
        self.log.write("evaluation", train_step=self.model.num_timesteps,
                       training_metrics=training_metrics, **result)
        write_json(self.args.run_dir / "evaluation_latest.json", result)
        if score > self.best_score:
            self.best_score = score
            atomic_save(self.model, self.args.run_dir / "best.zip")
            write_json(self.args.run_dir / "evaluation_best.json", {
                "train_step": self.model.num_timesteps, **result,
            })
        self.checkpoint()
        elapsed = max(time.monotonic() - self.started, 1e-6)
        fps = (self.model.num_timesteps - self.initial_steps) / elapsed
        print(f"Step {self.model.num_timesteps:,} | held-out orbits "
              f"{result['success_rate']:.0%} | return {result['mean_return']:.2f} "
              f"| {fps:.0f} steps/s", flush=True)

    def _on_training_start(self) -> None:
        self.initial_steps = self.model.num_timesteps
        self.next_evaluation = self.model.num_timesteps + self.args.eval_freq

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_count += 1
                self.log.write("episode", train_step=self.model.num_timesteps,
                               episode=self.episode_count, success=info.get("success", False),
                               **info["episode"],
                               termination_reason=info.get("termination_reason", info.get("reason", "")))
        if self.model.num_timesteps >= self.next_evaluation:
            self.evaluate()
            self.next_evaluation = self.model.num_timesteps + self.args.eval_freq
        return not self.stop_requested[0]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", choices=["train", "evaluate"], default="train")
    parser.add_argument("--algo", choices=ALGORITHMS, default="ppo")
    parser.add_argument("--steps", type=int, default=200_000, help="Additional environment transitions")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/ppo"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=100_000)
    parser.add_argument("--resume", type=Path, help="Model zip checkpoint")
    parser.add_argument("--save-replay", action="store_true", help="Persist off-policy replay for exact data continuation")
    parser.add_argument("--load-replay", type=Path)
    parser.add_argument("--bc-episodes", type=int, default=30, help="Optional scripted teacher initialization")
    parser.add_argument("--bc-epochs", type=int, default=100)
    parser.add_argument("--value-epochs", type=int, default=100, help="PPO critic MC warmstart epochs; 0 disables")
    parser.add_argument("--ppo-log-std", type=float, default=-3.0)
    parser.add_argument("--gae-lambda", type=float, default=1.0, help="PPO advantage trace; 1 preserves sparse task credit")
    parser.add_argument("--config", type=Path, help="Captured or custom EnvConfig JSON")
    parser.add_argument("--residual", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--randomization", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--learning-starts", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--ent-coef", default="auto_0.02", help="SAC/TQC entropy coefficient")
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args(argv)
    for key in ["steps", "n_envs", "eval_freq", "eval_episodes", "batch_size", "buffer_size", "hidden_size"]:
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.bc_episodes < 0 or args.bc_epochs <= 0:
        parser.error("--bc-episodes must be nonnegative and --bc-epochs must be positive")
    if args.value_epochs < 0 or not 0 <= args.gae_lambda <= 1:
        parser.error("--value-epochs must be nonnegative and --gae-lambda must be in [0, 1]")
    if args.command == "evaluate" and args.resume is None:
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
    # Recover the source contract before opening environments or overwriting metadata.
    metadata = None
    if args.resume:
        metadata_path = args.resume.with_suffix(".metadata.json")
        source_config_path = args.resume.parent / "config.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
        elif source_config_path.exists():
            source = json.loads(source_config_path.read_text())
            metadata = {"environment": source["environment"], "algorithm": source["arguments"]["algo"]}
        if metadata is None:
            raise ValueError("Resume requires checkpoint .metadata.json or its source config.json; keep the environment contract with the model")
        args.algo = str(metadata["algorithm"]).lower()
        if args.algo not in ALGORITHMS:
            raise ValueError(f"Unsupported checkpoint algorithm {args.algo!r}")
    config = EnvConfig.load(args.config) if args.config else (
        EnvConfig.from_dict(metadata["environment"]) if metadata else EnvConfig())
    overrides = {}
    if args.randomization is not None:
        overrides["randomization"] = args.randomization
    if args.residual is not None:
        overrides["residual"] = args.residual
    config = replace(config, **overrides, curriculum=False)
    if args.resume and metadata and asdict(config) != asdict(EnvConfig.from_dict(metadata["environment"])):
        raise ValueError("Resume configuration differs from checkpoint environment; use a fresh run for a changed physics/reward contract")
    if args.resume:
        # Reject a swapped sidecar before creating logs or replacing run metadata.
        # Loading again with the vector environment below rebuilds SB3's buffers
        # correctly when the requested number of environments differs.
        source_model = ALGORITHMS[args.algo].load(args.resume, device="cpu")
        validate_checkpoint_contract(source_model, metadata)
        del source_model
    args.run_dir.mkdir(parents=True, exist_ok=True)
    seeds = [args.eval_seed + index for index in range(args.eval_episodes)]
    log = EventLog(args.run_dir / "progress.jsonl")
    stop_requested = [False]
    previous_handlers = {}
    def request_stop(signum, frame):
        stop_requested[0] = True
        print(f"Signal {signum}: checkpointing at the next training step.", flush=True)
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, request_stop)
    vector_env = None
    callback = None
    model = None
    try:
        if args.command == "evaluate":
            model = ALGORITHMS[args.algo].load(args.resume, device="cpu")
            result = evaluate_policy(model, config, seeds)
            log.write("standalone_evaluation", model=str(args.resume), **result)
            write_json(args.run_dir / "evaluation.json", result)
            print(json.dumps(_jsonable(result), indent=2), flush=True)
            return
        write_json(args.run_dir / "config.json", {
            "arguments": vars(args), "environment": asdict(config),
            "evaluation_seeds": seeds,
        })
        if not args.skip_baselines:
            for kind in ("random", "scripted"):
                result = evaluate_policy(None, config, seeds, kind=kind)
                log.write("baseline", **result)
                write_json(args.run_dir / f"baseline_{kind}.json", result)
                print(f"{kind.capitalize()} baseline: {result['success_rate']:.0%} orbits, "
                      f"return {result['mean_return']:.2f}", flush=True)
        vector_env = DummyVecEnv([
            lambda: Monitor(HeadlessOrbiterEnv(config)) for _ in range(args.n_envs)
        ])
        vector_env.seed(args.seed)
        if args.resume:
            model = ALGORITHMS[args.algo].load(args.resume, env=vector_env, device="cpu")
            validate_checkpoint_contract(model, metadata)
            # Save parameters are authoritative on resume; fail instead of silently
            # changing the reward/discount contract represented by replay targets.
            if not math.isclose(model.gamma, config.gamma):
                raise ValueError(f"Checkpoint gamma {model.gamma} differs from environment gamma {config.gamma}")
            if args.load_replay:
                model.load_replay_buffer(args.load_replay)
            elif hasattr(model, "replay_buffer"):
                print("Resuming weights/optimizers with fresh replay; use --load-replay for saved transitions.", flush=True)
        else:
            shared = dict(policy="MlpPolicy", env=vector_env, learning_rate=args.learning_rate,
                          gamma=config.gamma, seed=args.seed, device="cpu", verbose=0)
            if args.algo == "ppo":
                rollout_steps = 512
                model = PPO(**shared, n_steps=rollout_steps, batch_size=min(args.batch_size, rollout_steps * args.n_envs),
                            n_epochs=10, ent_coef=0.0, clip_range=0.2, max_grad_norm=0.5, target_kl=args.target_kl,
                            gae_lambda=args.gae_lambda,
                            policy_kwargs={"log_std_init": args.ppo_log_std, "net_arch": dict(pi=[args.hidden_size] * 2, vf=[args.hidden_size] * 2)})
            else:
                model = ALGORITHMS[args.algo](
                    **shared, buffer_size=args.buffer_size, learning_starts=args.learning_starts,
                    batch_size=args.batch_size, train_freq=1, gradient_steps=1, tau=0.005,
                    ent_coef=args.ent_coef, policy_kwargs={"net_arch": [args.hidden_size] * 2, "log_std_init": -3.0},
                )
            if args.bc_episodes:
                behavior_clone(model, config, args.bc_episodes, args.seed, log, epochs=args.bc_epochs,
                               value_epochs=args.value_epochs)
        model.orbiter_config = asdict(config)
        model.orbiter_algorithm = args.algo
        state_path = args.run_dir / "state.json"
        initial_best = None
        if args.resume and state_path.exists():
            initial_best = json.loads(state_path.read_text()).get("best_score")
        callback = RunCallback(args, config, seeds, log, stop_requested, initial_best=initial_best)
        callback.model = model
        callback.evaluate()
        log.write("training_start", train_step=model.num_timesteps,
                  requested_additional_steps=args.steps, algorithm=args.algo)
        model.learn(total_timesteps=args.steps, callback=callback, reset_num_timesteps=not bool(args.resume))
        callback.evaluate()
        log.write("training_stopped" if stop_requested[0] else "training_complete", train_step=model.num_timesteps)
    finally:
        if model is not None and callback is not None:
            callback.checkpoint()
        if vector_env is not None:
            vector_env.close()
        log.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
