"""Experimental residual imitation of independently qualified ascent teachers.

The student receives the normal vehicle observation and its previous action.
Its bounded residual augments DEFAULT stage-aware feedback, whose parameters do
not require a per-craft optimization at inference. GRU history is reset for every
craft. This is supervised simulator research, never an approved KSP checkpoint.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch import nn

from .ascent_planner import (
    DEFAULT,
    PARAMETER_NAMES,
    SCHEMA as TEACHER_SCHEMA,
    config_fingerprint,
    guidance_action,
    qualify_witness,
)
from .env import EnvConfig
from .generalized_aero_env import BatchedAeroAscent, FEATURE_SCHEMA, OBSERVATIONS


TASK = "stage_aware_residual_distillation_v1"


def vehicle_fingerprint(config):
    """Physical craft identity, independent of target/deadline/body/timestep."""
    fields = (
        "payload_mass", "stages", "drag_area", "drag_alpha_area", "lift_area",
        "attitude_rate", "attitude_tau",
    )
    values = asdict(config)

    def physical(value):
        if isinstance(value, dict):
            return {key: physical(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [physical(item) for item in value]
        return float(value)

    return hashlib.sha256(json.dumps(
        {name: physical(values[name]) for name in fields}, sort_keys=True, allow_nan=False
    ).encode()).hexdigest()


def _write_json(path, value):
    def finite(item):
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, dict):
            return {key: finite(part) for key, part in item.items()}
        if isinstance(item, (list, tuple)):
            return [finite(part) for part in item]
        return item

    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(finite(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _pressure_limit(*limits):
    selected = []
    for limit in limits:
        if limit is not None:
            if isinstance(limit, bool) or not math.isfinite(limit) or limit <= 0:
                raise ValueError("pressure limit must be positive and finite")
            selected.append(float(limit))
    return min(selected) if selected else None


@dataclass(frozen=True)
class Trajectory:
    config: EnvConfig
    config_sha256: str
    vehicle_sha256: str
    observations: torch.Tensor
    previous_actions: torch.Tensor
    baseline_actions: torch.Tensor
    teacher_actions: torch.Tensor
    qualification: dict


@dataclass(frozen=True)
class WitnessBank:
    trajectories: tuple[Trajectory, ...]
    configs: tuple[EnvConfig, ...]
    source_sha256: str
    report_craft_sha256: tuple[str, ...]
    report_vehicle_sha256: tuple[str, ...]
    reported_qualified: tuple[bool, ...]
    max_dynamic_pressure: float | None
    excluded_unqualified: int


def _recorded_qualification(row, fingerprint, action_count):
    qualification = row.get("qualification")
    if not isinstance(qualification, dict):
        raise ValueError("qualified teacher lacks scalar qualification")
    coast = qualification.get("coast")
    if not all(
        qualification.get(key) is True
        for key in ("qualified", "scalar_success", "pressure_constraint_satisfied")
    ) or not isinstance(coast, dict) or coast.get("success") is not True:
        raise ValueError("teacher requires successful scalar replay and full coast")
    duration, period = coast.get("duration_s"), coast.get("requested_period_s")
    if (
        not isinstance(duration, (int, float))
        or not isinstance(period, (int, float))
        or not math.isfinite(duration)
        or not math.isfinite(period)
        or period <= 0
        or duration < period - 1e-6
    ):
        raise ValueError("teacher coast must cover a complete finite orbit")
    if (
        qualification.get("config_sha256") != fingerprint
        or qualification.get("actions_consumed") != action_count
        or qualification.get("total_actions") != action_count
    ):
        raise ValueError("teacher qualification does not cover this exact witness")


@torch.no_grad()
def replay_labels(config, actions, qualification):
    """Collect ordered pre-action observations; never relabel a future action."""
    env = BatchedAeroAscent(1, seed=0)
    env.load_configs([config])
    parameters = torch.tensor(DEFAULT, dtype=env.dtype)[None]
    previous = torch.zeros(2)
    observations, previous_actions, baseline_actions = [], [], []
    for index, action in enumerate(actions):
        observations.append(env.observation()[0].clone())
        previous_actions.append(previous.clone())
        baseline_actions.append(guidance_action(env, parameters)[0].clone())
        _, _, terminated, truncated, info = env.step(action[None])
        previous = action
        if bool(terminated[0] | truncated[0]):
            if index != len(actions) - 1 or not bool(info["success"][0]):
                raise ValueError("teacher actions do not finish the same successful batched episode")
            break
    else:
        raise ValueError("teacher actions end before a successful terminal")
    return Trajectory(
        config=config,
        config_sha256=config_fingerprint(config),
        vehicle_sha256=vehicle_fingerprint(config),
        observations=torch.stack(observations),
        previous_actions=torch.stack(previous_actions),
        baseline_actions=torch.stack(baseline_actions),
        teacher_actions=actions,
        qualification=qualification,
    )


@torch.no_grad()
def replay_labels_batch(configs, actions, qualifications):
    """Reconstruct independent ordered labels in compatible physics batches.

    Scalar/coast qualification remains the caller's responsibility. A batch
    shares only the global values required by ``load_configs``; craft, targets,
    stage counts and witness lengths remain independent. Completed rows receive
    zero-duration padded steps and contribute no further labels.
    """
    configs, actions, qualifications = tuple(configs), tuple(actions), tuple(qualifications)
    if not configs or len(configs) != len(actions) or len(configs) != len(qualifications):
        raise ValueError("provide a config, action witness and qualification for each craft")
    commands = tuple(torch.as_tensor(row).detach().cpu() for row in actions)
    for config, row in zip(configs, commands):
        if (
            not isinstance(config, EnvConfig) or row.ndim != 2 or row.shape[1] != 2
            or not len(row) or not row.is_floating_point()
            or not bool(torch.isfinite(row).all()) or bool((row.abs() > 1).any())
        ):
            raise ValueError("teacher actions must be a bounded finite ordered [steps,2] array")
    shared = (
        "mu", "radius", "rotation_period", "atmosphere_depth", "atmosphere",
        "control_dt", "physics_dt", "max_time", "gamma", "orbit_tolerance",
    )
    groups = {}
    for index, config in enumerate(configs):
        key = json.dumps({name: getattr(config, name) for name in shared}, sort_keys=True, allow_nan=False)
        groups.setdefault(key, []).append(index)
    trajectories = [None] * len(configs)
    for indices in groups.values():
        env = BatchedAeroAscent(len(indices), seed=0)
        env.load_configs([configs[index] for index in indices])
        parameters = torch.tensor(DEFAULT, dtype=env.dtype).expand(env.count, -1)
        lengths = torch.tensor([len(commands[index]) for index in indices])
        steps = int(lengths.max())
        padded = torch.zeros(steps, env.count, 2, dtype=env.dtype)
        for world, index in enumerate(indices):
            padded[:lengths[world], world] = commands[index]
        observations = torch.empty(steps, env.count, OBSERVATIONS)
        baseline_actions = torch.empty(steps, env.count, 2)
        done = torch.zeros(env.count, dtype=torch.bool)
        for step in range(steps):
            observations[step] = env.observation()
            baseline_actions[step] = guidance_action(env, parameters)
            _, _, terminated, truncated, info = env.step(padded[step])
            ended = terminated | truncated
            if bool((~done & ended & ((lengths != step + 1) | ~info["success"])).any()):
                raise ValueError("teacher actions do not finish the same successful batched episode")
            if bool((~done & ~ended & (lengths == step + 1)).any()):
                raise ValueError("teacher actions end before a successful terminal")
            done |= ended
            # step() has no persistent done mask. Expiring only completed worlds
            # makes every subsequent padded integration interval exactly zero.
            env.time[done] = env.max_time
        if not bool(done.all()):
            raise ValueError("teacher actions end before a successful terminal")
        for world, index in enumerate(indices):
            row = commands[index]
            trajectories[index] = Trajectory(
                config=configs[index], config_sha256=config_fingerprint(configs[index]),
                vehicle_sha256=vehicle_fingerprint(configs[index]),
                observations=observations[:len(row), world].clone(),
                previous_actions=torch.cat((torch.zeros(1, 2), row[:-1])),
                baseline_actions=baseline_actions[:len(row), world].clone(),
                teacher_actions=row, qualification=qualifications[index],
            )
    return tuple(trajectories)


def load_witnesses(path, *, max_dynamic_pressure=None, maximum_crafts=None):
    """Accept only exact, freshly scalar-requalified teacher trajectories.

    Failed searches remain excluded rows. A claimed qualification that cannot
    be reproduced is an error rather than a silently accepted training label.
    """
    path = Path(path)
    raw = path.read_bytes()
    report = json.loads(raw)
    if report.get("schema") != TEACHER_SCHEMA or report.get("qualification_requested") is not True:
        raise ValueError("a scalar-qualified ascent planner report is required")
    if maximum_crafts is not None and (
        isinstance(maximum_crafts, bool) or not isinstance(maximum_crafts, int) or maximum_crafts < 1
    ):
        raise ValueError("maximum crafts must be a positive integer")
    pressure = _pressure_limit(report.get("budget", {}).get("max_dynamic_pressure"), max_dynamic_pressure)
    rows = report.get("results")
    if not isinstance(rows, list) or not rows:
        raise ValueError("teacher report has no craft rows")
    fingerprints, configs, vehicles, reported_qualified = [], [], [], []
    label_configs, label_actions, label_qualifications = [], [], []
    excluded = 0
    for row in rows:
        config = EnvConfig.from_dict(row["config"])
        fingerprint = config_fingerprint(config)
        if row.get("config_sha256") != fingerprint or fingerprint in fingerprints:
            raise ValueError("teacher config hash differs or a craft is duplicated")
        fingerprints.append(fingerprint)
        configs.append(config)
        vehicles.append(vehicle_fingerprint(config))
        reported_qualified.append(row.get("qualified") is True)
        if row.get("qualified") is not True:
            excluded += 1
            continue
        if maximum_crafts is not None and len(label_configs) >= maximum_crafts:
            continue
        actions = torch.as_tensor(row.get("actions"), dtype=torch.float32)
        if (
            actions.ndim != 2 or actions.shape[1] != 2 or not len(actions)
            or len(actions) > math.ceil(config.max_time / config.control_dt) + 1
            or not bool(torch.isfinite(actions).all()) or bool((actions.abs() > 1).any())
        ):
            raise ValueError("teacher actions must be a bounded finite ordered [steps,2] array")
        _recorded_qualification(row, fingerprint, len(actions))
        qualification = qualify_witness(config, actions.tolist(), max_dynamic_pressure=pressure)
        if not qualification["qualified"] or qualification["actions_consumed"] != len(actions):
            raise ValueError("claimed teacher failed fresh scalar replay/coast or pressure gate")
        label_configs.append(config)
        label_actions.append(actions)
        label_qualifications.append(qualification)
    if not label_configs:
        raise ValueError("teacher report has no qualified labels")
    trajectories = replay_labels_batch(label_configs, label_actions, label_qualifications)
    return WitnessBank(
        tuple(trajectories), tuple(configs), hashlib.sha256(raw).hexdigest(),
        tuple(fingerprints), tuple(vehicles), tuple(reported_qualified), pressure, excluded
    )


def require_disjoint_banks(*banks):
    seen = set()
    for bank in banks:
        if seen.intersection(bank.report_vehicle_sha256):
            raise ValueError("craft fingerprints overlap between training/held-out banks")
        seen.update(bank.report_vehicle_sha256)


def student_contract(
    architecture, *, hidden, residual_bounds, hide_aero, bptt_steps,
    source_witness_sha256, train_crafts, validation_crafts,
    qualification_pressure_limits=None,
):
    if architecture not in ("feedforward", "gru"):
        raise ValueError("unknown student architecture")
    if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden < 1:
        raise ValueError("hidden size must be a positive integer")
    if isinstance(bptt_steps, bool) or not isinstance(bptt_steps, int) or bptt_steps < 1:
        raise ValueError("BPTT steps must be a positive integer")
    if len(residual_bounds) != 2 or any(
        isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 2
        for value in residual_bounds
    ):
        raise ValueError("two residual bounds in (0,2] are required")
    if not isinstance(hide_aero, bool):
        raise ValueError("aero feature mask must be explicit")
    if set(source_witness_sha256) != {"train", "validation"} or not all(
        _sha256(value) for value in source_witness_sha256.values()
    ):
        raise ValueError("source witness hashes must bind both dataset reports")
    train_crafts, validation_crafts = list(train_crafts), list(validation_crafts)
    if not train_crafts or not validation_crafts or not all(
        _sha256(value) for value in train_crafts + validation_crafts
    ) or len(set(train_crafts + validation_crafts)) != len(train_crafts + validation_crafts):
        raise ValueError("student contract requires distinct train and validation craft hashes")
    qualification_pressure_limits = qualification_pressure_limits or dict(train=None, validation=None)
    if set(qualification_pressure_limits) != {"train", "validation"}:
        raise ValueError("qualification pressure limits must describe both banks")
    pressures = {key: _pressure_limit(value) for key, value in qualification_pressure_limits.items()}
    return dict(
        task=TASK, version=1, architecture=architecture, hidden=hidden,
        observations=OBSERVATIONS, feature_schema=FEATURE_SCHEMA, input_features=58,
        masked_observation_indices=[54, 55] if hide_aero else [],
        previous_action_dimensions=2, initial_previous_action=[0.0, 0.0],
        previous_action_training="teacher_forcing", previous_action_inference="student_output",
        recurrent_reset="zero_at_each_craft", sequence_training="ordered_truncated_bptt",
        bptt_steps=bptt_steps, residual_bounds=[float(value) for value in residual_bounds],
        action="clip(DEFAULT_guidance + bounds*tanh(residual), -1, 1)",
        baseline_parameter_names=list(PARAMETER_NAMES), baseline_parameters=list(DEFAULT),
        source_witness_sha256=dict(source_witness_sha256),
        qualification_pressure_limits_pa=pressures,
        train_craft_sha256=train_crafts, validation_craft_sha256=validation_crafts,
        objective="qualified_teacher_action_mse", live_ksp_validated=False,
    )


class ResidualStudent(nn.Module):
    def __init__(self, contract):
        super().__init__()
        self.contract = contract
        hidden = contract["hidden"]
        self.encoder = nn.Sequential(nn.Linear(58, hidden), nn.Tanh())
        self.memory = (
            nn.GRU(hidden, hidden, batch_first=True)
            if contract["architecture"] == "gru"
            else nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        )
        self.output = nn.Linear(hidden, 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.register_buffer("bounds", torch.tensor(contract["residual_bounds"]))

    def forward(self, observations, previous_actions, baseline_actions, hidden=None, *, clamp=True):
        if observations.ndim != 3 or observations.shape[-1] != OBSERVATIONS:
            raise ValueError("student observations must have [crafts,time,56] shape")
        if previous_actions.shape != (*observations.shape[:2], 2) or baseline_actions.shape != previous_actions.shape:
            raise ValueError("previous and baseline actions must match the ordered sequence")
        if self.contract["masked_observation_indices"]:
            observations = observations.clone()
            observations[..., self.contract["masked_observation_indices"]] = 0
        features = self.encoder(torch.cat((observations, previous_actions), -1))
        if self.contract["architecture"] == "gru":
            features, hidden = self.memory(features, hidden)
        else:
            if hidden is not None:
                raise ValueError("feedforward student has no recurrent state")
            features, hidden = self.memory(features), None
        action = baseline_actions + self.bounds * self.output(features).tanh()
        # Training fits the unclamped sum: past a bound, clamp(-1, 1) has zero
        # gradient and a wrong-side residual could never be corrected.
        return (action.clamp(-1, 1) if clamp else action), hidden

    def step(self, observation, previous_action, baseline_action, hidden=None):
        action, hidden = self(observation[:, None], previous_action[:, None], baseline_action[:, None], hidden)
        return action[:, 0], hidden


def save_student(path, model, ancestry):
    path = Path(path)
    temporary = path.with_suffix(".tmp.pt")
    torch.save(dict(task=TASK, contract=model.contract, state_dict=model.state_dict(), ancestry=ancestry), temporary)
    temporary.replace(path)
    metadata = dict(task=TASK, contract=model.contract, ancestry=ancestry,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(), live_ksp_validated=False)
    _write_json(path.with_suffix(".metadata.json"), metadata)


def load_student(path, *, device="cpu"):
    path = Path(path)
    metadata_path = path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
    if metadata is not None and metadata.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("student checkpoint hash differs from metadata")
    record = torch.load(path, weights_only=True, map_location=device)
    contract = record.get("contract", {})
    try:
        expected = student_contract(
            contract["architecture"], hidden=contract["hidden"],
            residual_bounds=contract["residual_bounds"],
            hide_aero=contract["masked_observation_indices"] == [54, 55],
            bptt_steps=contract["bptt_steps"],
            source_witness_sha256=contract["source_witness_sha256"],
            train_crafts=contract["train_craft_sha256"], validation_crafts=contract["validation_craft_sha256"],
            qualification_pressure_limits=contract["qualification_pressure_limits_pa"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid residual student contract") from error
    if record.get("task") != TASK or contract != expected:
        raise ValueError("student task/history/feature contract differs")
    if metadata is not None and any(
        metadata.get(key) != record.get(key) for key in ("task", "contract", "ancestry")
    ):
        raise ValueError("student checkpoint and metadata disagree")
    if not all(
        isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all())
        for value in record["state_dict"].values()
    ):
        raise ValueError("student weights must be finite tensors")
    model = ResidualStudent(contract).to(device)
    model.load_state_dict(record["state_dict"], strict=True)
    if not all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise ValueError("student weights overflow their runtime tensor dtype")
    if not torch.equal(model.bounds, torch.tensor(contract["residual_bounds"], device=device)):
        raise ValueError("student residual bounds disagree with contract")
    return model, record


def sequence_batch(trajectories, *, device):
    length = max(len(trajectory.teacher_actions) for trajectory in trajectories)
    batch = {}
    for name, width in (("observations", 56), ("previous_actions", 2), ("baseline_actions", 2), ("teacher_actions", 2)):
        values = torch.zeros(len(trajectories), length, width, device=device)
        for index, trajectory in enumerate(trajectories):
            tensor = getattr(trajectory, name)
            values[index, :len(tensor)] = tensor.to(device)
        batch[name] = values
    batch["valid"] = torch.arange(length, device=device)[None] < torch.tensor(
        [len(trajectory.teacher_actions) for trajectory in trajectories], device=device
    )[:, None]
    return batch


def _sequence_pass(model, trajectories, *, batch_size, bptt_steps, optimizer=None, generator=None, stop_requested=None):
    device = next(model.parameters()).device
    order = torch.randperm(len(trajectories), generator=generator).tolist() if optimizer is not None else list(range(len(trajectories)))
    squared_error, action_count = 0.0, 0
    model.train(optimizer is not None)
    for offset in range(0, len(order), batch_size):
        batch = sequence_batch([trajectories[index] for index in order[offset:offset + batch_size]], device=device)
        hidden = None  # New craft batch: never carry another craft's history.
        for start in range(0, batch["valid"].shape[1], bptt_steps):
            selected = slice(start, start + bptt_steps)
            with torch.set_grad_enabled(optimizer is not None):
                predicted, hidden = model(
                    batch["observations"][:, selected], batch["previous_actions"][:, selected],
                    batch["baseline_actions"][:, selected], hidden, clamp=optimizer is None,
                )
                valid = batch["valid"][:, selected]
                error = (predicted - batch["teacher_actions"][:, selected]).square()[valid]
                loss = error.mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("student loss is nonfinite")
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            if hidden is not None:
                hidden = hidden.detach()  # Preserve order/state; truncate gradients only.
            squared_error += float(error.detach().sum())
            action_count += error.numel()
            if stop_requested is not None and stop_requested():
                return squared_error / action_count
    return squared_error / action_count


def train_student(model, train, validation, *, epochs, batch_size, learning_rate, seed, checkpoint):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    bptt_steps = model.contract["bptt_steps"]
    best = _sequence_pass(model, validation, batch_size=batch_size, bptt_steps=bptt_steps)
    history = [dict(epoch=0, validation_action_mse=best)]
    save_student(checkpoint, model, dict(seed=seed, selected_epoch=0, validation_action_mse=best))
    for epoch in range(1, epochs + 1):
        loss = _sequence_pass(model, train, batch_size=batch_size, bptt_steps=bptt_steps, optimizer=optimizer, generator=generator)
        validation_loss = _sequence_pass(model, validation, batch_size=batch_size, bptt_steps=bptt_steps)
        history.append(dict(epoch=epoch, training_action_mse=loss, validation_action_mse=validation_loss))
        if validation_loss < best:
            best = validation_loss
            save_student(checkpoint, model, dict(seed=seed, selected_epoch=epoch, validation_action_mse=best))
    return history


@torch.no_grad()
def evaluate_controller(model, bank, *, device="cpu", stop_requested=None):
    """Closed-loop student actions, followed by independent scalar/coast proof."""
    configs = bank.configs
    env = BatchedAeroAscent(len(configs), device=device, seed=0)
    env.load_configs(configs)
    parameters = torch.tensor(DEFAULT, device=device, dtype=env.dtype).expand(env.count, -1)
    previous = torch.zeros(env.count, 2, device=device)
    done = torch.zeros(env.count, dtype=torch.bool, device=device)
    actions = [[] for _ in configs]
    hidden = None
    if model is not None:
        model.eval()
    for _ in range(math.ceil(env.max_time / env.control_dt) + 1):
        if stop_requested is not None and stop_requested():
            raise InterruptedError("student evaluation interrupted")
        baseline = guidance_action(env, parameters)
        if model is None:
            action = baseline
        else:
            action, hidden = model.step(env.observation(), previous, baseline, hidden)
        active_indices = torch.nonzero(~done).flatten().cpu().tolist()
        action_rows = action.cpu().tolist()
        for index in active_indices:
            actions[index].append(action_rows[index])
        _, _, terminated, truncated, _ = env.step(action)
        done |= terminated | truncated
        previous = action
        if bool(done.all()):
            break
        env.time[done] = env.max_time
    if not bool(done.all()):
        raise RuntimeError("student evaluation did not finish its finite mission")
    rows = []
    for config, commands, teacher_qualified in zip(configs, actions, bank.reported_qualified):
        if stop_requested is not None and stop_requested():
            raise InterruptedError("student qualification interrupted")
        qualification = qualify_witness(
            config, commands, max_dynamic_pressure=bank.max_dynamic_pressure
        )
        rows.append(dict(config_sha256=config_fingerprint(config), vehicle_sha256=vehicle_fingerprint(config),
                         reported_teacher_qualified=teacher_qualified,
                         qualification=qualification, actions=commands))
    successes = sum(row["qualification"]["qualified"] for row in rows)
    return dict(episodes=len(rows), qualified_successes=successes, success_rate=successes / len(rows),
                results=rows, live_ksp_validated=False, full_unpowered_scalar_coast_required=True,
                includes_unqualified_teacher_crafts=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-report", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=707)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bptt-steps", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--residual-bounds", type=float, nargs=2, default=(2.0, 1.0))
    parser.add_argument("--hide-aero", action="store_true")
    parser.add_argument("--max-dynamic-pressure", type=float)
    parser.add_argument("--max-train-crafts", type=int)
    parser.add_argument("--max-validation-crafts", type=int)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args(argv)
    if min(args.epochs, args.hidden, args.batch_size, args.bptt_steps, args.cpu_threads) < 1:
        parser.error("training counts must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning rate must be positive and finite")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        parser.error("run directory must be empty")
    torch.set_num_threads(args.cpu_threads)
    start = time.monotonic()
    train = load_witnesses(args.train_report, max_dynamic_pressure=args.max_dynamic_pressure, maximum_crafts=args.max_train_crafts)
    validation = load_witnesses(args.validation_report, max_dynamic_pressure=args.max_dynamic_pressure, maximum_crafts=args.max_validation_crafts)
    require_disjoint_banks(train, validation)
    contract_options = dict(
        hidden=args.hidden, residual_bounds=args.residual_bounds,
        hide_aero=args.hide_aero, bptt_steps=args.bptt_steps,
        source_witness_sha256=dict(train=train.source_sha256, validation=validation.source_sha256),
        train_crafts=list(dict.fromkeys(row.vehicle_sha256 for row in train.trajectories)),
        validation_crafts=list(dict.fromkeys(validation.report_vehicle_sha256)),
        qualification_pressure_limits=dict(train=train.max_dynamic_pressure, validation=validation.max_dynamic_pressure),
    )
    # Validate every architecture contract before creating the output directory.
    contracts = {architecture: student_contract(architecture, **contract_options) for architecture in ("feedforward", "gru")}
    args.run_dir.mkdir(parents=True, exist_ok=True)
    report = dict(
        task=TASK, source_witness_sha256=dict(train=train.source_sha256, validation=validation.source_sha256),
        train_crafts=len(train.trajectories), validation_crafts=len(validation.configs),
        validation_label_crafts=len(validation.trajectories),
        excluded_unqualified=dict(train=train.excluded_unqualified, validation=validation.excluded_unqualified),
        validation_is_final_unseen_test=False, validation_selects_epoch=True,
        reported_teacher_qualified_successes=sum(validation.reported_qualified), live_ksp_validated=False,
        default_guidance=evaluate_controller(None, validation, device=args.device), students={},
        teacher_residual_coverage={
            name: float(torch.cat([
                ((trajectory.teacher_actions - trajectory.baseline_actions).abs() <= torch.tensor(args.residual_bounds) + 1e-6).all(-1)
                for trajectory in bank.trajectories
            ]).float().mean())
            for name, bank in (("train", train), ("validation", validation))
        },
        limitations=["Teacher-forced previous actions create exposure bias; no DAgger performed",
                     "Small recurrent history is an adaptation hypothesis, not a proven adaptive controller",
                     "Validation chooses the epoch; a separate final unseen bank and actual KSP flights remain required",
                     "All labels and tests use the planar sequential-liquid-stage surrogate"],
    )
    for architecture in ("feedforward", "gru"):
        contract = contracts[architecture]
        torch.manual_seed(args.seed)
        model = ResidualStudent(contract).to(args.device)
        path = args.run_dir / f"{architecture}.pt"
        history = train_student(
            model, train.trajectories, validation.trajectories, epochs=args.epochs,
            batch_size=args.batch_size, learning_rate=args.learning_rate, seed=args.seed,
            checkpoint=path,
        )
        model, record = load_student(path, device=args.device)
        report["students"][architecture] = dict(
            contract=contract, ancestry=record["ancestry"], training_history=history,
            closed_loop_validation=evaluate_controller(model, validation, device=args.device),
            parameters=sum(parameter.numel() for parameter in model.parameters()),
        )
    report["elapsed_s"] = time.monotonic() - start
    _write_json(args.run_dir / "summary.json", report)
    print(json.dumps(dict(
        train_crafts=report["train_crafts"], validation_crafts=report["validation_crafts"],
        default_qualified=report["default_guidance"]["qualified_successes"],
        students={key: value["closed_loop_validation"]["qualified_successes"] for key, value in report["students"].items()},
        elapsed_s=report["elapsed_s"], live_ksp_validated=False,
    )))


if __name__ == "__main__":
    main()
