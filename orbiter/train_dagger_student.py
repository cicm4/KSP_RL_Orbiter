"""On-policy feedback imitation for experimental stage-aware residual students.

Teachers are qualified only from the initial launch state. Their off-trajectory
feedback is a research oracle, not a recovery or safety guarantee. Validation
uses the student alone and independently replays/coasts all resulting flights.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import signal
import time

import torch

from .ascent_planner import DEFAULT, LOWER, UPPER, PARAMETER_NAMES, guidance_action
from .distill_ascent import (
    ResidualStudent, Trajectory, _sequence_pass, _write_json, evaluate_controller,
    load_witnesses, require_disjoint_banks, student_contract,
)
from .generalized_aero_env import BatchedAeroAscent

TASK = "stage_aware_onpolicy_imitation_v1"
_STOP = False


def _request_stop(_signum, _frame):
    global _STOP
    _STOP = True


@dataclass(frozen=True)
class FeedbackSpec:
    trajectory: Trajectory
    parameters: tuple[float, ...]
    heuristic: bool


def feedback_specs(path, bank):
    rows = {r["config_sha256"]: r for r in json.loads(Path(path).read_text())["results"]}
    specs = []
    for trajectory in bank.trajectories:
        row = rows[trajectory.config_sha256]
        if row.get("selected_plan") not in ("heuristic", "stage_aware_profile"):
            raise ValueError("feedback teacher must identify its guidance family")
        values = row.get("parameters", {})
        if set(values) != set(PARAMETER_NAMES):
            raise ValueError("feedback teacher parameters are incomplete")
        parameters = tuple(values[k] for k in PARAMETER_NAMES)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
               not math.isfinite(v) or not lo <= v <= hi
               for v, lo, hi in zip(parameters, LOWER, UPPER)):
            raise ValueError("feedback teacher parameters exceed the search bounds")
        specs.append(FeedbackSpec(trajectory, parameters, row["selected_plan"] == "heuristic"))
    return tuple(specs)


def make_contract(base, *, dataset_mode, beta_start, beta_decay, max_trajectories):
    if dataset_mode not in ("recent", "aggregate"):
        raise ValueError("dataset mode must be recent or aggregate")
    if any(isinstance(v, bool) or not math.isfinite(v) or not 0 <= v <= 1
           for v in (beta_start, beta_decay)):
        raise ValueError("teacher mixing coefficients must be finite in [0,1]")
    if isinstance(max_trajectories, bool) or not isinstance(max_trajectories, int) or max_trajectories < 1:
        raise ValueError("aggregate trajectory capacity must be positive")
    base = dict(base, task=TASK, previous_action_training="actual_executed_mixture_after_teacher_warmup",
                objective="current_state_feedback_action_mse")
    return dict(task=TASK, version=1, student=base, dataset_mode=dataset_mode,
                beta_start=float(beta_start), beta_decay=float(beta_decay),
                max_trajectories=max_trajectories,
                labels="feedback teacher queried at current visited pre-action state",
                previous_actions="actual executed actions, including teacher mixing",
                teacher_qualification="initial launch witness only; off-trajectory recovery unqualified",
                selection="student-only scalar/coast-qualified validation count, then action MSE",
                validation_is_final_unseen_test=False, live_ksp_validated=False)


def save_checkpoint(path, model, contract, ancestry):
    path = Path(path)
    temporary = path.with_suffix(".tmp.pt")
    torch.save(dict(task=TASK, contract=contract, ancestry=ancestry,
                    state_dict=model.state_dict()), temporary)
    temporary.replace(path)
    _write_json(path.with_suffix(".metadata.json"), dict(
        task=TASK, contract=contract, ancestry=ancestry,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), live_ksp_validated=False))


def load_checkpoint(path, *, device="cpu"):
    path = Path(path)
    metadata = json.loads(path.with_suffix(".metadata.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata.get("sha256"):
        raise ValueError("on-policy checkpoint hash mismatch")
    record = torch.load(path, weights_only=True, map_location=device)
    if any(metadata.get(k) != record.get(k) for k in ("task", "contract", "ancestry")):
        raise ValueError("on-policy metadata differs from checkpoint")
    try:
        c = record["contract"]
        b = c["student"]
        base = student_contract(
            b["architecture"], hidden=b["hidden"], residual_bounds=b["residual_bounds"],
            hide_aero=b["masked_observation_indices"] == [54, 55], bptt_steps=b["bptt_steps"],
            source_witness_sha256=b["source_witness_sha256"], train_crafts=b["train_craft_sha256"],
            validation_crafts=b["validation_craft_sha256"],
            qualification_pressure_limits=b["qualification_pressure_limits_pa"])
        expected = make_contract(base, dataset_mode=c["dataset_mode"], beta_start=c["beta_start"],
                                 beta_decay=c["beta_decay"], max_trajectories=c["max_trajectories"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError("invalid on-policy student contract") from e
    if record.get("task") != TASK or c != expected:
        raise ValueError("on-policy history or objective contract mismatch")
    if not all(isinstance(t, torch.Tensor) and bool(torch.isfinite(t).all())
               for t in record["state_dict"].values()):
        raise ValueError("on-policy weights must be finite tensors")
    model = ResidualStudent(c["student"]).to(device)
    model.load_state_dict(record["state_dict"], strict=True)
    if not all(bool(torch.isfinite(t).all()) for t in model.state_dict().values()):
        raise ValueError("on-policy weights overflow runtime dtype")
    if not torch.equal(model.bounds, torch.tensor(base["residual_bounds"], device=device)):
        raise ValueError("on-policy action bounds differ from contract")
    return model, record


@torch.no_grad()
def collect_rollouts(model, specs, *, device, beta, seed, repeats=1, verify_witness=False):
    """Query teachers at visited states; never replay indexed actions as labels."""
    if not 0 <= beta <= 1 or not math.isfinite(beta) or repeats < 1:
        raise ValueError("valid mixing probability and repeat count required")
    specs = tuple(specs) * repeats
    env = BatchedAeroAscent(len(specs), device=device, seed=seed)
    env.load_configs([s.trajectory.config for s in specs])
    parameters = torch.tensor([s.parameters for s in specs], device=device, dtype=env.dtype)
    default = torch.tensor(DEFAULT, device=device, dtype=env.dtype).expand(env.count, -1)
    heuristic = torch.tensor([s.heuristic for s in specs], device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    done = torch.zeros(env.count, device=device, dtype=torch.bool)
    lengths = torch.zeros(env.count, device=device, dtype=torch.long)
    previous = torch.zeros(env.count, 2, device=device)
    hidden = None
    histories = {k: [] for k in ("observations", "previous_actions", "baseline_actions", "teacher_actions")}
    if model is not None:
        model.eval()
    for step in range(math.ceil(env.max_time / env.control_dt) + 1):
        if _STOP:
            raise InterruptedError("feedback collection interrupted")
        observation = env.observation()
        teacher = guidance_action(env, parameters)
        if bool(heuristic.any()):
            teacher = torch.where(heuristic[:, None], env.teacher(), teacher)
        baseline = guidance_action(env, default)
        action = teacher
        if model is not None:
            student, hidden = model.step(observation, previous, baseline, hidden)
            use_teacher = torch.rand(env.count, generator=generator, device=device) < beta
            action = torch.where(use_teacher[:, None], teacher, student)
        if verify_witness:
            if model is not None:
                raise ValueError("feedback witness verification must use the teacher alone")
            labels = teacher.cpu()
            finished = done.cpu().tolist()
            for index, spec in enumerate(specs):
                if not finished[index]:
                    expected = spec.trajectory.teacher_actions
                    if step >= len(expected) or not torch.allclose(labels[index], expected[step], atol=1e-5, rtol=0):
                        raise ValueError("feedback parameters do not reproduce their qualified witness")
        for key, tensor in zip(histories, (observation, previous, baseline, teacher)):
            histories[key].append(tensor.clone())
        lengths += (~done).long()
        _, _, terminated, truncated, _ = env.step(action)
        done |= terminated | truncated
        previous = action
        if bool(done.all()):
            break
        env.time[done] = env.max_time
    if not bool(done.all()):
        raise RuntimeError("on-policy collection exceeded its finite episode")
    batches = {k: torch.stack(v).transpose(0, 1).cpu() for k, v in histories.items()}
    result = []
    for index, (spec, length) in enumerate(zip(specs, lengths.cpu().tolist())):
        if verify_witness and length != len(spec.trajectory.teacher_actions):
            raise ValueError("feedback witness has a different terminal time")
        t = spec.trajectory
        result.append(Trajectory(t.config, t.config_sha256, t.vehicle_sha256,
                                 **{k: v[index, :length].clone() for k, v in batches.items()},
                                 qualification=dict(qualified=False, scope="off-trajectory feedback labels; no recovery proof")))
    return tuple(result)


def update_dataset(original, accumulated, collected, *, mode, capacity):
    if capacity < len(original):
        raise ValueError("dataset capacity cannot discard the initial qualified bank")
    if mode == "recent":
        return tuple(collected)
    if mode != "aggregate":
        raise ValueError("unknown dataset mode")
    room = capacity - len(original)
    tail = tuple(accumulated) + tuple(collected)
    return tuple(original) + (tail[-room:] if room else ())


def main(argv=None):
    global _STOP
    _STOP = False
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-report", type=Path, required=True)
    p.add_argument("--validation-report", type=Path, required=True)
    p.add_argument("--test-report", type=Path, help="optional disjoint final comparison bank; never selects weights")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--architecture", choices=("feedforward", "gru"), default="gru")
    p.add_argument("--dataset-mode", choices=("recent", "aggregate"), default="recent")
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--epochs-per-round", type=int, default=10)
    p.add_argument("--rollout-repeats", type=int, default=1)
    p.add_argument("--max-trajectories", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--bptt-steps", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--beta-start", type=float, default=0.0)
    p.add_argument("--beta-decay", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=807)
    p.add_argument("--hide-aero", action="store_true")
    p.add_argument("--max-dynamic-pressure", type=float)
    p.add_argument("--cpu-threads", type=int, default=1)
    args = p.parse_args(argv)
    if min(args.rounds, args.epochs_per_round, args.warmup_epochs, args.rollout_repeats,
           args.hidden, args.batch_size, args.bptt_steps, args.cpu_threads, args.max_trajectories) < 1:
        p.error("training counts must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        p.error("learning rate must be positive and finite")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error("run directory must be empty")
    if args.rollout_repeats > 1 and args.beta_start == 0:
        p.error("pure deterministic student rollouts need only one repeat")
    torch.set_num_threads(args.cpu_threads)
    started = time.monotonic()
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    train = load_witnesses(args.train_report, max_dynamic_pressure=args.max_dynamic_pressure)
    validation = load_witnesses(args.validation_report, max_dynamic_pressure=args.max_dynamic_pressure)
    test = load_witnesses(args.test_report, max_dynamic_pressure=args.max_dynamic_pressure) if args.test_report else None
    require_disjoint_banks(train, validation, *([test] if test is not None else []))
    specs = feedback_specs(args.train_report, train)
    base = student_contract(
        args.architecture, hidden=args.hidden, residual_bounds=(2, 1), hide_aero=args.hide_aero,
        bptt_steps=args.bptt_steps,
        source_witness_sha256=dict(train=train.source_sha256, validation=validation.source_sha256),
        train_crafts=list(dict.fromkeys(t.vehicle_sha256 for t in train.trajectories)),
        validation_crafts=list(dict.fromkeys(validation.report_vehicle_sha256)),
        qualification_pressure_limits=dict(train=train.max_dynamic_pressure, validation=validation.max_dynamic_pressure))
    contract = make_contract(base, dataset_mode=args.dataset_mode, beta_start=args.beta_start,
                             beta_decay=args.beta_decay, max_trajectories=args.max_trajectories)
    if args.max_trajectories < len(train.trajectories):
        p.error("dataset capacity must retain the initial bank")
    base = contract["student"]
    torch.manual_seed(args.seed)
    model = ResidualStudent(base).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.run_dir / "contract.json", contract)
    history = []
    best = (-1, -math.inf)
    selected_flights = None
    dataset = train.trajectories
    accumulated = ()
    ancestry = dict(seed=args.seed, round=-1, phase="initialization", interrupted=False)
    save_checkpoint(args.run_dir / "best.pt", model, contract, ancestry)
    save_checkpoint(args.run_dir / "latest.pt", model, contract, ancestry)
    _write_json(args.run_dir / "runtime.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    try:
        initial_mse = _sequence_pass(model, validation.trajectories, batch_size=args.batch_size,
                                     bptt_steps=args.bptt_steps, stop_requested=lambda: _STOP)
        selected_flights = evaluate_controller(model, validation, device=args.device, stop_requested=lambda: _STOP)
        best = (selected_flights["qualified_successes"], -initial_mse)
        ancestry = dict(seed=args.seed, round=-1, phase="initialization", interrupted=False,
                        validation_qualified=best[0], validation_action_mse=initial_mse)
        save_checkpoint(args.run_dir / "best.pt", model, contract, ancestry)
        # Initial launch qualification does not establish off-trajectory recovery.
        collect_rollouts(None, specs, device=args.device, beta=1, seed=args.seed, verify_witness=True)
        for round_index in range(args.rounds + 1):
            if round_index:
                beta = args.beta_start * args.beta_decay ** (round_index - 1)
                collected = collect_rollouts(model, specs, device=args.device, beta=beta,
                                             seed=args.seed + 10000 * round_index,
                                             repeats=args.rollout_repeats if beta > 0 else 1)
                if args.dataset_mode == "aggregate":
                    room = args.max_trajectories - len(train.trajectories)
                    accumulated = (accumulated + collected)[-room:] if room else ()
                dataset = update_dataset(train.trajectories, (),
                                         accumulated if args.dataset_mode == "aggregate" else collected,
                                         mode=args.dataset_mode, capacity=args.max_trajectories)
            epochs = args.warmup_epochs if round_index == 0 else args.epochs_per_round
            loss = None
            for _ in range(epochs):
                loss = _sequence_pass(model, dataset, batch_size=args.batch_size, bptt_steps=args.bptt_steps,
                                      optimizer=optimizer, generator=generator, stop_requested=lambda: _STOP)
                if _STOP:
                    raise InterruptedError("student optimization interrupted")
            mse = _sequence_pass(model, validation.trajectories, batch_size=args.batch_size, bptt_steps=args.bptt_steps,
                                 stop_requested=lambda: _STOP)
            flights = evaluate_controller(model, validation, device=args.device, stop_requested=lambda: _STOP)
            ancestry = dict(seed=args.seed, round=round_index, training_trajectories=len(dataset),
                            validation_qualified=flights["qualified_successes"], validation_action_mse=mse,
                            source_witness_sha256=base["source_witness_sha256"], interrupted=_STOP)
            save_checkpoint(args.run_dir / "latest.pt", model, contract, ancestry)
            score = (flights["qualified_successes"], -mse)
            if score > best:
                best = score
                selected_flights = flights
                save_checkpoint(args.run_dir / "best.pt", model, contract, ancestry)
            row = dict(round=round_index, training_action_mse=loss, validation_action_mse=mse,
                       qualified=flights["qualified_successes"], episodes=flights["episodes"],
                       training_trajectories=len(dataset), beta=None if round_index == 0 else beta)
            history.append(row)
            with (args.run_dir / "events.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            if _STOP:
                break
    except InterruptedError:
        ancestry = dict(seed=args.seed, round=len(history) - 1, phase="interrupted",
                        interrupted=True, validation_pending=True)
        save_checkpoint(args.run_dir / "latest.pt", model, contract, ancestry)
    selected, record = load_checkpoint(args.run_dir / "best.pt", device=args.device)
    final_test = None
    if test is not None and not _STOP:
        try:
            final_test = evaluate_controller(selected, test, device=args.device, stop_requested=lambda: _STOP)
        except InterruptedError:
            pass  # Keep already saved checkpoints; this final test remains incomplete.
    summary = dict(task=TASK, contract=contract, selected_ancestry=record["ancestry"], history=history,
                   closed_loop_validation=selected_flights,
                   final_comparison_test=final_test,
                   final_test_source_sha256=test.source_sha256 if test is not None else None,
                   final_test_selects_checkpoint=False,
                   validation_selects_checkpoint=True, validation_is_final_unseen_test=False,
                   live_ksp_validated=False, elapsed_s=time.monotonic() - started, interrupted=_STOP,
                   limitations=["Only initial teacher launch witnesses are qualified; recovery feedback is unqualified",
                                "Known physical parameters and planar sequential liquid stages only",
                                "Separate final unseen evaluation and actual KSP flights required"])
    _write_json(args.run_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
