"""Run a finite, paired deadline experiment on independent CUDA devices.

This controller starts simulator training only. It neither runs KSP nor installs
experimental weights. Each device has at most one child, and failures stop the
queue rather than silently rerunning a different experiment.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time


ARMS = (("truncate", "truncate", 0), ("deadline", "terminal", 0), ("deadline_failure", "terminal", -30))


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_checkpoint(directory, label="best"):
    directory = Path(directory)
    metadata = read_json(directory / f"{label}.metadata.json")
    digest = sha256(directory / f"{label}.pt")
    if metadata["sha256"] != digest:
        raise ValueError(f"checkpoint hash mismatch: {directory}/{label}.pt")
    return digest


def verify_inputs(manifest):
    """A queued job must see exactly the frozen source and parent it declares."""
    if "parent" not in manifest:  # Small process-supervision harnesses have no model.
        return
    if sha256(manifest["parent"]) != manifest["parent_sha256"]:
        raise ValueError("parent checkpoint changed during the experiment")
    root = Path(manifest["source_root"])
    current = {str(p.relative_to(root)): sha256(p)
               for p in sorted((root / "orbiter").glob("*.py"))}
    if current != manifest["source_sha256"]:
        raise ValueError("training source changed during the experiment")


def make_jobs(root, parent, *, seeds, validation_seed, test_seed, training):
    """Rotate arm order across seeds; the device is assigned only at launch."""
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("training seeds must be nonempty and distinct")
    if validation_seed == test_seed or set(seeds) & {validation_seed, test_seed}:
        raise ValueError("training, selection and comparison seeds must be distinct")
    root, parent = Path(root).resolve(), Path(parent).resolve()
    jobs = []
    for offset, seed in enumerate(seeds):
        arms = ARMS[offset % len(ARMS):] + ARMS[:offset % len(ARMS)]
        for arm, mode, penalty in arms:
            name = f"{arm}_seed{seed}"
            run_dir = root / name
            command = [
                sys.executable, "-u", "-m", "orbiter.train_generalized",
                "--run-dir", str(run_dir), "--device", "cuda:0",
                "--physics", "aero", "--curriculum", "stock",
                "--actor-from", str(parent), "--ppo-only",
                "--deadline-mode", mode, "--timeout-penalty", str(penalty),
                "--seed", str(seed), "--validation-seed", str(validation_seed),
                "--test-seed", str(test_seed), "--evaluation-suite", "ablation",
            ]
            for key, value in training.items():
                command.extend(("--" + key.replace("_", "-"), str(value)))
            jobs.append(dict(name=name, arm=arm, seed=seed, run_dir=str(run_dir),
                             command=command, status="pending", pid=None, device=None))
    return jobs


def validated_rows(report):
    rows = report["results"]
    by_world = {row["world"]: row for row in rows}
    if (not rows or len(rows) != report["episodes"] or len(by_world) != len(rows)):
        raise ValueError("comparison banks have different worlds or duplicate rows")
    successes = sum(bool(row["success"]) for row in rows)
    if (successes != report["successes"]
            or abs(successes / len(rows) - report["success_rate"]) > 1e-12):
        raise ValueError("reported success rate disagrees with episode outcomes")
    return by_world


def paired_outcomes(first, second):
    """Pair by explicit world and exact craft, never by array position alone."""
    left, right = validated_rows(first), validated_rows(second)
    if left.keys() != right.keys():
        raise ValueError("comparison banks have different worlds")
    gained = lost = both = neither = 0
    for world, a in left.items():
        b = right[world]
        if a.get("seed") != b.get("seed") or "craft" not in a or a["craft"] != b.get("craft"):
            raise ValueError("comparison banks are not the same frozen craft")
        x, y = bool(a["success"]), bool(b["success"])
        gained += not x and y
        lost += x and not y
        both += x and y
        neither += not x and not y
    return dict(episodes=len(left), gained=gained, lost=lost, both=both,
                neither=neither, success_delta=(gained - lost) / len(left))


def summarize(manifest):
    """Descriptive research comparison; never a release or promotion decision."""
    verify_inputs(manifest)
    grouped = defaultdict(list)
    by_seed = defaultdict(dict)
    hashes = {}
    initializations = {}
    for job in manifest["jobs"]:
        if job["status"] != "complete":
            raise ValueError("all jobs must complete before comparing arms")
        rd = Path(job["run_dir"])
        hashes[job["name"]] = verify_checkpoint(rd)
        metadata = read_json(rd / "best.metadata.json")
        if "parent_sha256" in manifest:
            ancestry = metadata["ancestry"]
            if ancestry["upstream"]["sha256"] != manifest["parent_sha256"]:
                raise ValueError("checkpoint ancestry differs from the suite parent")
            init = ancestry["initialization"]
            if init["critic"] != "fresh_seeded" or init["critic_seed"] != job["seed"]:
                raise ValueError("checkpoint did not use the matched fresh critic")
            # A best checkpoint can legitimately remain the pre-warmup actor.
            # Audit work performed by the run from its events, without adding
            # later training history to that earlier checkpoint's ancestry.
            events = [json.loads(line) for line in (rd / "events.jsonl").read_text().splitlines() if line.strip()]
            warm = next((event for event in events if event.get("event") == "value_warm_start"), None)
            if warm is None:
                raise ValueError("missing frozen-actor critic warmup evidence")
            identity = (init["critic_sha256"], warm["initial_observation_sha256"],
                        warm["final_observation_sha256"])
            if job["seed"] in initializations and initializations[job["seed"]] != identity:
                raise ValueError("same-seed arms have different initial critics or warmup craft")
            initializations[job["seed"]] = identity
            objective = metadata["contract"].get("objective", {})
            _, mode, penalty = next(arm for arm in ARMS if arm[0] == job["arm"])
            if (objective.get("deadline_mode", "truncate") != mode
                    or objective.get("timeout_penalty", 0) != penalty):
                raise ValueError("checkpoint objective differs from its experiment arm")
        summary = read_json(rd / "summary.json")
        for case in ("nominal", "four_stage_fresh_crafts"):
            report = read_json(rd / f"test_{case}.json")
            validated_rows(report)
            grouped[(job["arm"], case)].append(report["success_rate"])
            by_seed[(job["seed"], case)][job["arm"]] = report
        if summary.get("live_ksp_validated"):
            raise ValueError("simulator suite must not claim KSP validation")
    arms = {}
    for (arm, case), rates in grouped.items():
        arms.setdefault(arm, {})[case] = dict(
            training_seeds=len(rates), mean_success_rate=statistics.mean(rates),
            seed_success_rates=rates,
            seed_stdev=statistics.stdev(rates) if len(rates) > 1 else None,
        )
    pairs = []
    for (seed, case), reports in sorted(by_seed.items()):
        if set(reports) != {arm[0] for arm in ARMS}:
            raise ValueError("missing arm in paired seed comparison")
        for arm in ("deadline", "deadline_failure"):
            pairs.append(dict(seed=seed, case=case, baseline="truncate", arm=arm,
                              **paired_outcomes(reports["truncate"], reports[arm])))
    return dict(scope="research comparison bank; not release qualification",
                checkpoint_selection="separate fixed validation bank",
                live_ksp_validated=False, promoted=False, arms=arms,
                paired_comparisons=pairs, best_sha256=hashes)


def cuda_devices(indices):
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    available = {int(row.split(",")[0]): row.split(",")[1].strip()
                 for row in result.stdout.splitlines() if row.strip()}
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("CUDA device indices must be distinct")
    return [dict(index=index, uuid=available[index]) for index in indices]


def terminate_owned(running, *, grace_seconds=60):
    """Allow atomic checkpoints, then reap only children created by this runner."""
    for process, job, _output in running.values():
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            job["stop_requested"] = True
    deadline = time.monotonic() + grace_seconds
    for process, job, output in running.values():
        if process.poll() is None:
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                job["forced_stop"] = True
                process.wait(timeout=10)
        job.update(exit_code=process.returncode, finished_at=time.time(), status="interrupted")
        output.close()


def execute(manifest, *, poll_seconds=2, stop_grace_seconds=60):
    """Own only the Popen handles created here; never signal discovered PIDs."""
    root = Path(manifest["root"])
    status_path = root / "suite.json"
    running = {}
    stop_signal = None
    stop_deadline = None

    def request_stop(signum, _frame):
        nonlocal stop_signal
        stop_signal = signum

    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        manifest.update(status="running", controller_pid=os.getpid(), started_at=time.time())
        write_json(status_path, manifest)
        while True:
            for uuid, (process, job, output) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                output.close()
                del running[uuid]
                job.update(exit_code=code, finished_at=time.time())
                try:
                    if job.get("stop_requested"):
                        job["status"] = "interrupted"
                        continue
                    if code:
                        raise RuntimeError(f"trainer exit {code}")
                    summary = read_json(Path(job["run_dir"]) / "summary.json")
                    job["best_sha256"] = verify_checkpoint(job["run_dir"])
                    if summary.get("live_ksp_validated"):
                        raise ValueError("unexpected KSP validation claim")
                    job["status"] = "complete"
                except (OSError, ValueError, KeyError, RuntimeError) as error:
                    job.update(status="failed", error=str(error))
                    stop_signal = stop_signal or signal.SIGTERM
                write_json(status_path, manifest)
            if stop_signal:
                manifest["status"] = "stopping"
                if stop_deadline is None:
                    stop_deadline = time.monotonic() + stop_grace_seconds
                for process, job, _output in running.values():
                    if not job.get("stop_requested"):
                        process.send_signal(signal.SIGTERM)
                        job["stop_requested"] = True
                    elif time.monotonic() >= stop_deadline and not job.get("forced_stop"):
                        process.kill()
                        job["forced_stop"] = True
                if not running:
                    manifest["status"] = "failed" if any(j["status"] == "failed" for j in manifest["jobs"]) else "interrupted"
                    manifest["finished_at"] = time.time()
                    write_json(status_path, manifest)
                    return 1
            else:
                for device in manifest["devices"]:
                    if device["uuid"] in running:
                        continue
                    job = next((j for j in manifest["jobs"] if j["status"] == "pending"), None)
                    if job is None:
                        break
                    verify_inputs(manifest)
                    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=device["uuid"], OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
                    output = (root / f"{job['name']}.log").open("xb")
                    try:
                        process = subprocess.Popen(job["command"], cwd=manifest["source_root"], env=environment,
                                                   stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                    except BaseException:
                        output.close()
                        raise
                    job.update(status="running", pid=process.pid, device=device, started_at=time.time())
                    running[device["uuid"]] = (process, job, output)
                    write_json(status_path, manifest)
                if not running and all(j["status"] == "complete" for j in manifest["jobs"]):
                    write_json(root / "comparison.json", summarize(manifest))
                    manifest.update(status="complete", finished_at=time.time())
                    write_json(status_path, manifest)
                    return 0
            time.sleep(poll_seconds)
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}", finished_at=time.time())
        write_json(status_path, manifest)
        raise
    finally:
        # Unexpected controller failures must not leave an orphan queue running.
        terminate_owned(running, grace_seconds=stop_grace_seconds)
        write_json(status_path, manifest)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--actor-from", type=Path)
    parser.add_argument("--parent-sha256")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--seeds", type=int, nargs="+", default=[601, 602, 603])
    parser.add_argument("--validation-seed", type=int, default=1500505)
    parser.add_argument("--test-seed", type=int, default=2500505)
    parser.add_argument("--ppo-updates", type=int, default=100)
    parser.add_argument("--worlds", type=int, default=8192)
    parser.add_argument("--eval-worlds", type=int, default=256)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args(argv)
    root = args.run_root.resolve()
    if args.summarize:
        write_json(root / "comparison.json", summarize(read_json(root / "suite.json")))
        return 0
    if not args.actor_from or not args.parent_sha256:
        parser.error("--actor-from and --parent-sha256 are required")
    if min(args.ppo_updates, args.worlds, args.eval_worlds) < 1:
        parser.error("training counts must be positive")
    if sha256(args.actor_from) != args.parent_sha256:
        parser.error("parent checkpoint SHA256 differs from the declared source")
    training = dict(worlds=args.worlds, label_steps=800, value_updates=5000,
                    ppo_updates=args.ppo_updates, horizon=128, batch_size=16384,
                    eval_worlds=args.eval_worlds, validation_interval=20,
                    target_kl=.02, learning_rate=1e-5)
    jobs = make_jobs(root, args.actor_from, seeds=args.seeds,
                     validation_seed=args.validation_seed, test_seed=args.test_seed,
                     training=training)
    devices = cuda_devices(args.gpus) if not args.plan_only else [dict(index=i, uuid=None) for i in args.gpus]
    if root.exists() and any(root.iterdir()):
        parser.error("suite root must be empty; existing experiments are never overwritten")
    source_root = Path(__file__).resolve().parent.parent
    sources = {str(p.relative_to(source_root)): sha256(p) for p in sorted((source_root / "orbiter").glob("*.py"))}
    manifest = dict(schema="deadline_ablation_v1", root=str(root), source_root=str(source_root),
                    parent=str(args.actor_from.resolve()), parent_sha256=args.parent_sha256,
                    source_sha256=sources, status="planned", created_at=time.time(),
                    devices=devices, jobs=jobs, live_ksp_validated=False,
                    evaluation_role="research comparison; no weight promotion")
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "suite.json", manifest)
    if args.plan_only:
        print(json.dumps(dict(root=str(root), jobs=len(jobs), status="planned")))
        return 0
    return execute(manifest)


if __name__ == "__main__":
    raise SystemExit(main())
