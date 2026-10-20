"""Execute a frozen, finite simulator research plan until a UTC deadline.

Only explicitly listed Python module commands run, with one owned child per
physical CUDA UUID. No game actions, checkpoint installation, or retries occur.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from .ablation_suite import cuda_devices, read_json, sha256, write_json


SCHEMA = "overnight_plan_v1"
MODULES = {
    "orbiter.train_generalized", "orbiter.distill_ascent",
    "orbiter.ascent_planner", "orbiter.train_dagger_student",
    "orbiter.train_architectures",
}
INPUT_FLAGS = {
    "--actor-from", "--warm-start", "--transfer-from", "--train-report",
    "--validation-report", "--test-report", "--bank", "--evaluate",
}
DEFAULT_CHECKPOINTS = {
    "orbiter.train_generalized": ["best.pt", "latest.pt"],
    "orbiter.distill_ascent": ["feedforward.pt", "gru.pt"],
    "orbiter.ascent_planner": [],
    "orbiter.train_dagger_student": ["best.pt", "latest.pt"],
    "orbiter.train_architectures": ["best.pt", "latest.pt"],
}


def utc_timestamp(value):
    """Require a timezone rather than guessing the machine's local timezone."""
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise ValueError("stop-at must be an ISO UTC timestamp") from error
    if result.utcoffset() != timezone.utc.utcoffset(result):
        raise ValueError("stop-at must explicitly use UTC")
    return result.timestamp()


def _digest(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return value


def _absolute(value, label):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    return Path(value).resolve()


def _inside(path, directory):
    resolved = Path(path).resolve()
    if resolved == directory or not resolved.is_relative_to(directory):
        raise ValueError(f"output path escapes its run directory: {path}")
    return resolved


def _option(command, flag, required=False):
    # Reject the equals form and duplicates, so validation sees argparse's value.
    matches = [i for i, item in enumerate(command) if item == flag or item.startswith(flag + "=")]
    if len(matches) > 1 or any(command[i] != flag for i in matches):
        raise ValueError(f"duplicate or ambiguous command option {flag}")
    if not matches:
        if required:
            raise ValueError(f"command requires {flag}")
        return None
    index = matches[0]
    if index + 1 >= len(command) or command[index + 1].startswith("--"):
        raise ValueError(f"command option {flag} requires a value")
    return command[index + 1]


def source_hashes(root):
    root = Path(root)
    return {str(path.relative_to(root)): sha256(path)
            for path in sorted((root / "orbiter").rglob("*"))
            if path.is_file() and path.suffix in {".py", ".json"}}


def verify_inputs(manifest):
    """Revalidate immutable plan, every input, and the complete package at launch."""
    if sha256(manifest["plan_path"]) != manifest["plan_sha256"]:
        raise ValueError("experiment plan changed")
    if source_hashes(manifest["source_root"]) != manifest["source_sha256"]:
        raise ValueError("frozen training source changed")
    for item in manifest["inputs"]:
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"frozen input changed: {item['path']}")


def prepare(plan_path, run_root, *, stop_at, devices, stop_grace_seconds=60,
            maximum_consecutive_failures=3, persist=True):
    """Validate the entire plan before creating any experiment outputs."""
    plan_path, root = Path(plan_path).resolve(), Path(run_root).resolve()
    raw = plan_path.read_bytes()
    plan = json.loads(raw)
    if plan.get("schema") != SCHEMA:
        raise ValueError(f"plan schema must be {SCHEMA}")
    source = _absolute(plan.get("source_root"), "source_root")
    declared = plan.get("source_sha256")
    if not isinstance(declared, dict) or not declared or any(not _digest(v) for v in declared.values()):
        raise ValueError("source_sha256 must contain the frozen package hashes")
    if source_hashes(source) != declared:
        raise ValueError("frozen training source changed")
    _number(stop_grace_seconds, "stop grace")
    if not isinstance(maximum_consecutive_failures, int) or isinstance(maximum_consecutive_failures, bool) or maximum_consecutive_failures < 1:
        raise ValueError("maximum consecutive failures must be a positive integer")
    if not math.isfinite(stop_at):
        raise ValueError("deadline must be finite")
    if not devices or len({item["uuid"] for item in devices}) != len(devices):
        raise ValueError("physical CUDA devices must be nonempty and distinct")
    if root.exists() and any(root.iterdir()):
        raise ValueError("suite root must be empty; experiments are never overwritten")
    inputs, input_paths = [], set()
    if not isinstance(plan.get("inputs", []), list):
        raise ValueError("inputs must be an array")
    for item in plan.get("inputs", []):
        if not isinstance(item, dict):
            raise ValueError("each input must declare a path and SHA256")
        path = str(_absolute(item.get("path"), "input"))
        if path in input_paths or not _digest(item.get("sha256")):
            raise ValueError("input paths must be unique and SHA256-bound")
        inputs.append(dict(path=path, sha256=item["sha256"]))
        input_paths.add(path)
    if not isinstance(plan.get("jobs"), list) or not plan["jobs"]:
        raise ValueError("plan needs a finite nonempty jobs list")
    jobs, names = [], set()
    for item in plan["jobs"]:
        if not isinstance(item, dict):
            raise ValueError("each job must be an explicit object")
        name = item.get("name")
        if not isinstance(name, str) or not re.fullmatch("[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name) or name in names:
            raise ValueError("job names must be unique simple path components")
        names.add(name)
        directory = _absolute(item.get("run_dir"), "job run_dir")
        if directory != root / name:
            raise ValueError("job run_dir must equal suite root / job name")
        command = item.get("command")
        if not isinstance(command, list) or any(not isinstance(arg, str) or not arg or "\0" in arg for arg in command):
            raise ValueError("command must be a nonempty argv array")
        python = _absolute(command[0] if command else None, "Python executable")
        if (not python.is_file() or not os.access(python, os.X_OK)
                or not re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", python.name)):
            raise ValueError("Python executable does not exist or is not executable")
        offset = 2 if len(command) > 1 and command[1] == "-u" else 1
        if len(command) <= offset + 1 or command[offset] != "-m" or command[offset + 1] not in MODULES:
            raise ValueError("only approved simulator Python modules may run")
        module = command[offset + 1]
        options = command[offset + 2:]
        protected = INPUT_FLAGS | {"--device", "--run-dir", "--output"}
        for token in options:
            option = token.split("=", 1)[0]
            if option.startswith("--") and any(flag.startswith(option) and flag != option for flag in protected):
                raise ValueError(f"abbreviated protected command option: {option}")
        if _option(options, "--device", required=True) != "cuda:0":
            raise ValueError("child device must be cuda:0 within its physical UUID")
        output_flag = "--output" if module == "orbiter.ascent_planner" else "--run-dir"
        output_path = _absolute(_option(options, output_flag, required=True), output_flag)
        expected = directory / "report.json" if module == "orbiter.ascent_planner" else directory
        if output_path != expected:
            raise ValueError("command output does not match the declared run directory")
        forbidden_output = "--run-dir" if output_flag == "--output" else "--output"
        if _option(options, forbidden_output) is not None:
            raise ValueError("unexpected extra output option")
        for flag in INPUT_FLAGS:
            value = _option(options, flag)
            if value is not None and str(_absolute(value, flag)) not in input_paths:
                raise ValueError(f"command input {flag} is absent from frozen inputs")
        checkpoints = item.get("checkpoints", DEFAULT_CHECKPOINTS.get(module))
        if not isinstance(checkpoints, list):
            raise ValueError("new training modules require an explicit checkpoints list")
        files = item.get("completion_files", ["report.json" if module == "orbiter.ascent_planner" else "summary.json"])
        for collection in (checkpoints, files):
            if not isinstance(collection, list) or len(collection) != len(set(collection)):
                raise ValueError("completion paths must be distinct arrays")
            for path in collection:
                if not isinstance(path, str) or Path(path).is_absolute():
                    raise ValueError("completion files must be relative paths")
                _inside(directory / path, directory)
        if not files or any(not path.endswith(".pt") for path in checkpoints):
            raise ValueError("completion files are required and checkpoints must end in .pt")
        if not set(DEFAULT_CHECKPOINTS[module]).issubset(checkpoints):
            raise ValueError("training jobs require their standard checkpoints")
        required_report = "report.json" if module == "orbiter.ascent_planner" else "summary.json"
        if required_report not in files:
            raise ValueError("job requires its standard completion report")
        parent_hash = item.get("expected_parent_sha256")
        if parent_hash is not None and not _digest(parent_hash):
            raise ValueError("expected parent must be a SHA256")
        if parent_hash is not None and parent_hash not in {entry["sha256"] for entry in inputs}:
            raise ValueError("expected parent is absent from frozen inputs")
        jobs.append(dict(name=name, method=item.get("method", module), module=module,
                         command=list(command), run_dir=str(directory), status="pending", pid=None,
                         max_seconds=_number(item.get("max_seconds"), "job max_seconds"),
                         checkpoints=checkpoints, completion_files=files,
                         expected_parent_sha256=parent_hash))
    manifest = dict(schema=SCHEMA, root=str(root), source_root=str(source),
                    source_sha256=declared, source_commit=plan.get("source_commit"),
                    plan_path=str(plan_path), plan_sha256=hashlib.sha256(raw).hexdigest(),
                    inputs=inputs, jobs=jobs, devices=devices, stop_at=stop_at,
                    stop_at_utc=datetime.fromtimestamp(stop_at, timezone.utc).isoformat(),
                    stop_grace_seconds=stop_grace_seconds,
                    maximum_consecutive_failures=maximum_consecutive_failures,
                    status="planned", created_at=time.time(), live_ksp_validated=False,
                    promoted=False, scope="simulator research; no release qualification")
    verify_inputs(manifest)
    if persist:
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "suite.json", manifest)
    return manifest


def audit_checkpoints(job, *, available_only=False):
    """Verify complete checkpoint/metadata pairs, including graceful-stop saves."""
    directory = Path(job["run_dir"])
    digests = {}
    for name in job["checkpoints"]:
        path = _inside(directory / name, directory)
        metadata_path = _inside(path.with_suffix(".metadata.json"), directory)
        if available_only and not path.exists() and not metadata_path.exists():
            continue
        metadata = read_json(metadata_path)
        if not path.stat().st_size:
            raise ValueError(f"checkpoint is empty: {name}")
        digest = sha256(path)
        if digest != metadata.get("sha256"):
            raise ValueError(f"checkpoint hash mismatch: {name}")
        parent = job.get("expected_parent_sha256")
        if parent and metadata.get("ancestry", {}).get("upstream", {}).get("sha256") != parent:
            raise ValueError("checkpoint ancestry differs from declared parent")
        if metadata.get("live_ksp_validated"):
            raise ValueError("simulator checkpoint claims live KSP validation")
        digests[name] = digest
        digests[str(metadata_path.relative_to(directory))] = sha256(metadata_path)
    return digests


def audit_outputs(job):
    """Hash completed artifacts without loading or promoting model weights."""
    directory = Path(job["run_dir"])
    digests = audit_checkpoints(job)
    for name in job["completion_files"]:
        path = _inside(directory / name, directory)
        report = read_json(path)
        if not isinstance(report, dict) or not report:
            raise ValueError("completion report must be a nonempty object")
        if report.get("live_ksp_validated") or report.get("promoted"):
            raise ValueError("simulator job made an unexpected deployment or KSP claim")
        digests[name] = sha256(path)
        required = "report.json" if job["module"] == "orbiter.ascent_planner" else "summary.json"
        if name == required:
            validate_report(job["module"], report)
    return digests


def validate_report(module, report):
    """Reject empty/partial successes without imposing a research success gate."""
    def episodes(row, successes="successes"):
        if not isinstance(row, dict) or type(row.get("episodes")) is not int or row["episodes"] < 1:
            raise ValueError("completed evaluation needs positive episode count")
        count, success = row["episodes"], row.get(successes)
        rate = row.get("success_rate")
        if (type(success) is not int or not 0 <= success <= count
                or not isinstance(rate, (int, float)) or not math.isfinite(rate)
                or abs(rate - success / count) > 1e-10):
            raise ValueError("completed evaluation counts disagree")

    if module == "orbiter.train_architectures":
        episodes(report.get("comparison"))
        if (report.get("task") != "matched_history_architecture_research_v1"
                or report.get("interrupted") is not False
                or report.get("live_ksp_validated") is not False
                or report.get("final_tests_select_checkpoint") is not False):
            raise ValueError("architecture completion lacks research scope or final comparison")
        if report.get("benchmark") == "orbital":
            from .system import SystemCatalog
            if set(report["comparison"].get("per_body", {})) != set(SystemCatalog.load().bodies):
                raise ValueError("orbital comparison must cover every stock body")
            for row in report["comparison"]["per_body"].values():
                episodes(row)
        elif report.get("benchmark") == "ascent":
            episodes(report.get("four_stage"))
        else:
            raise ValueError("unknown architecture benchmark")
    elif module == "orbiter.train_generalized":
        episodes(report.get("nominal"))
        episodes(report.get("four_stage_fresh_crafts", report.get("heldout_four_stage")))
        if not isinstance(report.get("selected_ancestry"), dict) or report.get("live_ksp_validated") is not False:
            raise ValueError("PPO completion lacks selected ancestry or simulator scope")
    elif module == "orbiter.distill_ascent":
        episodes(report.get("default_guidance"), "qualified_successes")
        for architecture in ("feedforward", "gru"):
            episodes(report.get("students", {}).get(architecture, {}).get("closed_loop_validation"), "qualified_successes")
        if not report.get("task") or report.get("live_ksp_validated") is not False:
            raise ValueError("student completion lacks task or simulator scope")
    elif module == "orbiter.train_dagger_student":
        episodes(report.get("closed_loop_validation"), "qualified_successes")
        if (not report.get("task") or not isinstance(report.get("history"), list)
                or not report["history"] or report.get("interrupted") is not False
                or report.get("live_ksp_validated") is not False):
            raise ValueError("DAgger completion lacks completed training history")
    elif module == "orbiter.ascent_planner":
        count, rows, successes = report.get("craft_count"), report.get("results"), report.get("qualified_successes")
        if (type(count) is not int or count < 1 or not isinstance(rows, list) or len(rows) != count
                or type(successes) is not int or not 0 <= successes <= count or not report.get("schema")):
            raise ValueError("planner completion lacks full craft results")


def retain_interrupted_artifacts(job):
    try:
        job["artifact_sha256"] = audit_checkpoints(job, available_only=True)
    except (OSError, ValueError, KeyError) as error:
        job["artifact_integrity_error"] = f"{type(error).__name__}: {error}"


def execute(manifest, *, poll_seconds=1, clock=time.monotonic, wall_clock=time.time,
            sleep=time.sleep):
    """Own each Popen handle until reaped, including deadline and exception paths."""
    root = Path(manifest["root"])
    state = root / "suite.json"
    running = {}
    stop_reason = None
    consecutive_failures = 0
    hard_deadline = clock() + max(0, manifest["stop_at"] - wall_clock())
    grace = manifest["stop_grace_seconds"]
    # Reserve part of the graceful interval for killing and reaping stubborn
    # children. Both wall and monotonic clocks prevent clock adjustments from
    # extending the authorized run window.
    reap_margin = min(5, grace / 4)

    def save():
        manifest["updated_at"] = wall_clock()
        write_json(state, manifest)

    def request_stop(signum, _frame):
        nonlocal stop_reason
        stop_reason = stop_reason or f"signal_{signum}"

    def stop_child(entry, reason):
        process, job, _output = entry[:3]
        if not job.get("stop_requested"):
            if process.poll() is None:
                try:
                    process.send_signal(signal.SIGTERM)
                except ProcessLookupError:
                    pass  # It exited between poll() and the signal.
            job.update(stop_requested=True, stop_reason=reason,
                       stop_requested_at=wall_clock())
            entry[3] = min(clock() + max(0, grace - reap_margin), hard_deadline - reap_margin)

    handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        manifest.update(status="running", controller_pid=os.getpid(), started_at=wall_clock())
        save()
        while True:
            now = clock()
            hard_deadline = min(hard_deadline, now + max(0, manifest["stop_at"] - wall_clock()))
            if now >= hard_deadline - grace or wall_clock() >= manifest["stop_at"] - grace:
                stop_reason = stop_reason or "deadline"
            for uuid, entry in list(running.items()):
                process, job, output, kill_at = entry
                code = process.poll()
                if code is None:
                    if now >= job["deadline_monotonic"]:
                        stop_child(entry, "job_timeout")
                    if stop_reason:
                        stop_child(entry, stop_reason)
                    if job.get("stop_requested"):
                        entry[3] = min(entry[3], hard_deadline - reap_margin)
                    if job.get("stop_requested") and clock() >= entry[3] and not job.get("forced_stop"):
                        try:
                            process.kill()
                        except ProcessLookupError:
                            pass
                        job["forced_stop"] = True
                    continue
                output.close()
                del running[uuid]
                job.update(exit_code=code, finished_at=wall_clock())
                try:
                    if job.get("stop_requested"):
                        job["status"] = "timed_out" if job["stop_reason"] == "job_timeout" else "interrupted"
                        retain_interrupted_artifacts(job)
                        if job["status"] == "timed_out":
                            consecutive_failures += 1
                    elif code:
                        raise RuntimeError(f"trainer exit {code}")
                    else:
                        job["artifact_sha256"] = audit_outputs(job)
                        job["status"] = "complete"
                        consecutive_failures = 0
                except (OSError, ValueError, KeyError, RuntimeError) as error:
                    job.update(status="failed", error=f"{type(error).__name__}: {error}")
                    consecutive_failures += 1
                if consecutive_failures >= manifest["maximum_consecutive_failures"]:
                    stop_reason = stop_reason or "consecutive_failures"
                manifest["consecutive_failures"] = consecutive_failures
                save()
            if stop_reason:
                manifest.update(status="stopping", stop_reason=stop_reason)
                for entry in running.values():
                    stop_child(entry, stop_reason)
                if not running:
                    for job in manifest["jobs"]:
                        if job["status"] == "pending":
                            job.update(status="skipped", skip_reason=stop_reason)
                    status = "deadline_reached" if stop_reason == "deadline" else "interrupted"
                    if stop_reason == "consecutive_failures":
                        status = "failed"
                    manifest.update(status=status, finished_at=wall_clock())
                    save()
                    return 2 if stop_reason == "deadline" else 1
            else:
                for device in manifest["devices"]:
                    if device["uuid"] in running:
                        continue
                    job = next((item for item in manifest["jobs"] if item["status"] == "pending"), None)
                    if job is None:
                        break
                    if clock() >= hard_deadline - grace or wall_clock() >= manifest["stop_at"] - grace:
                        stop_reason = "deadline"
                        break
                    verify_inputs(manifest)
                    Path(job["run_dir"]).mkdir(exist_ok=False)
                    output = (root / f"{job['name']}.log").open("xb")
                    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=device["uuid"],
                                       OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                                       PYTHONPATH=manifest["source_root"], PYTHONNOUSERSITE="1")
                    try:
                        process = subprocess.Popen(job["command"], cwd=manifest["source_root"],
                                                   env=environment, stdout=output, stderr=subprocess.STDOUT,
                                                   start_new_session=True)
                    except BaseException:
                        output.close()
                        raise
                    job.update(status="running", pid=process.pid, device=device, started_at=wall_clock(),
                               deadline_monotonic=clock() + job["max_seconds"],
                               environment={key: environment[key] for key in
                                            ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONPATH", "PYTHONNOUSERSITE")})
                    running[device["uuid"]] = [process, job, output, None]
                    save()
                if not running and not any(job["status"] == "pending" for job in manifest["jobs"]):
                    verify_inputs(manifest)
                    failed = any(job["status"] != "complete" for job in manifest["jobs"])
                    manifest.update(status="complete_with_failures" if failed else "complete", finished_at=wall_clock())
                    save()
                    return 1 if failed else 0
            sleep(max(.001, min(poll_seconds, max(.001, hard_deadline - clock()))))
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}", finished_at=wall_clock())
        for job in manifest["jobs"]:
            if job["status"] == "pending":
                job.update(status="skipped", skip_reason="controller_error")
        raise
    finally:
        # Never discover or signal arbitrary PIDs. These handles are exclusively
        # children launched by this controller, and every one is reaped.
        for entry in running.values():
            stop_child(entry, "controller_exit")
        cleanup_until = min(clock() + max(0, grace - reap_margin), hard_deadline - reap_margin)
        for process, job, output, _kill_at in running.values():
            if process.poll() is None:
                try:
                    process.wait(timeout=max(0, cleanup_until - clock()))
                except subprocess.TimeoutExpired:
                    process.kill()
                    job["forced_stop"] = True
            process.wait()
            job.update(status="interrupted", exit_code=process.returncode, finished_at=wall_clock())
            retain_interrupted_artifacts(job)
            output.close()
        save()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stop-at", required=True, help="UTC ISO timestamp, e.g. 2026-10-19T11:00:00Z")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--stop-grace-seconds", type=float, default=60)
    parser.add_argument("--maximum-consecutive-failures", type=int, default=3)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        deadline = utc_timestamp(args.stop_at)
        if deadline <= time.time():
            raise ValueError("deadline is already past")
        if not args.gpus or len(set(args.gpus)) != len(args.gpus) or any(i < 0 for i in args.gpus):
            raise ValueError("GPU indices must be distinct and nonnegative")
        devices = ([dict(index=i, uuid=f"unresolved-index-{i}") for i in args.gpus]
                   if args.plan_only else cuda_devices(args.gpus))
        manifest = prepare(args.plan, args.run_root, stop_at=deadline, devices=devices,
                           stop_grace_seconds=args.stop_grace_seconds,
                           maximum_consecutive_failures=args.maximum_consecutive_failures,
                           persist=not args.plan_only)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    if args.plan_only:
        print(json.dumps(dict(status="planned", jobs=len(manifest["jobs"]), root=manifest["root"],
                              stop_at=manifest["stop_at_utc"], plan_sha256=manifest["plan_sha256"])))
        return 0
    return execute(manifest)


if __name__ == "__main__":
    raise SystemExit(main())
