"""Finite overnight supervision, artifact integrity, and deadline ownership."""

import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from orbiter import overnight_suite as suite


TRAINER = '''
import argparse, hashlib, json, os, signal, time
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--run-dir')
p.add_argument('--device')
p.add_argument('--duration', type=float, default=.06)
p.add_argument('--exit-code', type=int, default=0)
p.add_argument('--stubborn', action='store_true')
p.add_argument('--parent-sha')
a = p.parse_args()
root = Path(a.run_dir)
if a.stubborn:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
(root / 'started.json').write_text(json.dumps(dict(time=time.time(), uuid=os.environ['CUDA_VISIBLE_DEVICES'])))
time.sleep(a.duration)
if a.exit_code:
    raise SystemExit(a.exit_code)
for label in ('best', 'latest'):
    path = root / (label + '.pt')
    path.write_bytes(label.encode())
    (root / (label + '.metadata.json')).write_text(json.dumps(dict(
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        ancestry=dict(upstream=dict(sha256=a.parent_sha)))))
(root / 'summary.json').write_text(json.dumps(dict(
    nominal=dict(episodes=2, successes=1, success_rate=.5),
    four_stage_fresh_crafts=dict(episodes=2, successes=0, success_rate=0),
    selected_ancestry={}, live_ksp_validated=False)))
'''


def plan_at(tmp_path, jobs=3, *, devices=2, grace=.1, limit=20):
    source = tmp_path / "source"
    package = source / "orbiter"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "train_generalized.py").write_text(TRAINER)
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"frozen parent")
    root = tmp_path / "runs"
    entries = []
    for i in range(jobs):
        name = f"job{i}"
        command = [sys.executable, "-u", "-m", "orbiter.train_generalized",
                   "--run-dir", str(root / name), "--device", "cuda:0",
                   "--parent-sha", suite.sha256(parent)]
        entries.append(dict(name=name, method="test", command=command,
                            run_dir=str(root / name), max_seconds=5,
                            expected_parent_sha256=suite.sha256(parent)))
    plan = dict(schema=suite.SCHEMA, source_root=str(source),
                source_sha256=suite.source_hashes(source),
                inputs=[dict(path=str(parent), sha256=suite.sha256(parent))], jobs=entries)
    path = tmp_path / "plan.json"
    suite.write_json(path, plan)
    manifest = suite.prepare(path, root, stop_at=time.time() + limit,
                             devices=[dict(index=i, uuid=f"GPU-physical-{i}") for i in range(devices)],
                             stop_grace_seconds=grace)
    return manifest, plan, path


def rewrite_plan(manifest, plan, path):
    # Rebuild a fresh validated manifest for intentionally varied test plans.
    (Path(manifest["root"]) / "suite.json").unlink()
    suite.write_json(path, plan)
    return suite.prepare(path, manifest["root"], stop_at=manifest["stop_at"],
                         devices=manifest["devices"], stop_grace_seconds=manifest["stop_grace_seconds"])


def test_utc_deadline_requires_explicit_utc():
    assert suite.utc_timestamp("2026-10-19T11:00:00Z") == 1792407600
    assert suite.utc_timestamp("2026-10-19T11:00:00+00:00") == 1792407600
    for value in ("2026-10-19T11:00:00", "2026-10-19T07:00:00-04:00", "tomorrow"):
        with pytest.raises(ValueError, match="UTC"):
            suite.utc_timestamp(value)


def test_real_children_overlap_once_per_physical_device_and_hash_outputs(tmp_path):
    manifest, _, _ = plan_at(tmp_path, jobs=4)
    assert suite.execute(manifest, poll_seconds=.005) == 0
    assert manifest["status"] == "complete"
    jobs = manifest["jobs"]
    assert max(jobs[0]["started_at"], jobs[1]["started_at"]) < min(jobs[0]["finished_at"], jobs[1]["finished_at"])
    for uuid in {job["device"]["uuid"] for job in jobs}:
        ordered = [job for job in jobs if job["device"]["uuid"] == uuid]
        assert all(a["finished_at"] <= b["started_at"] for a, b in zip(ordered, ordered[1:]))
    for job in jobs:
        data = suite.read_json(Path(job["run_dir"]) / "started.json")
        assert data["uuid"] == job["device"]["uuid"]
        assert job["environment"]["CUDA_VISIBLE_DEVICES"] == data["uuid"]
        assert job["artifact_sha256"]["best.pt"] == suite.sha256(Path(job["run_dir"]) / "best.pt")


@pytest.mark.parametrize("change", ["parent", "source", "extra_source", "plan"])
def test_every_frozen_input_and_plan_is_rechecked(tmp_path, change):
    manifest, _, path = plan_at(tmp_path)
    if change == "parent":
        Path(manifest["inputs"][0]["path"]).write_bytes(b"changed")
    elif change == "source":
        (Path(manifest["source_root"]) / "orbiter/train_generalized.py").write_text("changed")
    elif change == "extra_source":
        (Path(manifest["source_root"]) / "orbiter/new.py").write_text("changed")
    else:
        path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="changed"):
        suite.verify_inputs(manifest)


@pytest.mark.parametrize("kind", ["empty", "shell", "module", "output", "relative_python", "duplicate", "path_escape", "input", "device", "count", "abbreviation", "executable", "empty_checkpoints"])
def test_bad_plans_cannot_create_outputs(tmp_path, kind):
    manifest, plan, path = plan_at(tmp_path, jobs=2)
    root = Path(manifest["root"])
    (root / "suite.json").unlink()
    root.rmdir()
    job = plan["jobs"][0]
    if kind == "empty":
        job["command"] = []
    elif kind == "shell":
        job["command"] = [sys.executable, "-c", "print('unapproved')"]
    elif kind == "module":
        job["command"][3] = "http.server"
    elif kind == "output":
        job["command"][5] = str(tmp_path / "escape")
    elif kind == "relative_python":
        job["command"][0] = "python"
    elif kind == "duplicate":
        plan["jobs"][1]["name"] = job["name"]
    elif kind == "path_escape":
        job["completion_files"] = ["../elsewhere.json"]
    elif kind == "input":
        job["command"] += ["--actor-from", str(tmp_path / "undeclared.pt")]
    elif kind == "device":
        job["command"][7] = "cuda:1"
    elif kind == "count":
        job["max_seconds"] = float("inf")
    elif kind == "abbreviation":
        job["command"] += ["--run-d", str(tmp_path / "escape")]
    elif kind == "executable":
        job["command"][0] = "/bin/sh"
    elif kind == "empty_checkpoints":
        job["checkpoints"] = []
    if kind == "count":
        path.write_text(json.dumps(plan))
    else:
        suite.write_json(path, plan)
    with pytest.raises((ValueError, KeyError)):
        suite.prepare(path, root, stop_at=manifest["stop_at"], devices=manifest["devices"])
    assert not root.exists()


def test_plan_only_does_not_consume_the_reviewed_output_root(tmp_path):
    manifest, _, path = plan_at(tmp_path)
    root = Path(manifest["root"])
    (root / "suite.json").unlink()
    root.rmdir()
    reviewed = suite.prepare(path, root, stop_at=manifest["stop_at"],
                              devices=manifest["devices"], persist=False)
    assert reviewed["status"] == "planned" and not root.exists()
    created = suite.prepare(path, root, stop_at=manifest["stop_at"], devices=manifest["devices"])
    assert created["plan_sha256"] == reviewed["plan_sha256"]


def test_existing_experiment_is_never_overwritten(tmp_path):
    manifest, _, path = plan_at(tmp_path)
    before = (Path(manifest["root"]) / "suite.json").read_bytes()
    with pytest.raises(ValueError, match="never overwritten"):
        suite.prepare(path, manifest["root"], stop_at=manifest["stop_at"], devices=manifest["devices"])
    assert (Path(manifest["root"]) / "suite.json").read_bytes() == before


def test_bounded_failures_continue_independent_jobs_then_stop(tmp_path):
    manifest, plan, path = plan_at(tmp_path, jobs=5, devices=1)
    for job in plan["jobs"][:3]:
        job["command"] += ["--exit-code", "9"]
    manifest = rewrite_plan(manifest, plan, path)
    assert suite.execute(manifest, poll_seconds=.005) == 1
    assert manifest["status"] == "failed"
    assert [job["status"] for job in manifest["jobs"]] == ["failed"] * 3 + ["skipped"] * 2
    assert all(job["exit_code"] == 9 for job in manifest["jobs"][:3])


def test_independent_success_resets_failure_streak(tmp_path):
    manifest, plan, path = plan_at(tmp_path, jobs=4, devices=1)
    plan["jobs"][0]["command"] += ["--exit-code", "7"]
    plan["jobs"][2]["command"] += ["--exit-code", "8"]
    manifest = rewrite_plan(manifest, plan, path)
    manifest["maximum_consecutive_failures"] = 2
    assert suite.execute(manifest, poll_seconds=.005) == 1
    assert manifest["status"] == "complete_with_failures"
    assert [job["status"] for job in manifest["jobs"]] == ["failed", "complete", "failed", "complete"]


def test_fake_clock_past_deadline_launches_nothing(tmp_path):
    manifest, _, _ = plan_at(tmp_path)
    assert suite.execute(manifest, clock=lambda: 0, wall_clock=lambda: manifest["stop_at"] + 1) == 2
    assert manifest["status"] == "deadline_reached"
    assert all(job["status"] == "skipped" for job in manifest["jobs"])
    assert not list(Path(manifest["root"]).glob("*.log"))


def test_real_deadline_kills_and_reaps_only_owned_stubborn_child(tmp_path, monkeypatch):
    manifest, plan, path = plan_at(tmp_path, jobs=2, devices=1, grace=.15, limit=.65)
    plan["jobs"][0]["command"] += ["--stubborn", "--duration", "30"]
    manifest = rewrite_plan(manifest, plan, path)
    owned, original = [], subprocess.Popen

    def record(*args, **kwargs):
        process = original(*args, **kwargs)
        owned.append(process)
        return process

    monkeypatch.setattr(suite.subprocess, "Popen", record)
    started = time.monotonic()
    assert suite.execute(manifest, poll_seconds=.005) == 2
    assert time.monotonic() - started < 1.2
    assert len(owned) == 1 and owned[0].poll() is not None
    assert manifest["jobs"][0]["forced_stop"]
    assert manifest["jobs"][1]["status"] == "skipped"
    assert (Path(manifest["jobs"][0]["run_dir"]) / "started.json").exists()


def test_job_timeout_reaps_then_runs_next_independent_job(tmp_path):
    manifest, plan, path = plan_at(tmp_path, jobs=2, devices=1, grace=.05)
    plan["jobs"][0]["command"] += ["--stubborn", "--duration", "30"]
    plan["jobs"][0]["max_seconds"] = .1
    manifest = rewrite_plan(manifest, plan, path)
    assert suite.execute(manifest, poll_seconds=.005) == 1
    assert manifest["status"] == "complete_with_failures"
    assert [job["status"] for job in manifest["jobs"]] == ["timed_out", "complete"]
    assert manifest["jobs"][0]["forced_stop"]


def test_graceful_deadline_records_partial_checkpoint_hashes(tmp_path):
    manifest, plan, path = plan_at(tmp_path, jobs=1, devices=1, grace=.2, limit=.7)
    code = Path(plan["source_root"]) / "orbiter/train_generalized.py"
    code.write_text(TRAINER.replace("time.sleep(a.duration)", '''
def stopped(_signum, _frame):
    for label in ('best', 'latest'):
        checkpoint = root / (label + '.pt')
        checkpoint.write_bytes(b'partial checkpoint')
        (root / (label + '.metadata.json')).write_text(json.dumps(dict(
            sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            ancestry=dict(upstream=dict(sha256=a.parent_sha)))))
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stopped)
time.sleep(a.duration)
'''))
    plan["source_sha256"] = suite.source_hashes(plan["source_root"])
    plan["jobs"][0]["command"] += ["--duration", "30"]
    manifest = rewrite_plan(manifest, plan, path)
    assert suite.execute(manifest, poll_seconds=.005) == 2
    job = manifest["jobs"][0]
    assert job["status"] == "interrupted" and not job.get("forced_stop")
    assert set(job["artifact_sha256"]) == {"best.pt", "best.metadata.json", "latest.pt", "latest.metadata.json"}
    assert not (Path(job["run_dir"]) / "summary.json").exists()


def test_midqueue_source_drift_stops_before_next_launch(tmp_path, monkeypatch):
    manifest, _, _ = plan_at(tmp_path, jobs=2, devices=1)
    original = suite.audit_outputs

    def audit(job):
        result = original(job)
        (Path(manifest["source_root"]) / "orbiter/train_generalized.py").write_text("drift")
        return result

    monkeypatch.setattr(suite, "audit_outputs", audit)
    with pytest.raises(ValueError, match="source changed"):
        suite.execute(manifest, poll_seconds=.005)
    assert manifest["status"] == "failed"
    assert manifest["jobs"][0]["status"] == "complete"
    assert manifest["jobs"][1]["pid"] is None


def test_checkpoint_hash_and_parent_must_match(tmp_path):
    manifest, _, _ = plan_at(tmp_path, jobs=1, devices=1)
    assert suite.execute(manifest, poll_seconds=.005) == 0
    job = manifest["jobs"][0]
    metadata_path = Path(job["run_dir"]) / "best.metadata.json"
    metadata = suite.read_json(metadata_path)
    metadata["ancestry"]["upstream"]["sha256"] = "0" * 64
    suite.write_json(metadata_path, metadata)
    with pytest.raises(ValueError, match="ancestry"):
        suite.audit_outputs(job)
    metadata["ancestry"]["upstream"]["sha256"] = job["expected_parent_sha256"]
    suite.write_json(metadata_path, metadata)
    (Path(job["run_dir"]) / "best.pt").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash mismatch"):
        suite.audit_outputs(job)


def test_partial_artifact_or_false_ksp_claim_fails_completion(tmp_path):
    manifest, _, _ = plan_at(tmp_path, jobs=1, devices=1)
    assert suite.execute(manifest, poll_seconds=.005) == 0
    job = manifest["jobs"][0]
    report = Path(job["run_dir"]) / "summary.json"
    suite.write_json(report, dict(live_ksp_validated=True))
    with pytest.raises(ValueError, match="KSP claim"):
        suite.audit_outputs(job)
    report.unlink()
    with pytest.raises(OSError):
        suite.audit_outputs(job)


def test_empty_report_checkpoint_and_forged_evaluation_counts_rejected(tmp_path):
    manifest, _, _ = plan_at(tmp_path, jobs=1, devices=1)
    assert suite.execute(manifest, poll_seconds=.005) == 0
    job = manifest["jobs"][0]
    directory = Path(job["run_dir"])
    report = directory / "summary.json"
    original = suite.read_json(report)
    suite.write_json(report, {})
    with pytest.raises(ValueError, match="nonempty"):
        suite.audit_outputs(job)
    altered = dict(original, nominal=dict(episodes=2, successes=2, success_rate=.5))
    suite.write_json(report, altered)
    with pytest.raises(ValueError, match="counts disagree"):
        suite.audit_outputs(job)
    suite.write_json(report, original)
    (directory / "best.pt").write_bytes(b"")
    metadata = suite.read_json(directory / "best.metadata.json")
    metadata["sha256"] = suite.sha256(directory / "best.pt")
    suite.write_json(directory / "best.metadata.json", metadata)
    with pytest.raises(ValueError, match="empty"):
        suite.audit_outputs(job)
