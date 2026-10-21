"""Finite experiment scheduling and same-craft comparison safeguards."""

import json
import subprocess
import sys

import pytest

from orbiter import ablation_suite as suite


def report(successes, *, seed=901):
    return dict(episodes=len(successes), successes=sum(successes), success_rate=sum(successes) / len(successes),
                results=[dict(world=i, seed=seed, success=success,
                              craft=dict(payload_mass=100 + i))
                         for i, success in enumerate(successes)])


def test_comparison_requires_exact_craft_and_pairs_by_world():
    first = report([True, False, True, False])
    second = report([False, True, True, False])
    second["results"].reverse()
    assert suite.paired_outcomes(first, second) == dict(
        episodes=4, gained=1, lost=1, both=1, neither=1, success_delta=0,
    )
    second["results"][0]["craft"]["payload_mass"] += 1
    with pytest.raises(ValueError, match="same frozen craft"):
        suite.paired_outcomes(first, second)


def test_duplicate_worlds_are_rejected():
    first, second = report([True, False]), report([True, False])
    second["results"][1]["world"] = 0
    with pytest.raises(ValueError, match="different worlds"):
        suite.paired_outcomes(first, second)


def test_extra_duplicate_row_and_forged_rate_are_rejected():
    first, second = report([True, False]), report([True, False])
    second["results"].append(dict(second["results"][1], success=True))
    with pytest.raises(ValueError, match="duplicate"):
        suite.paired_outcomes(first, second)
    second = report([True, False])
    second["success_rate"] = 1
    with pytest.raises(ValueError, match="success rate"):
        suite.paired_outcomes(first, second)


def test_frozen_inputs_are_rechecked_after_parent_or_source_change(tmp_path):
    source = tmp_path / "orbiter"
    source.mkdir()
    code = source / "trainer.py"
    code.write_text("original")
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"parent")
    manifest = dict(source_root=str(tmp_path), parent=str(parent),
                    parent_sha256=suite.sha256(parent),
                    source_sha256={"orbiter/trainer.py": suite.sha256(code)})
    suite.verify_inputs(manifest)
    code.write_text("different")
    with pytest.raises(ValueError, match="source changed"):
        suite.verify_inputs(manifest)
    code.write_text("original")
    parent.write_bytes(b"different")
    with pytest.raises(ValueError, match="parent checkpoint changed"):
        suite.verify_inputs(manifest)


def test_three_arms_share_initialization_and_banks_but_not_training_seeds(tmp_path):
    jobs = suite.make_jobs(tmp_path, tmp_path / "parent.pt", seeds=[601, 602, 603],
                           validation_seed=1500505, test_seed=2500505,
                           training=dict(worlds=8192, ppo_updates=100))
    assert len(jobs) == 9
    assert [j["arm"] for j in jobs[:3]] == ["truncate", "deadline", "deadline_failure"]
    assert [j["arm"] for j in jobs[3:6]] == ["deadline", "deadline_failure", "truncate"]
    for job in jobs:
        cmd = job["command"]
        assert cmd[cmd.index("--actor-from") + 1] == str(tmp_path / "parent.pt")
        assert cmd[cmd.index("--validation-seed") + 1] == "1500505"
        assert cmd[cmd.index("--test-seed") + 1] == "2500505"
        assert cmd[cmd.index("--seed") + 1] == str(job["seed"])
        assert "orbiter.train_generalized" in cmd
        assert "--warm-start" not in cmd and "--transfer-from" not in cmd


@pytest.mark.parametrize("seeds,validation,test", [([1, 1], 2, 3), ([1], 2, 2), ([1], 1, 3), ([], 2, 3)])
def test_invalid_or_leaking_seed_banks_rejected(tmp_path, seeds, validation, test):
    with pytest.raises(ValueError):
        suite.make_jobs(tmp_path, tmp_path / "p.pt", seeds=seeds,
                        validation_seed=validation, test_seed=test, training={})


def test_parent_hash_failure_creates_no_output(tmp_path):
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"parent")
    root = tmp_path / "suite"
    with pytest.raises(SystemExit):
        suite.main(["--run-root", str(root), "--actor-from", str(parent),
                    "--parent-sha256", "0" * 64, "--plan-only"])
    assert not root.exists()


def test_plan_does_not_overwrite_existing_experiment(tmp_path):
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"parent")
    root = tmp_path / "suite"
    args = ["--run-root", str(root), "--actor-from", str(parent),
            "--parent-sha256", suite.sha256(parent), "--plan-only"]
    assert suite.main(args) == 0
    before = (root / "suite.json").read_bytes()
    with pytest.raises(SystemExit):
        suite.main(args)
    assert (root / "suite.json").read_bytes() == before


def write_completed_run(rd, successes):
    rd.mkdir()
    (rd / "best.pt").write_bytes(b"checkpoint")
    suite.write_json(rd / "best.metadata.json", dict(sha256=suite.sha256(rd / "best.pt")))
    suite.write_json(rd / "summary.json", dict(live_ksp_validated=False))
    for case in ("nominal", "four_stage_fresh_crafts"):
        suite.write_json(rd / f"test_{case}.json", report(successes))


def test_summary_checks_hashes_and_retains_training_seed_variability(tmp_path):
    jobs = []
    for seed in [11, 12]:
        for arm, _, _ in suite.ARMS:
            rd = tmp_path / f"{arm}{seed}"
            successes = [True, arm != "truncate" and seed == 11]
            write_completed_run(rd, successes)
            jobs.append(dict(name=rd.name, run_dir=str(rd), seed=seed, arm=arm, status="complete"))
    result = suite.summarize(dict(jobs=jobs))
    assert result["arms"]["deadline"]["nominal"]["seed_success_rates"] == [1, .5]
    assert result["arms"]["deadline"]["nominal"]["mean_success_rate"] == .75
    assert result["paired_comparisons"][0]["gained"] == 1
    assert not result["promoted"] and not result["live_ksp_validated"]
    (tmp_path / "truncate11" / "best.pt").write_bytes(b"replaced")
    with pytest.raises(ValueError, match="hash mismatch"):
        suite.summarize(dict(jobs=jobs))


def test_runtime_ancestry_audit_rejects_unmatched_initial_critic(tmp_path):
    package = tmp_path / "orbiter"
    package.mkdir()
    code = package / "trainer.py"
    code.write_text("frozen")
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"parent")
    manifest = dict(source_root=str(tmp_path), parent=str(parent),
                    parent_sha256=suite.sha256(parent),
                    source_sha256={"orbiter/trainer.py": suite.sha256(code)}, jobs=[])
    for arm, mode, penalty in suite.ARMS:
        rd = tmp_path / arm
        write_completed_run(rd, [True, False])
        metadata = suite.read_json(rd / "best.metadata.json")
        metadata.update(
            contract=dict(objective=dict(deadline_mode=mode, timeout_penalty=penalty)),
            ancestry=dict(upstream=dict(sha256=manifest["parent_sha256"]),
                          initialization=dict(critic="fresh_seeded", critic_seed=1, critic_sha256="same"),
                          value_warm_start=dict(initial_observation_sha256="initial", final_observation_sha256="final")),
        )
        # A selected initialization checkpoint predates critic warmup. Its
        # ancestry intentionally omits that later work; run events still prove
        # that all arms collected the same frozen-actor trajectories.
        warm = metadata["ancestry"]["value_warm_start"]
        (rd / "events.jsonl").write_text(json.dumps(dict(event="value_warm_start", **warm)) + "\n")
        if arm == "truncate":
            metadata["ancestry"].pop("value_warm_start")
        suite.write_json(rd / "best.metadata.json", metadata)
        manifest["jobs"].append(dict(name=arm, run_dir=str(rd), seed=1, arm=arm, status="complete"))
    assert len(suite.summarize(manifest)["best_sha256"]) == 3
    changed = tmp_path / "deadline" / "best.metadata.json"
    metadata = suite.read_json(changed)
    metadata["ancestry"]["initialization"]["critic_sha256"] = "different"
    suite.write_json(changed, metadata)
    with pytest.raises(ValueError, match="different initial critics"):
        suite.summarize(manifest)


def test_failed_child_stops_pending_queue_without_retry(tmp_path):
    jobs = [dict(name="failure", arm="truncate", seed=1, run_dir=str(tmp_path / "failure"),
                 command=[sys.executable, "-c", "raise SystemExit(7)"], status="pending"),
            dict(name="must_not_start", arm="deadline", seed=1, run_dir=str(tmp_path / "other"),
                 command=[sys.executable, "-c", "raise SystemExit(0)"], status="pending")]
    manifest = dict(root=str(tmp_path), source_root=str(tmp_path), jobs=jobs,
                    devices=[dict(index=0, uuid="GPU-test")])
    assert suite.execute(manifest, poll_seconds=.01) == 1
    assert manifest["status"] == "failed"
    assert jobs[0]["exit_code"] == 7
    assert jobs[1]["status"] == "pending"
    assert not (tmp_path / "must_not_start.log").exists()


def test_two_devices_overlap_but_each_has_only_one_owned_child(tmp_path):
    jobs = []
    for arm, _, _ in suite.ARMS:
        rd = tmp_path / arm
        payload = json.dumps(report([True, arm != "truncate"]))
        script = (
            "import pathlib,json,hashlib,time; "
            f"r=pathlib.Path({str(rd)!r}); r.mkdir(); time.sleep(.1); "
            "(r/'best.pt').write_bytes(b'model'); "
            "(r/'best.metadata.json').write_text(json.dumps({'sha256':hashlib.sha256(b'model').hexdigest()})); "
            "(r/'summary.json').write_text(json.dumps({'live_ksp_validated':False})); "
            f"(r/'test_nominal.json').write_text({payload!r}); "
            f"(r/'test_four_stage_fresh_crafts.json').write_text({payload!r})"
        )
        jobs.append(dict(name=arm, arm=arm, seed=1, run_dir=str(rd),
                         command=[sys.executable, "-c", script], status="pending"))
    manifest = dict(root=str(tmp_path), source_root=str(tmp_path), jobs=jobs,
                    devices=[dict(index=i, uuid=f"GPU-test{i}") for i in range(2)])
    assert suite.execute(manifest, poll_seconds=.01) == 0
    assert manifest["status"] == "complete"
    assert jobs[0]["device"] != jobs[1]["device"]
    assert jobs[1]["started_at"] < jobs[0]["finished_at"]
    for first in jobs:
        for second in jobs:
            if first is second or first["device"] != second["device"]:
                continue
            assert (first["finished_at"] <= second["started_at"]
                    or second["finished_at"] <= first["started_at"])
    assert (tmp_path / "comparison.json").exists()


def test_cleanup_escalates_and_reaps_only_owned_stubborn_child(tmp_path):
    ready = tmp_path / "ready"
    script = (
        "import signal,time,pathlib; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"pathlib.Path({str(ready)!r}).touch(); time.sleep(10)"
    )
    output = (tmp_path / "child.log").open("wb")
    child = subprocess.Popen([sys.executable, "-c", script], stdout=output)
    try:
        import time
        deadline = time.monotonic() + 3
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.exists()
        job = {}
        suite.terminate_owned({"owned": (child, job, output)}, grace_seconds=.02)
        assert child.poll() is not None
        assert job["forced_stop"] and job["status"] == "interrupted"
        assert output.closed
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=3)
        output.close()
