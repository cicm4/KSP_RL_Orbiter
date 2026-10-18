"""The capture learner cannot consume launch weights or train on test bodies."""

import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3 import PPO

from orbiter.capture_env import CaptureConfig, CaptureEnv
from orbiter.system import SystemCatalog
from orbiter import train_capture


@pytest.mark.parametrize(
    "result", [{"successes": 3}, SimpleNamespace(policy="trained-model")]
)
def test_console_entry_returns_zero_for_successful_library_results(monkeypatch, result):
    forwarded = []

    def main(argv):
        forwarded.append(argv)
        return result

    monkeypatch.setattr(train_capture, "main", main)
    assert train_capture.cli_main(["evaluate"]) == 0
    assert forwarded == [["evaluate"]]


def test_console_entry_does_not_swallow_training_errors(monkeypatch):
    def failed(argv):
        raise ValueError("Invalid capture contract")

    monkeypatch.setattr(train_capture, "main", failed)
    with pytest.raises(ValueError, match="Invalid capture contract"):
        train_capture.cli_main([])


@pytest.fixture
def catalog():
    bodies = SystemCatalog.load().bodies
    return {name: bodies[name] for name in ("Mun", "Duna")}


@pytest.fixture
def checkpoint(tmp_path, catalog):
    config = CaptureConfig.for_body(catalog["Mun"], randomization=0, max_time=10)
    env = CaptureEnv(config, body_catalog={"Mun": catalog["Mun"]})
    model = PPO(
        "MlpPolicy",
        env,
        gamma=config.gamma,
        n_steps=4,
        batch_size=4,
        policy_kwargs=dict(net_arch=dict(pi=[8], vf=[8])),
        device="cpu",
        seed=0,
    )
    model.orbiter_capture_contract = train_capture.make_contract(
        config, catalog, ["Mun"], ["Duna"]
    )
    path = tmp_path / "capture.zip"
    train_capture.save_checkpoint(model, path)
    env.close()
    return path, model, config, catalog


def test_body_split_rejects_overlap_and_unknown_targets(catalog):
    assert train_capture.body_sets(catalog, ["Mun"], ["Duna"]) == (("Mun",), ("Duna",))
    with pytest.raises(ValueError, match="overlap"):
        train_capture.body_sets(catalog, ["Mun"], ["Mun"])
    with pytest.raises(ValueError, match="missing"):
        train_capture.body_sets(catalog, ["Mun"], ["UnavailableMoon"])
    with pytest.raises(ValueError, match="distinct"):
        train_capture.body_sets(catalog, ["Mun", "Mun"], ["Duna"])


def test_wilson_intervals_remain_honest_for_small_per_body_samples():
    lower, upper = train_capture.wilson_interval(4, 4)
    assert 0.5 < lower < 0.6 and upper == pytest.approx(1)
    lower, upper = train_capture.wilson_interval(0, 4)
    assert lower == 0 and 0.4 < upper < 0.5
    assert train_capture.wilson_interval(0, 0) == [0, 1]


def test_long_gilly_episodes_cannot_dominate_multibody_imitation_batches():
    labels = np.array(["Gilly"] * 1000 + ["Mun"] * 10)
    indices = train_capture.body_balanced_indices(labels, 100, np.random.default_rng(1))
    assert np.count_nonzero(labels[indices] == "Mun") == 50
    assert np.count_nonzero(labels[indices] == "Gilly") == 50


def test_capture_checkpoint_roundtrip_preserves_task_body_and_goal_contract(checkpoint):
    path, _, config, catalog = checkpoint
    model, metadata = train_capture.load_capture_controller(path)
    assert metadata["task"] == train_capture.TASK
    assert metadata["body_catalog"] == catalog
    assert metadata["train_bodies"] == ["Mun"] and metadata["heldout_bodies"] == [
        "Duna"
    ]
    assert CaptureConfig.from_dict(metadata["environment"]) == config
    assert model.observation_space.shape == (27,) and model.action_space.shape == (3,)


@pytest.mark.parametrize("old_task", ["launch_v1", "orbital_capture_v1"])
def test_incompatible_task_is_rejected_before_model_deserialization(
    tmp_path, monkeypatch, old_task
):
    path = tmp_path / "launch.zip"
    path.with_suffix(".metadata.json").write_text(
        json.dumps(dict(task=old_task, algorithm="ppo"))
    )

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "A launch model must not be deserialized by the capture loader"
        )

    monkeypatch.setattr(PPO, "load", forbidden)
    with pytest.raises(ValueError, match=train_capture.TASK):
        train_capture.load_capture_controller(path)


@pytest.mark.parametrize("field", ["environment", "body_catalog", "train_bodies"])
def test_capture_sidecar_cannot_swap_body_physics_or_training_split(checkpoint, field):
    path, _, _, _ = checkpoint
    metadata_path = path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text())
    if field == "environment":
        metadata[field]["target_altitude"] += 1000
    elif field == "body_catalog":
        metadata[field]["Duna"]["gravitational_parameter"] *= 2
    else:
        metadata[field] = ["Duna"]
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="embedded|contract"):
        train_capture.load_capture_controller(path)


def test_capture_contract_rejects_gamma_and_action_space_changes(checkpoint):
    path, model, _, _ = checkpoint
    model.gamma = 0.9
    train_capture.save_checkpoint(model, path)
    with pytest.raises(ValueError, match="gamma|discount"):
        train_capture.load_capture_controller(path)
    model.gamma = model.orbiter_capture_contract["environment"]["gamma"]
    model.action_space = spaces.Box(0, 1, shape=(3,), dtype=np.float32)
    train_capture.save_checkpoint(model, path)
    with pytest.raises(ValueError, match="action_space|contract"):
        train_capture.load_capture_controller(path)


def test_moved_checkpoint_needs_only_its_own_sidecar(checkpoint, tmp_path):
    path, _, _, _ = checkpoint
    destination = tmp_path / "moved"
    destination.mkdir()
    moved = destination / "model.zip"
    shutil.copy2(path, moved)
    shutil.copy2(
        path.with_suffix(".metadata.json"), moved.with_suffix(".metadata.json")
    )
    _, metadata = train_capture.load_capture_controller(moved)
    assert metadata["task"] == train_capture.TASK


def test_v2_metadata_before_optional_arrival_curriculum_still_loads(checkpoint):
    path, model, _, _ = checkpoint
    model.orbiter_capture_contract["environment"].pop("arrival_vinf_by_body")
    train_capture.save_checkpoint(model, path)
    loaded, metadata = train_capture.load_capture_controller(path)
    config, _, _, _ = train_capture.validate_contract(loaded, metadata)
    assert config.arrival_vinf_by_body == {}
    assert "arrival_vinf_by_body" not in metadata["environment"]


class FakeCapture:
    """One-step outcomes isolate split accounting from orbital dynamics."""

    seen = []
    actions = []
    seen_options = []

    def __init__(self, config, body_catalog):
        self.config = config
        self.body_catalog = body_catalog
        self.action_space = spaces.Box(-1, 1, shape=(3,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.body = options["body"]
        self.seen_options.append(dict(options))
        assert self.body in self.body_catalog
        self.seen.append((self.body, seed))
        return np.zeros(27, dtype=np.float32), {}

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        success = self.body == "Mun"
        return (
            np.zeros(27, dtype=np.float32),
            1.0,
            True,
            False,
            dict(
                success=success,
                body=self.body,
                termination_reason="orbit_captured" if success else "out_of_fuel",
            ),
        )

    def close(self):
        pass


def test_evaluation_reports_each_held_out_body_failure_and_interval(
    catalog, monkeypatch
):
    FakeCapture.seen = []
    monkeypatch.setattr(train_capture, "CaptureEnv", FakeCapture)
    model = SimpleNamespace(predict=lambda *args, **kwargs: (np.zeros(3), None))
    result = train_capture.evaluate(
        model, CaptureConfig(), catalog, ["Mun", "Duna"], [100, 101]
    )
    assert result["per_body"]["Mun"]["successes"] == 2
    assert result["per_body"]["Duna"]["failure_counts"] == {"out_of_fuel": 2}
    assert result["successes"] == 2 and result["episodes"] == 4
    assert result["per_body"]["Duna"]["success_ci95"][1] > 0
    assert FakeCapture.seen == [
        ("Mun", 100),
        ("Mun", 101),
        ("Duna", 100),
        ("Duna", 101),
    ]
    assert result["scope"] == "local inbound orbital capture surrogate only"


def test_behavior_clone_uses_only_training_bodies_and_complete_value_targets(
    checkpoint, monkeypatch
):
    _, model, config, catalog = checkpoint
    FakeCapture.seen = []
    monkeypatch.setattr(train_capture, "CaptureEnv", FakeCapture)
    monkeypatch.setattr(
        train_capture,
        "capture_teacher",
        lambda env: np.array([0.1, -0.2, 0.05], dtype=np.float32),
    )
    entries = []
    log = SimpleNamespace(
        write=lambda event, **data: entries.append(dict(event=event, **data))
    )
    train_capture.behavior_clone(
        model,
        config,
        catalog,
        ["Mun"],
        episodes=3,
        seed=7,
        epochs=2,
        value_epochs=2,
        log=log,
    )
    assert [body for body, _ in FakeCapture.seen] == ["Mun"] * 3
    assert (
        entries[-1]["bodies"] == ["Mun"] and entries[-1]["complete_value_targets"] == 3
    )
    assert entries[-1]["per_body"] == {"Mun": {"successes": 3, "failures": 0}}
    assert np.isfinite(entries[-1]["final_action_mse"]) and np.isfinite(
        entries[-1]["final_value_mse"]
    )


def test_resume_rejects_changed_body_split_before_touching_outputs(
    checkpoint, tmp_path
):
    path, _, _, _ = checkpoint
    run = tmp_path / "existing"
    run.mkdir()
    record = run / "config.json"
    record.write_text("prior experiment evidence")
    with pytest.raises(ValueError, match="Body sets"):
        train_capture.main(
            ["--resume", str(path), "--run-dir", str(run), "--train-bodies", "Duna"]
        )
    assert record.read_text() == "prior experiment evidence"


def test_real_local_teacher_baseline_is_reported_separately_by_body(catalog):
    config = CaptureConfig.for_body(catalog["Mun"])
    result = train_capture.evaluate(
        None, config, catalog, ["Mun", "Duna"], [100_000], kind="teacher"
    )
    assert result["kind"] == "teacher" and result["success_rate"] == 1
    assert set(result["per_body"]) == {"Mun", "Duna"}
    assert all(row["success_ci95"][0] < 0.5 for row in result["per_body"].values())


def test_dagger_queries_training_states_and_balances_original_and_recovery_labels(
    checkpoint, monkeypatch, tmp_path
):
    _, model, config, catalog = checkpoint
    FakeCapture.seen, FakeCapture.actions = [], []
    monkeypatch.setattr(train_capture, "CaptureEnv", FakeCapture)
    queried_bodies = []

    def label(env):
        queried_bodies.append(env.body)
        return np.full(3, 0.2, dtype=np.float32)

    monkeypatch.setattr(train_capture, "capture_teacher", label)
    original_mse = train_capture.torch.nn.functional.mse_loss
    target_batches = []

    def record_mse(prediction, target, *args, **kwargs):
        target_batches.append(target.detach().cpu().numpy())
        return original_mse(prediction, target, *args, **kwargs)

    monkeypatch.setattr(train_capture.torch.nn.functional, "mse_loss", record_mse)
    entries = []
    log = SimpleNamespace(
        write=lambda event, **data: entries.append(dict(event=event, **data))
    )
    demonstrations = (
        np.zeros((4, 27), dtype=np.float32),
        np.full((4, 3), -0.7, dtype=np.float32),
    )
    result = train_capture.dagger(
        model,
        config,
        catalog,
        ["Mun"],
        demonstrations,
        rounds=2,
        episodes=2,
        epochs=1,
        beta=0.8,
        seed=4,
        validation_seeds=[200],
        run_dir=tmp_path,
        log=log,
        minimum_success=1,
    )
    assert queried_bodies == ["Mun"] * 4  # Duna remains genuinely unqueried
    assert result["robust"] and result["aggregated_states"] == 4
    assert [entry["expert_mixing"] for entry in entries] == [0.8, 0]
    assert all(
        np.count_nonzero(np.isclose(batch[:, 0], -0.7)) == len(batch) // 2
        for batch in target_batches
    )
    assert all(
        np.count_nonzero(np.isclose(batch[:, 0], 0.2)) == len(batch) // 2
        for batch in target_batches
    )
    assert (tmp_path / "dagger_best.zip").exists()
    assert (tmp_path / "dagger_evaluation_best.json").exists()


def test_current_policy_critic_warmstart_does_not_move_actor(checkpoint, monkeypatch):
    _, model, config, catalog = checkpoint
    monkeypatch.setattr(train_capture, "CaptureEnv", FakeCapture)
    actor_before = [
        parameter.detach().clone()
        for parameter in model.policy.mlp_extractor.policy_net.parameters()
    ]
    head_before = [
        parameter.detach().clone() for parameter in model.policy.action_net.parameters()
    ]
    values_before = [
        parameter.detach().clone() for parameter in model.policy.value_net.parameters()
    ]
    entries = []
    log = SimpleNamespace(
        write=lambda event, **data: entries.append(dict(event=event, **data))
    )
    train_capture.warmstart_current_values(
        model, config, catalog, ["Mun"], episodes=4, epochs=2, seed=0, log=log
    )
    assert all(
        train_capture.torch.equal(before, after)
        for before, after in zip(
            actor_before, model.policy.mlp_extractor.policy_net.parameters()
        )
    )
    assert all(
        train_capture.torch.equal(before, after)
        for before, after in zip(head_before, model.policy.action_net.parameters())
    )
    assert any(
        not train_capture.torch.equal(before, after)
        for before, after in zip(values_before, model.policy.value_net.parameters())
    )
    assert entries[-1]["terminal_transitions"] == 4


def test_dagger_requires_original_demos_and_a_fresh_run():
    with pytest.raises(SystemExit):
        train_capture.parse_args(["--dagger-rounds", "2", "--bc-episodes", "0"])
    with pytest.raises(SystemExit):
        train_capture.parse_args(["--dagger-rounds", "2", "--resume", "old.zip"])
    args = train_capture.parse_args(
        ["--dagger-rounds", "2", "--initialize", "old.zip", "--bc-epochs", "0"]
    )
    assert args.initialize.name == "old.zip" and args.bc_epochs == 0
    with pytest.raises(SystemExit):
        train_capture.parse_args(["--initialize", "old.zip", "--resume", "old.zip"])


def test_body_arrival_fractions_are_explicit_and_reject_ambiguous_input():
    args = train_capture.parse_args(["--arrival-vinf-by-body", "Mun=.7,Minmus=1.9"])
    assert args.arrival_vinf_by_body == {"Mun": 0.7, "Minmus": 1.9}
    for invalid in ("Mun=-1", "Mun=nan", "Mun=.7,Mun=.8", "Mun:.7"):
        with pytest.raises(SystemExit):
            train_capture.parse_args(["--arrival-vinf-by-body", invalid])


def test_label_collection_can_preserve_a_pretrained_actor(checkpoint, monkeypatch):
    _, model, config, catalog = checkpoint
    monkeypatch.setattr(train_capture, "CaptureEnv", FakeCapture)
    monkeypatch.setattr(
        train_capture, "capture_teacher", lambda env: np.full(3, 0.2, dtype=np.float32)
    )
    before = {
        name: value.detach().clone()
        for name, value in model.policy.state_dict().items()
    }
    entries = []
    labels = train_capture.behavior_clone(
        model,
        config,
        catalog,
        ["Mun"],
        episodes=2,
        seed=0,
        epochs=0,
        value_epochs=0,
        log=SimpleNamespace(write=lambda event, **data: entries.append(data)),
    )
    assert len(labels[0]) == 2 and entries[-1]["final_action_mse"] is None
    assert all(
        train_capture.torch.equal(before[name], value)
        for name, value in model.policy.state_dict().items()
    )


def test_imitation_initialization_reuses_weights_but_resets_ppo_state(
    checkpoint, tmp_path, monkeypatch
):
    path, original, _, _ = checkpoint
    original.num_timesteps = 12345
    loss = sum(parameter.square().sum() for parameter in original.policy.parameters())
    original.policy.optimizer.zero_grad()
    loss.backward()
    original.policy.optimizer.step()
    assert original.policy.optimizer.state
    train_capture.save_checkpoint(original, path)
    observed = []

    def inspect(self, *args, **kwargs):
        observed.append(self)
        assert self.num_timesteps == 0 and not self.policy.optimizer.state
        assert self.learning_rate == 1e-4
        assert all(
            train_capture.torch.equal(original.policy.state_dict()[name], value)
            for name, value in self.policy.state_dict().items()
        )
        return self

    monkeypatch.setattr(PPO, "learn", inspect)
    run = tmp_path / "initialized"
    train_capture.main(
        [
            "--initialize",
            str(path),
            "--run-dir",
            str(run),
            "--skip-baselines",
            "--mass",
            "2500",
            "--arrival-vinf-by-body",
            "Mun=.7,Duna=1.9",
            "--train-bodies",
            "Mun,Duna",
            "--heldout-bodies",
            "",
            "--bc-episodes",
            "0",
            "--learning-rate",
            "0.0001",
            "--steps",
            "4",
            "--n-envs",
            "1",
            "--rollout-steps",
            "4",
            "--batch-size",
            "4",
            "--eval-episodes",
            "1",
        ]
    )
    assert len(observed) == 1
    record = json.loads((run / "config.json").read_text())
    assert record["initialization"]["source_num_timesteps"] == 12345
    assert record["initialization"]["source_train_bodies"] == ["Mun"]
    assert record["initialization"]["optimizer_and_step_count_reset"] is True
    assert len(record["initialization"]["sha256"]) == 64
    assert record["contract"]["environment"]["mass"] == 2500
    assert record["contract"]["environment"]["arrival_vinf_by_body"] == {
        "Mun": 0.7,
        "Duna": 1.9,
    }
    assert record["contract"]["train_bodies"] == ["Mun", "Duna"]
    assert record["contract"]["heldout_bodies"] == []
    assert original.orbiter_capture_contract["environment"]["mass"] == 4000


def test_initialization_cannot_relabel_previously_trained_bodies_as_heldout(
    checkpoint, tmp_path
):
    path, _, _, _ = checkpoint
    run = tmp_path / "mislabelled"
    with pytest.raises(ValueError, match="Previously trained"):
        train_capture.main(
            [
                "--initialize",
                str(path),
                "--run-dir",
                str(run),
                "--train-bodies",
                "Duna",
                "--heldout-bodies",
                "Mun",
            ]
        )
    assert not run.exists()
    with pytest.raises(ValueError, match="retain previously trained"):
        train_capture.main(
            [
                "--initialize",
                str(path),
                "--run-dir",
                str(run),
                "--train-bodies",
                "Duna",
                "--heldout-bodies",
                "",
            ]
        )
    assert not run.exists()


def test_speed_stress_overrides_body_curriculum_and_reports_infeasible_mass_ranges(
    catalog, monkeypatch
):
    def evaluate(model, config, *args, **kwargs):
        return {
            "fraction": config.arrival_vinf_fraction,
            "overrides": config.arrival_vinf_by_body,
        }

    monkeypatch.setattr(train_capture, "evaluate", evaluate)
    config = CaptureConfig(
        mass=2500,
        dry_mass=1980,
        randomization=0.1,
        arrival_vinf_by_body={"Mun": 0.7, "Duna": 1.9},
    )
    cases = train_capture.evaluate_stress(None, config, catalog, ["Mun"], [0])["cases"]
    assert cases["arrival_vinf_0.2_circular_speed"] == {
        "fraction": 0.2,
        "overrides": {},
    }
    assert cases["arrival_vinf_0.7_circular_speed"] == {
        "fraction": 0.7,
        "overrides": {},
    }
    invalid = cases["craft_mass_2187.5_thrust_1_isp_1"]
    assert invalid["status"] == "infeasible_vehicle_randomization"
    assert invalid["minimum_randomized_mass"] < invalid["dry_mass"]


def test_lower_goal_stress_stays_above_jool_atmosphere(monkeypatch):
    catalog = {"Jool": SystemCatalog.load().bodies["Jool"]}
    FakeCapture.seen_options = []
    monkeypatch.setattr(train_capture, "CaptureEnv", FakeCapture)
    model = SimpleNamespace(predict=lambda *args, **kwargs: (np.zeros(3), None))
    train_capture.evaluate(
        model,
        CaptureConfig(),
        catalog,
        ["Jool"],
        [200],
        target_clearance_multiplier=0.6,
    )
    target = FakeCapture.seen_options[-1]["target_altitude"]
    assert target == pytest.approx(212_000)
    assert (
        target
        > catalog["Jool"]["atmosphere_depth"] + CaptureConfig().orbit_tolerance + 1000
    )


@pytest.mark.parametrize("with_dagger", [False, True])
def test_initial_success_gate_prevents_unready_ppo_updates_and_exits_two(
    tmp_path, monkeypatch, with_dagger
):
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "PPO updates must not start when the initial capture policy fails its gate"
        )

    monkeypatch.setattr(PPO, "learn", forbidden)
    run = tmp_path / "gated"
    extra = []
    if with_dagger:
        monkeypatch.setattr(train_capture, "behavior_clone", lambda *a, **k: None)
        monkeypatch.setattr(
            train_capture, "dagger", lambda *a, **k: {"robust": False, "validation": {}}
        )
        extra = ["--dagger-rounds", "1", "--bc-episodes", "1"]
    status = train_capture.cli_main(
        [
            "--run-dir",
            str(run),
            "--train-bodies",
            "Mun",
            "--heldout-bodies",
            "",
            "--steps",
            "1",
            "--n-envs",
            "1",
            "--rollout-steps",
            "2",
            "--batch-size",
            "2",
            "--bc-episodes",
            "0",
            "--eval-episodes",
            "1",
            "--skip-baselines",
            "--min-initial-success",
            "1",
        ]
        + extra
    )
    assert status == 2
    events = [
        json.loads(line) for line in (run / "progress.jsonl").read_text().splitlines()
    ]
    assert any(row["event"] == "training_blocked_initial_validation" for row in events)
    assert not any(row["event"] == "training_start" for row in events)
    assert (run / "latest.zip").exists()
    saved, _ = train_capture.load_capture_controller(run / "latest.zip")
    assert saved.orbiter_training_status == "initial_validation_failed"


@pytest.mark.parametrize("source_flag", ["--resume", "--initialize"])
def test_new_training_clears_stale_failed_status_and_marks_completion(
    checkpoint, tmp_path, monkeypatch, source_flag
):
    path, previous, _, _ = checkpoint
    previous.orbiter_training_status = "initial_validation_failed"
    train_capture.save_checkpoint(previous, path)
    inspected = []

    def inspect(self, *args, **kwargs):
        assert self.orbiter_training_status == "in_progress"
        inspected.append(True)
        return self

    monkeypatch.setattr(PPO, "learn", inspect)
    run = tmp_path / source_flag[2:]
    status = train_capture.cli_main(
        [
            source_flag,
            str(path),
            "--run-dir",
            str(run),
            "--steps",
            "4",
            "--n-envs",
            "1",
            "--rollout-steps",
            "4",
            "--batch-size",
            "4",
            "--bc-episodes",
            "0",
            "--eval-episodes",
            "1",
            "--skip-baselines",
        ]
    )
    assert status == 0 and inspected == [True]
    current, _ = train_capture.load_capture_controller(run / "latest.zip")
    assert current.orbiter_training_status == "complete"
