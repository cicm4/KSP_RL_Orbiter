"""Qualified-label, history, checkpoint, and closed-loop distillation checks."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import pytest

torch = pytest.importorskip("torch")

from orbiter import distill_ascent as distillation
from orbiter.ascent_planner import SearchBudget, config_fingerprint, plan_ascent
from orbiter.env import EnvConfig, Stage
from orbiter.generalized_aero_env import BatchedAeroAscent


@pytest.fixture(scope="module", autouse=True)
def small_tensor_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def qualified_reports(tmp_path_factory):
    two = replace(EnvConfig(), randomization=0, target_altitude=100000,
                  orbit_tolerance=20000, max_time=1600, physics_dt=1)
    three = replace(two, stages=(
        two.stages[0], Stage(600, 600, 120000, 345, 85),
        Stage(400, 5400, 120000, 345, 85),
    ))
    report = plan_ascent([two, three], budget=SearchBudget(population=8, iterations=2, seed=91))
    assert report["qualified_successes"] == 2
    directory = tmp_path_factory.mktemp("distill-witnesses")
    paths = []
    for index in range(2):
        selected = deepcopy(report)
        selected["results"] = [selected["results"][index]]
        selected["craft_count"] = 1
        path = directory / f"bank{index}.json"
        path.write_text(json.dumps(selected, allow_nan=False))
        paths.append(path)
    return paths, report


def contract(architecture="gru", *, hide_aero=False, residual_bounds=(2, 1)):
    return distillation.student_contract(
        architecture, hidden=8, residual_bounds=residual_bounds, hide_aero=hide_aero,
        bptt_steps=32, source_witness_sha256={"train": "a" * 64, "validation": "b" * 64},
        train_crafts=["c" * 64], validation_crafts=["d" * 64],
    )


def test_labels_replay_exact_qualified_actions_in_order_with_previous_action(qualified_reports):
    paths, report = qualified_reports
    bank = distillation.load_witnesses(paths[0])
    trajectory = bank.trajectories[0]
    assert bank.source_sha256 == hashlib.sha256(paths[0].read_bytes()).hexdigest()
    assert trajectory.teacher_actions.shape[1] == 2
    assert trajectory.observations.shape == (len(trajectory.teacher_actions), 56)
    torch.testing.assert_close(trajectory.previous_actions[0], torch.zeros(2))
    torch.testing.assert_close(trajectory.previous_actions[1:], trajectory.teacher_actions[:-1])
    torch.testing.assert_close(trajectory.teacher_actions, torch.tensor(report["results"][0]["actions"]))
    env = BatchedAeroAscent(1, seed=0)
    env.load_configs([trajectory.config])
    torch.testing.assert_close(trajectory.observations[0], env.observation()[0])
    assert trajectory.qualification["qualified"]
    assert trajectory.qualification["coast"]["duration_s"] == pytest.approx(
        trajectory.qualification["coast"]["requested_period_s"]
    )
    assert (trajectory.baseline_actions - trajectory.teacher_actions).abs().max() > 0.01


@pytest.mark.parametrize("failure", ["unqualified", "coast", "actions", "hash", "pressure"])
def test_untrusted_or_unqualified_teacher_cannot_become_labels(tmp_path, qualified_reports, failure):
    paths, _ = qualified_reports
    report = json.loads(paths[0].read_text())
    row = report["results"][0]
    kwargs = {}
    if failure == "unqualified":
        row["qualified"] = False
    elif failure == "coast":
        row["qualification"]["coast"]["duration_s"] /= 2
    elif failure == "actions":
        row["actions"] = [[-1, 1] for _ in row["actions"]]
    elif failure == "hash":
        row["config"]["target_altitude"] += 1000
    elif failure == "pressure":
        kwargs["max_dynamic_pressure"] = 1
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        distillation.load_witnesses(path, **kwargs)


def test_pressure_gate_cannot_be_weakened_by_cli(tmp_path, qualified_reports, monkeypatch):
    paths, _ = qualified_reports
    report = json.loads(paths[0].read_text())
    report["budget"]["max_dynamic_pressure"] = 60000
    path = tmp_path / "pressure.json"
    path.write_text(json.dumps(report))
    limits = []
    qualification = report["results"][0]["qualification"]

    def qualify(_config, _actions, *, max_dynamic_pressure):
        limits.append(max_dynamic_pressure)
        return qualification

    monkeypatch.setattr(distillation, "qualify_witness", qualify)
    distillation.load_witnesses(path, max_dynamic_pressure=90000)
    assert limits == [60000]


def test_disjoint_split_rejects_same_vehicle_with_a_different_mission(qualified_reports):
    paths, _ = qualified_reports
    train = distillation.load_witnesses(paths[0])
    validation = distillation.load_witnesses(paths[1])
    distillation.require_disjoint_banks(train, validation)
    original = train.configs[0]
    changed = replace(original, target_altitude=original.target_altitude + 1000,
                      physics_dt=original.physics_dt / 2, max_time=original.max_time + 50)
    assert config_fingerprint(original) != config_fingerprint(changed)
    assert distillation.vehicle_fingerprint(original) == distillation.vehicle_fingerprint(changed)
    numeric_alias = replace(original, payload_mass=int(original.payload_mass))
    assert distillation.vehicle_fingerprint(original) == distillation.vehicle_fingerprint(numeric_alias)
    disguised = replace(validation, report_vehicle_sha256=(distillation.vehicle_fingerprint(changed),))
    with pytest.raises(ValueError, match="overlap"):
        distillation.require_disjoint_banks(train, disguised)


@pytest.mark.parametrize("architecture", ["feedforward", "gru"])
def test_students_start_at_default_guidance_and_have_bounded_residuals(architecture):
    torch.manual_seed(41)
    model = distillation.ResidualStudent(contract(architecture, residual_bounds=(0.2, 0.3)))
    observation = torch.randn(2, 5, 56)
    previous = torch.randn(2, 5, 2)
    baseline = torch.rand(2, 5, 2) * 2 - 1
    initial, _ = model(observation, previous, baseline)
    torch.testing.assert_close(initial, baseline, rtol=0, atol=0)
    with torch.no_grad():
        model.output.bias.fill_(20)
    changed, _ = model(observation, previous, baseline)
    assert ((changed - baseline).abs() <= torch.tensor([0.2, 0.3]) + 1e-7).all()
    assert changed.abs().max() <= 1


@pytest.mark.parametrize("architecture", ["feedforward", "gru"])
def test_aero_feature_mask_is_real_and_explicit(architecture):
    torch.manual_seed(41)
    model = distillation.ResidualStudent(contract(architecture, hide_aero=True))
    with torch.no_grad():
        model.output.weight.normal_(0, 0.1)
    observation = torch.randn(2, 5, 56)
    altered = observation.clone()
    altered[..., 54:] += 10
    previous = torch.randn(2, 5, 2)
    baseline = torch.zeros(2, 5, 2)
    first, _ = model(observation, previous, baseline)
    second, _ = model(altered, previous, baseline)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert model.contract["masked_observation_indices"] == [54, 55]


def test_gru_history_preserves_time_order_and_resets_per_craft():
    torch.manual_seed(41)
    model = distillation.ResidualStudent(contract())
    with torch.no_grad():
        model.output.weight.normal_(0, 0.2)
    observations = torch.randn(2, 9, 56)
    previous = torch.randn(2, 9, 2)
    baseline = torch.zeros(2, 9, 2)
    together, _ = model(observations, previous, baseline)
    separate = torch.cat([
        model(observations[index:index + 1], previous[index:index + 1], baseline[index:index + 1])[0]
        for index in range(2)
    ])
    torch.testing.assert_close(together, separate)
    first, hidden = model(observations[:, :4], previous[:, :4], baseline[:, :4])
    last, _ = model(observations[:, 4:], previous[:, 4:], baseline[:, 4:], hidden)
    torch.testing.assert_close(together, torch.cat((first, last), 1))
    permutation = torch.arange(8, -1, -1)
    reversed_actions, _ = model(observations[:, permutation], previous[:, permutation], baseline[:, permutation])
    assert (together - reversed_actions[:, permutation]).abs().max() > 0.001


@pytest.mark.parametrize("corruption", ["history", "mask", "bounds", "nan", "overflow", "metadata"])
def test_checkpoint_rejects_contract_or_weight_corruption(tmp_path, corruption):
    model = distillation.ResidualStudent(contract())
    path = tmp_path / "student.pt"
    distillation.save_student(path, model, {"selected_epoch": 2})
    loaded, _ = distillation.load_student(path)
    torch.testing.assert_close(loaded.bounds, model.bounds)
    record = torch.load(path, weights_only=True)
    if corruption == "history":
        record["contract"]["previous_action_dimensions"] = 4
    elif corruption == "mask":
        record["contract"]["masked_observation_indices"] = [0]
    elif corruption == "bounds":
        record["state_dict"]["bounds"] = torch.tensor([99., 99.])
    elif corruption == "nan":
        record["state_dict"]["encoder.0.weight"][0, 0] = float("nan")
    elif corruption == "overflow":
        record["state_dict"]["encoder.0.weight"] = torch.full_like(
            record["state_dict"]["encoder.0.weight"], 1e100, dtype=torch.float64
        )
    else:
        record["ancestry"]["selected_epoch"] = 3
    torch.save(record, path)
    if corruption != "metadata":
        # Standalone checkpoint validation must catch this even without a pair.
        path.with_suffix(".metadata.json").unlink()
    with pytest.raises(ValueError):
        distillation.load_student(path)


def test_training_uses_both_architectures_and_selects_an_actual_checkpoint(tmp_path, qualified_reports):
    paths, _ = qualified_reports
    train = distillation.load_witnesses(paths[0])
    validation = distillation.load_witnesses(paths[1])
    for architecture in ("feedforward", "gru"):
        torch.manual_seed(707)
        model = distillation.ResidualStudent(contract(architecture))
        path = tmp_path / f"{architecture}.pt"
        history = distillation.train_student(
            model, train.trajectories, validation.trajectories, epochs=3,
            batch_size=1, learning_rate=0.001, seed=707, checkpoint=path,
        )
        loaded, record = distillation.load_student(path)
        assert len(history) == 4
        assert history[-1]["training_action_mse"] < history[1]["training_action_mse"]
        assert record["ancestry"]["validation_action_mse"] == min(row["validation_action_mse"] for row in history)
        assert record["ancestry"]["selected_epoch"] in range(4)
        assert loaded.contract["recurrent_reset"] == "zero_at_each_craft"


def test_closed_loop_evaluates_failed_teacher_crafts_too(qualified_reports):
    paths, report = qualified_reports
    bank = distillation.load_witnesses(paths[0])
    weak = replace(bank.configs[0], stages=(Stage(1000, 2000, 100, 320, 280),) * 2)
    bank = replace(
        bank, configs=(*bank.configs, weak),
        report_craft_sha256=(*bank.report_craft_sha256, config_fingerprint(weak)),
        report_vehicle_sha256=(*bank.report_vehicle_sha256, distillation.vehicle_fingerprint(weak)),
        reported_qualified=(*bank.reported_qualified, False),
    )
    result = distillation.evaluate_controller(None, bank)
    assert result["episodes"] == 2
    assert result["includes_unqualified_teacher_crafts"]
    assert result["results"][1]["reported_teacher_qualified"] is False
    assert result["results"][1]["qualification"]["qualified"] is False
    assert result["live_ksp_validated"] is False


def test_cli_cpu_pilot_writes_both_strict_models_and_closed_loop_results(tmp_path, qualified_reports):
    paths, _ = qualified_reports
    output = tmp_path / "pilot"
    distillation.main([
        "--train-report", str(paths[0]), "--validation-report", str(paths[1]),
        "--run-dir", str(output), "--epochs", "2", "--hidden", "8",
        "--batch-size", "1", "--bptt-steps", "32",
    ])
    summary = json.loads((output / "summary.json").read_text())
    assert set(summary["students"]) == {"feedforward", "gru"}
    assert summary["validation_is_final_unseen_test"] is False
    assert summary["validation_selects_epoch"] is True
    assert summary["default_guidance"]["episodes"] == 1
    for architecture, row in summary["students"].items():
        _, record = distillation.load_student(output / f"{architecture}.pt")
        assert record["contract"] == row["contract"]
        assert row["closed_loop_validation"]["full_unpowered_scalar_coast_required"]
        assert row["closed_loop_validation"]["live_ksp_validated"] is False
