"""Checkpoint migration and continuation safeguards for GPU experiments."""

import json

import pytest

torch = pytest.importorskip("torch")

from orbiter import train_generalized as training
from orbiter.generalized_aero_env import BatchedAeroAscent


def source_checkpoint(path, *, curriculum="stock"):
    torch.manual_seed(71)
    model = training.VehiclePolicy(16)
    _, _, contract = training.experiment_contract(curriculum)
    training.save_policy(path, model, contract, {"ppo_transitions": 123})
    return model


def test_v3_to_aero_migration_preserves_actor_and_critic_for_new_coefficients(tmp_path):
    source = source_checkpoint(tmp_path / "v3.pt")
    migrated, record = training.transfer_aero_policy(tmp_path / "v3.pt")
    observation = BatchedAeroAscent(12, seed=91).observation()
    assert observation[:, 54:].abs().sum() > 0
    torch.testing.assert_close(migrated(observation), source(observation[:, :54]))
    torch.testing.assert_close(
        migrated.critic(observation), source.critic(observation[:, :54])
    )
    assert record["ancestry"]["ppo_transitions"] == 123
    _, _, contract = training.experiment_contract("stock", "aero")
    path = tmp_path / "v4.pt"
    training.save_policy(path, migrated, contract, {})
    loaded, _ = training.load_policy(path)
    torch.testing.assert_close(loaded(observation), migrated(observation))
    with pytest.raises(ValueError, match="original v3"):
        training.transfer_aero_policy(path)


def test_aero_contract_trains_four_stage_channels_without_claiming_them_heldout():
    envelope, four_stage, contract = training.experiment_contract("stock", "aero")
    assert envelope.stages_min == 2 and envelope.stages_max == 4
    assert four_stage.stages_min == four_stage.stages_max == 4
    assert not contract["four_stage_is_heldout"]
    assert "heldout_stage_envelope" not in contract
    batch = training.make_environment(128, envelope=envelope, seed=100)
    for key in (
        "max_time",
        "orbit_tolerance",
        "mu",
        "radius",
        "rotation_period",
        "atmosphere_depth",
    ):
        assert getattr(batch, key) == contract[key]
    assert batch.atmosphere.tolist() == contract["atmosphere"]
    assert set(batch.stage_count.tolist()) == {2, 3, 4}
    assert batch.observation()[batch.stage_count == 4, 40:48].abs().sum() > 0


def test_warm_restart_cannot_overwrite_incumbent_best(tmp_path):
    run = tmp_path / "existing"
    run.mkdir()
    source_checkpoint(run / "best.pt")
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    with pytest.raises(SystemExit):
        training.main(
            [
                "--run-dir",
                str(run),
                "--device",
                "cpu",
                "--curriculum",
                "stock",
                "--warm-start",
                str(run / "best.pt"),
                "--ppo-only",
            ]
        )
    assert {p.name: p.read_bytes() for p in run.iterdir()} == before


def test_incompatible_transfer_fails_before_creating_output(tmp_path):
    source_checkpoint(tmp_path / "efficient.pt", curriculum="efficient")
    output = tmp_path / "new"
    with pytest.raises(SystemExit):
        training.main(
            [
                "--run-dir",
                str(output),
                "--device",
                "cpu",
                "--physics",
                "aero",
                "--curriculum",
                "stock",
                "--transfer-from",
                str(tmp_path / "efficient.pt"),
                "--ppo-only",
            ]
        )
    assert not output.exists()


def test_ppo_only_retains_weights_and_records_new_transition_ancestry(
    tmp_path, monkeypatch
):
    source_checkpoint(tmp_path / "v3.pt")

    def no_imitation(*_args, **_kwargs):
        pytest.fail(
            "PPO-only continuation must not replace trained weights with teacher fits"
        )

    validation_calls = 0

    def short_evaluation(_model, *, count, **_kwargs):
        nonlocal validation_calls
        validation_calls += 1
        return {
            "episodes": count,
            "successes": min(validation_calls, count),
            "success_rate": min(validation_calls, count) / count,
            "success_ci95": [0, 1],
            "failure_counts": {},
        }

    monkeypatch.setattr(training, "collect_labels", no_imitation)
    monkeypatch.setattr(training, "evaluate", short_evaluation)
    monkeypatch.setattr(
        training, "cpu_parity", lambda *_args, **_kwargs: {"passed": True}
    )
    monkeypatch.setattr(
        training,
        "evaluate_captured_transfer",
        lambda *_args, **_kwargs: {
            "policy": {"successes": 0, "episodes": 0},
            "teacher": {"successes": 0},
        },
    )
    output = tmp_path / "aero"
    training.main(
        [
            "--run-dir",
            str(output),
            "--device",
            "cpu",
            "--physics",
            "aero",
            "--curriculum",
            "stock",
            "--transfer-from",
            str(tmp_path / "v3.pt"),
            "--ppo-only",
            "--ppo-updates",
            "1",
            "--value-updates",
            "0",
            "--worlds",
            "4",
            "--horizon",
            "2",
            "--batch-size",
            "8",
            "--eval-worlds",
            "8",
        ]
    )
    _, record = training.load_policy(output / "best.pt")
    ancestry = record["ancestry"]
    assert ancestry["bc_transitions"] == ancestry["dagger_transitions"] == 0
    assert ancestry["ppo_transitions"] == 8
    assert ancestry["upstream"]["ancestry"]["ppo_transitions"] == 123
    assert ancestry["upstream"]["initialization"] == "zero_padded_v3_to_v4_inputs"
    assert not ancestry["upstream"]["optimizer_state_restored"]
    summary = json.loads((output / "summary.json").read_text())
    assert "four_stage_fresh_crafts" in summary
    assert "heldout_four_stage" not in summary
    assert not summary["live_ksp_validated"]


def test_ppo_kl_guard_stops_before_more_optimizer_steps_on_large_policy_change(
    monkeypatch,
):
    torch.manual_seed(16)
    model = training.VehiclePolicy(16)
    env = training.BatchedAscent(8, seed=16, max_time=4)
    events = []
    optimizer_steps = 0
    original_step = torch.optim.Adam.step

    def counted_step(optimizer, *args, **kwargs):
        nonlocal optimizer_steps
        optimizer_steps += 1
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", counted_step)
    training.ppo_train(
        model,
        env,
        updates=1,
        horizon=4,
        batch_size=8,
        learning_rate=0.1,
        target_kl=1e-8,
        selected_callback=lambda _: None,
        log=lambda _event, **values: events.append(values),
    )
    assert events[0]["kl_early_stop"]
    assert events[0]["optimization_epochs"] == 1
    assert events[0]["approximate_kl"] > 1e-8
    assert 0 <= events[0]["clip_fraction"] <= 1
    # A single epoch has four optimizer minibatches. A late epoch-only check
    # lets all four mutate the actor after it has already exceeded its budget.
    assert 0 < optimizer_steps < 4
    assert events[0]["optimization_minibatches"] == optimizer_steps
    assert events[0]["checked_minibatches"] == optimizer_steps + 1
    assert events[0]["kl_check"] == "before_minibatch_optimizer_step"
