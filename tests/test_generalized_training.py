"""Checkpoint migration and continuation safeguards for GPU experiments."""

import hashlib
import json
import math

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
    for name in (
        "actor_loss",
        "critic_loss",
        "gradient_norm_before_clip",
        "actor_gradient_norm_before_clip",
        "critic_gradient_norm_before_clip",
        "explained_variance",
        "mean_td_error",
        "mean_absolute_td_error",
        "action_tanh_saturation_fraction",
    ):
        assert math.isfinite(events[0][name])
    assert events[0]["cumulative_optimizer_steps"] == optimizer_steps
    assert optimizer_steps == 1
    assert events[0]["gradient_norm_before_clip"] == pytest.approx(
        math.hypot(
            events[0]["actor_gradient_norm_before_clip"],
            events[0]["critic_gradient_norm_before_clip"],
        ),
        rel=1e-6,
    )


@pytest.mark.parametrize(
    "curriculum,physics,sha256",
    [
        (
            "efficient", "constant-cda",
            "ed44d04ff5cbc8dcab30813531221edb1331b4b715ae56f9ced01bb03584ad01",
        ),
        (
            "efficient", "aero",
            "2726d219945cdf49611d653922198f73e5b8f3dc406ed5927cf6740007e64820",
        ),
        (
            "stock", "constant-cda",
            "c568695a91d0d28391c553c56c4f324660cae9a2b0b1af9018b0e20fd8e4fa80",
        ),
        (
            "stock", "aero",
            "69a97222db1ceea1613cbe91ea4d5fbfac772fae01b1336b3df11afb9eee8d4b",
        ),
    ],
)
def test_original_v3_v4_contract_bytes_stay_compatible(curriculum, physics, sha256):
    contract = training.experiment_contract(curriculum, physics)[2]
    assert hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest() == sha256
    assert "objective" not in contract


@pytest.mark.parametrize("penalty", [0.0, -30.0])
def test_finite_objective_has_strict_distinct_checkpoint_contract(tmp_path, penalty):
    _, _, contract = training.experiment_contract(
        "stock", "aero", deadline_mode="terminal", timeout_penalty=penalty
    )
    assert contract["task"] == training.AERO_DEADLINE_TASK
    assert contract["observations"] == 56
    assert contract["objective"]["timeout_penalty"] == penalty
    assert contract["objective"]["deadline_bootstrap"] is False
    path = tmp_path / "finite.pt"
    training.save_policy(path, training.VehiclePolicy(16, observations=56), contract, {})
    _, record = training.load_policy(path)
    assert record["contract"] == contract
    record["contract"]["objective"]["deadline_bootstrap"] = True
    torch.save(record, path)
    with pytest.raises(ValueError, match="contract differs"):
        training.load_policy(path)


def test_actor_only_transfer_resets_identical_critic_for_every_objective(tmp_path):
    source = source_checkpoint(tmp_path / "v3.pt")
    observation = BatchedAeroAscent(12, seed=91).observation()
    critics = []
    for mode, penalty in (("truncate", 0), ("terminal", 0), ("terminal", -30)):
        contract = training.experiment_contract(
            "stock", "aero", deadline_mode=mode, timeout_penalty=penalty
        )[2]
        model, _ = training.initialize_actor_policy(
            tmp_path / "v3.pt", contract, seed=601
        )
        torch.testing.assert_close(model(observation), source(observation[:, :54]))
        torch.testing.assert_close(model.log_std, source.log_std)
        critics.append(model.critic.state_dict())
        assert not torch.equal(model.critic[0].weight[:, :54], source.critic[0].weight)
    torch.manual_seed(601)
    expected = training.VehiclePolicy(16, observations=56)
    for critic in critics:
        for key, value in expected.critic.state_dict().items():
            torch.testing.assert_close(critic[key], value, rtol=0, atol=0)


def test_actor_source_requires_unmodified_original_contract(tmp_path):
    source_checkpoint(tmp_path / "v3.pt")
    record = torch.load(tmp_path / "v3.pt", weights_only=True)
    record["contract"]["gamma"] = 0.5
    torch.save(record, tmp_path / "altered.pt")
    contract = training.experiment_contract("stock", "aero")[2]
    with pytest.raises(ValueError, match="original curriculum contract"):
        training.initialize_actor_policy(tmp_path / "altered.pt", contract, seed=601)


def test_v4_actor_only_initialization_retains_aerodynamic_input_weights(tmp_path):
    torch.manual_seed(71)
    source = training.VehiclePolicy(16, observations=56)
    contract = training.experiment_contract("stock", "aero")[2]
    path = tmp_path / "v4.pt"
    training.save_policy(path, source, contract, {})
    finite_contract = training.experiment_contract(
        "stock", "aero", deadline_mode="terminal", timeout_penalty=-30
    )[2]
    initialized, _ = training.initialize_actor_policy(path, finite_contract, seed=601)
    observation = BatchedAeroAscent(12, seed=91).observation()
    torch.testing.assert_close(initialized(observation), source(observation))
    assert initialized.actor[0].weight[:, 54:].abs().sum() > 0
    assert not torch.equal(initialized.critic[0].weight, source.critic[0].weight)


def test_value_warmup_never_updates_the_inherited_actor_or_log_std():
    torch.manual_seed(11)
    model = training.VehiclePolicy(16)
    env = training.BatchedAscent(4, seed=11, max_time=4, deadline_mode="terminal")
    actor_before = {key: value.clone() for key, value in model.actor.state_dict().items()}
    critic_before = {key: value.clone() for key, value in model.critic.state_dict().items()}
    std_before = model.log_std.clone()
    result = training.warm_value(model, env, steps=2, updates=2, batch_size=8)
    assert result["complete_value_targets"] == 8
    assert result["actor_updated"] is False
    for key, value in model.actor.state_dict().items():
        torch.testing.assert_close(value, actor_before[key], rtol=0, atol=0)
    torch.testing.assert_close(model.log_std, std_before, rtol=0, atol=0)
    assert any(
        not torch.equal(value, critic_before[key])
        for key, value in model.critic.state_dict().items()
    )


def test_warmup_physics_rollout_is_paired_across_deadline_objectives():
    torch.manual_seed(11)
    model = training.VehiclePolicy(16)
    fingerprints = []
    target_counts = []
    for mode, penalty in (("truncate", 0), ("terminal", 0), ("terminal", -30)):
        env = training.BatchedAscent(
            4, seed=11, max_time=4, deadline_mode=mode, timeout_penalty=penalty
        )
        torch.manual_seed(6000601)
        result = training.warm_value(model, env, steps=2, updates=2, batch_size=8)
        fingerprints.append(
            (result["initial_observation_sha256"], result["final_observation_sha256"])
        )
        target_counts.append(result["complete_value_targets"])
    assert len(set(fingerprints)) == 1
    # The baseline deliberately bootstraps time limits. Finite mission arms
    # instead have known terminal returns over those same physical trajectories.
    assert target_counts == [0, 8, 8]


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--deadline-mode", "truncate", "--timeout-penalty", "-30"],
        ["--deadline-mode", "terminal", "--timeout-penalty", "nan"],
        ["--deadline-mode", "terminal", "--timeout-penalty", "1"],
        ["--validation-seed", "71", "--test-seed", "71"],
    ],
)
def test_invalid_objective_or_overlapping_selection_bank_fails_before_output(
    tmp_path, extra_args
):
    output = tmp_path / "invalid"
    with pytest.raises(SystemExit):
        training.main(["--run-dir", str(output), "--device", "cpu", *extra_args])
    assert not output.exists()


@pytest.mark.parametrize("mode,penalty", [("truncate", 0), ("terminal", 0), ("terminal", -30)])
@pytest.mark.parametrize("seed", [601, 602])
def test_ablation_propagates_objective_and_independent_paired_seeds(
    tmp_path, monkeypatch, mode, penalty, seed
):
    source_checkpoint(tmp_path / "v3.pt")
    evaluations = []
    training_environments = []

    def short_evaluation(_model, **kwargs):
        evaluations.append(kwargs)
        return dict(
            episodes=kwargs["count"], successes=1,
            success_rate=1 / kwargs["count"], success_ci95=[0, 1], failure_counts={},
            results=[dict(world=0, craft_sha256="same-craft-bank")],
        )

    def warmup(_model, env, **_kwargs):
        training_environments.append(env)
        return {"actor_updated": False}

    def ppo(_model, env, *, selected_callback, **_kwargs):
        training_environments.append(env)
        selected_callback(8)
        return 8

    def forbidden(*_args, **_kwargs):
        pytest.fail("ablation must not run imitation or the full CPU/capture suite")

    monkeypatch.setattr(training, "evaluate", short_evaluation)
    monkeypatch.setattr(training, "warm_value", warmup)
    monkeypatch.setattr(training, "ppo_train", ppo)
    for name in ("collect_labels", "cpu_parity", "evaluate_captured_transfer"):
        monkeypatch.setattr(training, name, forbidden)
    output = tmp_path / "ablation"
    training.main(
        [
            "--run-dir", str(output), "--device", "cpu", "--physics", "aero",
            "--curriculum", "stock", "--actor-from", str(tmp_path / "v3.pt"),
            "--ppo-only", "--ppo-updates", "1", "--value-updates", "1",
            "--worlds", "4", "--eval-worlds", "8", "--seed", str(seed),
            "--validation-seed", "1500505", "--test-seed", "2500505",
            "--evaluation-suite", "ablation", "--deadline-mode", mode,
            "--timeout-penalty", str(penalty),
        ]
    )
    assert len(training_environments) == 2
    assert all(env.deadline_mode == mode for env in training_environments)
    assert all(env.timeout_penalty == penalty for env in training_environments)
    assert [call["seed"] for call in evaluations] == [1500505, 1500505, 2500505, 2500505]
    assert all(call["deadline_mode"] == mode for call in evaluations)
    assert all(call["timeout_penalty"] == penalty for call in evaluations)
    assert all(call["export_crafts"] for call in evaluations[-2:])
    summary = json.loads((output / "summary.json").read_text())
    assert summary["evaluation_suite"] == "ablation"
    assert summary["final_tests_select_checkpoint"] is False
    assert "cpu_parity" not in summary and "captured_transfer" not in summary
    ancestry = summary["selected_ancestry"]
    assert ancestry["initialization"]["critic"] == "fresh_seeded"
    assert ancestry["initialization"]["critic_seed"] == seed
    assert ancestry["initialization"]["value_fit_seed"] == seed + 6000000
    assert ancestry["upstream"]["critic_restored"] is False
    assert ancestry["upstream"]["ancestry"]["ppo_transitions"] == 123


def test_paired_final_rows_have_identical_craft_fingerprints_across_objectives(monkeypatch):
    make_environment = training.make_environment

    def short_environment(count, **kwargs):
        return make_environment(count, max_time=4, **kwargs)

    monkeypatch.setattr(training, "make_environment", short_environment)
    envelope = training.experiment_contract("stock", "aero")[0]
    model = training.VehiclePolicy(16, observations=56)
    rows = []
    for mode, penalty in (("truncate", 0), ("terminal", 0), ("terminal", -30)):
        result = training.evaluate(
            model, count=8, device="cpu", seed=2500505, envelope=envelope,
            export_crafts=True, deadline_mode=mode, timeout_penalty=penalty,
        )
        rows.append(result["results"])
    for index in range(8):
        assert len({arm[index]["craft_sha256"] for arm in rows}) == 1
        assert len({arm[index]["elapsed"] for arm in rows}) == 1
    assert all(row["reason"] == 5 for arm in rows for row in arm)
    assert rows[2][0]["task_reward"] == rows[1][0]["task_reward"] - 30
