from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from orbiter.env import EnvConfig, Stage
from orbiter.generalized_env import (
    BatchedAscent,
    FEATURE_SCHEMA,
    OBSERVATIONS,
    VehicleEnvelope,
    assess_vehicle,
    cpu_observation,
    normalization_audit,
)
from orbiter.train_generalized import (
    VehiclePolicy,
    complete_return_targets,
    cpu_parity,
    load_policy,
    save_policy,
    ppo_train,
    main,
)


def test_conditioned_observation_matches_independent_scalar_builder():
    batch = BatchedAscent(7, seed=47)
    assert batch.observation().shape == (7, OBSERVATIONS)
    assert torch.isfinite(batch.observation()).all()
    assert batch.initial_mass.max() / batch.initial_mass.min() > 3
    for index in range(7):
        reference = batch.cpu_reference(index)
        np.testing.assert_allclose(
            batch.observation()[index], cpu_observation(reference), atol=1e-6
        )
        assert assess_vehicle(reference.config)["accepted"]


@pytest.mark.parametrize("near_stage_boundary", [False, True])
def test_rk4_fuel_consumption_and_dry_mass_staging_match_scalar(near_stage_boundary):
    batch = BatchedAscent(3, seed=4)
    references = [batch.cpu_reference(index) for index in range(3)]
    if near_stage_boundary:
        batch.fuel[:] = 0.0123
        for reference in references:
            reference.fuel = 0.0123
    for _ in range(8):
        action = batch.teacher()
        batch.step(action)
        for index, reference in enumerate(references):
            reference.step(action[index].numpy())
            expected = [reference.x, reference.y, reference.vx, reference.vy]
            np.testing.assert_allclose(
                batch.state[index], expected, rtol=1e-12, atol=1e-7
            )
            assert float(batch.fuel[index]) == pytest.approx(reference.fuel, abs=1e-8)
            assert float(batch.mass()[index]) == pytest.approx(reference.mass, abs=1e-8)
            assert int(batch.stage[index]) == reference.stage
    if near_stage_boundary:
        assert batch.stage.tolist() == [1, 1, 1]


def test_partial_reset_preserves_other_worlds_and_changes_craft():
    batch = BatchedAscent(3, seed=81)
    state, stages = batch.state.clone(), batch.stages.clone()
    batch.reset(torch.tensor([False, True, False]))
    torch.testing.assert_close(batch.state[[0, 2]], state[[0, 2]])
    torch.testing.assert_close(batch.stages[[0, 2]], stages[[0, 2]])
    assert not torch.equal(batch.stages[1], stages[1])


def test_impossible_launch_screen_is_distinct_from_policy_failure():
    config = EnvConfig(
        stages=(Stage(1000, 10, 100, 300, 280),),
        target_altitude=100000,
        orbit_tolerance=10000,
    )
    result = assess_vehicle(config)
    assert not result["accepted"]
    assert result["screen_only"]
    assert "insufficient_pad_twr" in result["failures"]
    assert "insufficient_ideal_delta_v" in result["failures"]
    assert not result["live_ksp_validated"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"stages_min": 0},
        {"stages_min": 2.5},
        {"payload_min": float("nan")},
        {"initial_twr_min": 1},
        {"stage_dry_fraction_max": 1},
        {"stages_min": 1, "stages_max": 1},
    ],
)
def test_envelope_rejects_invalid_or_infeasible_ranges(overrides):
    with pytest.raises(ValueError):
        VehicleEnvelope(**overrides)


def test_terminal_crash_cancels_potential_instead_of_cashing_out_height():
    batch = BatchedAscent(1, seed=8)
    batch.state[0] = torch.tensor([batch.radius + 0.01, 0, -100, 0])
    batch.previous_potential[:] = batch.potential()
    previous = float(batch.previous_potential[0])
    _, reward, terminated, truncated, info = batch.step(torch.tensor([[-1.0, 1.0]]))
    assert terminated.item() and not truncated.item()
    assert not info["success"].item()
    assert reward.item() == pytest.approx(
        -previous - 30 - 0.003 * float(batch.time[0]), abs=1e-5
    )


def test_policy_contract_cannot_be_confused_with_old_checkpoints(tmp_path):
    model = VehiclePolicy(32)
    path = tmp_path / "candidate.pt"
    save_policy(
        path,
        model,
        {"observations": OBSERVATIONS, "feature_schema": FEATURE_SCHEMA},
        {"ppo_transitions": 0},
    )
    restored, record = load_policy(path)
    sample = BatchedAscent(2).observation()
    torch.testing.assert_close(model(sample), restored(sample))
    assert record["contract"]["observations"] == 54
    assert (
        '"live_ksp_validated": false' in path.with_suffix(".metadata.json").read_text()
    )


def test_cpu_parity_handles_early_policy_crashes_without_stepping_ended_reference():
    torch.manual_seed(8)
    result = cpu_parity(
        VehiclePolicy(16), envelope=VehicleEnvelope(), seed=4, count=2, steps=40
    )
    assert result["passed"]


def test_unseen_four_stage_family_is_separate_from_training_stage_counts():
    train = VehicleEnvelope(stages_min=2, stages_max=3)
    holdout = replace(train, stages_min=4, stages_max=4)
    assert set(BatchedAscent(12, seed=4, envelope=train).stage_count.tolist()) <= {2, 3}
    assert set(BatchedAscent(12, seed=4, envelope=holdout).stage_count.tolist()) == {4}


def test_high_attitude_rates_do_not_alias_through_observation_clipping():
    low = BatchedAscent(
        1,
        seed=101,
        envelope=VehicleEnvelope(attitude_rate_min=12, attitude_rate_max=12),
    )
    high = BatchedAscent(
        1,
        seed=101,
        envelope=VehicleEnvelope(attitude_rate_min=18, attitude_rate_max=18),
    )
    assert 0 < low.observation()[0, 50] < high.observation()[0, 50] < 10
    for batch in (low, high):
        np.testing.assert_allclose(
            batch.observation()[0], cpu_observation(batch.cpu_reference(0)), atol=1e-6
        )


def test_value_targets_retain_unknown_bootstrap_at_time_limits():
    rewards = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    terminal = torch.tensor([[False, False], [True, False], [False, False]])
    truncated = torch.tensor([[False, False], [False, True], [False, False]])
    returns, valid = complete_return_targets(rewards, terminal, truncated, 0.5)
    assert valid[:, 0].tolist() == [True, True, False]
    assert valid[:, 1].tolist() == [False, False, False]
    assert returns[0, 0] == 2
    assert returns[1, 0] == 2


def test_ppo_updates_use_finite_device_rollouts_and_time_limit_resets():
    model = VehiclePolicy(16)
    env = BatchedAscent(4, seed=5, max_time=4)
    before = [parameter.detach().clone() for parameter in model.critic.parameters()]
    events = []
    selected = []
    count = ppo_train(
        model,
        env,
        updates=1,
        horizon=4,
        batch_size=8,
        learning_rate=1e-4,
        log=lambda event, **values: events.append((event, values)),
        selected_callback=selected.append,
    )
    assert count == 16 and selected == [16]
    assert events[0][0] == "ppo"
    assert events[0][1]["completed_episodes"] == 8
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert any(
        not torch.equal(old, new) for old, new in zip(before, model.critic.parameters())
    )


def test_high_thrust_context_and_instantaneous_twr_do_not_clip_at_ten_g():
    envelope = VehicleEnvelope(
        stages_min=2,
        stages_max=3,
        initial_twr_min=2.5,
        initial_twr_max=2.5,
        sea_level_isp_ratio_min=0.15,
        sea_level_isp_ratio_max=0.15,
    )
    batch = BatchedAscent(1, seed=32, envelope=envelope)
    assert batch.stages[0, 0, 2] / batch.initial_mass[0] / 9.80665 > 10
    assert 0 < batch.context()[0, 3] < 10
    batch.state[0, 0] = batch.radius + 80000
    assert batch.available_thrust()[0] / batch.mass()[0] / 9.80665 > 10
    assert batch.observation()[0, 13] > np.log1p(10)
    assert batch.observation()[0, 13] < 10
    assert normalization_audit(envelope)["all_craft_and_actuator_features_below_clip"]


def test_all_declared_stock_context_bounds_are_audited_before_clipping():
    envelope = VehicleEnvelope(
        stages_min=2,
        stages_max=3,
        stage_wet_to_upper_max=6,
        stage_dry_fraction_max=0.3,
        initial_twr_min=1.15,
        initial_twr_max=2.5,
        upper_twr_min=0.6,
        upper_twr_max=3,
        sea_level_isp_ratio_min=0.15,
        sea_level_isp_ratio_max=1,
        isp_min=280,
        isp_max=380,
    )
    audit = normalization_audit(envelope)
    assert audit["all_craft_and_actuator_features_below_clip"]
    assert max(audit["upper_bounds"].values()) < 10
    batch = BatchedAscent(256, seed=917, envelope=envelope)
    assert torch.max(batch.context()) < 10


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware required")
def test_actual_cuda_rk4_matches_scalar_across_staging_and_gravity_turn():
    batch = BatchedAscent(4, seed=47, device="cuda:0")
    references = [batch.cpu_reference(index) for index in range(4)]
    batch.fuel[0] = 0.0123
    references[0].fuel = 0.0123
    finished = [False] * 4
    for _ in range(80):
        action = batch.teacher()
        expected_actions = action.cpu().numpy()
        batch.step(action)
        for index, reference in enumerate(references):
            if finished[index]:
                continue
            _, _, terminated, truncated, _ = reference.step(expected_actions[index])
            finished[index] = terminated or truncated
            expected = [reference.x, reference.y, reference.vx, reference.vy]
            np.testing.assert_allclose(
                batch.state[index].cpu(), expected, rtol=1e-11, atol=1e-6
            )
            np.testing.assert_allclose(
                batch.observation()[index].cpu(), cpu_observation(reference), atol=1e-5
            )
        for index, done in enumerate(finished):
            if done:
                batch.time[index] = batch.max_time
    assert int(batch.stage[0]) >= 1


def test_evaluation_rejects_a_different_nominal_curriculum_before_writing_contract(
    tmp_path,
):
    path = tmp_path / "stock.pt"
    save_policy(
        path,
        VehiclePolicy(16),
        {
            "observations": OBSERVATIONS,
            "feature_schema": FEATURE_SCHEMA,
            "curriculum": "stock",
        },
        {},
    )
    destination = tmp_path / "evaluation"
    with pytest.raises(SystemExit):
        main(
            ["--run-dir", str(destination), "--device", "cpu", "--evaluate", str(path)]
        )
    assert not (destination / "contract.json").exists()
