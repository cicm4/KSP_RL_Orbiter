from dataclasses import replace
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from orbiter.ascent_planner import (
    DEFAULT,
    LOWER,
    PARAMETER_NAMES,
    UPPER,
    SearchBudget,
    config_fingerprint,
    generate_bank,
    guidance_action,
    plan_ascent,
    qualify_witness,
    remaining_burn_time,
)
from orbiter.env import EnvConfig, G0, Stage
from orbiter.generalized_aero_env import BatchedAeroAscent


@pytest.fixture(scope="module", autouse=True)
def small_tensor_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def solved_bank():
    two = replace(
        EnvConfig(), randomization=0, target_altitude=100000,
        orbit_tolerance=20000, max_time=1600, physics_dt=1,
    )
    # Same total propellant and first-stage mass as the two-stage craft, but
    # the short middle burn forces the successful witness across both stage
    # boundaries. No stored trajectory or specially selected plan is supplied.
    three = replace(two, stages=(
        two.stages[0], Stage(600, 600, 120000, 345, 85),
        Stage(400, 5400, 120000, 345, 85),
    ))
    configs = [two, three]
    budget = SearchBudget(population=8, iterations=2, seed=91)
    return configs, budget, plan_ascent(configs, budget=budget)


def test_solved_two_and_three_stage_witnesses_independently_coast_one_orbit(solved_bank):
    configs, _, report = solved_bank
    assert report["baseline_successes"] == 0
    assert report["qualified_successes"] == 2
    assert [row["stage"] for row in report["results"]] == [1, 2]
    for config, row in zip(configs, report["results"]):
        assert row["success"] and row["feasible"] and row["qualified"]
        assert row["selected_plan"] == "stage_aware_profile"
        assert row["reachability"] == "feasible_independent_scalar_witness"
        assert row["config_sha256"] == config_fingerprint(config)
        q = row["qualification"]
        assert q["scalar_success"] and q["actions_consumed"] == len(row["actions"])
        assert q["stage"] == row["stage"]
        assert q["search_state_max_abs_difference"] < 1e-6
        coast = q["coast"]
        assert coast["success"]
        assert coast["duration_s"] == pytest.approx(coast["requested_period_s"])
        assert coast["min_periapsis_m"] >= 80000 - .01
        assert coast["max_apoapsis_m"] <= 120000 + .01
        assert coast["return_position_error_m"] < .1
        assert coast["relative_energy_drift"] < 1e-8
        assert q["proof_scope"].endswith("not actual KSP")


def test_bounded_plans_repeat_exactly_and_search_incumbent_never_regresses(solved_bank):
    configs, budget, report = solved_bank
    repeated = plan_ascent(configs, budget=budget)
    np.testing.assert_array_equal(report["incumbent_cost_history"], repeated["incumbent_cost_history"])
    history = np.asarray(report["incumbent_cost_history"])
    assert history.shape == (budget.iterations + 1, len(configs))
    assert (np.diff(history, axis=0) <= 0).all()
    for first, second in zip(report["results"], repeated["results"]):
        assert first["parameters"] == second["parameters"]
        assert first["actions"] == second["actions"]
        assert first["cost"] <= first["baseline"]["cost"]
        actions = np.asarray(first["actions"])
        assert np.isfinite(actions).all() and (abs(actions) <= 1).all()
        for name, low, high in zip(PARAMETER_NAMES, LOWER, UPPER):
            assert low <= first["parameters"][name] <= high
    assert report["selection_bank_only"]


def test_existing_successful_heuristic_is_kept_as_an_incumbent():
    sampled = BatchedAeroAscent(12, seed=818, physics_dt=1)
    config = sampled.export_config(3)
    report = plan_ascent([config], budget=SearchBudget(population=3, iterations=1, seed=18))
    row = report["results"][0]
    assert row["baseline"]["qualified"]
    assert row["qualified"]
    assert row["cost"] <= row["baseline"]["cost"]


def test_weak_vehicle_failure_does_not_claim_infeasibility():
    stage = Stage(1000, 2000, 100, 320, 280)
    config = replace(EnvConfig(), stages=(stage, stage), randomization=0,
                     target_altitude=100000, max_time=1600, physics_dt=1)
    report = plan_ascent([config], budget=SearchBudget(population=3, iterations=1))
    row = report["results"][0]
    assert not row["success"] and not row["qualified"]
    assert row["termination_reason"] == "impact"
    assert row["reachability"] == "unknown_search_failed"
    assert row["qualification"] is None
    assert math.isfinite(row["cost"])


def test_scalar_qualification_rejects_changed_actions_and_real_pressure_limit(solved_bank):
    configs, _, report = solved_bank
    config, row = configs[0], report["results"][0]
    failed = qualify_witness(config, [[-1, 1]])
    assert not failed["qualified"] and not failed["scalar_success"]
    pressure = qualify_witness(config, row["actions"], max_dynamic_pressure=1)
    assert pressure["scalar_success"]
    assert isinstance(pressure["pressure_constraint_satisfied"], bool)
    assert not pressure["qualified"] and not pressure["pressure_constraint_satisfied"]
    assert pressure["coast"] is None


def test_stage_aware_burn_estimate_crosses_boundaries_and_reports_missing_delta_v():
    env = BatchedAeroAscent(1, seed=16)
    stages = (Stage(100, 100, 10000, 300, 250), Stage(80, 300, 5000, 350, 200))
    config = replace(EnvConfig(), payload_mass=100, stages=stages, randomization=0,
                     target_altitude=100000, max_time=1600, physics_dt=1)
    env.load_configs([config])
    initial_mass = 680
    first_dv = stages[0].isp_vac * G0 * math.log(initial_mass / (initial_mass - 100))
    desired = first_dv + 200
    time, missing = remaining_burn_time(env, torch.tensor([desired], dtype=torch.float64))
    second_mass = initial_mass - stages[0].fuel_mass - stages[0].dry_mass
    second_fuel = second_mass * (1 - math.exp(-200 / (stages[1].isp_vac * G0)))
    expected = 100 * stages[0].isp_vac * G0 / stages[0].thrust_vac + second_fuel * stages[1].isp_vac * G0 / stages[1].thrust_vac
    assert float(time[0]) == pytest.approx(expected)
    assert float(missing[0]) == pytest.approx(0, abs=1e-9)
    _, missing = remaining_burn_time(env, torch.tensor([1e6]))
    assert float(missing[0]) > 900000
    # After staging, discarded dry mass and consumed fuel must not reappear.
    env.stage.fill_(1)
    env.fuel.fill_(150)
    time, missing = remaining_burn_time(env, torch.tensor([200.], dtype=torch.float64))
    fuel = (100 + 80 + 150) * (1 - math.exp(-200 / (350 * G0)))
    assert float(time[0]) == pytest.approx(fuel * 350 * G0 / 5000)
    assert float(missing[0]) == pytest.approx(0, abs=1e-9)


def test_guidance_bounds_and_bad_parameters():
    env = BatchedAeroAscent(6, seed=7)
    parameters = torch.tensor([DEFAULT] * 6)
    for _ in range(3):
        actions = guidance_action(env, parameters)
        assert actions.shape == (6, 2)
        assert bool(torch.isfinite(actions).all() & (actions.abs() <= 1).all())
        env.step(actions)
    with pytest.raises(ValueError, match="parameters"):
        guidance_action(env, parameters[:, :3])
    parameters[0, 0] = float("nan")
    with pytest.raises(ValueError, match="parameters"):
        guidance_action(env, parameters)


@pytest.mark.parametrize("kwargs", [
    {"population": 2}, {"population": True}, {"iterations": 0},
    {"seed": -1}, {"elite_fraction": 0}, {"elite_fraction": True},
    {"elite_fraction": float("nan")}, {"physics_dt": True},
    {"physics_dt": 0}, {"physics_dt": 3}, {"max_dynamic_pressure": float("inf")},
    {"max_dynamic_pressure": True},
])
def test_invalid_budget(kwargs):
    with pytest.raises(ValueError):
        SearchBudget(**kwargs)


def test_frozen_configs_only_and_no_qualification_claim_when_skipped():
    with pytest.raises(ValueError, match="at least one"):
        plan_ascent([])
    with pytest.raises(ValueError, match="explicit"):
        plan_ascent([EnvConfig()])
    stage = Stage(1000, 2000, 100, 320, 280)
    config = replace(EnvConfig(), stages=(stage, stage), randomization=0, physics_dt=1)
    result = plan_ascent([config], qualify=False, budget=SearchBudget(population=3, iterations=1))
    assert not result["qualification_requested"]
    assert result["baseline_qualified_successes"] is None
    assert result["qualified_successes"] == 0


@pytest.mark.parametrize("curriculum", ["efficient", "stock"])
def test_generated_bank_uses_exact_trainer_aero_envelope(curriculum):
    from dataclasses import asdict
    from orbiter.train_generalized import experiment_contract

    configs, contract = generate_bank(8, seed=83, curriculum=curriculum)
    envelope, _, expected = experiment_contract(curriculum, "aero")
    assert contract == expected
    assert contract["train_envelope"] == asdict(envelope)
    reference = BatchedAeroAscent(8, seed=83, envelope=envelope)
    assert [config_fingerprint(c) for c in configs] == [
        config_fingerprint(reference.export_config(i)) for i in range(8)
    ]
    if curriculum == "stock":
        assert envelope.stage_wet_to_upper_max == 6
        assert envelope.stage_dry_fraction_max == .3
        assert envelope.sea_level_isp_ratio_min == .2


def test_cli_defaults_to_stock_and_records_the_generation_contract(monkeypatch, tmp_path):
    import json
    import orbiter.ascent_planner as planner

    captured = []

    def capture_bank(configs, **kwargs):
        captured.extend(configs)
        return dict(craft_count=len(configs), baseline_successes=0,
                    search_successes=0, qualified_successes=0,
                    candidate_rollouts=0, elapsed_s=0.)

    monkeypatch.setattr(planner, "plan_ascent", capture_bank)
    output = tmp_path / "report.json"
    planner.main(["--count", "2", "--seed", "83", "--output", str(output)])
    report = json.loads(output.read_text())
    assert report["generation_contract"]["curriculum"] == "stock"
    assert report["generation_contract"]["train_envelope"]["stage_wet_to_upper_max"] == 6
    expected, _ = generate_bank(2, seed=83, curriculum="stock")
    assert [config_fingerprint(c) for c in captured] == [config_fingerprint(c) for c in expected]
