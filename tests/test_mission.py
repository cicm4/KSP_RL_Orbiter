from types import SimpleNamespace

import numpy as np
import pytest

from orbiter.mission import (
    capture_delta_v,
    plan_child_transfer,
    plan_interplanetary_window,
    require_delta_v,
)
from orbiter.orbital import propagate
from orbiter.system import SystemCatalog


def test_mun_candidate_has_safe_offset_and_reaches_aim_point():
    system = SystemCatalog.load()
    mu = system.bodies["Kerbin"]["gravitational_parameter"]
    r = np.array([680000.0, 0.0, 0.0])
    v = np.array([0.0, 0.0, np.sqrt(mu / 680000.0)])
    plan = plan_child_transfer(
        system,
        "Kerbin",
        "Mun",
        r,
        v,
        5000.0,
        target_altitude=50000.0,
        departure_samples=16,
        duration_samples=8,
    )
    assert 700 < plan.departure_delta_v < 1500
    assert plan.estimated_capture_delta_v > 100
    assert np.linalg.norm(plan.aim_offset) > plan.target_periapsis_radius
    arrival, _ = propagate(
        plan.departure_position,
        np.asarray(plan.departure_velocity) + plan.delta_v,
        plan.arrival_ut - plan.departure_ut,
        mu,
    )
    target, _ = system.state("Mun", plan.arrival_ut, relative_to="Kerbin")
    np.testing.assert_allclose(arrival, target + plan.aim_offset, atol=0.1)
    assert plan.to_dict()["actual_ksp_verified"] is False


def test_range_gate_refuses_current_used_upper_stage():
    estimate = SimpleNamespace(estimated_total_delta_v=1165.0)
    with pytest.raises(ValueError, match="Insufficient"):
        require_delta_v(estimate, mass=2364.256, dry_mass=1730.0, isp=345.0)
    assert (
        require_delta_v(estimate, mass=3300.0, dry_mass=1980.0, isp=345.0)[
            "remaining_margin"
        ]
        > 0
    )


def test_inclined_minmus_candidate_is_3d():
    system = SystemCatalog.load()
    mu = system.bodies["Kerbin"]["gravitational_parameter"]
    plan = plan_child_transfer(
        system,
        "Kerbin",
        "Minmus",
        [680000.0, 0.0, 0.0],
        [0.0, 0.0, np.sqrt(mu / 680000.0)],
        5000.0,
        target_altitude=50000.0,
        departure_samples=12,
        duration_samples=6,
    )
    assert abs(plan.delta_v[1]) > 1
    assert plan.route == ["Kerbin", "Minmus"]


def test_duna_window_is_explicitly_an_estimate_not_an_executable_node():
    plan = plan_interplanetary_window(
        SystemCatalog.load(),
        "Kerbin",
        "Duna",
        5000.0,
        departure_samples=12,
        duration_samples=6,
    )
    assert plan["kind"] == "sibling_window_estimate"
    assert not plan["executable"] and not plan["actual_ksp_verified"]
    assert plan["route"] == ["Kerbin", "Sun", "Duna"]
    assert 500 < plan["estimated_departure_delta_v"] < 5000


def test_invalid_body_relationship_cannot_be_executed_as_child_transfer():
    with pytest.raises(ValueError, match="Direct-child"):
        plan_child_transfer(
            SystemCatalog.load(),
            "Kerbin",
            "Duna",
            [680000.0, 0, 0],
            [0, 0, 2200],
            5000.0,
        )
    with pytest.raises(ValueError):
        capture_delta_v(1, 1, -1)


@pytest.mark.parametrize("total", [float("nan"), float("inf"), -float("inf"), -1.0])
def test_budget_rejects_invalid_plan_cost_instead_of_passing_comparison(total):
    with pytest.raises(ValueError, match="estimated.*finite.*nonnegative"):
        require_delta_v(
            SimpleNamespace(estimated_total_delta_v=total),
            mass=3300.0,
            dry_mass=1980.0,
            isp=345.0,
        )


def test_zero_delta_v_and_zero_reserve_remain_valid():
    budget = require_delta_v(
        SimpleNamespace(estimated_total_delta_v=0.0),
        mass=1980.0,
        dry_mass=1980.0,
        isp=345.0,
        reserve=0.0,
    )
    assert budget["estimated_required_delta_v"] == budget["remaining_margin"] == 0.0


@pytest.mark.parametrize(
    "target_altitude",
    [-50000.0, 0.0, 50000.0, 51000.0, float("nan"), float("inf"), -float("inf"), 1e10],
)
def test_interplanetary_window_rejects_unsafe_destination_before_search(
    monkeypatch, target_altitude
):
    system = SystemCatalog.load()
    monkeypatch.setattr(
        system,
        "state",
        lambda *args, **kwargs: pytest.fail(
            "Unsafe goal must be rejected before ephemeris search"
        ),
    )
    with pytest.raises(ValueError, match="Target orbit"):
        plan_interplanetary_window(
            system,
            "Kerbin",
            "Duna",
            5000.0,
            target_altitude=target_altitude,
            departure_samples=2,
            duration_samples=2,
        )


@pytest.mark.parametrize(
    "parking_altitude",
    [-1.0, 0.0, 70000.0, 71000.0, float("nan"), float("inf"), -float("inf"), 1e10],
)
def test_interplanetary_window_rejects_unsafe_source_parking_before_search(
    monkeypatch, parking_altitude
):
    system = SystemCatalog.load()
    monkeypatch.setattr(
        system,
        "state",
        lambda *args, **kwargs: pytest.fail(
            "Unsafe parking must be rejected before ephemeris search"
        ),
    )
    with pytest.raises(ValueError, match="Departure parking orbit"):
        plan_interplanetary_window(
            system,
            "Kerbin",
            "Duna",
            5000.0,
            parking_altitude=parking_altitude,
            departure_samples=2,
            duration_samples=2,
        )
