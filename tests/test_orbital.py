"""Frame-independent coast/transfer mathematics, not a game-success test."""

import math

import numpy as np
import pytest

from orbiter.orbital import elements, lambert, propagate, rocket_delta_v


def test_circular_propagation_and_reversal():
    mu, radius = 3.5316e12, 680000.0
    p, v = np.array([radius, 0.0, 0.0]), np.array([0.0, 0.0, math.sqrt(mu / radius)])
    period = 2 * math.pi * math.sqrt(radius**3 / mu)
    quarter, speed = propagate(p, v, period / 4, mu)
    np.testing.assert_allclose(quarter, [0.0, 0.0, radius], atol=1e-5)
    np.testing.assert_allclose(speed, [-math.sqrt(mu / radius), 0.0, 0.0], atol=1e-7)
    back, _ = propagate(quarter, speed, -period / 4, mu)
    np.testing.assert_allclose(back, p, atol=1e-5)


@pytest.mark.parametrize("long_way", [False, True])
def test_lambert_solution_reaches_3d_endpoint_under_independent_propagator(long_way):
    mu = 3.986004418e14
    start = np.array([5000.0, 10000.0, 2100.0]) * 1000
    finish = np.array([-14600.0, 2500.0, 7000.0]) * 1000
    departure, arrival = lambert(start, finish, 3600.0, mu, long_way=long_way)
    predicted, predicted_velocity = propagate(start, departure, 3600.0, mu)
    np.testing.assert_allclose(predicted, finish, atol=0.01)
    np.testing.assert_allclose(predicted_velocity, arrival, atol=1e-5)


def test_lambert_known_vallado_example():
    departure, arrival = lambert(
        np.array([5000.0, 10000.0, 2100.0]) * 1000,
        np.array([-14600.0, 2500.0, 7000.0]) * 1000,
        3600.0,
        3.986004418e14,
    )
    np.testing.assert_allclose(departure / 1000, [-5.9925, 1.9254, 3.2456], atol=0.0001)
    np.testing.assert_allclose(
        arrival / 1000, [-3.3125, -4.1966, -0.38529], atol=0.0001
    )


def test_hyperbolic_coast_preserves_invariants():
    mu = 6.513839752e10
    p, v = np.array([230000.0, 0.0, 0.0]), np.array([0.0, 1000.0, 100.0])
    initial = elements(p, v, mu)
    final_position, final_velocity = propagate(p, v, 12000.0, mu)
    final = elements(final_position, final_velocity, mu)
    assert final["energy"] == pytest.approx(initial["energy"], rel=1e-10)
    assert final["periapsis"] == pytest.approx(initial["periapsis"], rel=1e-10)
    assert final["eccentricity"] > 1


def test_common_orthogonal_frame_change_preserves_transfer():
    p, q, mu = np.array([680000.0, 0.0, 0.0]), np.array([0.0, 1e7, 3e6]), 3.5316e12
    first = lambert(p, q, 16000.0, mu)
    # A reflection changes handedness; algebraic propagation must still agree.
    reflection = np.diag([1.0, -1.0, 1.0])
    second = lambert(reflection @ p, reflection @ q, 16000.0, mu)
    for left, right in zip(first, second):
        np.testing.assert_allclose(reflection @ left, right, atol=1e-8)


def test_delta_v_is_finite_capacity_not_an_infinite_control_budget():
    assert rocket_delta_v(2364.256, 1730.0, 345.0) == pytest.approx(1056.74, abs=0.1)
    assert rocket_delta_v(1730.0, 1730.0, 345.0) == 0
    with pytest.raises(ValueError):
        rocket_delta_v(100, 200, 345)


def test_singular_lambert_and_invalid_inputs_fail():
    with pytest.raises(ValueError, match="singular"):
        lambert([1, 0, 0], [-1, 0, 0], 10, 1)
    with pytest.raises(ValueError):
        propagate([0, 0, 0], [1, 2, 3], 10, 1)
