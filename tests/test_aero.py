"""Force direction, units, and invariance for optional planar lift/induced drag.

These are physics-contract checks; measured KSP fit quality is evaluated against
flight captures separately, not established by synthetic unit-test forces.
"""
from dataclasses import replace
import math

import numpy as np
import pytest

from orbiter.env import EnvConfig, HeadlessOrbiterEnv


def environment(**changes):
    config = EnvConfig(randomization=0, atmosphere=((0., 1., 101325.), (70000., 1., 101325.)),
                       **changes)
    return HeadlessOrbiterEnv(config)


@pytest.mark.parametrize("alpha", [-20., 0., 20.])
def test_effective_area_force_units_and_signed_lift(alpha):
    env = environment(drag_area=.8, drag_alpha_area=6.7, lift_area=4.5)
    radius, speed, mass = env.config.radius + 10000., 100., 1000.
    omega = 2 * math.pi / env.config.rotation_period
    derivative = env._derivative(np.array([radius, 0., 0., omega * radius + speed]), mass, alpha, 0.)
    aerodynamic_acceleration = derivative[2:] + [env.config.mu / radius**2, 0.]
    q = .5 * speed**2  # density=1 kg/m^3
    assert aerodynamic_acceleration[0] == pytest.approx(q * 4.5 * math.sin(math.radians(2 * alpha)) / mass)
    assert aerodynamic_acceleration[1] == pytest.approx(-q * (.8 + 6.7 * math.sin(math.radians(alpha))**2) / mass)


def test_lift_is_perpendicular_and_cannot_add_air_relative_kinetic_energy():
    bare, lifted = environment(drag_area=0.), environment(drag_area=0., lift_area=5.)
    radius = bare.config.radius + 10000.
    air_velocity = np.array([70., 90.])
    omega = 2 * math.pi / bare.config.rotation_period
    state = np.array([radius, 0., air_velocity[0], omega * radius + air_velocity[1]])
    pitch = math.degrees(math.atan2(*air_velocity)) + 15.
    difference = lifted._derivative(state, 1000., pitch, 0.)[2:] - bare._derivative(state, 1000., pitch, 0.)[2:]
    assert np.linalg.norm(difference) > 1.
    assert float(difference @ air_velocity) == pytest.approx(0., abs=1e-10)
    assert difference[0] > 0 and difference[1] < 0


def test_drag_always_removes_air_relative_kinetic_energy():
    env = environment(drag_area=.8, drag_alpha_area=6.7)
    radius, mass = env.config.radius + 10000., 1000.
    omega = 2 * math.pi / env.config.rotation_period
    air_velocity = np.array([70., -90.])
    state = np.array([radius, 0., air_velocity[0], omega * radius + air_velocity[1]])
    acceleration = env._derivative(state, mass, -10., 0.)[2:] + [env.config.mu / radius**2, 0.]
    assert float(acceleration @ air_velocity) < 0


def test_corotating_air_produces_no_drag_or_lift():
    env = environment(drag_area=.8, drag_alpha_area=6.7, lift_area=4.5)
    radius = env.config.radius + 10000.
    state = np.array([radius, 0., 0., 2 * math.pi / env.config.rotation_period * radius])
    derivative = env._derivative(state, 1000., 20., 0.)
    np.testing.assert_allclose(derivative[2:], [-env.config.mu / radius**2, 0.], atol=1e-12)
    assert np.isfinite(derivative).all()


def test_force_law_rotates_with_the_local_orbital_frame():
    env = environment(drag_area=.8, drag_alpha_area=6.7, lift_area=4.5)
    radius, angle = env.config.radius + 10000., .73
    state = np.array([radius, 0., 70., 300.])
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    rotated_state = np.concatenate([rotation @ state[:2], rotation @ state[2:]])
    first = env._derivative(state, 1000., 35., 10000.)
    second = env._derivative(rotated_state, 1000., 35., 10000.)
    np.testing.assert_allclose(second[:2], rotation @ first[:2], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(second[2:], rotation @ first[2:], rtol=1e-12, atol=1e-12)


def test_aerodynamic_coefficients_have_no_effect_in_vacuum():
    base = environment(atmosphere_depth=0)
    changed = HeadlessOrbiterEnv(replace(base.config, drag_alpha_area=100., lift_area=100.))
    state = np.array([base.config.radius + 100000., 0., 400., 2000.])
    np.testing.assert_array_equal(base._derivative(state, 1000., 10., 5000.),
                                  changed._derivative(state, 1000., 10., 5000.))


@pytest.mark.parametrize("field", ["drag_alpha_area", "lift_area"])
@pytest.mark.parametrize("value", [-1., math.inf, math.nan])
def test_invalid_aerodynamic_areas_are_rejected(field, value):
    with pytest.raises(ValueError, match="aerodynamic areas"):
        EnvConfig(**{field: value})


def test_optional_coefficients_roundtrip_and_old_profiles_keep_zero_defaults(tmp_path):
    config = EnvConfig(drag_alpha_area=6.7, lift_area=4.5)
    target = tmp_path / "aero.json"
    config.save(target)
    assert EnvConfig.load(target) == config
    import json
    data = json.loads(target.read_text())
    del data["drag_alpha_area"], data["lift_area"]
    old = EnvConfig.from_dict(data)
    assert old.drag_alpha_area == old.lift_area == 0.
