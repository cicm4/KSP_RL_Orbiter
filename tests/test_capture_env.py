"""Physics and task checks for local arrival/capture, not Kerbin transfer proof."""

from dataclasses import asdict
import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from orbiter.capture_env import (
    CaptureConfig,
    CaptureEnv,
    G0,
    build_capture_observation,
    control_acceleration,
    decode_capture_action,
    rtn_basis,
)
from orbiter.system import SystemCatalog


def fixed_config(**kwargs):
    return CaptureConfig(randomization=0.0, target_altitude_randomization=0.0, **kwargs)


def circular_options(config, radius=None):
    radius = radius or config.goal_radius
    return dict(
        r=[radius, 0.0, 0.0],
        v=[0.0, math.sqrt(config.mu / radius), 0.0],
        goal_normal=[0.0, 0.0, 1.0],
    )


def test_gymnasium_contract_and_reproducible_inbound_reset():
    env = CaptureEnv()
    check_env(env, skip_render_check=True)
    observation, info = env.reset(seed=21)
    first = env.telemetry()
    repeated, _ = env.reset(seed=21)
    np.testing.assert_array_equal(observation, repeated)
    assert observation.shape == (27,) and observation.dtype == np.float32
    assert env.action_space.shape == (3,) and env.observation_space.contains(
        observation
    )
    assert first["energy"] > 0 and first["r"] @ first["v"] < 0
    assert (
        info["task"] == "orbital_capture_v2" and info["arrival_already_at_destination"]
    )


@pytest.mark.parametrize(
    "field",
    [
        "mu",
        "radius",
        "thrust",
        "isp",
        "control_dt",
        "physics_dt",
        "max_time",
        "sphere_of_influence",
        "gamma",
        "acceleration_gravity_limit",
        "target_altitude_randomization",
    ],
)
@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_config_rejects_nonfinite_parameters(field, value):
    with pytest.raises(ValueError):
        CaptureConfig(**{field: value})


def test_config_target_safety_and_roundtrip(tmp_path):
    with pytest.raises(ValueError, match="clearance"):
        CaptureConfig(target_altitude=4000.0)
    with pytest.raises(ValueError, match="sphere of influence"):
        CaptureConfig(target_altitude=3_000_000.0)
    with pytest.raises(ValueError, match="mass"):
        CaptureConfig(mass=1000.0)
    config = fixed_config()
    path = tmp_path / "capture.json"
    config.save(path)
    assert asdict(CaptureConfig.load(path)) == asdict(config)


def test_actual_catalog_import_uses_si_and_rejects_star():
    bodies = SystemCatalog.load().bodies
    config = CaptureConfig.for_body(bodies["Kerbin"])
    assert config.mu == pytest.approx(3.5316e12)
    assert config.radius == 600000.0 and config.atmosphere_depth == 70000.0
    assert config.target_altitude == 90000.0
    with pytest.raises(ValueError, match="sphere_of_influence"):
        CaptureConfig.for_body(bodies["Sun"])


def test_randomized_goals_stay_safe_and_live_overrides_are_exact():
    bodies = SystemCatalog.load().bodies
    env = CaptureEnv(CaptureConfig(), body_catalog={"Eve": bodies["Eve"]})
    goals = []
    for seed in range(20):
        env.reset(seed=seed)
        goals.append(env.config.target_altitude)
        assert (
            env.config.goal_radius - env.config.orbit_tolerance > env.config.safe_radius
        )
        assert (
            env.config.goal_radius + env.config.orbit_tolerance
            < env.config.sphere_of_influence
        )
    assert len(set(goals)) == 20
    env.reset(
        seed=8,
        options=dict(
            target_altitude=140000.0,
            mass=3200.0,
            dry_mass=1980.0,
            thrust=60000.0,
            isp=345.0,
        ),
    )
    assert env.config.target_altitude == 140000.0
    assert (env.mass, env.initial_mass, env.dry_mass, env.thrust, env.isp) == (
        3200.0,
        3200.0,
        1980.0,
        60000.0,
        345.0,
    )
    with pytest.raises(ValueError):
        env.reset(options={"mass": 2000.0, "dry_mass": 2500.0})


def test_per_body_arrival_fractions_preserve_fallback_and_config_map():
    bodies = SystemCatalog.load().bodies
    original = {"Mun": 0.7, "Minmus": 1.9}
    config = fixed_config(arrival_vinf_by_body=original)
    env = CaptureEnv(config, {name: bodies[name] for name in ("Mun", "Minmus", "Ike")})
    for name, fraction in (("Mun", 0.7), ("Minmus", 1.9), ("Ike", 0.4), ("Mun", 0.7)):
        env.reset(seed=12, options={"body": name})
        actual = (
            math.sqrt(max(0.0, 2 * env.telemetry()["energy"]))
            / env.config.velocity_scale
        )
        assert actual == pytest.approx(fraction, abs=1e-12)
        assert env.config.arrival_vinf_by_body == original
    assert config.arrival_vinf_by_body == original
    original["Mun"] = 3.0
    assert config.arrival_vinf_by_body["Mun"] == 0.7


@pytest.mark.parametrize(
    "mapping",
    [
        [],
        {"": 0.7},
        {" Mun": 0.7},
        {1: 0.7},
        {"Mun": math.nan},
        {"Mun": math.inf},
        {"Mun": -0.1},
        {"Mun": True},
        {"Mun": "0.7"},
    ],
)
def test_invalid_per_body_arrival_map_is_rejected(mapping):
    with pytest.raises(ValueError, match="arrival_vinf_by_body"):
        CaptureConfig(arrival_vinf_by_body=mapping)


def test_config_missing_optional_arrival_map_preserves_old_v2_distribution():
    old = asdict(fixed_config())
    old.pop("arrival_vinf_by_body")
    expanded = CaptureConfig.from_dict(old)
    assert expanded.arrival_vinf_by_body == {}
    env = CaptureEnv(expanded)
    env.reset(seed=11)
    actual = math.sqrt(2 * env.telemetry()["energy"]) / expanded.velocity_scale
    assert actual == pytest.approx(expanded.arrival_vinf_fraction)


def test_observation_is_rotation_invariant_and_does_not_mutate_live_telemetry():
    env = CaptureEnv(fixed_config())
    data = env.telemetry()
    data["goal_normal"] *= 2
    data["direction"] *= 2
    original = {key: data[key].copy() for key in ("r", "v", "direction", "goal_normal")}
    observation = build_capture_observation(env.config, data)
    for key, value in original.items():
        np.testing.assert_array_equal(data[key], value)
    rotation, _ = np.linalg.qr(np.random.default_rng(2).normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 2] *= -1
    rotated = dict(data)
    for key, value in original.items():
        rotated[key] = rotation @ value
    np.testing.assert_allclose(
        build_capture_observation(env.config, rotated), observation, atol=1e-6
    )


def test_rtn_basis_is_right_handed_and_transverse_not_full_prograde():
    basis = rtn_basis(np.array([1.0, 0.0, 0.0]), np.array([-1.0, 2.0, 0.0]))
    np.testing.assert_allclose(basis, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(
        np.cross(basis[:, 0], basis[:, 1]), basis[:, 2], atol=1e-12
    )


def test_zero_action_coast_conserves_energy_momentum_and_mass_for_three_orbits():
    config = fixed_config(control_dt=30.0, physics_dt=2.0, max_time=25000.0)
    env = CaptureEnv(config)
    radius = config.goal_radius + 50000.0
    env.reset(seed=0, options=circular_options(config, radius))
    before = env.telemetry()
    period = 2 * math.pi * math.sqrt(radius**3 / config.mu)
    for _ in range(math.ceil(3 * period / config.control_dt)):
        observation, reward, terminated, truncated, _ = env.step(np.zeros(3))
        assert not terminated and not truncated
        assert np.isfinite(observation).all() and math.isfinite(reward)
    after = env.telemetry()
    assert after["energy"] == pytest.approx(before["energy"], rel=1e-9)
    assert after["momentum"] == pytest.approx(before["momentum"], rel=1e-9)
    assert env.mass == before["mass"]


def test_throttle_is_action_norm_with_exact_fuel_mass_flow():
    config = fixed_config(control_dt=0.25, physics_dt=0.25)
    env = CaptureEnv(config)
    mass = env.mass
    env.step(np.array([0.3, 0.4, 0.0]))
    assert mass - env.mass == pytest.approx(
        0.5 * config.thrust / (config.isp * G0) * config.control_dt, rel=1e-10
    )


def test_small_body_action_decodes_to_gravity_limited_physical_throttle_and_fuel():
    body = SystemCatalog.load().bodies["Gilly"]
    config = CaptureConfig.for_body(
        body,
        randomization=0.0,
        target_altitude_randomization=0.0,
        control_dt=0.25,
        physics_dt=0.25,
    )
    env = CaptureEnv(config)
    data = env.telemetry()
    expected_acceleration = (
        config.acceleration_gravity_limit * config.mu / config.goal_radius**2
    )
    assert expected_acceleration < data["physical_max_acceleration"] / 100
    assert data["control_acceleration"] == pytest.approx(expected_acceleration)
    throttle, direction = decode_capture_action(config, data, [0.3, 0.4, 0.0])
    np.testing.assert_allclose(direction, [0.6, 0.8, 0.0], atol=1e-12)
    assert throttle == pytest.approx(
        0.5 * expected_acceleration / data["physical_max_acceleration"]
    )
    assert throttle * env.thrust / env.mass == pytest.approx(
        0.5 * expected_acceleration
    )
    mass = env.mass
    env.step([0.3, 0.4, 0.0])
    assert mass - env.mass == pytest.approx(
        throttle * env.thrust / (env.isp * G0) * config.control_dt, rel=1e-8
    )


def test_action_saturation_preserves_direction_and_exhausted_engine_cannot_thrust():
    config = fixed_config()
    env = CaptureEnv(config)
    data = env.telemetry()
    throttle, direction = decode_capture_action(config, data, [2.0, 1.0, 0.0])
    np.testing.assert_allclose(direction, np.array([2.0, 1.0, 0.0]) / math.sqrt(5.0))
    assert throttle == pytest.approx(1.0)  # Mun cap exceeds physical acceleration.
    zero, direction = decode_capture_action(config, data, np.zeros(3))
    assert zero == 0 and not direction.any()
    data["mass"] = data["dry_mass"]
    assert control_acceleration(config, data) == 0.0
    throttle, _ = decode_capture_action(config, data, [0.0, 1.0, 0.0])
    assert throttle == 0.0
    with pytest.raises(ValueError):
        decode_capture_action(config, data, [math.nan, 0.0, 0.0])


def test_antipodal_attitude_command_turns_at_bounded_angular_rate():
    config = fixed_config(control_dt=0.25, physics_dt=0.25, pointing_rate_deg_s=30.0)
    env = CaptureEnv(config)
    target = rtn_basis(env.r, env.v)[:, 1]
    env.direction = -target.copy()
    before = env.direction.copy()
    env.step([0.0, 1.0, 0.0])
    turn = math.degrees(math.acos(float(np.clip(before @ env.direction, -1.0, 1.0))))
    assert turn == pytest.approx(7.5, abs=1e-10)
    assert np.linalg.norm(env.direction) == pytest.approx(1.0)


def test_capture_success_needs_safe_apses_goal_plane_and_multiple_steps():
    config = fixed_config()
    env = CaptureEnv(config)
    env.reset(seed=0, options=circular_options(config))
    assert env.telemetry()["orbit_in_band"]
    _, _, terminated, _, _ = env.step([0.0, 0.0, 0.0])
    assert not terminated
    _, _, terminated, truncated, info = env.step([0.0, 0.0, 0.0])
    assert terminated and not truncated and info["success"]
    assert info["termination_reason"] == "orbit_captured"
    with pytest.raises(RuntimeError, match="reset"):
        env.step([0.0, 0.0, 0.0])
    wrong_plane = circular_options(config)
    wrong_plane["goal_normal"] = [0.0, 0.0, -1.0]
    env.reset(options=wrong_plane)
    assert not env.telemetry()["orbit_in_band"]


def test_terminal_potential_is_cancelled_but_time_limit_retains_bootstrap_potential():
    env = CaptureEnv(fixed_config())
    env.reset(options=circular_options(env.config))
    # A nonzero potential occurs for a valid, slightly elliptical target orbit.
    env.v *= 0.999
    env._potential_prev = env.potential()
    initial = env._potential_prev
    discounted = 0.0
    for index in range(3):
        _, _, terminated, truncated, info = env.step(np.zeros(3))
        discounted += env.config.gamma**index * info["reward_shaping"]
        if terminated or truncated:
            break
    assert terminated and not truncated and info["success"]
    assert discounted == pytest.approx(-initial, abs=1e-12)
    env = CaptureEnv(fixed_config(max_time=0.1))
    initial = env.potential()
    _, _, terminated, truncated, info = env.step(np.zeros(3))
    assert truncated and not terminated
    assert info["reward_shaping"] == pytest.approx(
        env.config.gamma * env.potential() - initial
    )


def test_physically_insufficient_fuel_is_failure_not_partial_capture_success():
    config = fixed_config(mass=3200.0, dry_mass=3199.0)
    env = CaptureEnv(config)
    initial = env.telemetry()
    assert initial["available_delta_v"] < 2.0
    for _ in range(20):
        _, _, terminated, truncated, info = env.step(env.teacher_action())
        if terminated or truncated:
            break
    assert terminated and not truncated and not info["success"]
    assert info["termination_reason"] == "out_of_fuel"
    assert env.mass == pytest.approx(env.dry_mass)
    assert info["energy"] > 0


@pytest.mark.parametrize(
    "name",
    [
        name
        for name, row in SystemCatalog.load().bodies.items()
        if row["parent"] is not None
    ],
)
def test_teacher_captures_randomized_incoming_state_on_each_catalog_body(name):
    bodies = SystemCatalog.load().bodies
    env = CaptureEnv(CaptureConfig(max_time=1800.0), body_catalog={name: bodies[name]})
    env.reset(seed=100000)
    assert env.telemetry()["energy"] > 0
    for _ in range(math.ceil(env.config.max_time / env.config.control_dt) + 1):
        observation, reward, terminated, truncated, info = env.step(
            env.teacher_action()
        )
        assert np.isfinite(observation).all() and math.isfinite(reward)
        if terminated or truncated:
            break
    assert info["success"], info
    final = env.telemetry()
    assert final["energy"] < 0 and final["periapsis_radius"] > env.config.safe_radius
    assert (
        abs(final["apoapsis_radius"] - env.config.goal_radius)
        <= env.config.orbit_tolerance
    )
    assert (
        abs(final["periapsis_radius"] - env.config.goal_radius)
        <= env.config.orbit_tolerance
    )
    assert final["plane_error_deg"] <= env.config.plane_tolerance_deg
