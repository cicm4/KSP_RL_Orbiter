"""Physical and task invariants for the accelerated launch surrogate.

These checks validate the surrogate's own equations. They do not establish
agreement with KSP's craft aerodynamics, attitude controller, or joint physics.
"""
import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from orbiter.env import G0, EnvConfig, HeadlessOrbiterEnv, Stage, shaped_reward
from orbiter.guidance import expert_action


def set_circular_state(env, altitude=300_000.0):
    """Place an unfired, fueled vehicle on a vacuum orbit outside the goal."""
    env.x = env.config.radius + altitude
    env.y = env.vx = 0.0
    env.vy = math.sqrt(env.config.mu / env.x)
    env._potential_prev = env.potential()


def test_gymnasium_contract():
    check_env(HeadlessOrbiterEnv(EnvConfig(randomization=0)), skip_render_check=True)


@pytest.mark.parametrize('field', [
    'control_dt', 'physics_dt', 'max_time', 'mu', 'radius', 'rotation_period',
    'target_altitude', 'orbit_tolerance', 'payload_mass', 'drag_area',
    'attitude_rate', 'attitude_tau', 'circularization_margin',
])
@pytest.mark.parametrize('value', [math.nan, math.inf])
def test_nonfinite_environment_values_are_rejected_before_rollout(field, value):
    # In particular, a NaN control interval used to produce t=0 forever with
    # neither termination nor truncation, hanging baseline evaluation.
    with pytest.raises(ValueError):
        EnvConfig(**{field: value})


@pytest.mark.parametrize('field', ['dry_mass', 'fuel_mass', 'thrust_vac', 'isp_vac', 'isp_sl'])
@pytest.mark.parametrize('value', [math.nan, math.inf])
def test_nonfinite_stage_values_are_rejected(field, value):
    from dataclasses import replace
    stage = replace(EnvConfig().stages[0], **{field: value})
    with pytest.raises(ValueError):
        EnvConfig(stages=(stage,))


@pytest.mark.parametrize('field', [
    'control_dt', 'physics_dt', 'max_time', 'mu', 'radius',
    'rotation_period', 'target_altitude', 'attitude_tau',
])
def test_zero_environment_denominators_are_rejected(field):
    with pytest.raises(ValueError):
        EnvConfig(**{field: 0.})


def test_valid_zero_coefficients_and_retrograde_rotation_remain_supported():
    config = EnvConfig(randomization=0., atmosphere_depth=0., drag_area=0.,
                       drag_alpha_area=0., lift_area=0., payload_mass=0.,
                       attitude_rate=0., orbit_tolerance=0., circularization_margin=0.,
                       rotation_period=-21549.425)
    env = HeadlessOrbiterEnv(config)
    observation, reward, _, _, _ = env.step([1., 1.])
    assert env.t == 2. and np.isfinite(observation).all() and math.isfinite(reward)


@pytest.mark.parametrize('atmosphere', [
    ((0., math.nan, 101325.), (70000., 0., 0.)),
    ((0., 1., math.inf), (70000., 0., 0.)),
    ((0., 1., 101325.), (0., .5, 50000.)),
])
def test_invalid_atmosphere_values_and_repeated_altitudes_are_rejected(atmosphere):
    with pytest.raises(ValueError):
        EnvConfig(atmosphere=atmosphere)


def test_vacuum_coast_conserves_energy_and_momentum_over_three_orbits():
    config = EnvConfig(
        randomization=0, atmosphere_depth=0, control_dt=30,
        physics_dt=2, max_time=20_000,
    )
    env = HeadlessOrbiterEnv(config)
    set_circular_state(env)
    initial = env.telemetry()
    fuel = env.fuel
    period = 2 * math.pi * math.sqrt(env.x ** 3 / config.mu)
    for _ in range(math.ceil(3 * period / config.control_dt)):
        observation, reward, terminated, truncated, _ = env.step([-1, 1])
        assert not terminated and not truncated
        assert np.isfinite(observation).all() and math.isfinite(reward)
    final = env.telemetry()
    assert final['energy'] == pytest.approx(initial['energy'], rel=1e-9)
    assert final['momentum'] == pytest.approx(initial['momentum'], rel=1e-9)
    assert final['altitude'] == pytest.approx(300_000, abs=0.01)
    assert env.fuel == fuel


def test_atmosphere_rotates_with_planet():
    env = HeadlessOrbiterEnv(EnvConfig(randomization=0))
    # reset() initializes the pad velocity in the inertial frame; that same
    # state must have no aerodynamic speed relative to the rotating atmosphere.
    assert env.telemetry()['dynamic_pressure'] == pytest.approx(0, abs=1e-20)


def test_engine_thrust_depends_on_pressure_and_fuel_flow_is_consistent():
    config = EnvConfig(randomization=0, control_dt=0.1, physics_dt=0.01)
    results = []
    for altitude in (0.0, config.atmosphere_depth + 1):
        env = HeadlessOrbiterEnv(config)
        env.x = config.radius + altitude
        stage = env.stages[0]
        available = env.available_thrust()
        if altitude == 0:
            assert available == pytest.approx(stage.thrust_vac * stage.isp_sl / stage.isp_vac)
            # Move a little off the surface before advancing the integration.
            env.x += 100
        else:
            assert available == pytest.approx(stage.thrust_vac)
        env._potential_prev = env.potential()
        fuel_before = env.fuel
        _, _, terminated, truncated, _ = env.step([0, 1])  # 50% throttle
        assert not terminated and not truncated
        consumed = fuel_before - env.fuel
        expected = 0.5 * stage.thrust_vac / (stage.isp_vac * G0) * config.control_dt
        assert consumed == pytest.approx(expected, rel=1e-10)
        results.append(consumed)
    # This engine model decreases thrust with Isp at pressure; its mass flow
    # therefore stays constant for a fixed throttle and engine configuration.
    assert results[0] == pytest.approx(results[1], rel=1e-10)


def test_stage_exhaustion_jettisons_dry_mass_and_never_makes_fuel_negative():
    config = EnvConfig(
        randomization=0, atmosphere_depth=0, control_dt=2, physics_dt=0.25,
        payload_mass=1000,
        stages=(Stage(100, 0.2, 1000, 300, 280), Stage(80, 0.3, 2000, 300, 280)),
    )
    env = HeadlessOrbiterEnv(config)
    set_circular_state(env)
    _, reward, terminated, truncated, info = env.step([1, 1])
    assert terminated and not truncated
    assert info['termination_reason'] == 'out_of_fuel'
    assert env.stage == 1 and 0 <= env.fuel <= 1e-7
    assert env.mass == pytest.approx(config.payload_mass + config.stages[-1].dry_mass, abs=1e-7)
    assert env.available_thrust() == 0
    assert math.isfinite(reward)


def test_exhausted_fuel_cannot_produce_thrust_or_free_impulse():
    config = EnvConfig(
        randomization=0, atmosphere_depth=0,
        stages=(Stage(1000, 1000, 650_000, 320, 280),),
    )
    nearly_empty = HeadlessOrbiterEnv(config)
    empty = HeadlessOrbiterEnv(config)
    for env, fuel in ((nearly_empty, 5e-8), (empty, 0.0)):
        set_circular_state(env)
        env.fuel = fuel
        env._potential_prev = env.potential()
    # The exhaustion threshold used for staging/mass flow must also gate thrust.
    assert nearly_empty.available_thrust() == 0
    nearly_empty.step([1, 1])
    empty.step([1, 1])
    np.testing.assert_allclose(
        [nearly_empty.x, nearly_empty.y, nearly_empty.vx, nearly_empty.vy],
        [empty.x, empty.y, empty.vx, empty.vy], rtol=1e-12, atol=1e-8,
    )


def test_escape_is_failure_and_observation_remains_finite():
    env = HeadlessOrbiterEnv(EnvConfig(randomization=0, atmosphere_depth=0))
    set_circular_state(env)
    env.vy = 1.1 * math.sqrt(2 * env.config.mu / env.x)
    env._potential_prev = env.potential()
    observation, reward, terminated, truncated, info = env.step([-1, 1])
    assert terminated and not truncated and not info['success']
    assert info['termination_reason'] == 'escape'
    assert math.isinf(info['apoapsis']) and info['energy'] > 0
    assert np.isfinite(observation).all() and math.isfinite(reward)
    assert env.observation_space.contains(observation)


@pytest.mark.parametrize('success', [False, True])
def test_terminal_potential_cancels_over_the_discounted_episode(success):
    # Telescoping shaping cannot let a partial ascent cash out its final Phi.
    potentials = (-16.0, -3.0, -7.0, -1.0)
    gamma = 0.93
    discounted_shaping = 0.0
    for step, (previous, current) in enumerate(zip(potentials, potentials[1:])):
        terminal = step == len(potentials) - 2
        _, shaping, _ = shaped_reward(previous, current, gamma, terminal, success and terminal, 1)
        discounted_shaping += gamma ** step * shaping
    assert discounted_shaping == pytest.approx(-potentials[0])


def test_impact_applies_terminal_cancellation_in_environment():
    env = HeadlessOrbiterEnv(EnvConfig(randomization=0))
    env.x = env.config.radius + 1
    env.y = 0
    env.vx = -100
    env._potential_prev = previous = env.potential()
    _, _, terminated, truncated, info = env.step([-1, 1])
    assert terminated and not truncated
    assert info['termination_reason'] == 'impact' and not info['success']
    assert info['reward_shaping'] == pytest.approx(-previous)
    assert info['reward_task'] < 0


def test_external_time_limit_keeps_next_potential_for_bootstrap():
    config = EnvConfig(randomization=0, max_time=1, control_dt=1, physics_dt=0.1)
    env = HeadlessOrbiterEnv(config)
    previous = env.potential()
    observation, _, terminated, truncated, info = env.step([-1, 1])
    assert not terminated and truncated
    assert info['termination_reason'] == 'time_limit'
    assert info['reward_task'] == 0
    assert info['reward_shaping'] == pytest.approx(config.gamma * env.potential() - previous)
    assert info['reward_shaping'] != pytest.approx(-previous)
    assert np.isfinite(observation).all()


def test_seed_reproduces_randomized_dynamics_and_trajectory():
    first = HeadlessOrbiterEnv(EnvConfig(randomization=0.05))
    second = HeadlessOrbiterEnv(EnvConfig(randomization=0.05))
    first_observation, _ = first.reset(seed=2026)
    second_observation, _ = second.reset(seed=2026)
    assert first.stages == second.stages
    assert first.drag_area == second.drag_area
    assert first.attitude_rate == second.attitude_rate
    np.testing.assert_array_equal(first_observation, second_observation)
    for _ in range(8):
        action = expert_action(first)
        left, right = first.step(action), second.step(action)
        np.testing.assert_array_equal(left[0], right[0])
        assert left[1:] == right[1:]
    stages = first.stages
    first.reset(seed=2027)
    assert first.stages != stages


def test_reducing_physics_timestep_converges_during_powered_flight():
    final_positions = []
    for physics_dt in (2.0, 0.5, 0.03125):
        env = HeadlessOrbiterEnv(EnvConfig(
            randomization=0, atmosphere_depth=0, physics_dt=physics_dt,
        ))
        set_circular_state(env)
        for _ in range(10):
            _, _, terminated, truncated, _ = env.step([1, 1])
            assert not terminated and not truncated
        final_positions.append(np.array([env.x, env.y]))
    coarse_error = np.linalg.norm(final_positions[0] - final_positions[-1])
    fine_error = np.linalg.norm(final_positions[1] - final_positions[-1])
    assert 0 < fine_error < coarse_error / 4
    assert fine_error < 0.1  # metres, against the smaller-step reference


@pytest.mark.parametrize('randomization', [0.0, 0.05])
@pytest.mark.parametrize('seed', [11, 29, 47, 83, 101])
def test_expert_reaches_bounded_orbit_from_launch_on_held_out_seeds(randomization, seed):
    config = EnvConfig(randomization=randomization)
    env = HeadlessOrbiterEnv(config)
    env.reset(seed=seed)
    for _ in range(math.ceil(config.max_time / config.control_dt) + 1):
        action = expert_action(env)
        assert env.action_space.contains(action)
        observation, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(observation) and math.isfinite(reward)
        if terminated or truncated:
            break
    else:
        pytest.fail('expert rollout failed to respect the episode limit')
    assert terminated and not truncated, info
    assert info['success'] and info['termination_reason'] == 'orbit_achieved', info
    assert info['energy'] < 0
    assert info['periapsis'] >= max(config.atmosphere_depth + 1000, config.target_altitude - config.orbit_tolerance)
    assert info['apoapsis'] <= config.target_altitude + config.orbit_tolerance
    assert info['fuel_remaining'] >= 0
