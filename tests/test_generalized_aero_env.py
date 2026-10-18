from dataclasses import asdict, replace
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from orbiter.env import EnvConfig, HeadlessOrbiterEnv
from orbiter.generalized_aero_env import (
    AeroVehicleEnvelope,
    BatchedAeroAscent,
    FEATURE_SCHEMA,
    OBSERVATIONS,
    TASK,
    cpu_observation,
    normalization_audit,
)
from orbiter.generalized_env import BatchedAscent, VehicleEnvelope


DEVICES = [
    "cpu",
    pytest.param(
        "cuda:0",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA hardware required"
        ),
    ),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("near_stage_boundary", [False, True])
def test_aerodynamic_rk4_and_staging_match_independent_scalar(
    device, near_stage_boundary
):
    batch = BatchedAeroAscent(3, seed=71, device=device)
    references = [batch.cpu_reference(index) for index in range(3)]
    # Give each world substantial airspeed and nonzero angle of attack, so the
    # test exercises lift/alpha drag rather than only the pad's near-zero force.
    for index, reference in enumerate(references):
        state = [
            batch.radius + 15000,
            0,
            40 + index * 5,
            (batch.radius + 15000) * 2 * math.pi / batch.rotation_period + 250,
        ]
        batch.state[index] = torch.tensor(state, device=device, dtype=batch.dtype)
        reference.x, reference.y, reference.vx, reference.vy = state
        batch.pitch[index] = reference.pitch = 30
        if near_stage_boundary:
            batch.fuel[index] = reference.fuel = 0.0123
        reference._potential_prev = reference.potential()
    batch.previous_potential.copy_(batch.potential())
    for _ in range(8):
        action = torch.tensor([[0.6, -0.2], [0.4, 0.1], [0.9, -0.3]], device=device)
        observation, reward, terminated, truncated, _ = batch.step(action)
        for index, reference in enumerate(references):
            _, expected_reward, expected_term, expected_trunc, _ = reference.step(
                action[index].cpu().numpy()
            )
            np.testing.assert_allclose(
                batch.state[index].cpu(),
                [reference.x, reference.y, reference.vx, reference.vy],
                rtol=1e-12,
                atol=2e-7,
            )
            assert float(batch.fuel[index]) == pytest.approx(reference.fuel, abs=1e-8)
            assert float(batch.mass()[index]) == pytest.approx(reference.mass, abs=1e-8)
            assert int(batch.stage[index]) == reference.stage
            assert float(reward[index]) == pytest.approx(expected_reward, abs=2e-5)
            assert bool(terminated[index]) == expected_term
            assert bool(truncated[index]) == expected_trunc
            np.testing.assert_allclose(
                observation[index].cpu(), cpu_observation(reference), atol=2e-6
            )
    if near_stage_boundary:
        assert batch.stage.cpu().tolist() == [1, 1, 1]


@pytest.mark.parametrize("device", DEVICES)
def test_lift_changes_sign_across_relative_velocity_and_drag_removes_energy(device):
    batch = BatchedAeroAscent(2, device=device)
    batch.drag_alpha_area[:] = 4
    batch.lift_area[:] = 5
    batch.drag_area.zero_()
    altitude, east_speed = 10000.0, 200.0
    radius = batch.radius + altitude
    state = torch.tensor(
        [[radius, 0, 0, radius * 2 * math.pi / batch.rotation_period + east_speed]] * 2,
        device=device,
        dtype=batch.dtype,
    )
    mass = torch.full((2,), 10000.0, device=device, dtype=batch.dtype)
    pitch = torch.tensor([30.0, -30.0], device=device, dtype=batch.dtype)
    actual = batch.derivative(state, mass, pitch, torch.zeros_like(pitch))
    rho = batch.atmosphere_at(torch.full_like(pitch, altitude))[:, 0]
    gravity = -batch.mu / radius**2
    expected_lift = 0.5 * rho * 5 * math.sin(math.pi / 3) * east_speed**2 / mass
    torch.testing.assert_close(
        actual[:, 2] - gravity,
        expected_lift * torch.tensor([1.0, -1.0], device=device, dtype=batch.dtype),
    )
    assert bool((actual[:, 3] < 0).all())
    torch.testing.assert_close(actual[0, 3], actual[1, 3])


def test_zero_aerodynamic_coefficients_are_exact_v3_ablation_including_partial_reset():
    base = BatchedAscent(4, seed=48)
    aero = BatchedAeroAscent(
        4,
        seed=48,
        envelope=AeroVehicleEnvelope(
            drag_alpha_area_per_tonne_max=0,
            lift_area_per_tonne_max=0,
        ),
    )
    for index in range(8):
        action = base.teacher()
        expected = base.step(action)
        actual = aero.step(action)
        torch.testing.assert_close(aero.state, base.state, rtol=0, atol=0)
        torch.testing.assert_close(actual[0][:, :54], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
        if index == 3:
            mask = torch.tensor([False, True, False, True])
            base.reset(mask)
            aero.reset(mask)
            torch.testing.assert_close(aero.stages, base.stages, rtol=0, atol=0)


def test_partial_reset_changes_only_selected_coefficient_and_craft_rows():
    batch = BatchedAeroAscent(6, seed=47)
    before = {
        name: getattr(batch, name).clone()
        for name in (
            "state",
            "stages",
            "drag_alpha_area",
            "lift_area",
            "initial_mass",
        )
    }
    mask = torch.tensor([False, True, False, True, False, False])
    batch.reset(mask)
    for name, previous in before.items():
        torch.testing.assert_close(
            getattr(batch, name)[~mask], previous[~mask], rtol=0, atol=0
        )
    for name in ("drag_alpha_area", "lift_area", "stages"):
        assert not torch.equal(getattr(batch, name)[mask], before[name][mask])


def test_coefficient_sampling_is_independent_and_mass_normalized_without_aliasing():
    batch = BatchedAeroAscent(256, seed=21)
    obs = batch.observation()
    assert obs.shape == (256, OBSERVATIONS)
    assert TASK.endswith("v4") and FEATURE_SCHEMA.endswith("v4")
    assert torch.isfinite(obs).all()
    assert float(batch.initial_mass.max() / batch.initial_mass.min()) > 10
    assert abs(float(torch.corrcoef(obs[:, -2:].T)[0, 1])) < 0.2
    for index in (0, 19, 120, 255):
        np.testing.assert_allclose(
            obs[index], cpu_observation(batch.cpu_reference(index)), atol=1e-6
        )
    batch.drag_alpha_area[:2] = batch.initial_mass[:2] / 1000 * torch.tensor([0.1, 0.9])
    batch.lift_area[:2] = batch.initial_mass[:2] / 1000 * torch.tensor([0.6, 0.2])
    np.testing.assert_allclose(
        batch.observation()[:2, -2:], [[0.1, 0.6], [0.9, 0.2]], atol=1e-6
    )
    audit = normalization_audit(batch.envelope)
    assert audit["feature_schema"] == FEATURE_SCHEMA
    assert audit["all_craft_and_actuator_features_below_clip"]
    assert not normalization_audit(replace(batch.envelope, lift_area_per_tonne_max=11))[
        "all_craft_and_actuator_features_below_clip"
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"drag_alpha_area_per_tonne_min": -0.1},
        {"drag_alpha_area_per_tonne_max": float("nan")},
        {"lift_area_per_tonne_min": True},
        {"lift_area_per_tonne_max": float("inf")},
        {"lift_area_per_tonne_min": 0.8, "lift_area_per_tonne_max": 0.2},
        {"payload_min": -1},
        {"stages_min": 1.5},
    ],
)
def test_envelope_validates_both_inherited_and_new_ranges(kwargs):
    with pytest.raises(ValueError):
        AeroVehicleEnvelope(**kwargs)


def test_zero_ranges_valid_and_v3_envelopes_are_rejected():
    envelope = AeroVehicleEnvelope(
        drag_alpha_area_per_tonne_max=0, lift_area_per_tonne_max=0
    )
    assert not BatchedAeroAscent(2, envelope=envelope).observation()[:, -2:].any()
    with pytest.raises(TypeError, match="AeroVehicleEnvelope"):
        BatchedAeroAscent(2, envelope=VehicleEnvelope())


@pytest.mark.parametrize("device", DEVICES)
def test_exact_config_bank_retains_atmosphere_coefficients_and_scalar_physics(device):
    atmosphere = tuple(
        (altitude, density * 1.2, pressure * 1.1)
        for altitude, density, pressure in EnvConfig().atmosphere
    )
    config = EnvConfig(
        atmosphere=atmosphere,
        drag_alpha_area=6.845,
        lift_area=4.434,
        target_altitude=100000,
        max_time=1600,
        randomization=0,
        physics_dt=0.125,
        control_dt=1.5,
        attitude_rate=7.3,
    )
    cases = [
        config,
        replace(config, payload_mass=1800, drag_alpha_area=3.4, lift_area=1.7),
    ]
    batch = BatchedAeroAscent(2, seed=13, device=device)
    initial = batch.load_configs(cases)
    references = [HeadlessOrbiterEnv(case) for case in cases]
    for index, case in enumerate(cases):
        assert asdict(batch.export_config(index)) == asdict(case)
        np.testing.assert_allclose(
            initial[index].cpu(), cpu_observation(references[index]), atol=1e-6
        )
    for _ in range(12):
        action = batch.teacher()
        batch.step(action)
        for index, reference in enumerate(references):
            reference.step(action[index].cpu().numpy())
            np.testing.assert_allclose(
                batch.state[index].cpu(),
                [reference.x, reference.y, reference.vx, reference.vy],
                rtol=1e-12,
                atol=2e-7,
            )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: replace(c, randomization=0.1),
        lambda c: replace(c, residual=True),
        lambda c: replace(c, curriculum=True),
        lambda c: replace(c, atmosphere_depth=60000),
        lambda c: replace(c, physics_dt=3),
        lambda c: replace(c, gamma=0.98),
    ],
)
def test_config_bank_rejects_incompatible_cases_before_mutation(mutation):
    batch = BatchedAeroAscent(2, seed=3)
    config = batch.export_config(0)
    state, stages, atmosphere = (
        batch.state.clone(),
        batch.stages.clone(),
        batch.atmosphere.clone(),
    )
    with pytest.raises(ValueError):
        batch.load_configs([config, mutation(config)])
    torch.testing.assert_close(batch.state, state, rtol=0, atol=0)
    torch.testing.assert_close(batch.stages, stages, rtol=0, atol=0)
    torch.testing.assert_close(batch.atmosphere, atmosphere, rtol=0, atol=0)
