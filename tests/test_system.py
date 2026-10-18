import numpy as np
import pytest

from orbiter.system import SystemCatalog


def test_complete_stock_hierarchy_and_multi_primary_routes():
    system = SystemCatalog.load()
    assert len(system.bodies) == 17
    assert system.route("Kerbin", "Mun") == ["Kerbin", "Mun"]
    assert system.route("Kerbin", "Duna") == ["Kerbin", "Sun", "Duna"]
    assert system.route("Mun", "Tylo") == ["Mun", "Kerbin", "Sun", "Jool", "Tylo"]
    assert system.route("Kerbin", "Kerbin") == ["Kerbin"]
    with pytest.raises(ValueError):
        system.route("Kerbin", "Unknown")


@pytest.mark.parametrize(
    "name",
    [name for name, body in SystemCatalog.load().bodies.items() if body["orbit"]],
)
def test_ephemerides_reproduce_actual_krpc_parent_frame_samples(name):
    system = SystemCatalog.load()
    body = system.bodies[name]
    for sample in body["reference_samples"]:
        position, velocity = system.state(
            name, sample["ut_s"], relative_to=body["parent"]
        )
        np.testing.assert_allclose(
            position, sample["position_m"], atol=0.05, rtol=1e-10
        )
        assert np.isfinite(velocity).all()


def test_moon_positions_and_velocities_compose_through_parent():
    system = SystemCatalog.load()
    moon_position, moon_velocity = system.state("Mun", 100000.0)
    local_position, local_velocity = system.state("Mun", 100000.0, relative_to="Kerbin")
    parent_position, parent_velocity = system.state("Kerbin", 100000.0)
    np.testing.assert_allclose(moon_position, local_position + parent_position)
    np.testing.assert_allclose(moon_velocity, local_velocity + parent_velocity)


def test_ephemeris_velocity_agrees_with_independent_position_derivative():
    system = SystemCatalog.load()
    for name in system.bodies:
        if system.bodies[name]["parent"] is None:
            continue
        _, velocity = system.state(name, 100000.0)
        before, _ = system.state(name, 99999.0)
        after, _ = system.state(name, 100001.0)
        np.testing.assert_allclose((after - before) / 2, velocity, atol=1e-4, rtol=1e-7)
