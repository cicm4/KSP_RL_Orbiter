from types import SimpleNamespace

import numpy as np
import pytest

from orbiter.mission_nodes import (
    encounter_prediction,
    set_node_vector,
    usable_liquid_propellant,
)


def test_node_anti_radial_axis_sign_is_explicit():
    node = SimpleNamespace(
        orbital_reference_frame="node", prograde=0.0, normal=0.0, radial=0.0
    )
    sc = SimpleNamespace(
        transform_direction=lambda vector, source, target: (
            -vector[0],
            vector[1],
            vector[2],
        )
    )
    node.burn_vector = lambda frame: (
        -node.radial if frame == "node" else node.radial,
        node.prograde,
        node.normal,
    )
    set_node_vector(sc, node, np.array([5.0, 100.0, -2.0]), "inertial")
    assert (node.radial, node.prograde, node.normal) == (5.0, 100.0, -2.0)


def test_node_encoding_check_survives_inertial_frame_drift_between_rpc_calls():
    # Actual KSP Minmus diagnostic: a correct encoding drifted 0.0023 m/s
    # between the transform/set/read RPCs, while node-local agreement was 1e-12.
    requested = np.array([-847.655173, 10.072517, 273.951955])
    local = (-70.052246, 888.053501, 11.139306)
    node = SimpleNamespace(orbital_reference_frame="node")
    sc = SimpleNamespace(transform_direction=lambda *args: local)
    node.burn_vector = lambda frame: (
        (-node.radial, node.prograde, node.normal)
        if frame == "node"
        else requested + np.array([-0.000744, 0.0, -0.002302])
    )
    set_node_vector(sc, node, requested, "inertial")
    np.testing.assert_allclose(node.burn_vector("node"), local, atol=1e-9)


def test_node_local_radial_sign_error_is_still_rejected():
    node = SimpleNamespace(orbital_reference_frame="node")
    sc = SimpleNamespace(transform_direction=lambda *args: (-5.0, 100.0, -2.0))
    node.burn_vector = lambda frame: (node.radial, node.prograde, node.normal)
    with pytest.raises(RuntimeError, match="frame conversion"):
        set_node_vector(sc, node, np.array([5.0, 100.0, -2.0]), "inertial")


def test_frame_mismatch_is_not_silently_flown():
    node = SimpleNamespace(
        orbital_reference_frame="node", burn_vector=lambda frame: (0.0, 0.0, 0.0)
    )
    sc = SimpleNamespace(transform_direction=lambda vector, source, target: vector)
    with pytest.raises(RuntimeError, match="frame conversion"):
        set_node_vector(sc, node, np.array([5.0, 100.0, -2.0]), "inertial")


def test_prediction_requires_target_soi_not_just_source_apoapsis():
    def orbit(name, next_orbit=None):
        return SimpleNamespace(
            body=SimpleNamespace(name=name),
            periapsis_altitude=50000.0,
            apoapsis_altitude=1e7,
            eccentricity=1.2,
            next_orbit=next_orbit,
        )

    assert not encounter_prediction(SimpleNamespace(orbit=orbit("Kerbin")), "Mun")[
        "encounter"
    ]
    result = encounter_prediction(
        SimpleNamespace(orbit=orbit("Kerbin", orbit("Mun"))), "Mun"
    )
    assert result["encounter"] and [x["body"] for x in result["patches"]] == [
        "Kerbin",
        "Mun",
    ]


def test_excess_unburnable_oxidizer_is_not_counted_as_delta_v():
    resource = SimpleNamespace(
        names=["LiquidFuel", "Oxidizer"],
        amount=lambda name: 9 if name == "LiquidFuel" else 1100,
        density=lambda name: 5.0,
    )
    assert usable_liquid_propellant(resource) == 100.0


@pytest.mark.parametrize(
    "periapsis,eccentricity",
    [
        (float("nan"), 1.2),
        (50000.0, float("nan")),
        (float("inf"), 1.2),
        (50000.0, -1.0),
    ],
)
def test_nonfinite_target_patch_cannot_authorize_a_burn(periapsis, eccentricity):
    orbit = SimpleNamespace(
        body=SimpleNamespace(name="Mun"),
        periapsis_altitude=periapsis,
        apoapsis_altitude=float("inf"),
        eccentricity=eccentricity,
        next_orbit=None,
    )
    with pytest.raises(ValueError, match="nonfinite|invalid"):
        encounter_prediction(SimpleNamespace(orbit=orbit), "Mun")


def test_encounter_math_failure_cannot_leave_an_unowned_node(monkeypatch):
    from orbiter import mission_nodes as module

    nodes = []

    def create(*args, **kwargs):
        node = SimpleNamespace(removed=False)
        node.remove = lambda: setattr(node, "removed", True)
        nodes.append(node)
        return node

    def singular(*args, **kwargs):
        raise ValueError("Collinear Lambert endpoints are singular")

    monkeypatch.setattr(module, "create_node", create)
    monkeypatch.setattr(module, "lambert", singular)
    body = SimpleNamespace(name="Kerbin", non_rotating_reference_frame="frame")
    sc = SimpleNamespace(
        ut=0.0, active_vessel=SimpleNamespace(orbit=SimpleNamespace(body=body))
    )
    plan = SimpleNamespace(
        source="Kerbin",
        target="Mun",
        departure_ut=1000.0,
        arrival_ut=20000.0,
        departure_position=[700000.0, 0.0, 0.0],
        departure_velocity=[0.0, 2200.0, 0.0],
        aim_offset=np.array([0.0, 1.0, 0.0]),
        delta_v=[0.0, 0.0, 0.0],
    )
    with pytest.raises(ValueError, match="Collinear"):
        module.prepare_encounter_node(SimpleNamespace(space_center=sc), plan)
    assert all(node.removed for node in nodes)


@pytest.mark.parametrize("coast_orbits", [-1, 0.5, True, float("nan")])
def test_invalid_coast_window_is_rejected_before_any_rpc(monkeypatch, coast_orbits):
    from orbiter import mission_nodes as module

    monkeypatch.setattr(
        module,
        "read_parking_state",
        lambda *args, **kwargs: pytest.fail("Invalid window read KSP"),
    )
    with pytest.raises(ValueError, match="coast_orbits"):
        module.plan_live_transfer(None, "Minmus", coast_orbits=coast_orbits)


@pytest.mark.parametrize("coast_orbits", [0, 2])
def test_planning_window_coasts_actual_state_and_retains_current_budget(
    monkeypatch, coast_orbits
):
    from orbiter import mission_nodes as module

    system = module.SystemCatalog.load()
    mu = system.bodies["Kerbin"]["gravitational_parameter"]
    r = np.array([680000.0, 0.0, 0.0])
    v = np.array([0.0, 0.0, np.sqrt(mu / 680000.0)])
    state = dict(
        source="Kerbin", r=r, v=v, ut=5000.0, mass=3300.0, dry_mass=1980.0, isp=345.0
    )
    monkeypatch.setattr(
        module, "read_parking_state", lambda *args, **kwargs: state.copy()
    )
    bodies = {
        name: SimpleNamespace(
            gravitational_parameter=row["gravitational_parameter"],
            equatorial_radius=row["equatorial_radius"],
        )
        for name, row in system.bodies.items()
    }
    connection = SimpleNamespace(space_center=SimpleNamespace(bodies=bodies))
    calls = []

    def plan(catalog, source, target, position, velocity, epoch, **kwargs):
        calls.append((np.array(position), np.array(velocity), epoch))
        return SimpleNamespace(estimated_total_delta_v=1000.0)

    monkeypatch.setattr(module, "plan_child_transfer", plan)
    _, budget = module.plan_live_transfer(
        connection, "Minmus", coast_orbits=coast_orbits
    )
    period = 2 * np.pi * np.sqrt(680000.0**3 / mu)
    assert calls[0][2] == pytest.approx(5000.0 + coast_orbits * period)
    np.testing.assert_allclose(calls[0][0], r, atol=1e-5)
    np.testing.assert_allclose(calls[0][1], v, atol=1e-6)
    assert budget["available_delta_v"] == pytest.approx(
        345.0 * 9.80665 * np.log(3300.0 / 1980.0)
    )


def encounter_rig(monkeypatch, paths, *, target_periapsis=50000.0):
    from orbiter import mission_nodes as module

    nodes = []

    def patch(names):
        if not names:
            return None
        return SimpleNamespace(
            body=SimpleNamespace(name=names[0]),
            periapsis_altitude=target_periapsis,
            apoapsis_altitude=-1e6,
            eccentricity=1.2,
            next_orbit=patch(names[1:]),
        )

    def create(*args, **kwargs):
        node = SimpleNamespace(
            orbit=patch(paths[len(nodes)]), delta_v=100.0, removed=False
        )
        node.remove = lambda: setattr(node, "removed", True)
        nodes.append(node)
        return node

    monkeypatch.setattr(module, "create_node", create)
    monkeypatch.setattr(module, "set_node_vector", lambda *args: None)
    monkeypatch.setattr(
        module,
        "lambert",
        lambda *args, **kwargs: (np.array([0.0, 100.0, 0.0]), np.zeros(3)),
    )
    monkeypatch.setattr(module.time, "sleep", lambda *args: None)
    body = SimpleNamespace(name="Kerbin", non_rotating_reference_frame="frame")
    sc = SimpleNamespace(
        ut=0.0, active_vessel=SimpleNamespace(orbit=SimpleNamespace(body=body))
    )
    plan = SimpleNamespace(
        source="Kerbin",
        target="Minmus",
        kind="direct_child",
        departure_ut=1000.0,
        arrival_ut=20000.0,
        departure_position=[700000.0, 0.0, 0.0],
        departure_velocity=[0.0, 0.0, 0.0],
        aim_offset=np.array([0.0, 1.0, 0.0]),
        delta_v=[0.0, 100.0, 0.0],
        target_altitude=50000.0,
    )
    return SimpleNamespace(space_center=sc), plan, nodes


def test_intermediate_mun_encounter_is_rejected_and_owned_node_removed(monkeypatch):
    from orbiter import mission_nodes as module

    conn, plan, nodes = encounter_rig(monkeypatch, [["Kerbin", "Mun", "Minmus"]])
    with pytest.raises(
        module.EncounterUnavailable, match="unmodeled|admissible|intermediate"
    ):
        module.prepare_encounter_node(conn, plan)
    assert len(nodes) == 1 and nodes[0].removed


def test_direct_minmus_encounter_still_accepts_first_window(monkeypatch):
    from orbiter import mission_nodes as module

    conn, plan, nodes = encounter_rig(monkeypatch, [["Kerbin", "Minmus"]])
    node, prediction = module.prepare_encounter_node(conn, plan)
    assert node is nodes[0] and not node.removed and prediction["encounter"]


def test_unreachable_periapsis_is_retryable_but_invalid_telemetry_is_not(monkeypatch):
    from orbiter import mission_nodes as module

    conn, plan, nodes = encounter_rig(
        monkeypatch, [["Kerbin", "Minmus"]], target_periapsis=500000.0
    )
    with pytest.raises(module.EncounterUnavailable, match="periapsis differs"):
        module.prepare_encounter_node(conn, plan)
    assert nodes[0].removed
    conn, plan, nodes = encounter_rig(
        monkeypatch, [["Kerbin", "Minmus"]], target_periapsis=float("nan")
    )
    with pytest.raises(ValueError, match="nonfinite"):
        module.prepare_encounter_node(conn, plan)
    assert nodes[0].removed
