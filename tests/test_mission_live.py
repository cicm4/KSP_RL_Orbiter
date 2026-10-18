"""Mission phases and capture-frame contracts with fake RPCs only."""

from contextlib import contextmanager
import json
import math
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from orbiter.capture_env import CaptureConfig, rtn_basis
from orbiter.orbital import propagate
from orbiter import mission_live as live


class Log:
    def __init__(self):
        self.events = []

    def write(self, event, **data):
        self.events.append(dict(event=event, **data))


@pytest.mark.parametrize(
    "body,altitude", [("Mun", 50000.0), ("Duna", 70000.0), ("Jool", 220000.0)]
)
def test_default_destination_goal_matches_safe_capture_training_goal(body, altitude):
    assert live.resolve_target_altitude(body) == altitude


@pytest.mark.parametrize(
    "body,altitude", [("Sun", None), ("missing", None), ("Duna", 50000.0)]
)
def test_invalid_destination_goals_fail_before_launch(body, altitude):
    with pytest.raises(ValueError):
        live.resolve_target_altitude(body, altitude)


class AutoPilot:
    def __init__(self):
        self.error = 0.0
        self.engages = self.disengages = 0

    def engage(self):
        self.engages += 1

    def disengage(self):
        self.disengages += 1


class Connection:
    def __init__(self, config=None):
        self.config = config or CaptureConfig(randomization=0.0)
        c = self.config
        self.body = SimpleNamespace(
            name=c.body_name,
            gravitational_parameter=c.mu,
            equatorial_radius=c.radius,
            atmosphere_depth=c.atmosphere_depth,
            sphere_of_influence=c.sphere_of_influence,
            non_rotating_reference_frame="native inertial",
        )
        self.vessel = SimpleNamespace(
            name="Mission vessel",
            mass=3000.0,
            auto_pilot=AutoPilot(),
            control=SimpleNamespace(throttle=0.0, sas=False),
            direction=lambda frame: (0.0, 1.0, 0.0),
            situation="VesselSituation.orbiting",
        )
        self.vessel.orbit = SimpleNamespace(
            body=self.body,
            semi_major_axis=c.goal_radius,
            apoapsis_altitude=c.target_altitude,
            periapsis_altitude=c.target_altitude,
            eccentricity=0.0,
            period=1000.0,
            next_orbit=None,
            time_to_periapsis=400.0,
        )
        self.space_center = SimpleNamespace(
            active_vessel=self.vessel,
            ut=100.0,
            bodies={c.body_name: self.body},
            rails_warp_factor=0,
            physics_warp_factor=0,
        )
        self.closed = False

    def state(self):
        c = self.config
        return dict(
            ut=self.space_center.ut,
            source=self.body.name,
            r=[c.goal_radius, 0.0, 0.0],
            v=[0.0, 0.0, c.velocity_scale],
            mass=3000.0,
            dry_mass=1000.0,
            thrust=60000.0,
            isp=345.0,
        )

    def close(self):
        self.closed = True


def test_reflection_and_capture_action_preserve_native_rtn_geometry():
    native_r, native_v = np.array([3.0, 4.0, 5.0]), np.array([6.0, -4.0, 2.0])
    r, v = live.reflect(native_r), live.reflect(native_v)
    np.testing.assert_array_equal(live.reflect(r), native_r)
    for local in np.eye(3):
        native = live.capture_direction_native(r, v, local)
        np.testing.assert_allclose(live.reflect(native), rtn_basis(r, v) @ local)
        assert np.linalg.norm(native) == pytest.approx(1.0)


def test_27_observation_bridge_uses_reflected_vectors_and_fixed_goal_plane(monkeypatch):
    conn = Connection()
    monkeypatch.setattr(live, "_state", lambda connection: connection.state())
    c = conn.config
    normal = np.cross(live.reflect(conn.state()["r"]), live.reflect(conn.state()["v"]))
    normal /= np.linalg.norm(normal)
    data, observation = live.read_capture_state(
        conn, conn.vessel, c, initial_mass=3000.0, goal_normal=normal, start_ut=100.0
    )
    assert observation.shape == (27,) and np.isfinite(observation).all()
    np.testing.assert_array_equal(data["direction"], [0.0, -1.0, 0.0])
    np.testing.assert_allclose(observation[7:10], [0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(observation[10:13], [0.0, 0.0, 1.0], atol=1e-6)
    assert data["orbit_in_band"]


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"body": "Kerbin"}, "wrong_body"),
        ({"energy": 1.0}, "not_bound"),
        ({"periapsis": 40000.0}, "outside_target_band"),
        ({"apoapsis": 1e8}, "outside_soi"),
        ({"apoapsis": math.nan}, "invalid_telemetry"),
    ],
)
def test_actual_orbit_gate_rejects_wrong_body_energy_or_apses(change, reason):
    conn = Connection()
    sample = live.read_actual_orbit(conn.space_center, conn.vessel)
    assert reason in live.orbit_failures({**sample, **change}, conn.config)


def test_incoming_radius_bisection_matches_universal_propagation():
    c = CaptureConfig()
    peri_r = np.array([c.goal_radius, 0.0, 0.0])
    peri_v = np.array([0.0, math.sqrt(200.0**2 + 2 * c.mu / c.goal_radius), 0.0])
    r, v = propagate(peri_r, peri_v, -400.0, c.mu)
    dt = live.incoming_radius_time(r, v, c.mu, 1.04 * c.goal_radius, 400.0)
    arrived, velocity = propagate(r, v, dt, c.mu)
    assert 0 < dt < 400
    assert np.linalg.norm(arrived) == pytest.approx(1.04 * c.goal_radius, abs=1e-5)
    assert arrived @ velocity < 0
    with pytest.raises(ValueError, match="incoming"):
        live.incoming_radius_time(r, -v, c.mu, 1.04 * c.goal_radius, 400.0)


def test_actual_patched_conic_is_required_for_soi_warp():
    conn = Connection()
    with pytest.raises(RuntimeError, match="no destination encounter"):
        live.target_entry_ut(conn.space_center, conn.vessel, "Minmus")
    conn.vessel.orbit.next_orbit = SimpleNamespace(body=SimpleNamespace(name="Minmus"))
    conn.vessel.orbit.time_to_soi_change = 500.0
    assert live.target_entry_ut(conn.space_center, conn.vessel, "Minmus") == 600.0


def test_full_orbital_period_verification_uses_four_uncontrolled_probes(monkeypatch):
    conn, log, warps = Connection(), Log(), []

    def warp(sc, target, **kwargs):
        warps.append(target)
        sc.ut = target

    monkeypatch.setattr(live, "bounded_warp", warp)
    result = live.verify_orbital_period(conn, conn.vessel, conn.config, log=log)
    assert result["success"] and result["observed_coast_s"] == 1000.0
    assert warps == [350.0, 600.0, 850.0, 1100.0]
    assert (
        conn.vessel.control.throttle == 0.0 and conn.vessel.auto_pilot.disengages == 1
    )
    assert len(result["samples"]) == 5


def test_period_verification_fails_if_target_soi_changes(monkeypatch):
    conn = Connection()

    def warp(sc, target, **kwargs):
        sc.ut = target
        conn.body.name = "Kerbin"

    monkeypatch.setattr(live, "bounded_warp", warp)
    result = live.verify_orbital_period(conn, conn.vessel, conn.config, log=Log())
    assert not result["success"] and result["failures"] == ["wrong_body"]


def test_capture_calls_neural_policy_directly_and_engages_only_once(monkeypatch):
    conn, log, observations = Connection(), Log(), []
    conn.vessel.auto_pilot.target_roll = 120.0
    monkeypatch.setattr(live, "_state", lambda connection: connection.state())
    monkeypatch.setattr(
        live,
        "wait_for_ut",
        lambda sc, target, *args, **kwargs: setattr(sc, "ut", target),
    )

    def predict(observation, deterministic):
        observations.append(observation)
        return np.zeros(3), None

    result = live.run_capture(
        conn, conn.vessel, conn.config, log=log, model=SimpleNamespace(predict=predict)
    )
    assert result["success"] and result["controller"] == "learned_capture_policy"
    assert observations and all(obs.shape == (27,) for obs in observations)
    assert conn.vessel.auto_pilot.engages == conn.vessel.auto_pilot.disengages == 1
    assert math.isnan(conn.vessel.auto_pilot.target_roll)
    assert conn.vessel.control.throttle == 0.0
    assert any(event["event"] == "capture_action" for event in log.events)


def test_capture_scripted_controller_has_an_explicit_baseline_label(monkeypatch):
    conn = Connection()
    monkeypatch.setattr(live, "_state", lambda connection: connection.state())
    monkeypatch.setattr(
        live,
        "wait_for_ut",
        lambda sc, target, *args, **kwargs: setattr(sc, "ut", target),
    )
    result = live.run_capture(conn, conn.vessel, conn.config, log=Log(), scripted=True)
    assert result["success"] and result["controller"] == "scripted_capture_teacher"


def test_capture_wrong_body_releases_controls_without_a_fallback(monkeypatch):
    conn = Connection()
    monkeypatch.setattr(
        live, "_state", lambda connection: {**connection.state(), "source": "Kerbin"}
    )
    with pytest.raises(RuntimeError, match="actual target SOI"):
        live.run_capture(conn, conn.vessel, conn.config, log=Log(), scripted=True)
    assert (
        conn.vessel.control.throttle == 0.0 and conn.vessel.auto_pilot.disengages == 1
    )


@pytest.mark.parametrize("pointing_error", [20.0, -172.35])
def test_capture_throttle_is_zero_for_either_sign_of_misalignment(
    monkeypatch, pointing_error
):
    conn, log = Connection(), Log()
    conn.vessel.auto_pilot.error = pointing_error
    monkeypatch.setattr(live, "_state", lambda connection: connection.state())
    monkeypatch.setattr(live, "orbit_failures", lambda *args: ["outside_target_band"])
    monkeypatch.setattr(
        live, "decode_capture_action", lambda *args: (1.0, np.array([0.0, -1.0, 0.0]))
    )

    def inspect_command(*args, **kwargs):
        assert (
            conn.vessel.control.throttle == 0.0
        ), "Misaligned capture must not apply throttle"
        raise TimeoutError("Stop after one fake capture command")

    monkeypatch.setattr(live, "wait_for_ut", inspect_command)
    with pytest.raises(TimeoutError, match="one fake capture command"):
        live.run_capture(
            conn,
            conn.vessel,
            conn.config,
            log=log,
            model=SimpleNamespace(predict=lambda *args, **kwargs: (np.zeros(3), None)),
        )
    assert conn.vessel.control.throttle == 0.0


@pytest.mark.parametrize("pointing_error", [20.0, -172.35])
def test_capture_coast_waits_for_either_sign_of_misalignment(
    monkeypatch, pointing_error
):
    conn = Connection()
    conn.vessel.auto_pilot.target_roll = 120.0
    c = conn.config
    peri_r = np.array([c.goal_radius, 0.0, 0.0])
    peri_v = np.array([0.0, 0.0, math.sqrt(200.0**2 + 2 * c.mu / c.goal_radius)])
    r, v = propagate(peri_r, peri_v, -400.0, c.mu)
    monkeypatch.setattr(
        live, "_state", lambda connection: {**connection.state(), "r": r, "v": v}
    )
    conn.vessel.auto_pilot.error = pointing_error
    waits, warps = [], []

    def align(sc, ut, *args, **kwargs):
        waits.append(ut)
        sc.ut = ut
        conn.vessel.auto_pilot.error = 0.0

    monkeypatch.setattr(live, "wait_for_ut", align)
    monkeypatch.setattr(
        live, "bounded_warp", lambda *args, **kwargs: warps.append(args[1])
    )
    live.coast_to_capture(conn, conn.vessel, c, log=Log())
    assert (
        len(waits) == 1
    ), "Coasting warp must wait until the pointing error magnitude is small"
    assert len(warps) == 1 and conn.vessel.control.throttle == 0.0
    assert math.isnan(conn.vessel.auto_pilot.target_roll)


def test_read_only_cli_plans_without_flight_calls(tmp_path, monkeypatch):
    import krpc
    from orbiter import mission_nodes

    conn = Connection()
    monkeypatch.setattr(krpc, "connect", lambda **kwargs: conn)
    monkeypatch.setattr(
        mission_nodes,
        "plan_live_transfer",
        lambda *args, **kwargs: (
            SimpleNamespace(to_dict=lambda: {"target": "Mun"}),
            {"available_delta_v": 2000.0},
        ),
    )
    monkeypatch.setattr(
        live,
        "fly_mission",
        lambda *args, **kwargs: pytest.fail("Read-only invocation attempted flight"),
    )
    output = tmp_path / "plan"
    assert (
        live.main(
            [
                "--target",
                "Mun",
                "--output-dir",
                str(output),
                "--address",
                f"mission-test-{uuid4().hex}.invalid",
            ]
        )
        == 0
    )
    result = json.loads((output / "mission.json").read_text())
    assert result["planning_complete"] and not result["success"]
    assert not result["end_to_end_from_launch_verified"]
    assert conn.vessel.auto_pilot.engages == conn.vessel.auto_pilot.disengages == 0
    assert conn.closed


def test_flight_controller_must_be_selected_before_any_mutation():
    with pytest.raises(SystemExit):
        live.parse_args(["--target", "Mun", "--output-dir", "unused", "--fly"])
    conn = Connection()
    with pytest.raises(ValueError, match="before flight"):
        live.fly_mission(conn, "Mun", 50000.0, log=Log())
    assert conn.vessel.auto_pilot.disengages == 0


def test_failure_removes_only_the_node_created_for_this_mission(monkeypatch):
    from orbiter import mission_nodes

    conn = Connection()
    monkeypatch.setattr(live, "_state", lambda connection: connection.state())
    counts = dict(owned=0, other=0)
    owned = SimpleNamespace(
        remove=lambda: counts.__setitem__("owned", counts["owned"] + 1)
    )
    other = SimpleNamespace(
        remove=lambda: counts.__setitem__("other", counts["other"] + 1)
    )
    conn.vessel.control.nodes = [other]
    plan = SimpleNamespace(
        source="Mun", estimated_total_delta_v=50.0, to_dict=lambda: {}
    )
    monkeypatch.setattr(
        mission_nodes,
        "plan_live_transfer",
        lambda *args, **kwargs: (plan, {"reserve_delta_v": 100.0}),
    )
    monkeypatch.setattr(
        mission_nodes, "prepare_encounter_node", lambda *args, **kwargs: (owned, {})
    )
    monkeypatch.setattr(
        live,
        "execute_node",
        lambda *args, **kwargs: {"success": False, "error": "test"},
    )
    with pytest.raises(RuntimeError, match="Maneuver execution failed"):
        live.fly_mission(conn, "Minmus", 50000.0, log=Log(), scripted=True)
    assert counts == dict(owned=1, other=0)


def test_warp_timeout_resets_both_factors():
    def timeout(*args, **kwargs):
        raise TimeoutError("test warp timeout")

    sc = SimpleNamespace(
        ut=0.0, rails_warp_factor=7, physics_warp_factor=2, warp_to=timeout
    )
    with pytest.raises(TimeoutError):
        live.bounded_warp(sc, 100.0)
    assert sc.rails_warp_factor == sc.physics_warp_factor == 0


def test_full_pipeline_requires_target_soi_capture_and_independent_period(monkeypatch):
    from orbiter import mission_nodes

    conn, log = Connection(), Log()
    destination = conn.body
    source = SimpleNamespace(
        name="Kerbin",
        gravitational_parameter=3.5316e12,
        non_rotating_reference_frame="Kerbin inertial",
    )
    conn.body = source
    conn.vessel.orbit.body = source
    conn.vessel.orbit.next_orbit = SimpleNamespace(body=destination)
    conn.vessel.orbit.time_to_soi_change = 20.0
    monkeypatch.setattr(live, "_state", lambda connection: connection.state())
    plan = SimpleNamespace(
        source="Kerbin",
        estimated_total_delta_v=1000.0,
        to_dict=lambda: {"source": "Kerbin", "target": "Mun"},
    )
    monkeypatch.setattr(
        mission_nodes,
        "plan_live_transfer",
        lambda *args, **kwargs: (plan, {"reserve_delta_v": 100.0}),
    )
    removed, burn_bodies = [], []
    node = SimpleNamespace(remove=lambda: removed.append("owned"))
    monkeypatch.setattr(
        mission_nodes,
        "prepare_encounter_node",
        lambda *args, **kwargs: (node, {"encounter": True, "target": "Mun"}),
    )
    monkeypatch.setattr(
        mission_nodes, "correct_periapsis_node", lambda *args, **kwargs: None
    )

    def burn(*args, **kwargs):
        burn_bodies.append(kwargs["expected_body"])
        return dict(success=True)

    monkeypatch.setattr(live, "execute_node", burn)

    def warp(sc, ut, **kwargs):
        sc.ut = ut
        conn.body = destination
        conn.vessel.orbit.body = destination

    monkeypatch.setattr(live, "bounded_warp", warp)

    def incoming(*args, **kwargs):
        assert conn.vessel.orbit.body.name == "Mun"
        conn.vessel.auto_pilot.engage()
        return dict(autopilot_prepared=True)

    monkeypatch.setattr(live, "coast_to_capture", incoming)
    monkeypatch.setattr(
        live,
        "wait_for_ut",
        lambda sc, target, *args, **kwargs: setattr(sc, "ut", target),
    )
    result = live.fly_mission(conn, "Mun", 50000.0, log=log, scripted=True)
    assert result["success"] and result["capture"]["success"]
    assert result["independent_coast"]["success"]
    assert result["independent_coast"]["observed_coast_s"] == 1000.0
    assert burn_bodies == ["Kerbin"] and removed == ["owned"]
    assert [row["stage"] for row in result["handoffs"]] == [
        "actual_target_soi",
        "actual_target_orbit",
    ]
    assert not result["end_to_end_from_launch_verified"]


def window_retry_rig(monkeypatch, paths):
    """Use real node-admissibility/cleanup with fake patched conics and burns."""
    from orbiter import mission_nodes

    conn, log, windows, nodes, burns = Connection(), Log(), [], [], []
    conn.body.name = "Kerbin"

    def patch(names):
        if not names:
            return None
        return SimpleNamespace(
            body=SimpleNamespace(name=names[0]),
            periapsis_altitude=50000.0,
            apoapsis_altitude=-1e6,
            eccentricity=1.2,
            next_orbit=patch(names[1:]),
        )

    def plan(*args, **kwargs):
        window = kwargs.get("coast_orbits", 0)
        windows.append(window)
        candidate = SimpleNamespace(
            source="Kerbin",
            target="Minmus",
            kind="direct_child",
            departure_ut=1000.0 + window * 2000.0,
            arrival_ut=20000.0 + window * 2000.0,
            departure_position=[700000.0, 0.0, 0.0],
            departure_velocity=[0.0, 0.0, 0.0],
            aim_offset=np.array([0.0, 1.0, 0.0]),
            delta_v=[0.0, 100.0, 0.0],
            target_altitude=50000.0,
            estimated_total_delta_v=1000.0,
            to_dict=lambda: {"coast_orbits": window},
        )
        return candidate, {"reserve_delta_v": 100.0}

    def create(*args, **kwargs):
        node = SimpleNamespace(
            orbit=patch(paths[len(nodes)]), delta_v=100.0, removed=False
        )
        node.remove = lambda: setattr(node, "removed", True)
        nodes.append(node)
        return node

    def burn(connection, vessel, node, **kwargs):
        burns.append(node)
        return {"success": False, "error": "fake stop after verified departure"}

    monkeypatch.setattr(mission_nodes, "plan_live_transfer", plan)
    monkeypatch.setattr(mission_nodes, "create_node", create)
    monkeypatch.setattr(mission_nodes, "set_node_vector", lambda *args: None)
    monkeypatch.setattr(
        mission_nodes,
        "lambert",
        lambda *args, **kwargs: (np.array([0.0, 100.0, 0.0]), np.zeros(3)),
    )
    monkeypatch.setattr(mission_nodes.time, "sleep", lambda *args: None)
    monkeypatch.setattr(live, "_state", lambda connection: connection.state())
    monkeypatch.setattr(live, "execute_node", burn)
    return conn, log, windows, nodes, burns


def test_later_window_rejects_mun_interception_then_accepts_direct_minmus(monkeypatch):
    conn, log, windows, nodes, burns = window_retry_rig(
        monkeypatch, [["Kerbin", "Mun", "Minmus"], ["Kerbin", "Minmus"]]
    )
    with pytest.raises(RuntimeError, match="fake stop after verified departure"):
        live.fly_mission(conn, "Minmus", 50000.0, log=log, scripted=True)
    assert windows == [0, 1]
    assert len(nodes) == 2 and all(node.removed for node in nodes)
    assert burns == [nodes[1]]
    rejects = [row for row in log.events if row["event"] == "mission_window_rejected"]
    assert len(rejects) == 1 and rejects[0]["coast_orbits"] == 0
    assert any(row["event"] == "encounter_node_verified" for row in log.events)


def test_all_eight_unavailable_windows_remove_their_nodes_without_burning(monkeypatch):
    from orbiter.mission_nodes import EncounterUnavailable

    conn, log, windows, nodes, burns = window_retry_rig(
        monkeypatch, [["Kerbin", "Mun", "Minmus"]] * 8
    )
    with pytest.raises(EncounterUnavailable, match="8.*window"):
        live.fly_mission(conn, "Minmus", 50000.0, log=log, scripted=True)
    assert (
        windows == list(range(8))
        and len(nodes) == 8
        and all(node.removed for node in nodes)
    )
    assert not burns and conn.vessel.control.throttle == 0.0
    assert (
        len([row for row in log.events if row["event"] == "mission_window_rejected"])
        == 8
    )


@pytest.mark.parametrize(
    "error",
    [
        ValueError("invalid telemetry"),
        TimeoutError("RPC stalled"),
        RuntimeError("frame mismatch"),
        ValueError("pre-existing node"),
        ValueError("insufficient fuel"),
    ],
)
def test_unrelated_window_error_is_not_retried(monkeypatch, error):
    from orbiter import mission_nodes

    conn, log, windows, nodes, burns = window_retry_rig(
        monkeypatch, [["Kerbin", "Minmus"]]
    )

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(mission_nodes, "prepare_encounter_node", fail)
    with pytest.raises(type(error), match=str(error)):
        live.fly_mission(conn, "Minmus", 50000.0, log=log, scripted=True)
    assert windows == [0] and not nodes and not burns


def test_window_retry_stops_at_parent_wall_deadline(monkeypatch):
    from orbiter import mission_nodes

    conn, log, windows, nodes, burns = window_retry_rig(
        monkeypatch, [["Kerbin", "Mun", "Minmus"]]
    )
    elapsed = [0.0]
    monkeypatch.setattr(live.time, "monotonic", lambda: elapsed[0])
    prepare = mission_nodes.prepare_encounter_node

    def rejected(*args, **kwargs):
        try:
            return prepare(*args, **kwargs)
        finally:
            elapsed[0] = 2.0

    monkeypatch.setattr(mission_nodes, "prepare_encounter_node", rejected)
    with pytest.raises(TimeoutError, match="Mission wall-clock"):
        live.fly_mission(
            conn, "Minmus", 50000.0, log=log, scripted=True, wall_timeout=1.0
        )
    assert windows == [0] and len(nodes) == 1 and nodes[0].removed and not burns


def test_initial_vessel_lookup_is_bounded_and_failure_is_recorded(monkeypatch):
    depth, log = [], Log()

    @contextmanager
    def deadline(seconds):
        depth.append(seconds)
        try:
            yield
        finally:
            depth.pop()

    class SpaceCenter:
        @property
        def active_vessel(self):
            assert depth, "active_vessel RPC must be bounded"
            raise TimeoutError("initial vessel RPC")

    monkeypatch.setattr(live, "rpc_deadline", deadline)
    with pytest.raises(TimeoutError, match="initial vessel RPC"):
        live.fly_mission(
            SimpleNamespace(space_center=SpaceCenter()),
            "Mun",
            50000.0,
            log=log,
            scripted=True,
        )
    assert log.events[-1]["event"] == "mission_end"
    assert not log.events[-1]["success"]


def test_capture_configuration_body_reads_have_a_deadline(monkeypatch):
    conn, depth = Connection(), []
    body = conn.body

    @contextmanager
    def deadline(seconds):
        depth.append(seconds)
        try:
            yield
        finally:
            depth.pop()

    class GuardedBodies:
        def __getitem__(self, key):
            assert depth, "body lookup RPC must be bounded"
            return body

    conn.space_center.bodies = GuardedBodies()
    monkeypatch.setattr(live, "rpc_deadline", deadline)
    monkeypatch.setattr(live, "_state", lambda connection: connection.state())
    assert live.capture_config(conn, "Mun", 50000.0).body_name == "Mun"


def test_connection_close_failure_preserves_completed_summary(tmp_path, monkeypatch):
    import krpc
    from orbiter import mission_nodes

    conn = Connection()

    def failed_close():
        raise TimeoutError("socket close stalled")

    conn.close = failed_close
    monkeypatch.setattr(krpc, "connect", lambda **kwargs: conn)
    monkeypatch.setattr(
        mission_nodes,
        "plan_live_transfer",
        lambda *args, **kwargs: (
            SimpleNamespace(to_dict=lambda: {"target": "Mun"}),
            {"available_delta_v": 2000.0},
        ),
    )
    assert (
        live.main(
            [
                "--target",
                "Mun",
                "--output-dir",
                str(tmp_path),
                "--address",
                f"mission-test-{uuid4().hex}.invalid",
            ]
        )
        == 0
    )
    result = json.loads((tmp_path / "mission.json").read_text())
    assert result["planning_complete"]
    assert "socket close stalled" in result["connection_close_error"]


def launch_connection():
    conn, launched = Connection(), []
    conn.vessel.situation = "VesselSituation.pre_launch"
    part = SimpleNamespace(name="mk1pod", title="Mk1 Command Pod", vessel=conn.vessel)
    part.position = lambda frame: (conn.body.equatorial_radius + 86.5, 0.0, 0.0)
    conn.vessel.parts = SimpleNamespace(controlling=part, root=part, all=[part])
    conn.vessel.crew = [SimpleNamespace(name="Valentina Kerman")]
    conn.vessel.flight = lambda: SimpleNamespace(
        latitude=0.0, longitude=0.0, mean_altitude=86.5
    )
    conn.space_center.launch_vessel_from_vab = lambda name, **kwargs: launched.append(
        (name, kwargs)
    )
    return conn, part, launched


def active_transfer_pad():
    from orbiter.env import EnvConfig
    from pathlib import Path

    conn, part, launched = launch_connection()
    config = EnvConfig.load(
        Path(__file__).resolve().parents[1] / "orbiter/configs/orbiter_rl_transfer.json"
    )
    profile = json.loads(
        (
            Path(__file__).resolve().parents[1] / "orbiter/data/transfer_vehicle.json"
        ).read_text()
    )
    conn.body.name = "Kerbin"
    conn.body.gravitational_parameter = config.mu
    conn.body.equatorial_radius = config.radius
    conn.vessel.mass = 14310.0
    conn.vessel.control.current_stage = 4
    conn.vessel.control.state = "ControlState.full"
    conn.vessel.control.source = "ControlSource.kerbal"
    conn.vessel.control.nodes = []
    conn.vessel.parts.all = [SimpleNamespace(name=name) for name in profile["parts"]]
    conn.vessel.parts.decouplers = [
        SimpleNamespace(
            part=SimpleNamespace(name="Decoupler.1", stage=stage, decouple_stage=stage)
        )
        for stage in (1, 2)
    ]
    conn.vessel.parts.engines = [
        SimpleNamespace(
            part=SimpleNamespace(name=name, stage=stage, decouple_stage=stage - 1),
            active=False,
            has_fuel=True,
            available_thrust=thrust,
            max_vacuum_thrust=thrust,
            vacuum_specific_impulse=isp,
            thrust_limit=1.0,
            can_shutdown=True,
            propellant_names=["LiquidFuel", "Oxidizer"],
        )
        for name, stage, thrust, isp in (
            ("liquidEngine2", 3, 215000.0, 320.0),
            ("liquidEngine3.v2", 2, 60000.0, 345.0),
        )
    ]

    def fuel(stage, cumulative=False):
        mass = {2: 6000.0, 1: 4000.0}[stage]
        return SimpleNamespace(
            names=["LiquidFuel", "Oxidizer"],
            amount=lambda name: mass * (0.09 if name == "LiquidFuel" else 0.11),
            density=lambda name: 5.0,
        )

    conn.vessel.resources_in_decouple_stage = fuel
    # Actual inactive-engine available thrust at the Kerbin pad is atmospheric,
    # while max_vacuum_thrust remains the engine rating used by the model.
    conn.vessel.parts.engines[0].available_thrust = 168405.21875
    conn.vessel.parts.engines[1].available_thrust = 15104.4863
    return conn, config, part, launched


def test_active_pad_inspection_and_launch_never_load_or_create_a_craft(monkeypatch):
    conn, config, part, launched = active_transfer_pad()
    conn.space_center.load = lambda *args: pytest.fail(
        "Active pad attempted a save load"
    )
    preflight = live.inspect_active_pad(conn, config)
    assert preflight["ready"] and not preflight["failures"]
    assert (
        preflight["supported_profile"] == "OrbiterRL Transfer"
        and len(preflight["part_names"]) == 20
    )
    assert conn.vessel.auto_pilot.engages == conn.vessel.auto_pilot.disengages == 0
    monkeypatch.setattr(live, "fly", lambda *args, **kwargs: {"success": True})
    result, original = live.launch_ascent(
        conn,
        config,
        object(),
        plan=(3, 2),
        fuel_stages=(2, 1),
        log=Log(),
        launch_active=True,
    )
    assert result["success"] and original is part and not launched
    assert result["active_pad_preflight"]["ready"]


@pytest.mark.parametrize(
    "mismatch",
    [
        "unstaged_engines",
        "swapped_engines",
        "engine_decouple",
        "propellant",
        "engine_count",
        "decoupler_stage",
        "decoupler_group",
        "decoupler_count",
        "active_engine",
        "unavailable_thrust",
        "nan_thrust",
        "thrust_limit",
        "vacuum_thrust",
        "vacuum_isp",
        "nan_limit",
        "infinite_vacuum_thrust",
        "nan_vacuum_isp",
        "no_engine_fuel",
        "engine_cannot_shutdown",
    ],
)
def test_active_pad_rejects_changed_staging_or_propulsion_before_controls(
    monkeypatch, mismatch
):
    conn, config, part, launched = active_transfer_pad()
    engines = conn.vessel.parts.engines
    decouplers = conn.vessel.parts.decouplers
    if mismatch == "unstaged_engines":
        for engine in engines:
            engine.part.stage = 0
    elif mismatch == "swapped_engines":
        engines[0].part.stage, engines[1].part.stage = 2, 3
    elif mismatch == "engine_decouple":
        engines[0].part.decouple_stage = 1
    elif mismatch == "propellant":
        engines[0].propellant_names = ["LiquidFuel"]
    elif mismatch == "engine_count":
        engines.pop()
    elif mismatch == "decoupler_stage":
        decouplers[0].part.stage = 2
    elif mismatch == "decoupler_group":
        decouplers[0].part.decouple_stage = 2
    elif mismatch == "decoupler_count":
        decouplers.pop()
    elif mismatch == "active_engine":
        engines[0].active = True
    elif mismatch == "unavailable_thrust":
        for engine in engines:
            engine.available_thrust = 0.0
    elif mismatch == "nan_thrust":
        engines[0].available_thrust = float("nan")
    elif mismatch == "thrust_limit":
        engines[0].thrust_limit = 0.5
    elif mismatch == "vacuum_thrust":
        engines[0].max_vacuum_thrust *= 0.5
    elif mismatch == "vacuum_isp":
        engines[0].vacuum_specific_impulse *= 0.5
    elif mismatch == "nan_limit":
        engines[0].thrust_limit = float("nan")
    elif mismatch == "infinite_vacuum_thrust":
        engines[0].max_vacuum_thrust = float("inf")
    elif mismatch == "nan_vacuum_isp":
        engines[0].vacuum_specific_impulse = float("nan")
    elif mismatch == "no_engine_fuel":
        engines[0].has_fuel = False
    elif mismatch == "engine_cannot_shutdown":
        engines[0].can_shutdown = False
    monkeypatch.setattr(
        live, "fly", lambda *args, **kwargs: pytest.fail("Rejected pad started flight")
    )
    assert not live.inspect_active_pad(conn, config)["ready"]
    with pytest.raises(ValueError, match="Active pad"):
        live.launch_ascent(
            conn,
            config,
            object(),
            plan=(3, 2),
            fuel_stages=(2, 1),
            log=Log(),
            launch_active=True,
        )
    assert not launched
    assert conn.vessel.auto_pilot.engages == conn.vessel.auto_pilot.disengages == 0


@pytest.mark.parametrize(
    "mismatch",
    [
        "part",
        "mass",
        "body",
        "stage",
        "situation",
        "source",
        "control",
        "crew",
        "fuel",
        "model",
    ],
)
def test_active_pad_rejects_unsupported_or_uncontrollable_ship_before_any_controls(
    monkeypatch, mismatch
):
    conn, config, part, launched = active_transfer_pad()
    if mismatch == "part":
        conn.vessel.parts.all[0].name = "unsupported-part"
    elif mismatch == "mass":
        conn.vessel.mass = 10000.0
    elif mismatch == "body":
        conn.body.gravitational_parameter *= 2
    elif mismatch == "stage":
        conn.vessel.control.current_stage = 3
    elif mismatch == "situation":
        conn.vessel.situation = "VesselSituation.orbiting"
    elif mismatch == "source":
        conn.vessel.control.source = "ControlSource.none"
    elif mismatch == "control":
        conn.vessel.control.state = "ControlState.partial"
    elif mismatch == "crew":
        conn.vessel.crew = []
    elif mismatch == "fuel":
        conn.vessel.resources_in_decouple_stage = (
            lambda *args, **kwargs: SimpleNamespace(names=[])
        )
    elif mismatch == "model":
        config.drag_area *= 2
    monkeypatch.setattr(
        live,
        "fly",
        lambda *args, **kwargs: pytest.fail("Rejected active pad started flight"),
    )
    preflight = live.inspect_active_pad(conn, config)
    assert not preflight["ready"] and preflight["failures"]
    with pytest.raises(ValueError, match="Active pad"):
        live.launch_ascent(
            conn,
            config,
            object(),
            plan=(3, 2),
            fuel_stages=(2, 1),
            log=Log(),
            launch_active=True,
        )
    assert (
        not launched
        and conn.vessel.auto_pilot.engages == conn.vessel.auto_pilot.disengages == 0
    )


def test_launch_active_cli_requires_explicit_model_stage_and_no_save_options():
    valid = [
        "--fly",
        "--target",
        "Mun",
        "--output-dir",
        "unused",
        "--capture-scripted",
        "--launch-active",
        "--launch-model",
        "launch.zip",
        "--stage-plan",
        "3,2",
        "--fuel-stages",
        "2,1",
    ]
    assert live.parse_args(valid).launch_active
    for extra in (
        ["--ksp-root", "/tmp"],
        ["--pad-save-name", "pad"],
        ["--launch-pad-save", "pad"],
        ["--launch-craft", "craft"],
    ):
        with pytest.raises(SystemExit):
            live.parse_args(valid + extra)
    for flag, value in (
        ("--launch-model", "launch.zip"),
        ("--stage-plan", "3,2"),
        ("--fuel-stages", "2,1"),
    ):
        shortened = valid.copy()
        shortened.remove(flag)
        shortened.remove(value)
        with pytest.raises(SystemExit):
            live.parse_args(shortened)


def pad_snapshot(
    tmp_path, *, name="crewed-pad", craft="Mission vessel", situation="PRELAUNCH"
):
    folder = tmp_path / "saves" / "OrbiterRL"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (name + ".sfs")
    path.write_text(
        "GAME\n{\n FLIGHTSTATE\n {\n UT = 100\n activeVessel = 0\n VESSEL\n {\n name = "
        + craft
        + "\n sit = "
        + situation
        + "\n stg = 4\n lat = 0\n lon = 0\n alt = 86.5\n PART\n {\n name = mk1pod\n crew = Valentina Kerman\n }\n }\n }\n}\n"
    )
    return path


def test_named_pad_load_uses_same_connection_handoff_without_craft_launch(
    tmp_path, monkeypatch
):
    from orbiter.env import EnvConfig

    conn, part, launched = launch_connection()
    config = EnvConfig()
    conn.body.gravitational_parameter = config.mu
    conn.body.equatorial_radius = config.radius
    conn.vessel.mass = config.payload_mass + sum(
        stage.dry_mass + stage.fuel_mass for stage in config.stages
    )
    snapshot = pad_snapshot(tmp_path, craft=conn.vessel.name)
    before = snapshot.read_bytes()
    loads = []
    deadlines = []
    conn.space_center.load = lambda name: loads.append(name)
    conn.vessel.control.current_stage = 4
    original = live.rpc_deadline

    @contextmanager
    def deadline(seconds):
        deadlines.append(seconds)
        with original(seconds):
            yield

    monkeypatch.setattr(live, "rpc_deadline", deadline)

    def ascent(connection, *args, **kwargs):
        assert connection is conn
        return {"success": True}

    monkeypatch.setattr(live, "fly", ascent)
    result, original_part = live.launch_ascent(
        conn,
        config,
        object(),
        None,
        plan=(3, 2),
        fuel_stages=(2, 1),
        log=Log(),
        launch_pad_save="crewed-pad",
        ksp_root=tmp_path,
    )
    assert result["success"] and original_part is part
    assert loads == ["crewed-pad"] and not launched and 120.0 in deadlines
    assert result["loaded_pad_save"]["name"] == "crewed-pad"
    assert result["snapshot_state_consistent"] is True
    assert snapshot.read_bytes() == before


def test_snapshot_basename_collision_in_another_save_is_rejected_before_rpc(tmp_path):
    chosen = pad_snapshot(tmp_path)
    other = tmp_path / "saves" / "CS5100"
    other.mkdir()
    (other / chosen.name).write_bytes(chosen.read_bytes())
    with pytest.raises(ValueError, match="other.*save|ambiguous"):
        live.launch_pad_snapshot(tmp_path, "crewed-pad")


@pytest.mark.parametrize(
    "mean_altitude,root_altitude",
    [
        (79.90004638244864, 86.71293014497496),
        (79.78000262274873, 86.58666763245128),
        (79.7771716582356, 86.57977898686659),
    ],
)
def test_saved_pad_altitude_matches_root_origin_not_live_center_of_mass(
    tmp_path, mean_altitude, root_altitude
):
    """Actual KSP three-sample pad reload regression, captured 2026-10-18."""
    from orbiter.env import EnvConfig

    conn, part, _ = launch_connection()
    config = EnvConfig()
    conn.body.gravitational_parameter = config.mu
    conn.body.equatorial_radius = config.radius
    conn.vessel.control.current_stage = 4
    source = pad_snapshot(tmp_path, craft=conn.vessel.name)
    source.write_text(
        source.read_text().replace("alt = 86.5", "alt = 86.50665373238735")
    )
    snapshot = live.launch_pad_snapshot(tmp_path, "crewed-pad")
    conn.vessel.flight = lambda: SimpleNamespace(
        latitude=0.0, longitude=0.0, mean_altitude=mean_altitude
    )
    part.position = lambda frame: (config.radius + root_altitude, 0.0, 0.0)
    result = live.verify_loaded_pad_state(
        conn.space_center, conn.vessel, config, snapshot, elapsed_wall=4.14547
    )
    assert result["snapshot_state_consistent"]
    assert result["loaded_state"]["altitude"] == pytest.approx(root_altitude)
    assert result["loaded_state"]["mean_altitude"] == mean_altitude
    assert result["loaded_state"]["altitude_reference"] == "root_part_origin"
    assert result["tolerances"]["altitude_metres"] == 5.0


def test_wrong_root_altitude_is_rejected_even_when_center_of_mass_matches(tmp_path):
    from orbiter.env import EnvConfig

    conn, part, _ = launch_connection()
    config = EnvConfig()
    conn.body.gravitational_parameter = config.mu
    conn.body.equatorial_radius = config.radius
    conn.vessel.control.current_stage = 4
    pad_snapshot(tmp_path, craft=conn.vessel.name)
    snapshot = live.launch_pad_snapshot(tmp_path, "crewed-pad")
    part.position = lambda frame: (config.radius + 100.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="snapshot state differs"):
        live.verify_loaded_pad_state(
            conn.space_center, conn.vessel, config, snapshot, elapsed_wall=1.0
        )


@pytest.mark.parametrize(
    "mismatch", ["ut", "crew", "parts", "location", "stage", "body"]
)
def test_loaded_snapshot_state_mismatch_is_rejected_without_ascent(
    tmp_path, monkeypatch, mismatch
):
    from orbiter.env import EnvConfig

    conn, part, _ = launch_connection()
    config = EnvConfig()
    conn.body.gravitational_parameter = config.mu
    conn.body.equatorial_radius = config.radius
    conn.vessel.mass = config.payload_mass + sum(
        stage.dry_mass + stage.fuel_mass for stage in config.stages
    )
    conn.vessel.control.current_stage = 4
    conn.space_center.load = lambda name: None
    pad_snapshot(tmp_path, craft=conn.vessel.name)
    if mismatch == "ut":
        conn.space_center.ut = 1000.0
    elif mismatch == "crew":
        conn.vessel.crew = []
    elif mismatch == "parts":
        conn.vessel.parts.all = []
    elif mismatch == "location":
        conn.vessel.flight = lambda: SimpleNamespace(
            latitude=1.0, longitude=0.0, mean_altitude=86.5
        )
    elif mismatch == "stage":
        conn.vessel.control.current_stage = 3
    elif mismatch == "body":
        conn.body.equatorial_radius *= 2
    monkeypatch.setattr(
        live,
        "fly",
        lambda *args, **kwargs: pytest.fail("Snapshot mismatch started ascent"),
    )
    with pytest.raises(ValueError, match="snapshot|staging"):
        live.launch_ascent(
            conn,
            config,
            object(),
            None,
            plan=(3, 2),
            fuel_stages=(2, 1),
            log=Log(),
            launch_pad_save="crewed-pad",
            ksp_root=tmp_path,
        )


@pytest.mark.parametrize(
    "name", ["quicksave", "persistent", "../other", "path/pad", "path\\pad", "absent"]
)
def test_named_pad_invalid_name_or_missing_file_fails_before_rpc(tmp_path, name):
    from orbiter.env import EnvConfig

    pad_snapshot(tmp_path)
    with pytest.raises((ValueError, FileNotFoundError)):
        live.launch_ascent(
            None,
            EnvConfig(),
            object(),
            None,
            plan=(3, 2),
            fuel_stages=(2, 1),
            log=Log(),
            launch_pad_save=name,
            ksp_root=tmp_path,
        )


def test_named_pad_requires_root_and_prelaunch_snapshot_before_rpc(tmp_path):
    from orbiter.env import EnvConfig

    for root in (None, tmp_path):
        with pytest.raises((ValueError, FileNotFoundError)):
            live.launch_ascent(
                None,
                EnvConfig(),
                object(),
                None,
                plan=(3, 2),
                fuel_stages=(2, 1),
                log=Log(),
                launch_pad_save="crewed-pad",
                ksp_root=root,
            )
    pad_snapshot(tmp_path, situation="ORBITING")
    with pytest.raises(ValueError, match="pre.launch"):
        live.launch_ascent(
            None,
            EnvConfig(),
            object(),
            None,
            plan=(3, 2),
            fuel_stages=(2, 1),
            log=Log(),
            launch_pad_save="crewed-pad",
            ksp_root=tmp_path,
        )


def test_pad_load_cli_contract_is_explicit_and_mutually_exclusive(tmp_path):
    pad_snapshot(tmp_path)
    valid = [
        "--target",
        "Minmus",
        "--output-dir",
        "unused",
        "--fly",
        "--capture-scripted",
        "--launch-model",
        "launch.zip",
        "--launch-pad-save",
        "crewed-pad",
        "--ksp-root",
        str(tmp_path),
        "--stage-plan",
        "3,2",
        "--fuel-stages",
        "2,1",
    ]
    parsed = live.parse_args(valid)
    assert parsed.launch_pad_save == "crewed-pad" and parsed.launch_craft is None
    for extra in (["--launch-craft", "another"], ["--pad-save-name", "new-pad"]):
        with pytest.raises(SystemExit):
            live.parse_args(valid + extra)
    with pytest.raises(SystemExit):
        live.parse_args([value for value in valid if value != "--fly"])


@pytest.mark.parametrize("mass", [3000.0, float("nan")])
def test_loaded_pad_model_mass_mismatch_never_starts_ascent(
    tmp_path, monkeypatch, mass
):
    from orbiter.env import EnvConfig

    conn, _, launched = launch_connection()
    pad_snapshot(tmp_path, craft=conn.vessel.name)
    conn.space_center.load = lambda name: None
    conn.vessel.mass = mass
    conn.vessel.control.current_stage = 4
    monkeypatch.setattr(
        live,
        "fly",
        lambda *args, **kwargs: pytest.fail("Invalid pad mass started ascent"),
    )
    with pytest.raises(ValueError, match="wet mass"):
        live.launch_ascent(
            conn,
            EnvConfig(),
            object(),
            None,
            plan=(3, 2),
            fuel_stages=(2, 1),
            log=Log(),
            launch_pad_save="crewed-pad",
            ksp_root=tmp_path,
        )
    assert not launched


def test_pad_snapshot_cannot_resolve_outside_dedicated_save(tmp_path):
    path = pad_snapshot(tmp_path)
    outside = tmp_path / "other.sfs"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="only.*OrbiterRL"):
        live.launch_pad_snapshot(tmp_path, "crewed-pad")


def test_fresh_launch_preserves_controlling_part_and_prefixes_ascent_log(monkeypatch):
    from orbiter.env import EnvConfig

    conn, part, launched = launch_connection()

    def ascent(connection, config, **kwargs):
        assert connection is conn and kwargs["model"] == "learned launch"
        kwargs["log"].write("action", throttle=0.7)
        return dict(success=True, termination_reason="orbit_achieved")

    monkeypatch.setattr(live, "fly", ascent)
    log = Log()
    result, original_part = live.launch_ascent(
        conn,
        EnvConfig(),
        "learned launch",
        conn.vessel.name,
        plan=(3, 2),
        fuel_stages=(2, 1),
        log=log,
    )
    assert result["success"] and original_part is part
    assert launched == [(conn.vessel.name, {"recover": False})]
    assert result["controlling_part"] == result["parking_controlling_part"]
    assert any(
        row["event"] == "ascent_action" and row["phase"] == "ascent"
        for row in log.events
    )
    assert (
        conn.vessel.control.throttle == 0.0 and conn.vessel.auto_pilot.disengages == 1
    )


def test_fresh_launch_rejects_wrong_handoff_vessel_and_releases_controls(monkeypatch):
    from orbiter.env import EnvConfig

    conn, part, launched = launch_connection()

    def ascent(*args, **kwargs):
        part.vessel = SimpleNamespace(
            parts=SimpleNamespace(controlling=part),
            control=SimpleNamespace(throttle=0.7),
            auto_pilot=AutoPilot(),
        )
        return dict(success=True)

    monkeypatch.setattr(live, "fly", ascent)
    with pytest.raises(RuntimeError, match="controlling part"):
        live.launch_ascent(
            conn,
            EnvConfig(),
            object(),
            conn.vessel.name,
            plan=(3, 2),
            fuel_stages=(2, 1),
            log=Log(),
        )
    assert len(launched) == 1 and conn.vessel.control.throttle == 0.0
    assert part.vessel.control.throttle == 0.0


def test_launch_cleanup_follows_original_part_after_serial_staging(monkeypatch):
    from orbiter.env import EnvConfig

    conn, part, _ = launch_connection()
    upper = SimpleNamespace(
        name="Upper stage",
        parts=SimpleNamespace(controlling=part),
        control=SimpleNamespace(throttle=0.6),
        auto_pilot=AutoPilot(),
    )

    def ascent(*args, **kwargs):
        part.vessel = upper
        conn.space_center.active_vessel = upper
        return dict(success=True)

    monkeypatch.setattr(live, "fly", ascent)
    result, _ = live.launch_ascent(
        conn,
        EnvConfig(),
        object(),
        conn.vessel.name,
        plan=(3, 2),
        fuel_stages=(2, 1),
        log=Log(),
    )
    assert result["success"] and not result["cleanup_errors"]
    assert upper.control.throttle == 0.0 and upper.auto_pilot.disengages == 1
    assert conn.vessel.auto_pilot.disengages == 0


def test_launch_stage_contract_is_checked_before_launch(monkeypatch):
    from orbiter.env import EnvConfig

    conn, _, launched = launch_connection()
    with pytest.raises(ValueError, match="stage mappings"):
        live.launch_ascent(
            conn,
            EnvConfig(),
            object(),
            conn.vessel.name,
            plan=(3, 2),
            fuel_stages=(2,),
            log=Log(),
        )
    assert not launched


@pytest.mark.parametrize(
    "extra",
    [
        ["--launch-model", "launch.zip"],
        ["--stage-plan", "3,2"],
        [
            "--launch-model",
            "launch.zip",
            "--launch-craft",
            "craft",
            "--stage-plan",
            "3,2",
            "--fuel-stages",
            "2,1",
        ],
    ],
)
def test_launch_cli_requires_fly_and_complete_explicit_launch_contract(extra):
    with pytest.raises(SystemExit):
        live.parse_args(
            ["--target", "Mun", "--output-dir", "unused", "--capture-scripted", *extra]
        )


def test_pad_save_refuses_existing_name_before_launch(tmp_path):
    from orbiter.env import EnvConfig

    save = tmp_path / "saves" / "OrbiterRL"
    save.mkdir(parents=True)
    existing = save / "moon-pad.sfs"
    existing.write_text("previous save")
    conn, _, launched = launch_connection()
    with pytest.raises(FileExistsError, match="overwrite"):
        live.launch_ascent(
            conn,
            EnvConfig(),
            object(),
            conn.vessel.name,
            plan=(3, 2),
            fuel_stages=(2, 1),
            log=Log(),
            pad_save_name="moon-pad",
            ksp_root=tmp_path,
        )
    assert existing.read_text() == "previous save" and not launched
    for name in ("persistent", "quicksave", "../escape"):
        with pytest.raises(ValueError, match="dedicated"):
            live.check_new_pad_save(tmp_path, name)


@pytest.mark.parametrize("mission_success", [True, False])
@pytest.mark.parametrize("launch_source", ["craft", "pad", "active"])
def test_end_to_end_flag_requires_same_invocation_ascent_and_verified_mission(
    tmp_path, monkeypatch, mission_success, launch_source
):
    import krpc
    from orbiter.env import EnvConfig

    conn, part, _ = launch_connection()
    checkpoint = tmp_path / "launch.zip"
    checkpoint.write_bytes(b"immutable launch checkpoint")
    monkeypatch.setattr(krpc, "connect", lambda **kwargs: conn)
    monkeypatch.setattr(
        live,
        "load_controller",
        lambda path: (SimpleNamespace(num_timesteps=123), EnvConfig()),
    )

    def launch(connection, *args, **kwargs):
        assert connection is conn
        assert kwargs["launch_active"] is (launch_source == "active")
        assert kwargs["launch_pad_save"] == (
            "crewed-pad" if launch_source == "pad" else None
        )
        return dict(success=True, ascent={"success": True}), part

    def mission(connection, *args, **kwargs):
        assert connection is conn and kwargs["expected_control_part"] is part
        return dict(
            success=mission_success, independent_coast={"success": mission_success}
        )

    monkeypatch.setattr(live, "launch_ascent", launch)
    monkeypatch.setattr(live, "fly_mission", mission)
    directory = tmp_path / "flight"
    source_options = ["--launch-craft", conn.vessel.name]
    if launch_source == "pad":
        pad_snapshot(tmp_path, craft=conn.vessel.name)
        source_options = [
            "--launch-pad-save",
            "crewed-pad",
            "--ksp-root",
            str(tmp_path),
        ]
    elif launch_source == "active":
        source_options = ["--launch-active"]
    code = live.main(
        [
            "--address",
            f"mission-test-{uuid4().hex}.invalid",
            "--target",
            "Mun",
            "--output-dir",
            str(directory),
            "--fly",
            "--capture-scripted",
            "--launch-model",
            str(checkpoint),
            *source_options,
            "--stage-plan",
            "3,2",
            "--fuel-stages",
            "2,1",
        ]
    )
    result = json.loads((directory / "mission.json").read_text())
    assert code == (0 if mission_success else 1)
    assert result["end_to_end_from_launch_verified"] is mission_success
    assert result["launch_num_timesteps"] == 123 and result["launch_model_sha256"]
