"""Maneuver timing/control bounds with fake kRPC physics, never actual KSP."""

import math
from types import SimpleNamespace

import pytest

from orbiter.maneuvers import burn_duration, execute_node


class Log:
    def __init__(self):
        self.events = []

    def write(self, event, **data):
        self.events.append(dict(event=event, **data))


class Clock:
    def __init__(self):
        self.wall = 0.0
        self.sc = None
        self.paused = False
        self.no_progress = False

    def now(self):
        return self.wall

    def sleep(self, dt):
        self.wall += dt
        if self.paused:
            return
        self.sc.ut += dt
        vessel = self.sc.active_vessel
        throttle = vessel.control.throttle
        if throttle > 0:
            before = vessel.mass
            vessel.mass -= (
                vessel.available_thrust
                / (vessel.specific_impulse * 9.80665)
                * throttle
                * dt
            )
            if not self.no_progress:
                delta = (
                    vessel.specific_impulse * 9.80665 * math.log(before / vessel.mass)
                )
                vessel.node.remaining = max(0.0, vessel.node.remaining - delta)


class AutoPilot:
    def __init__(self, clock):
        self.clock = clock
        self.engages = self.disengages = 0
        self.error_at = lambda ut: 0.0

    @property
    def error(self):
        return self.error_at(self.clock.sc.ut)

    def engage(self):
        self.engages += 1

    def disengage(self):
        self.disengages += 1


class Control:
    def __init__(self, vessel):
        self.vessel = vessel
        self.commands = []
        self._throttle = 0.0
        self.sas = False

    @property
    def throttle(self):
        return self._throttle

    @throttle.setter
    def throttle(self, value):
        self.commands.append(
            (self.vessel.clock.sc.ut, value, self.vessel.auto_pilot.error)
        )
        self._throttle = value


class SpaceCenter:
    def __init__(self, clock):
        self.ut = 0.0
        self.rails_warp_factor = self.physics_warp_factor = 0
        self.warps = []
        self.warp_effect = None
        body = SimpleNamespace(
            name="Kerbin", non_rotating_reference_frame="Kerbin inertial"
        )
        self.active_vessel = SimpleNamespace(
            clock=clock,
            mass=1000.0,
            specific_impulse=345.0,
            available_thrust=10000.0,
            orbit=SimpleNamespace(body=body),
        )
        self.active_vessel.auto_pilot = AutoPilot(clock)
        self.active_vessel.control = Control(self.active_vessel)
        self.active_vessel.node = Node(body)

    def warp_to(self, ut, **kwargs):
        self.warps.append((ut, kwargs))
        self.rails_warp_factor = 7
        if self.warp_effect:
            self.warp_effect()
        else:
            self.ut = ut


class Node:
    def __init__(self, body):
        self.ut = 50.0
        self.remaining = 50.0
        self.orbit = SimpleNamespace(body=body)
        self.removes = 0

    def burn_vector(self, frame):
        assert frame == "Kerbin inertial"
        return (50.0, 0.0, 0.0)

    def remaining_burn_vector(self, frame):
        assert frame == "Kerbin inertial"
        return (self.remaining, 0.0, 0.0)

    def remove(self):
        self.removes += 1


@pytest.fixture
def rig():
    clock = Clock()
    sc = SpaceCenter(clock)
    clock.sc = sc
    return clock, sc, sc.active_vessel, Log()


def run(rig, **kwargs):
    clock, sc, vessel, log = rig
    return execute_node(
        SimpleNamespace(space_center=sc),
        vessel,
        vessel.node,
        log=log,
        clock=clock.now,
        sleeper=clock.sleep,
        **kwargs,
    )


def assert_released(rig):
    _, sc, vessel, _ = rig
    assert vessel.control.throttle == 0.0
    assert vessel.auto_pilot.disengages == 1
    assert sc.rails_warp_factor == sc.physics_warp_factor == 0
    assert vessel.node.removes == 0


def test_ideal_burn_time_uses_rocket_equation_mass_flow():
    duration = burn_duration(500.0, 4000.0, 60000.0, 345.0)
    fuel = duration * 60000.0 / (345.0 * 9.80665)
    assert 345.0 * 9.80665 * math.log(4000.0 / (4000.0 - fuel)) == pytest.approx(500.0)


def test_burn_centers_on_node_tapers_throttle_and_engages_once(rig):
    _, sc, vessel, log = rig
    vessel.auto_pilot.target_roll = (
        120.0  # Inherited attitude contract from a previous phase.
    )
    result = run(rig, available_delta_v=200.0, reserve_delta_v=100.0)
    assert result["success"] and result["remaining_delta_v"] <= 0.1
    assert result["actual_burn_start_ut"] == pytest.approx(
        result["planned_burn_start_ut"], abs=0.021
    )
    assert result["propellant_consumed_kg"] > 0
    assert vessel.auto_pilot.engages == 1
    assert vessel.auto_pilot.reference_frame == "Kerbin inertial"
    assert math.isnan(vessel.auto_pilot.target_roll)
    assert len(sc.warps) == 1 and sc.warps[0][0] < result["actual_burn_start_ut"]
    assert any(0 < value < 1 for _, value, _ in vessel.control.commands)
    assert any(event["event"] == "maneuver_burn" for event in log.events)
    assert_released(rig)


@pytest.mark.parametrize("pointing_error", [20.0, -172.35])
def test_no_throttle_is_applied_while_pointing_error_exceeds_threshold(
    rig, pointing_error
):
    _, _, vessel, _ = rig
    start = vessel.node.ut - burn_duration(50.0, 1000.0, 10000.0, 345.0) / 2
    vessel.auto_pilot.error_at = lambda ut: (
        pointing_error if start + 0.2 < ut < start + 0.7 else 0.0
    )
    assert run(rig)["success"]
    assert any(abs(error) > 3.0 for _, _, error in vessel.control.commands)
    assert all(
        throttle == 0.0
        for _, throttle, error in vessel.control.commands
        if abs(error) > 3.0
    )
    assert_released(rig)


def test_paused_simulation_is_bounded_and_cleans_up(rig):
    rig[0].paused = True
    result = run(rig, wall_timeout=0.3)
    assert not result["success"] and result["termination_reason"] == "timeout"
    assert_released(rig)


@pytest.mark.parametrize("wrong_node", [False, True])
def test_wrong_body_stops_before_warp_or_burn(rig, wrong_node):
    _, sc, vessel, _ = rig
    if wrong_node:
        vessel.node.orbit.body = SimpleNamespace(name="Mun")
        result = run(rig)
    else:
        result = run(rig, expected_body="Mun")
    assert not result["success"] and result["termination_reason"] == "wrong_body"
    assert not sc.warps and not any(
        value > 0 for _, value, _ in vessel.control.commands
    )
    assert_released(rig)


def test_supplied_budget_and_reserve_are_checked_before_burn(rig):
    result = run(rig, available_delta_v=60.0, reserve_delta_v=20.0)
    assert (
        not result["success"] and result["termination_reason"] == "insufficient_delta_v"
    )
    assert not rig[1].warps
    assert_released(rig)


def test_nonfinite_burn_vector_is_rejected_and_cleans_up(rig):
    rig[2].node.remaining = math.nan
    result = run(rig)
    assert not result["success"] and result["termination_reason"] == "invalid_telemetry"
    assert_released(rig)


def test_malformed_burn_vector_is_reported_as_failure_after_cleanup(rig):
    rig[2].node.remaining_burn_vector = lambda frame: None
    result = run(rig)
    assert not result["success"] and result["termination_reason"] == "invalid_telemetry"
    assert_released(rig)


def test_blocking_warp_has_a_longer_bounded_rpc_deadline_and_timeout_cleanup(
    rig, monkeypatch
):
    from contextlib import contextmanager
    from orbiter import maneuvers

    deadlines = []
    original = maneuvers.rpc_deadline

    @contextmanager
    def record_deadline(seconds):
        deadlines.append(seconds)
        with original(seconds):
            yield

    monkeypatch.setattr(maneuvers, "rpc_deadline", record_deadline)

    def failed_warp():
        raise TimeoutError("Warp RPC deadline expired")

    rig[1].warp_effect = failed_warp
    result = run(rig)
    assert not result["success"] and result["termination_reason"] == "timeout"
    assert 120.0 in deadlines and all(0 < value <= 120 for value in deadlines)
    assert_released(rig)


def test_soi_change_during_coast_is_detected_before_throttle(rig):
    _, sc, vessel, _ = rig
    sc.warp_effect = lambda: setattr(vessel.orbit, "body", SimpleNamespace(name="Mun"))
    result = run(rig)
    assert not result["success"] and result["termination_reason"] == "wrong_body"
    assert not any(value > 0 for _, value, _ in vessel.control.commands)
    assert_released(rig)


def test_simulation_time_reversal_during_warp_fails(rig):
    rig[1].warp_effect = lambda: setattr(rig[1], "ut", -10.0)
    result = run(rig)
    assert not result["success"] and result["termination_reason"] == "time_reversal"
    assert_released(rig)


@pytest.mark.parametrize("pointing_error", [20.0, -172.35])
def test_persistent_misalignment_has_its_own_deadline(rig, pointing_error):
    rig[2].auto_pilot.error_at = lambda ut: pointing_error
    result = run(rig, alignment_timeout=0.3)
    assert not result["success"] and result["termination_reason"] == "alignment_timeout"
    assert not any(value > 0 for _, value, _ in rig[2].control.commands)
    assert_released(rig)


def test_fueled_but_unresponsive_burn_does_not_continue_indefinitely(rig):
    rig[0].no_progress = True
    result = run(rig)
    assert not result["success"] and result["termination_reason"] == "no_progress"
    assert_released(rig)


def test_interrupt_during_blocking_warp_propagates_after_cleanup(rig):
    def interrupted():
        raise KeyboardInterrupt()

    rig[1].warp_effect = interrupted
    with pytest.raises(KeyboardInterrupt):
        run(rig)
    assert_released(rig)
