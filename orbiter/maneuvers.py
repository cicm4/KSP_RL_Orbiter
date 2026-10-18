"""Bounded actual-KSP maneuver-node burns; node creation/removal is caller owned.

This deterministic executor is separate from the learned ascent controller.
It operates only when called with an existing kRPC connection and node.
"""

from __future__ import annotations

import math
import time

from .live import rpc_deadline, stop_controls, wait_for_ut


class ManeuverFailure(RuntimeError):
    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason


def _finite(value, name, *, positive=False):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ManeuverFailure("invalid_telemetry", f"{name} must be numeric") from error
    if not math.isfinite(value) or (positive and value <= 0):
        raise ManeuverFailure(
            "invalid_telemetry",
            f"{name} must be finite" + (" and positive" if positive else ""),
        )
    return value


def _vector(value):
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ManeuverFailure(
            "invalid_telemetry", "Burn vector must contain three coordinates"
        )
    result = tuple(_finite(item, "burn vector coordinate") for item in value)
    if not math.isfinite(math.hypot(*result)):
        raise ManeuverFailure(
            "invalid_telemetry", "Burn vector magnitude must be finite"
        )
    return result


def burn_duration(delta_v, mass, thrust, specific_impulse):
    """Ideal full-throttle duration with changing mass, in SI units."""
    delta_v = _finite(delta_v, "delta-v")
    mass = _finite(mass, "mass", positive=True)
    thrust = _finite(thrust, "available thrust", positive=True)
    exhaust_speed = (
        _finite(specific_impulse, "specific impulse", positive=True) * 9.80665
    )
    if delta_v < 0:
        raise ValueError("delta-v must be nonnegative")
    return _finite(
        mass * exhaust_speed / thrust * (-math.expm1(-delta_v / exhaust_speed)),
        "burn duration",
    )


def execute_node(
    connection,
    vessel,
    node,
    *,
    log,
    rpc_timeout=15.0,
    wall_timeout=600.0,
    tolerance=0.1,
    alignment_tolerance=3.0,
    alignment_timeout=60.0,
    control_dt=0.1,
    warp_rpc_timeout=120.0,
    warp_margin=10.0,
    expected_body=None,
    available_delta_v=None,
    reserve_delta_v=0.0,
    clock=time.monotonic,
    sleeper=time.sleep,
):
    """Execute one existing node, returning success/failure and measured results.

    Uses actual KSP remaining-burn vectors in the body's nonrotating frame.
    Coasts may warp before the centered burn; throttle is zero while aligning.
    The optional available_delta_v is a caller-computed usable budget, never an
    estimate from dry mass that might count unusable resources. This function
    does not create/remove nodes, stage engines, load saves, or change targets.
    All operational failures return success=False; interrupts propagate after
    throttle/autopilot/warp cleanup. log supports FlightLog.write(event, **data).
    """
    for name, value in [
        ("rpc_timeout", rpc_timeout),
        ("wall_timeout", wall_timeout),
        ("tolerance", tolerance),
        ("alignment_tolerance", alignment_tolerance),
        ("alignment_timeout", alignment_timeout),
        ("control_dt", control_dt),
        ("warp_rpc_timeout", warp_rpc_timeout),
        ("warp_margin", warp_margin),
    ]:
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not math.isfinite(reserve_delta_v) or reserve_delta_v < 0:
        raise ValueError("reserve_delta_v must be finite and nonnegative")
    if available_delta_v is not None and (
        not math.isfinite(available_delta_v) or available_delta_v < 0
    ):
        raise ValueError("available_delta_v must be finite and nonnegative")
    sc = connection.space_center
    deadline = clock() + wall_timeout
    result = dict(
        success=False,
        termination_reason="not_started",
        controller="deterministic_node_executor",
        cleanup_errors=[],
        samples=0,
        tolerance_m_s=tolerance,
    )
    previous_ut = None
    initial_mass = None
    last_state = None

    def emit(event, **data):
        if log is not None:
            log.write(event, **data)

    def remaining_wall():
        remaining = deadline - clock()
        if remaining <= 0:
            raise TimeoutError("Maneuver wall-clock deadline expired")
        return remaining

    def scope(limit=rpc_timeout):
        return rpc_deadline(min(limit, remaining_wall()))

    def read_state():
        nonlocal previous_ut, last_state
        with scope():
            ut = _finite(sc.ut, "universal time")
            if previous_ut is not None and ut < previous_ut - 1e-6:
                raise ManeuverFailure(
                    "time_reversal", "KSP universal time moved backwards"
                )
            previous_ut = ut
            if sc.active_vessel != vessel:
                raise ManeuverFailure(
                    "vessel_changed", "Active vessel changed during maneuver"
                )
            if vessel.orbit.body.name != body_name:
                raise ManeuverFailure(
                    "wrong_body", "Vessel changed sphere of influence during maneuver"
                )
            vector = _vector(node.remaining_burn_vector(frame))
            state = dict(
                ut=ut,
                remaining_vector=vector,
                remaining_delta_v=math.hypot(*vector),
                mass=_finite(vessel.mass, "mass", positive=True),
                available_thrust=_finite(vessel.available_thrust, "available thrust"),
            )
        last_state = state
        result["samples"] += 1
        return state

    def target_error(vector):
        magnitude = math.hypot(*vector)
        with scope():
            ap.target_direction = tuple(value / magnitude for value in vector)
            # kRPC 0.5.4 may normalize orientation errors into [-180, 180].
            return abs(_finite(ap.error, "autopilot pointing error"))

    def tick(ut, duration=control_dt):
        wait_for_ut(
            sc,
            ut + duration,
            remaining_wall(),
            rpc_timeout=rpc_timeout,
            clock=clock,
            sleeper=sleeper,
        )

    def zero_warp():
        with scope():
            sc.rails_warp_factor = 0
            sc.physics_warp_factor = 0

    def align():
        align_deadline = clock() + alignment_timeout
        while True:
            state = read_state()
            if state["remaining_delta_v"] <= tolerance:
                return state
            error = target_error(state["remaining_vector"])
            with scope():
                vessel.control.throttle = 0.0
            emit("maneuver_alignment", pointing_error_deg=error, **state)
            if error <= alignment_tolerance:
                return state
            if clock() >= align_deadline:
                raise ManeuverFailure(
                    "alignment_timeout", "Autopilot did not align before its deadline"
                )
            tick(state["ut"])

    try:
        with scope():
            body = vessel.orbit.body
            body_name = body.name
            if expected_body is not None and body_name != expected_body:
                raise ManeuverFailure(
                    "wrong_body", f"Expected {expected_body}, received {body_name}"
                )
            if node.orbit.body.name != body_name:
                raise ManeuverFailure(
                    "wrong_body", "Node and vessel orbit different bodies"
                )
            frame = body.non_rotating_reference_frame
            node_ut = _finite(node.ut, "node UT")
            planned_vector = _vector(node.burn_vector(frame))
            planned_delta_v = math.hypot(*planned_vector)
            ap = vessel.auto_pilot
        state = read_state()
        initial_mass = state["mass"]
        result.update(
            body=body_name,
            node_ut=node_ut,
            planned_delta_v=planned_delta_v,
            initial_remaining_delta_v=state["remaining_delta_v"],
            initial_mass_kg=initial_mass,
            available_delta_v=available_delta_v,
            reserve_delta_v=reserve_delta_v,
        )
        if (
            available_delta_v is not None
            and state["remaining_delta_v"] + reserve_delta_v > available_delta_v
        ):
            raise ManeuverFailure(
                "insufficient_delta_v",
                "Node plus reserve exceeds the supplied usable delta-v budget",
            )
        with scope():
            vessel.control.throttle = 0.0
            vessel.control.sas = False
            ap.reference_frame = frame
            ap.target_roll = (
                math.nan
            )  # Direction-only burns must not inherit a prior phase's roll target.
            if state["remaining_delta_v"] > tolerance:
                ap.target_direction = tuple(
                    value / state["remaining_delta_v"]
                    for value in state["remaining_vector"]
                )
                ap.engage()  # Updating targets below must never restart the PID.
        if state["remaining_delta_v"] > tolerance:
            if state["available_thrust"] <= 0:
                raise ManeuverFailure(
                    "out_of_fuel", "No active fueled engine can provide thrust"
                )
            with scope():
                isp = _finite(
                    vessel.specific_impulse, "specific impulse", positive=True
                )
            duration = burn_duration(
                state["remaining_delta_v"], initial_mass, state["available_thrust"], isp
            )
            burn_start = node_ut - duration / 2
            result.update(
                estimated_full_throttle_seconds=duration,
                planned_burn_start_ut=burn_start,
            )
            emit("maneuver_start", **result)
            state = align()
            if burn_start < state["ut"] - 5.0:
                raise ManeuverFailure(
                    "late_start",
                    "Alignment finished more than five seconds after the centered burn start",
                )
            if burn_start - state["ut"] > warp_margin + 5.0:
                with scope(warp_rpc_timeout):
                    sc.warp_to(
                        burn_start - warp_margin,
                        max_rails_rate=100000.0,
                        max_physics_rate=2.0,
                    )
                zero_warp()
                state = align()  # Verify attitude/body again after warp.
            if state["ut"] < burn_start:
                wait_for_ut(
                    sc,
                    burn_start,
                    remaining_wall(),
                    rpc_timeout=rpc_timeout,
                    clock=clock,
                    sleeper=sleeper,
                )
            zero_warp()
            state = read_state()
            result["actual_burn_start_ut"] = state["ut"]
            best, progress_ut = state["remaining_delta_v"], state["ut"]
            misaligned_since = None
            while state["remaining_delta_v"] > tolerance:
                if state["available_thrust"] <= 0:
                    raise ManeuverFailure(
                        "out_of_fuel",
                        "Active engine exhausted available thrust before completing node",
                    )
                error = target_error(state["remaining_vector"])
                acceleration = _finite(
                    state["available_thrust"] / state["mass"],
                    "acceleration",
                    positive=True,
                )
                throttle = min(1.0, state["remaining_delta_v"] / acceleration)
                if error > alignment_tolerance:
                    throttle = 0.0
                    misaligned_since = (
                        state["ut"] if misaligned_since is None else misaligned_since
                    )
                    if state["ut"] - misaligned_since > alignment_timeout:
                        raise ManeuverFailure(
                            "alignment_timeout",
                            "Pointing error remained excessive during burn",
                        )
                else:
                    misaligned_since = None
                    if state["remaining_delta_v"] < best - max(0.001, tolerance * 0.1):
                        best, progress_ut = state["remaining_delta_v"], state["ut"]
                    if state["ut"] - progress_ut > 10.0:
                        raise ManeuverFailure(
                            "no_progress",
                            "Remaining delta-v did not decrease during ten seconds of aligned burn",
                        )
                with scope():
                    if sc.rails_warp_factor != 0 or sc.physics_warp_factor != 0:
                        raise ManeuverFailure(
                            "unexpected_warp", "Time warp changed during maneuver burn"
                        )
                    vessel.control.throttle = throttle
                emit(
                    "maneuver_burn",
                    pointing_error_deg=error,
                    throttle=throttle,
                    **state,
                )
                tick(state["ut"])
                state = read_state()
        result.update(
            success=True,
            termination_reason="node_complete",
            remaining_delta_v=state["remaining_delta_v"],
            final_ut=state["ut"],
            final_mass_kg=state["mass"],
            propellant_consumed_kg=max(0.0, initial_mass - state["mass"]),
        )
    except (OSError, RuntimeError, ValueError, TimeoutError) as error:
        result.update(
            success=False,
            termination_reason=getattr(
                error,
                "reason",
                "timeout" if isinstance(error, TimeoutError) else "operation_error",
            ),
            error=f"{type(error).__name__}: {error}",
        )
        if last_state is not None:
            result.update(
                final_ut=last_state["ut"],
                remaining_delta_v=last_state["remaining_delta_v"],
            )
    finally:
        if last_state is not None and initial_mass is not None:
            result.update(
                final_mass_kg=last_state["mass"],
                propellant_consumed_kg=max(0.0, initial_mass - last_state["mass"]),
            )
        result["cleanup_errors"] = stop_controls([vessel], min(rpc_timeout, 3.0))
        for name in ("rails_warp_factor", "physics_warp_factor"):
            try:
                with rpc_deadline(min(rpc_timeout, 3.0)):
                    setattr(sc, name, 0)
            except Exception as error:
                result["cleanup_errors"].append(
                    f"{name}: {type(error).__name__}: {error}"
                )
        if result["cleanup_errors"]:
            result.update(success=False, termination_reason="cleanup_failed")
        emit("maneuver_end", **result)
    return result
