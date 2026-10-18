"""Offline one-flight diagnostics; no kRPC import or game interaction.

Run with Python and NumPy: check_vacuum.py CAPTURE_JSONL --output PATH.
Plain or gzip-compressed JSONL is supported. The command/mass time offset is
fitted on the same flight, not an independent measurement of actuator latency.
"""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

G0 = 9.80665


def rtn_basis(position, velocity):
    radial = position / np.linalg.norm(position)
    normal = np.cross(position, velocity)
    normal /= np.linalg.norm(normal)
    return np.column_stack((radial, np.cross(normal, radial), normal))


def propagate(position, velocity, duration, mu):
    """Independent two-body RK4 coast with substeps no larger than 50ms."""
    state = np.concatenate((position, velocity)).astype(float)
    steps = max(1, math.ceil(duration / 0.05))
    dt = duration / steps

    def derivative(value):
        return np.concatenate(
            (value[3:], -mu * value[:3] / np.linalg.norm(value[:3]) ** 3)
        )

    for _ in range(steps):
        first = derivative(state)
        second = derivative(state + 0.5 * dt * first)
        third = derivative(state + 0.5 * dt * second)
        fourth = derivative(state + dt * third)
        state += dt * (first + 2 * second + 2 * third + fourth) / 6
    return state[:3], state[3:]


def statistics(values):
    values = np.asarray(values)
    if not len(values):
        return dict(count=0)
    return dict(
        count=len(values),
        mean=float(values.mean()),
        median=float(np.median(values)),
        p95_abs=float(np.quantile(abs(values), 0.95)),
        max_abs=float(max(abs(values))),
    )


def diagnose(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        events = [json.loads(line) for line in source]
    starts = [row for row in events if row["event"] == "capture_start"]
    if len(starts) != 1:
        raise ValueError("The log must contain exactly one capture_start event")
    start = starts[0]
    states = [row for row in events if row["event"] == "capture_telemetry"]
    actions = [row for row in events if row["event"] == "capture_action"]
    if len(states) < 2 or len(actions) != len(states) - 1:
        raise ValueError(
            "Need consecutive capture telemetry with one action per interval"
        )
    if any(b["time"] <= a["time"] for a, b in zip(states, states[1:])):
        raise ValueError("Capture telemetry time must increase")
    ut0, mu = start["start_ut"], start["config"]["mu"]

    def throttle_integral(lo, hi):
        current, previous, value = 0.0, lo, 0.0
        for action in actions:
            if action["ut"] <= lo:
                current = action["throttle"]
            elif action["ut"] < hi:
                value += current * (action["ut"] - previous)
                previous, current = action["ut"], action["throttle"]
        return value + current * (hi - previous)

    def mass_errors(lag):
        return np.array(
            [
                a["mass"]
                - b["mass"]
                - a["thrust"]
                / (a["isp"] * G0)
                * throttle_integral(ut0 + a["time"] - lag, ut0 + b["time"] - lag)
                for a, b in zip(states, states[1:])
            ]
        )

    has_burn = any(action["throttle"] > 0 for action in actions)
    fitted_lag = (
        min(
            np.linspace(-0.1, 0.1, 801),
            key=lambda value: float(np.mean(mass_errors(value) ** 2)),
        )
        if has_burn
        else None
    )
    lag = 0.0 if fitted_lag is None else fitted_lag
    intervals = []
    for a, b in zip(states, states[1:]):
        t0, t1 = ut0 + a["time"], ut0 + b["time"]
        dt, integral = t1 - t0, throttle_integral(t0, t1)
        expected = a["thrust"] / (a["isp"] * G0) * integral
        observed = a["mass"] - b["mass"]
        r0, r1, v0, v1 = (
            np.asarray(value) for value in (a["r"], b["r"], a["v"], b["v"])
        )
        gravity = lambda position: -mu * position / np.linalg.norm(position) ** 3
        gravity_dv = dt * (gravity(r0) + 4 * gravity((r0 + r1) / 2) + gravity(r1)) / 6
        estimated = v1 - v0 - gravity_dv
        rocket_dv = a["isp"] * G0 * math.log(a["mass"] / b["mass"])
        row = dict(
            step=a["step"],
            duration=dt,
            commanded_throttle_integral=integral,
            mass_consumed=observed,
            predicted_mass_consumed=expected,
            mass_error=observed - expected,
            effective_lag_mass_error=observed
            - a["thrust"] / (a["isp"] * G0) * throttle_integral(t0 - lag, t1 - lag),
            thrust_dv_estimated=float(np.linalg.norm(estimated)),
            rocket_equation_dv=rocket_dv,
        )
        if integral < 1e-9 and abs(observed) <= 0.0005:
            predicted_r, predicted_v = propagate(r0, v0, dt, mu)
            row.update(
                coast_position_error=float(np.linalg.norm(predicted_r - r1)),
                coast_velocity_error=float(np.linalg.norm(predicted_v - v1)),
            )
        if rocket_dv > 0.1:
            action = next(action for action in actions if action["step"] == a["step"])
            direction = rtn_basis(r0, v0) @ np.asarray(action["direction_rtn"])
            row.update(
                thrust_impulse_relative_error=float(
                    np.linalg.norm(estimated) / rocket_dv - 1
                ),
                thrust_angle_error_deg=math.degrees(
                    math.acos(
                        float(
                            np.clip(
                                estimated
                                @ direction
                                / max(np.linalg.norm(estimated), 1e-15),
                                -1,
                                1,
                            )
                        )
                    )
                ),
            )
        intervals.append(row)
    observed = sum(row["mass_consumed"] for row in intervals)
    expected = sum(row["predicted_mass_consumed"] for row in intervals)
    summary = dict(
        interval_duration_s=statistics([row["duration"] for row in intervals]),
        mass_consumed_kg=observed,
        predicted_mass_consumed_kg=expected,
        total_mass_relative_error=observed / expected - 1 if expected else None,
        float32_mass_increment_kg=float(np.spacing(np.float32(states[0]["mass"]))),
        per_interval_mass_error_kg=statistics([row["mass_error"] for row in intervals]),
        fitted_command_mass_time_offset_s=float(fitted_lag)
        if fitted_lag is not None
        else None,
        offset_fitted_on_same_flight=has_burn,
        independent_latency_validation=False,
        per_interval_mass_error_with_fitted_lag_kg=statistics(mass_errors(lag)),
        powered_thrust_impulse_relative_error=statistics(
            [
                row["thrust_impulse_relative_error"]
                for row in intervals
                if "thrust_impulse_relative_error" in row
            ]
        ),
        powered_command_angle_error_deg=statistics(
            [
                row["thrust_angle_error_deg"]
                for row in intervals
                if "thrust_angle_error_deg" in row
                and actions[row["step"]]["throttle"] > 0
            ]
        ),
        coast_position_error_m=statistics(
            [
                row["coast_position_error"]
                for row in intervals
                if "coast_position_error" in row
            ]
        ),
        coast_velocity_error_m_s=statistics(
            [
                row["coast_velocity_error"]
                for row in intervals
                if "coast_velocity_error" in row
            ]
        ),
        autopilot_gated_commands=sum(
            action["pointing_error_deg"] > 3 for action in actions
        ),
        commands=len(actions),
        zero_throttle_command_burn_transients=sum(
            row["commanded_throttle_integral"] < 1e-9
            and abs(row["mass_consumed"]) > 0.0005
            for row in intervals
        ),
    )
    return dict(
        source=str(path),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        summary=summary,
        scope="Descriptive same-flight consistency diagnostic. Sequential RPC timing confounds inferred lag.",
        method="Piecewise-constant command UT throttle integral; F/(Isp*g0) mass flow. Actual delta-v minus Simpson midpoint gravity versus actual-mass rocket equation. True coasts require zero throttle integral and <=0.0005kg mass loss; independent two-body RK4 uses<=50ms substeps. Powered impulses<=0.1m/s excluded. Time offset fitted on the same intervals over +/-100ms.",
        intervals=intervals,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = diagnose(args.capture_log)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
