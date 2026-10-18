"""Read-only kRPC telemetry export for calibrating the fast simulator.

Run ``python -m orbiter.capture --output data/ksp_capture.json`` with the
kRPC server enabled. This module never stages, saves, loads, or writes controls.
Distances, masses, forces, pressures, and times in its JSON use SI units.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import signal
import time
from typing import Any, Callable


def _read(label: str, getter: Callable[[], Any], warnings: list[str]) -> Any:
    try:
        value = getter()
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, tuple):
            return list(value)
        return value
    except TimeoutError:
        raise
    except Exception as exc:
        warnings.append(f"{label}: {type(exc).__name__}: {exc}")
        return None


def _resources(resources: Any, warnings: list[str], label: str) -> dict:
    names = _read(f"{label}.names", lambda: resources.names, warnings) or []
    return {
        name: {
            "amount_units": _read(f"{label}.{name}.amount", lambda n=name: resources.amount(n), warnings),
            "capacity_units": _read(f"{label}.{name}.max", lambda n=name: resources.max(n), warnings),
            "density_kg_per_unit": _read(f"{label}.{name}.density", lambda n=name: resources.density(n), warnings),
        }
        for name in names
    }


def flight_snapshot(space_center: Any, vessel: Any, body: Any, warnings: list[str]) -> dict:
    """Sample both rotating and inertial frames without changing game state.

    ``ut_begin``/``ut_end`` expose RPC sampling skew; these reads are not an
    atomic physics frame. Vector ordering is kRPC's native (x, y, z) ordering.
    """
    snapshot = {"ut_begin_s": space_center.ut}
    rotating = body.reference_frame
    inertial = body.non_rotating_reference_frame
    flight = vessel.flight(rotating)
    attitude = vessel.flight(vessel.surface_reference_frame)
    for key, getter in {
        "position_inertial_m": lambda: vessel.position(inertial),
        "velocity_inertial_m_s": lambda: vessel.velocity(inertial),
        "position_body_fixed_m": lambda: vessel.position(rotating),
        "velocity_body_fixed_m_s": lambda: vessel.velocity(rotating),
        "direction_inertial": lambda: vessel.direction(inertial),
        "angular_velocity_inertial_rad_s": lambda: vessel.angular_velocity(inertial),
        "mass_kg": lambda: vessel.mass,
        "thrust_n": lambda: vessel.thrust,
        "available_thrust_n": lambda: vessel.available_thrust,
        "situation": lambda: str(vessel.situation),
        "stage": lambda: vessel.control.current_stage,
        "throttle": lambda: vessel.control.throttle,
        "pitch_control": lambda: vessel.control.pitch,
        "yaw_control": lambda: vessel.control.yaw,
        "roll_control": lambda: vessel.control.roll,
        "pitch_deg": lambda: attitude.pitch,
        "heading_deg": lambda: attitude.heading,
        "roll_deg": lambda: attitude.roll,
        "altitude_m": lambda: flight.mean_altitude,
        "surface_altitude_m": lambda: flight.surface_altitude,
        "latitude_deg": lambda: flight.latitude,
        "longitude_deg": lambda: flight.longitude,
        "vertical_speed_m_s": lambda: flight.vertical_speed,
        "horizontal_speed_m_s": lambda: flight.horizontal_speed,
        "dynamic_pressure_pa": lambda: flight.dynamic_pressure,
        "static_pressure_pa": lambda: flight.static_pressure,
        "atmosphere_density_kg_m3": lambda: flight.atmosphere_density,
        "mach": lambda: flight.mach,
        "angle_of_attack_deg": lambda: flight.angle_of_attack,
        "drag_body_fixed_n": lambda: flight.drag,
        "lift_body_fixed_n": lambda: flight.lift,
        "apoapsis_altitude_m": lambda: vessel.orbit.apoapsis_altitude,
        "periapsis_altitude_m": lambda: vessel.orbit.periapsis_altitude,
        "time_to_apoapsis_s": lambda: vessel.orbit.time_to_apoapsis,
    }.items():
        snapshot[key] = _read(key, getter, warnings)
    snapshot["ut_end_s"] = space_center.ut
    return snapshot


def capture(connection: Any, *, body_name: str | None = None, samples: int = 1,
            interval: float = 0.5, atmosphere_step: float = 1000.0) -> dict:
    """Collect a portable calibration record using only read-only RPCs."""
    warnings: list[str] = []
    sc = connection.space_center
    vessel = _read("active_vessel", lambda: sc.active_vessel, warnings)
    body = sc.bodies[body_name] if body_name else (
        vessel.orbit.body if vessel is not None else sc.bodies["Kerbin"]
    )
    atmosphere_depth = body.atmosphere_depth
    body_data = {
        "name": body.name,
        "gravitational_parameter_m3_s2": body.gravitational_parameter,
        "equatorial_radius_m": body.equatorial_radius,
        "rotational_period_s": body.rotational_period,
        "rotational_speed_rad_s": body.rotational_speed,
        "has_atmosphere": body.has_atmosphere,
        "atmosphere_depth_m": atmosphere_depth,
        "atmosphere": [],
    }
    # DensityAt is the stock altitude-only approximation. Position-aware density
    # below includes the current sun exposure at the vessel latitude/longitude.
    latitude, longitude = 0.0, 0.0
    if vessel is not None:
        flight = vessel.flight(body.reference_frame)
        latitude = _read("latitude", lambda: flight.latitude, warnings) or 0.0
        longitude = _read("longitude", lambda: flight.longitude, warnings) or 0.0
    body_data["atmosphere_sample_latitude_deg"] = latitude
    body_data["atmosphere_sample_longitude_deg"] = longitude
    n_altitudes = int(math.ceil(atmosphere_depth / atmosphere_step))
    for i in range(n_altitudes + 1):
        altitude = min(i * atmosphere_step, atmosphere_depth)
        position = body.position_at_altitude(latitude, longitude, altitude, body.reference_frame)
        body_data["atmosphere"].append({
            "altitude_m": altitude,
            "pressure_pa": body.pressure_at(altitude),
            "density_kg_m3": body.density_at(altitude),
            "position_aware_density_kg_m3": _read(
                f"density_at_position[{altitude}]",
                lambda p=position: body.atmospheric_density_at_position(p, body.reference_frame),
                warnings,
            ),
        })
    output = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "live_krpc_read_only",
        "notes": [
            "This export does not establish that a simplified simulator matches KSP.",
            "Atmosphere density depends on position and sun exposure; the table is a snapshot.",
            "Flight samples are sequential RPC reads; use ut_begin_s/ut_end_s to assess skew.",
            "kRPC vector coordinates are left-handed, with y along the rotation axis.",
            "Engine curve RPC inputs are atmospheres; exported pressure_pa converts these to Pascals.",
            "Engine specific_impulse_at may return zero for inactive engines; vacuum/sea-level values are also exported.",
        ],
        "body": body_data,
        "vessel": None,
        "flight_samples": [],
        "warnings": warnings,
    }
    if vessel is None:
        return output
    parts = vessel.parts.all
    vehicle = {
        "name": vessel.name,
        "body": vessel.orbit.body.name,
        "mass_kg": vessel.mass,
        "dry_mass_kg": vessel.dry_mass,
        "part_count": len(parts),
        "resources": _resources(vessel.resources, warnings, "vessel.resources"),
        "parts": [],
        "engines": [],
        "decouple_stages": [],
    }
    output["vessel"] = vehicle
    for index, part in enumerate(parts):
        vehicle["parts"].append({
            "index": index, "name": part.name, "title": part.title,
            "mass_kg": part.mass, "dry_mass_kg": part.dry_mass,
            "activation_stage": part.stage, "decouple_stage": part.decouple_stage,
            "resources": _resources(part.resources, warnings, f"parts[{index}].resources"),
        })
    for index, engine in enumerate(vessel.parts.engines):
        data = {"part_name": engine.part.name,
                "activation_stage": engine.part.stage,
                "decouple_stage": engine.part.decouple_stage}
        for name in ["active", "has_fuel", "throttle_locked", "thrust_limit",
                     "thrust", "available_thrust", "max_thrust", "max_vacuum_thrust",
                     "specific_impulse", "vacuum_specific_impulse",
                     "kerbin_sea_level_specific_impulse", "propellant_names", "propellant_ratios"]:
            data[name] = _read(f"engines[{index}].{name}", lambda n=name: getattr(engine, n), warnings)
        data["pressure_curve"] = [
            {"pressure_pa": pressure,
             "max_thrust_n": _read(f"engines[{index}].max_thrust_at[{pressure}]",
                                    lambda p=pressure: engine.max_thrust_at(p / 101325.0), warnings),
             "specific_impulse_s": _read(f"engines[{index}].specific_impulse_at[{pressure}]",
                                         lambda p=pressure: engine.specific_impulse_at(p / 101325.0), warnings)}
            for pressure in [0.0, 1013.25, 10132.5, 50662.5, 101325.0]
        ]
        vehicle["engines"].append(data)
    stages = sorted({p.decouple_stage for p in parts})
    for stage in stages:
        resources = _read(f"decouple_stage[{stage}]",
                          lambda s=stage: vessel.resources_in_decouple_stage(s, cumulative=False), warnings)
        vehicle["decouple_stages"].append({
            "stage": stage,
            "resources": _resources(resources, warnings, f"stage[{stage}]") if resources is not None else {},
        })
    for i in range(samples):
        if i:
            time.sleep(interval)
        output["flight_samples"].append(flight_snapshot(sc, vessel, body, warnings))
    return output


@contextmanager
def _deadline(seconds: float):
    """Bound kRPC handshakes too: its partial_receive can otherwise wait forever."""
    if not hasattr(signal, "SIGALRM"):
        raise RuntimeError("This timeout implementation requires macOS/Linux.")
    def expired(signum, frame):
        raise TimeoutError(f"kRPC capture exceeded {seconds:g} seconds")
    previous_handler = signal.signal(signal.SIGALRM, expired)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--address", default="127.0.0.1")
    parser.add_argument("--rpc-port", type=int, default=50000)
    parser.add_argument("--body", default=None, help="Defaults to the active vessel's body, or Kerbin")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--interval", type=float, default=0.5, help="Wall-clock seconds between samples")
    parser.add_argument("--atmosphere-step", type=float, default=1000.0)
    parser.add_argument("--timeout", type=float, default=60.0, help="Maximum total wall-clock seconds")
    args = parser.parse_args(argv)
    if args.samples < 0 or args.interval < 0 or args.atmosphere_step <= 0 or args.timeout <= 0:
        parser.error("samples/interval must be nonnegative; atmosphere-step/timeout must be positive")
    import krpc
    connection = None
    try:
        with _deadline(args.timeout):
            # Server-provided stubs support the installed kRPC 0.5.x even when
            # the local Python client ships newer pregenerated service classes.
            connection = krpc.connect(name="Read-only orbiter calibration", address=args.address,
                                      rpc_port=args.rpc_port, stream_port=None,
                                      use_pregenerated_stubs=False)
            data = capture(connection, body_name=args.body, samples=args.samples,
                           interval=args.interval, atmosphere_step=args.atmosphere_step)
    except (OSError, RuntimeError, TimeoutError) as exc:
        parser.exit(1, f"Capture failed: {exc}\n")
    finally:
        if connection is not None:
            connection.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.output}; {len(data['flight_samples'])} samples; {len(data['warnings'])} warnings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
