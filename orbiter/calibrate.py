"""Import read-only kRPC body measurements into a headless EnvConfig.

    python -m orbiter.calibrate --capture data/ksp_capture.json --output data/kerbin.json

Craft masses, engines, staging, and attitude dynamics remain those of the base
configuration. A separate provenance JSON identifies everything left unmeasured.
This is body calibration, not evidence that the surrogate reproduces KSP.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

from .env import EnvConfig


BODY_KEYS = {
    "mu": "gravitational_parameter_m3_s2",
    "radius": "equatorial_radius_m",
    "rotation_period": "rotational_period_s",
    "atmosphere_depth": "atmosphere_depth_m",
}


def _number(value: Any, name: str, *, minimum: float | None = None,
            strictly_positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if strictly_positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_tree(value: Any, name="configuration") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_tree(item, f"{name}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        _number(value, name)


def validate_config(config: EnvConfig) -> None:
    """Reject invalid base inputs too; never serialize NaN/Infinity as physics."""
    _finite_tree(asdict(config))
    for name in ("mu", "radius", "target_altitude", "orbit_tolerance", "payload_mass",
                 "drag_area", "attitude_rate", "attitude_tau"):
        _number(getattr(config, name), name, strictly_positive=True)
    if config.rotation_period == 0:
        raise ValueError("rotation_period must be nonzero")
    _number(config.atmosphere_depth, "atmosphere_depth", minimum=0)
    if config.target_altitude + config.orbit_tolerance <= config.atmosphere_depth + 1000:
        raise ValueError("target orbit must extend above atmosphere_depth + 1000 m; supply a suitable --base-config")
    config.__post_init__()


def atmosphere_table(body: dict, *, density_source="altitude") -> tuple[tuple[float, float, float], ...]:
    """Return (metres, kg/m^3, Pascals); capture.py already exports SI pressure."""
    depth = _number(body.get("atmosphere_depth_m"), "body.atmosphere_depth_m", minimum=0)
    has_atmosphere = body.get("has_atmosphere")
    if not isinstance(has_atmosphere, bool):
        raise ValueError("body.has_atmosphere must be boolean")
    if not has_atmosphere:
        if depth != 0:
            raise ValueError("airless body has a nonzero atmosphere depth")
        return ((0.0, 0.0, 0.0),)
    if depth == 0:
        raise ValueError("atmospheric body must have positive atmosphere depth")
    rows = body.get("atmosphere")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("body.atmosphere requires at least two samples")
    density_key = {"altitude": "density_kg_m3", "position": "position_aware_density_kg_m3"}.get(density_source)
    if density_key is None:
        raise ValueError("density_source must be altitude or position")
    output = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"body.atmosphere[{index}] must be an object")
        altitude = _number(row.get("altitude_m"), f"atmosphere[{index}].altitude_m", minimum=0)
        density = _number(row.get(density_key), f"atmosphere[{index}].{density_key}", minimum=0)
        pressure = _number(row.get("pressure_pa"), f"atmosphere[{index}].pressure_pa", minimum=0)
        if altitude > depth:
            raise ValueError("atmosphere sample exceeds atmosphere depth")
        if output:
            previous = output[-1]
            if altitude <= previous[0]:
                raise ValueError("atmosphere altitudes must be strictly increasing")
            if density > previous[1] + max(1e-12, previous[1] * 1e-8):
                raise ValueError("atmosphere density must be nonincreasing")
            if pressure > previous[2] + max(1e-9, previous[2] * 1e-8):
                raise ValueError("atmosphere pressure must be nonincreasing")
        output.append((altitude, density, pressure))
    if not math.isclose(output[0][0], 0, abs_tol=1e-8) or not math.isclose(output[-1][0], depth, abs_tol=1e-8):
        raise ValueError("atmosphere table must span sea level through atmosphere depth")
    return tuple(output)


def _vector(value, name):
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f"{name} must contain three coordinates")
    return tuple(_number(item, name) for item in value)


def drag_diagnostics(samples: list[dict], *, max_skew=0.25) -> dict:
    """Estimate local effective Cd*A; body-fixed airspeed assumes no winds.

    Discard pad/coast samples and force/velocity pairs that cannot represent the
    same drag observation. These estimates do not identify Mach/AoA/staging laws.
    """
    estimates, rejected = [], Counter()
    for index, sample in enumerate(samples):
        try:
            if not isinstance(sample, dict):
                raise ValueError("invalid_sample")
            begin = _number(sample.get("ut_begin_s"), "ut_begin_s")
            end = _number(sample.get("ut_end_s"), "ut_end_s")
            if not 0 <= end - begin <= max_skew:
                raise ValueError("rpc_sampling_skew")
            rho = _number(sample.get("atmosphere_density_kg_m3"), "density", strictly_positive=True)
            velocity = _vector(sample.get("velocity_body_fixed_m_s"), "velocity")
            drag = _vector(sample.get("drag_body_fixed_n"), "drag")
            speed = math.sqrt(sum(value * value for value in velocity))
            force = math.sqrt(sum(value * value for value in drag))
            q = 0.5 * rho * speed ** 2
            if speed < 30 or q < 20 or force <= 0:
                raise ValueError("insufficient_aerodynamic_signal")
            alignment = -sum(a * b for a, b in zip(velocity, drag)) / (speed * force)
            if alignment < 0.95:
                raise ValueError("drag_not_opposed_to_air_velocity")
            reported_q = sample.get("dynamic_pressure_pa")
            if reported_q is not None:
                reported_q = _number(reported_q, "dynamic_pressure", minimum=0)
                if abs(q - reported_q) > 0.2 * max(q, reported_q):
                    raise ValueError("inconsistent_dynamic_pressure")
            angle_of_attack = sample.get("angle_of_attack_deg")
            if angle_of_attack is not None and abs(_number(angle_of_attack, "angle_of_attack")) > 5:
                raise ValueError("large_angle_of_attack")
            estimates.append({"sample": index, "effective_drag_area_m2": force / q,
                              "speed_m_s": speed, "dynamic_pressure_pa": q,
                              "rpc_skew_s": end - begin})
        except (ValueError, TypeError) as exc:
            rejected[str(exc)] += 1
    values = [sample["effective_drag_area_m2"] for sample in estimates]
    estimate = median(values) if values else None
    relative_spread = ((max(values) - min(values)) / estimate) if values else None
    return {"accepted_samples": estimates, "rejected_counts": dict(rejected),
            "median_effective_drag_area_m2": estimate,
            "relative_range": relative_spread,
            "eligible_for_constant_fit": len(values) >= 3 and relative_spread <= 0.5,
            "assumptions": ["Rigidly rotating atmosphere without winds",
                            "Low angle of attack when captured AoA is available",
                            "A local effective Cd*A estimate, not a KSP aerodynamic model"]}


def calibrate(record: dict, base_config: EnvConfig | None = None, *, density_source="altitude",
              fit_drag=False, max_skew=0.25) -> tuple[EnvConfig, dict]:
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise ValueError("capture schema_version must equal 1")
    if not isinstance(record.get("body"), dict):
        raise ValueError("capture must contain a body object")
    body = record["body"]
    base = base_config or EnvConfig()
    validate_config(base)
    updates = {field: _number(body.get(key), f"body.{key}", minimum=0 if field == "atmosphere_depth" else None,
                             strictly_positive=field != "atmosphere_depth")
               for field, key in BODY_KEYS.items()}
    omega = body.get("rotational_speed_rad_s")
    if omega is not None:
        omega = _number(omega, "body.rotational_speed_rad_s")
        if omega == 0:
            raise ValueError("nonrotating bodies require a simulator angular-speed parameter; finite rotation_period cannot represent zero spin")
        if not math.isclose(abs(omega) * updates["rotation_period"], 2 * math.pi, rel_tol=1e-4):
            raise ValueError("captured rotation period and angular speed disagree")
        updates["rotation_period"] = math.copysign(updates["rotation_period"], omega)
    updates["atmosphere"] = atmosphere_table(body, density_source=density_source)
    samples = record.get("flight_samples", [])
    if not isinstance(samples, list):
        raise ValueError("flight_samples must be a list")
    diagnostics = drag_diagnostics(samples, max_skew=max_skew)
    if fit_drag:
        if not diagnostics["eligible_for_constant_fit"]:
            raise ValueError("--fit-drag needs at least three coherent low-AoA flight samples with effective Cd*A range <= 50% of its median")
        updates["drag_area"] = diagnostics["median_effective_drag_area_m2"]
    config = replace(base, **updates)
    validate_config(config)
    provenance = {
        "schema_version": 1,
        "calibration_scope": "body_and_local_effective_drag" if fit_drag else "body_only",
        "capture_source": record.get("source"),
        "captured_at_utc": record.get("captured_at_utc"),
        "body_name": body.get("name"),
        "vessel_name": (record.get("vessel") or {}).get("name"),
        "imported_fields": list(updates),
        "units": {"distance": "m", "mass": "kg", "time": "s", "pressure": "Pa", "density": "kg/m^3"},
        "density_source": density_source,
        "craft_imported": False,
        "craft_source": "supplied_base_config" if base_config else "generic_two_stage_defaults",
        "retained_base_fields": [field for field in asdict(base) if field not in updates],
        "drag_diagnostics": diagnostics,
        "capture_warnings": record.get("warnings", []),
        "limitations": [
            "The captured vessel's masses, engine curves, resources, and staging have not been imported.",
            "The raw capture lacks sufficient crossfeed/topology evidence for automatic sequential-stage reconstruction.",
            "Attitude response, aero Mach/AoA/staging effects, terrain, 3D motion, and Unity physics remain unvalidated.",
            "An altitude atmosphere table is a snapshot and does not reproduce temperature changes with position or time.",
            "No agreement with a live KSP flight trajectory has been established by this import.",
        ],
    }
    return config, provenance


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--density-source", choices=("altitude", "position"), default="altitude")
    parser.add_argument("--fit-drag", action="store_true")
    parser.add_argument("--max-sample-skew", type=float, default=0.25)
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_sample_skew) or args.max_sample_skew < 0:
        parser.error("--max-sample-skew must be finite and nonnegative")
    try:
        output_path = args.output.resolve()
        provenance_path = args.output.with_suffix('.provenance.json').resolve()
        if output_path == provenance_path:
            raise ValueError('configuration and provenance must have distinct output paths')
        destinations = {output_path, provenance_path}
        inputs = {args.capture.resolve()}
        if args.base_config:
            inputs.add(args.base_config.resolve())
        if destinations & inputs:
            raise ValueError('output and its provenance must differ from both input files')
        capture_bytes = args.capture.read_bytes()
        base = EnvConfig.load(args.base_config) if args.base_config else None
        config, provenance = calibrate(json.loads(capture_bytes), base, density_source=args.density_source,
                                       fit_drag=args.fit_drag, max_skew=args.max_sample_skew)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(1, f"Calibration failed: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    provenance.update(capture_path=str(args.capture.resolve()),
                      capture_sha256=hashlib.sha256(capture_bytes).hexdigest(),
                      base_config_path=str(args.base_config.resolve()) if args.base_config else None)
    config.save(args.output)
    provenance_path = args.output.with_suffix(".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.output} and {provenance_path}; {provenance['calibration_scope']}; "
          f"craft retained from {provenance['craft_source']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
