"""Fit a local planar aerodynamic approximation from read-only kRPC forces.

    python -m orbiter.fit_aero --capture runs/ksp_terrier_ascent_capture.json \
        --base-config orbiter/configs/orbiter_rl_terrier.json --output runs/terrier_aero_fit.json

Every fourth coherent sample is held out. The remaining first-observed-stage
samples fit CdA = base + induced*sin(alpha)^2 and ClA = lift*sin(2*alpha).
These global simulator parameters are not validated for later stages or outside
the sampled flight regime. Existing profiles and KSP controls are never changed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .calibrate import validate_config
from .env import EnvConfig


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _vector(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must contain three finite coordinates")
    return np.asarray([_finite(item, label) for item in value])


def measured_samples(record, *, min_q=20., max_skew=.25):
    """Project measured forces onto drag and the signed local planar normal."""
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise ValueError("capture schema_version must equal 1")
    samples = record.get("flight_samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("capture requires flight_samples")
    stage = samples[0].get("stage") if isinstance(samples[0], dict) else None
    if isinstance(stage, bool) or not isinstance(stage, int):
        raise ValueError("first sample must identify its integer KSP stage")
    accepted, rejected = [], Counter()
    stage_ended = False
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            rejected["invalid sample"] += 1
            continue
        if sample.get("stage") != stage:
            stage_ended = True
        if stage_ended:
            rejected["outside first observed stage"] += 1
            continue
        try:
            begin = _finite(sample.get("ut_begin_s"), "ut_begin_s")
            end = _finite(sample.get("ut_end_s"), "ut_end_s")
            if not 0 <= end - begin <= max_skew:
                raise ValueError("RPC sample skew")
            q = _finite(sample.get("dynamic_pressure_pa"), "dynamic pressure")
            rho = _finite(sample.get("atmosphere_density_kg_m3"), "density")
            if q < min_q or rho <= 0:
                raise ValueError("insufficient aerodynamic signal")
            position = _vector(sample.get("position_body_fixed_m"), "body-fixed position")
            velocity = _vector(sample.get("velocity_body_fixed_m_s"), "body-fixed velocity")
            inertial_position = _vector(sample.get("position_inertial_m"), "inertial position")
            direction = _vector(sample.get("direction_inertial"), "inertial direction")
            drag = _vector(sample.get("drag_body_fixed_n"), "drag")
            lift = _vector(sample.get("lift_body_fixed_n"), "lift")
            radius, inertial_radius = np.linalg.norm(position), np.linalg.norm(inertial_position)
            speed, direction_norm = np.linalg.norm(velocity), np.linalg.norm(direction)
            if min(radius, inertial_radius, speed, direction_norm) <= 0:
                raise ValueError("zero direction/position/airspeed")
            inertial_east = np.array([-inertial_position[2], 0., inertial_position[0]])
            if np.linalg.norm(inertial_east) <= 1e-6:
                raise ValueError("undefined prograde direction at pole")
            if float(direction @ inertial_east / np.linalg.norm(inertial_east) / direction_norm) < -.01:
                raise ValueError("retrograde craft direction is outside planar fit assumptions")
            if abs(.5 * rho * speed**2 - q) > .2 * q:
                raise ValueError("inconsistent dynamic pressure")
            radial = position / radius
            vertical_speed = float(velocity @ radial)
            horizontal = velocity - vertical_speed * radial
            horizontal_speed = np.linalg.norm(horizontal)
            if horizontal_speed < 1e-6:
                raise ValueError("undefined horizontal flight direction")
            normal = (horizontal_speed * radial - vertical_speed * horizontal / horizontal_speed) / speed
            drag_magnitude, lift_magnitude = np.linalg.norm(drag), np.linalg.norm(lift)
            if drag_magnitude <= 0 or -float(drag @ velocity) / (drag_magnitude * speed) < .95:
                raise ValueError("drag is not opposed to air velocity")
            if lift_magnitude > 1e-6 and abs(float(lift @ velocity)) / (lift_magnitude * speed) > .05:
                raise ValueError("lift is not perpendicular to air velocity")
            # Local pitch is invariant to kRPC's inertial/body-fixed azimuth.
            # This planar projection assumes prograde horizontal craft direction.
            pitch = math.asin(float(np.clip(direction @ inertial_position / direction_norm / inertial_radius, -1, 1)))
            alpha = pitch - math.atan2(vertical_speed, horizontal_speed)
            planar_lift_force = float(lift @ normal)
            accepted.append(dict(source_index=index, ut=begin, alpha=alpha, q=q,
                                 drag_area=float(drag_magnitude / q),
                                 signed_lift_area=planar_lift_force / q,
                                 orthogonal_lift_force_n=float(np.linalg.norm(lift-planar_lift_force*normal)),
                                 altitude=sample.get("altitude_m"), mach=sample.get("mach")))
        except ValueError as error:
            rejected[str(error)] += 1
    return accepted, stage, dict(rejected)


def _design(samples):
    alpha = np.asarray([sample["alpha"] for sample in samples])
    return np.column_stack([np.ones(len(alpha)), np.sin(alpha)**2]), np.sin(2 * alpha)


def _fit(samples):
    drag_design, lift_design = _design(samples)
    if np.linalg.matrix_rank(drag_design) < 2 or np.linalg.norm(lift_design) < 1e-5:
        raise ValueError("observed angles cannot identify induced drag and lift")
    drag_target = np.asarray([sample["drag_area"] for sample in samples])
    lift_target = np.asarray([sample["signed_lift_area"] for sample in samples])
    drag = np.linalg.lstsq(drag_design, drag_target, rcond=None)[0]
    lift = np.linalg.lstsq(lift_design[:, None], lift_target, rcond=None)[0][0]
    coefficients = dict(drag_area=float(drag[0]), drag_alpha_area=float(drag[1]), lift_area=float(lift))
    if any(not math.isfinite(value) or value < 0 for value in coefficients.values()):
        raise ValueError("unconstrained fit produced negative/nonfinite aerodynamic areas; this law is unsupported by the samples")
    return coefficients


def _metrics(samples, coefficients):
    drag_design, lift_design = _design(samples)
    q = np.asarray([sample["q"] for sample in samples])
    drag_error = drag_design @ [coefficients["drag_area"], coefficients["drag_alpha_area"]] - [sample["drag_area"] for sample in samples]
    lift_error = lift_design * coefficients["lift_area"] - [sample["signed_lift_area"] for sample in samples]
    rms = lambda value: float(np.sqrt(np.mean(np.asarray(value)**2)))
    planar_lift_error = lift_error * q
    orthogonal_lift_force = np.asarray([sample['orthogonal_lift_force_n'] for sample in samples])
    planar_lift_rmse = rms(planar_lift_error)
    return dict(samples=len(samples), drag_area_rmse_m2=rms(drag_error),
                lift_area_rmse_m2=rms(lift_error), drag_force_rmse_n=rms(drag_error * q),
                planar_lift_force_rmse_n=planar_lift_rmse,
                lift_force_rmse_n=planar_lift_rmse,  # Compatibility alias.
                ignored_orthogonal_lift_force_rms_n=rms(orthogonal_lift_force),
                lift_vector_force_rmse_n=rms(np.hypot(planar_lift_error, orthogonal_lift_force)))


def fit_aerodynamics(record, base_config, *, holdout_every=4, min_q=20., max_skew=.25):
    validate_config(base_config)
    if not isinstance(holdout_every, int) or holdout_every < 2:
        raise ValueError("holdout_every must be an integer of at least two")
    if not math.isfinite(min_q) or min_q <= 0 or not math.isfinite(max_skew) or max_skew < 0:
        raise ValueError("min_q must be positive and max_skew nonnegative, both finite")
    samples, stage, rejected = measured_samples(record, min_q=min_q, max_skew=max_skew)
    train = [sample for index, sample in enumerate(samples) if index % holdout_every]
    holdout = [sample for index, sample in enumerate(samples) if not index % holdout_every]
    if len(train) < 6 or len(holdout) < 2:
        raise ValueError("need at least six fitting samples and two held-out samples")
    coefficients = _fit(train)
    config = replace(base_config, **coefficients)
    validate_config(config)
    baseline = {key: getattr(base_config, key) for key in coefficients}
    finite_range = lambda key: [float(function([sample[key] for sample in samples])) for function in (min, max)]
    provenance = dict(
        schema_version=1, calibration_scope="local_first_stage_aerodynamics",
        capture_source=record.get("source"), captured_at_utc=record.get("captured_at_utc"),
        first_observed_krpc_stage=stage, coherent_samples=len(samples), rejected_counts=rejected,
        coefficients=coefficients, coefficient_units="effective m^2",
        fitting_method="Unweighted least squares on measured Cd*A and signed Cl*A; no held-out samples used for fitting",
        force_law=dict(drag="CdA = drag_area + drag_alpha_area*sin(alpha)^2",
                       lift="ClA = lift_area*sin(2*alpha)",
                       alpha="local pitch - atan2(radial airspeed, horizontal airspeed)"),
        metric_definitions=dict(
            planar_lift_force_rmse_n="Error in the modeled local planar lift component only",
            lift_force_rmse_n="Compatibility alias for planar_lift_force_rmse_n",
            ignored_orthogonal_lift_force_rms_n="Measured lift perpendicular to the modeled local planar normal, which this model omits",
            lift_vector_force_rmse_n="Full lift-vector error, combining planar fit error and omitted orthogonal force"),
        split=dict(holdout_every=holdout_every, holdout_offset=0,
                   training_source_indices=[sample["source_index"] for sample in train],
                   heldout_source_indices=[sample["source_index"] for sample in holdout]),
        observed_ranges=dict(ut_s=finite_range("ut"), alpha_deg=[math.degrees(value) for value in finite_range("alpha")],
                             dynamic_pressure_pa=finite_range("q")),
        fit_metrics=_metrics(train, coefficients), heldout_metrics=_metrics(holdout, coefficients),
        baseline_heldout_metrics=_metrics(holdout, baseline),
        thresholds=dict(min_dynamic_pressure_pa=min_q, max_rpc_skew_s=max_skew,
                        max_relative_q_error=.2, min_drag_velocity_alignment=.95,
                        max_lift_velocity_alignment=.05),
        applied_globally_to_all_stages=True, later_stages_validated=False,
        limitations=[
            "Held-out samples are interleaved within the same flight, not an independent held-out flight.",
            "The coefficients interpolate the observed first-stage regime; high-angle/Mach extrapolation is unvalidated.",
            "The simulator applies these scalar coefficients globally; later-stage geometry has not been validated.",
            "The projection assumes prograde planar flight and a rigidly rotating atmosphere without winds.",
            "This fit does not validate 3D attitude, staging transients, joints, terrain, or Unity physics.",
        ],
    )
    return config, provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout-every", type=int, default=4)
    parser.add_argument("--min-q", type=float, default=20.)
    parser.add_argument("--max-sample-skew", type=float, default=.25)
    args = parser.parse_args(argv)
    try:
        output_path = args.output.resolve()
        provenance_path = args.output.with_suffix('.provenance.json').resolve()
        if output_path == provenance_path:
            raise ValueError('configuration and provenance must have distinct output paths')
        destinations = {output_path, provenance_path}
        if destinations & {args.capture.resolve(), args.base_config.resolve()}:
            raise ValueError("output and its provenance must differ from both input files")
        raw_capture, raw_base = args.capture.read_bytes(), args.base_config.read_bytes()
        config, provenance = fit_aerodynamics(json.loads(raw_capture), EnvConfig.load(args.base_config),
                                             holdout_every=args.holdout_every, min_q=args.min_q,
                                             max_skew=args.max_sample_skew)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(1, f"Aerodynamic fit failed: {error}\n")
    provenance.update(capture_path=str(args.capture.resolve()), capture_sha256=hashlib.sha256(raw_capture).hexdigest(),
                      base_config_path=str(args.base_config.resolve()), base_config_sha256=hashlib.sha256(raw_base).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config.save(args.output)
    provenance_path = args.output.with_suffix(".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.output} and {provenance_path}; {len(provenance['split']['training_source_indices'])} fit / "
          f"{len(provenance['split']['heldout_source_indices'])} held out; coefficients={provenance['coefficients']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
