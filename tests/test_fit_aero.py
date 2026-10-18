"""Inverse-force fitting, independent holdouts, and calibration scope checks."""
from copy import deepcopy
from dataclasses import asdict
import json
import math

import numpy as np
import pytest

from orbiter.env import EnvConfig
from orbiter.fit_aero import fit_aerodynamics, main, measured_samples


def known_force_capture(count=20, *, induced=6.7):
    """Build measured vectors from known coefficients in a fixed local frame."""
    samples = []
    gamma = math.radians(35.)
    relative_direction = np.array([math.sin(gamma), 0., math.cos(gamma)])
    normal = np.array([math.cos(gamma), 0., -math.sin(gamma)])
    for index, alpha in enumerate(np.linspace(-18., 5., count)):
        speed = 100. + 5. * index
        q = .5 * speed**2
        angle = math.radians(alpha)
        drag_area = .8 + induced * math.sin(angle)**2
        lift_area = 4.5 * math.sin(2 * angle)
        pitch = gamma + angle
        samples.append({
            "stage": 3, "ut_begin_s": 10. + index, "ut_end_s": 10. + index,
            "dynamic_pressure_pa": q, "atmosphere_density_kg_m3": 1.,
            "position_body_fixed_m": [610000., 0., 0.],
            "velocity_body_fixed_m_s": (speed * relative_direction).tolist(),
            "position_inertial_m": [610000., 0., 0.],
            "direction_inertial": [math.sin(pitch), 0., math.cos(pitch)],
            "drag_body_fixed_n": (-q * drag_area * relative_direction).tolist(),
            "lift_body_fixed_n": (q * lift_area * normal).tolist(),
            "altitude_m": 10000., "mach": speed / 300.,
        })
    return {"schema_version": 1, "source": "synthetic_unit_test", "flight_samples": samples}


def test_inverse_problem_recovers_known_coefficients_and_signed_lift():
    base = EnvConfig()
    config, provenance = fit_aerodynamics(known_force_capture(), base)
    assert config.drag_area == pytest.approx(.8, abs=1e-12)
    assert config.drag_alpha_area == pytest.approx(6.7, abs=1e-11)
    assert config.lift_area == pytest.approx(4.5, abs=1e-12)
    assert provenance["fit_metrics"]["samples"] == 15
    assert provenance["heldout_metrics"]["samples"] == 5
    assert provenance["heldout_metrics"]["drag_force_rmse_n"] < 1e-9
    assert provenance["heldout_metrics"]["lift_force_rmse_n"] < 1e-9
    assert provenance["baseline_heldout_metrics"]["lift_force_rmse_n"] > 1000
    assert config.stages == base.stages and config.atmosphere == base.atmosphere
    assert provenance["applied_globally_to_all_stages"]
    assert not provenance["later_stages_validated"]
    assert provenance["calibration_scope"] == "local_first_stage_aerodynamics"


def test_holdout_force_labels_do_not_influence_fitted_coefficients():
    record = known_force_capture()
    for sample in record["flight_samples"][::4]:
        sample["drag_body_fixed_n"] = [value * 2 for value in sample["drag_body_fixed_n"]]
        sample["lift_body_fixed_n"] = [value * 2 for value in sample["lift_body_fixed_n"]]
    config, provenance = fit_aerodynamics(record, EnvConfig())
    assert config.drag_area == pytest.approx(.8)
    assert config.drag_alpha_area == pytest.approx(6.7)
    assert config.lift_area == pytest.approx(4.5)
    assert provenance["heldout_metrics"]["drag_force_rmse_n"] > 1000
    assert provenance["heldout_metrics"]["lift_force_rmse_n"] > 1000
    assert provenance["fit_metrics"]["drag_force_rmse_n"] < 1e-9
    assert provenance["split"]["heldout_source_indices"] == [0, 4, 8, 12, 16]


def test_ignored_out_of_plane_lift_is_reported_separately_from_planar_fit_error():
    record = known_force_capture()
    for sample in record['flight_samples']:
        # North is perpendicular to the modeled east/radial plane. This force
        # must not change the planar fit, or disappear from fit diagnostics.
        sample['lift_body_fixed_n'][1] += 10 * sample['dynamic_pressure_pa']
    config, provenance = fit_aerodynamics(record, EnvConfig())
    assert config.drag_area == pytest.approx(.8)
    assert config.drag_alpha_area == pytest.approx(6.7)
    assert config.lift_area == pytest.approx(4.5)
    metrics = provenance['heldout_metrics']
    assert metrics['planar_lift_force_rmse_n'] < 1e-9
    assert metrics['lift_force_rmse_n'] == metrics['planar_lift_force_rmse_n']
    indices = provenance['split']['heldout_source_indices']
    expected = np.sqrt(np.mean([record['flight_samples'][i]['lift_body_fixed_n'][1]**2
                                for i in indices]))
    assert metrics['ignored_orthogonal_lift_force_rms_n'] == pytest.approx(expected)
    assert metrics['lift_vector_force_rmse_n'] == pytest.approx(expected)
    assert metrics['lift_vector_force_rmse_n'] > 100000.


def test_later_stages_and_returns_to_old_stage_never_enter_fit():
    record = known_force_capture()
    for stage in (2, 3):
        extra = deepcopy(record["flight_samples"][-1])
        extra["stage"] = stage
        extra["drag_body_fixed_n"] = [-1e9, 0., 0.]
        record["flight_samples"].append(extra)
    config, provenance = fit_aerodynamics(record, EnvConfig())
    assert config.drag_area == pytest.approx(.8)
    assert provenance["coherent_samples"] == 20
    assert provenance["rejected_counts"]["outside first observed stage"] == 2


@pytest.mark.parametrize("mutation", [
    {"dynamic_pressure_pa": 1.}, {"dynamic_pressure_pa": 1e8},
    {"ut_end_s": 100.}, {"drag_body_fixed_n": [math.nan, 0., 0.]},
    {"drag_body_fixed_n": [1e6, 1e6, 1e6]},
    {"lift_body_fixed_n": [1e6, 1e6, 1e6]},
    {"direction_inertial": [0., 0., -1.]},
    {"position_body_fixed_m": [0., 0., 0.]},
])
def test_incoherent_vectors_pressure_timing_and_retrograde_are_excluded(mutation):
    record = known_force_capture()
    record["flight_samples"][1].update(mutation)
    samples, stage, rejected = measured_samples(record)
    assert stage == 3
    assert len(samples) == 19
    assert rejected
    assert 1 not in [sample["source_index"] for sample in samples]


def test_degenerate_angles_are_rejected_instead_of_arbitrary_coefficients():
    record = known_force_capture()
    record["flight_samples"] = [deepcopy(record["flight_samples"][0]) for _ in range(20)]
    with pytest.raises(ValueError, match="cannot identify"):
        fit_aerodynamics(record, EnvConfig())


def test_negative_unphysical_fitted_area_is_rejected():
    with pytest.raises(ValueError, match="negative/nonfinite"):
        fit_aerodynamics(known_force_capture(induced=-1.), EnvConfig())


@pytest.mark.parametrize("kwargs", [
    {"holdout_every": 1}, {"min_q": math.nan}, {"min_q": -1}, {"max_skew": -.1},
])
def test_invalid_fit_controls_are_rejected(kwargs):
    with pytest.raises(ValueError):
        fit_aerodynamics(known_force_capture(), EnvConfig(), **kwargs)


def test_not_enough_coherent_samples_is_explicit():
    with pytest.raises(ValueError, match="six fitting samples"):
        fit_aerodynamics(known_force_capture(count=5), EnvConfig())


def test_cli_output_roundtrips_and_provenance_has_split_and_source_hashes(tmp_path):
    capture, base, output = [tmp_path / name for name in ("capture.json", "base.json", "fit.json")]
    capture.write_text(json.dumps(known_force_capture()))
    EnvConfig().save(base)
    original_base = base.read_bytes()
    assert main(["--capture", str(capture), "--base-config", str(base), "--output", str(output)]) == 0
    fitted = EnvConfig.load(output)
    assert fitted.lift_area == pytest.approx(4.5)
    provenance = json.loads(output.with_suffix(".provenance.json").read_text())
    assert len(provenance["capture_sha256"]) == len(provenance["base_config_sha256"]) == 64
    assert provenance["heldout_metrics"]["samples"] == 5
    assert base.read_bytes() == original_base
    expected, _ = fit_aerodynamics(known_force_capture(), EnvConfig())
    assert asdict(fitted) == asdict(expected)


def test_cli_rejects_overwriting_base_profile(tmp_path):
    capture, base = tmp_path / "capture.json", tmp_path / "base.json"
    capture.write_text(json.dumps(known_force_capture()))
    EnvConfig().save(base)
    original = base.read_bytes()
    with pytest.raises(SystemExit) as error:
        main(["--capture", str(capture), "--base-config", str(base), "--output", str(base)])
    assert error.value.code == 1
    assert base.read_bytes() == original


def test_cli_provenance_cannot_overwrite_capture(tmp_path):
    capture, base = tmp_path / "fit.provenance.json", tmp_path / "base.json"
    capture.write_text(json.dumps(known_force_capture()))
    EnvConfig().save(base)
    original = capture.read_bytes()
    with pytest.raises(SystemExit):
        main(["--capture", str(capture), "--base-config", str(base),
              "--output", str(tmp_path / "fit.json")])
    assert capture.read_bytes() == original


def test_config_and_provenance_cannot_share_symlink_target(tmp_path):
    capture, base, output = [tmp_path/name for name in ('capture.json', 'base.json', 'fit.json')]
    capture.write_text(json.dumps(known_force_capture()))
    EnvConfig().save(base)
    output.with_suffix('.provenance.json').symlink_to(output)
    with pytest.raises(SystemExit):
        main(['--capture', str(capture), '--base-config', str(base), '--output', str(output)])
    assert not output.exists()


def test_output_named_provenance_json_has_distinct_valid_sidecar(tmp_path):
    capture, base, output = [tmp_path/name for name in (
        'capture.json', 'base.json', 'fit.provenance.json')]
    capture.write_text(json.dumps(known_force_capture()))
    EnvConfig().save(base)
    assert main(['--capture', str(capture), '--base-config', str(base), '--output', str(output)]) == 0
    assert EnvConfig.load(output).drag_area == pytest.approx(.8)
    assert output.with_suffix('.provenance.json').name == 'fit.provenance.provenance.json'
