"""Capture-to-simulator contracts, SI units, and calibration proof boundaries."""
from dataclasses import asdict, replace
import hashlib
import json
import math

import pytest

from orbiter.calibrate import calibrate, drag_diagnostics, main
from orbiter.env import EnvConfig, HeadlessOrbiterEnv, Stage


@pytest.fixture
def capture_record():
    return {
        "schema_version": 1, "source": "synthetic_unit_test", "captured_at_utc": "test",
        "body": {
            "name": "Test body", "gravitational_parameter_m3_s2": 4e12,
            "equatorial_radius_m": 700_000., "rotational_period_s": 20_000.,
            "rotational_speed_rad_s": 2 * math.pi / 20_000.,
            "has_atmosphere": True, "atmosphere_depth_m": 50_000.,
            "atmosphere": [
                {"altitude_m": 0., "pressure_pa": 101325., "density_kg_m3": 1.2,
                 "position_aware_density_kg_m3": 1.3},
                {"altitude_m": 25_000., "pressure_pa": 2500., "density_kg_m3": .04,
                 "position_aware_density_kg_m3": .05},
                {"altitude_m": 50_000., "pressure_pa": 0., "density_kg_m3": 0.,
                 "position_aware_density_kg_m3": 0.},
            ],
        },
        "vessel": {"name": "unimported craft", "mass_kg": 123456.},
        "flight_samples": [], "warnings": [],
    }


def drag_sample(area=4., **overrides):
    # q = .5 * .25 * 100^2 = 1250 Pa; F = q * CdA = 5000 N.
    sample = {"ut_begin_s": 10., "ut_end_s": 10.1,
              "atmosphere_density_kg_m3": .25,
              "velocity_body_fixed_m_s": [100., 0., 0.],
              "drag_body_fixed_n": [-1250. * area, 0., 0.],
              "dynamic_pressure_pa": 1250., "angle_of_attack_deg": 2.}
    return {**sample, **overrides}


def test_body_import_preserves_si_pressure_and_generic_craft(capture_record):
    config, provenance = calibrate(capture_record)
    assert config.mu == 4e12 and config.radius == 700_000.
    assert config.rotation_period == 20_000. and config.atmosphere_depth == 50_000.
    assert config.atmosphere == ((0., 1.2, 101325.), (25_000., .04, 2500.), (50_000., 0., 0.))
    # This detects multiplying already-SI captured pressures by 101325 again.
    assert HeadlessOrbiterEnv(config).atmosphere_at(0.) == pytest.approx((1.2, 101325.))
    assert config.stages == EnvConfig().stages
    assert config.payload_mass == EnvConfig().payload_mass
    assert not provenance["craft_imported"]
    assert provenance["craft_source"] == "generic_two_stage_defaults"
    assert provenance["calibration_scope"] == "body_only"
    assert "stages" in provenance["retained_base_fields"]


def test_custom_base_craft_and_task_are_preserved(capture_record):
    base = EnvConfig(payload_mass=1234., stages=(Stage(90, 900, 50000, 300, 250),),
                     target_altitude=100_000., randomization=.1, drag_area=2.5)
    config, provenance = calibrate(capture_record, base)
    assert config.stages == base.stages and config.payload_mass == 1234.
    assert config.target_altitude == 100_000. and config.randomization == .1
    assert config.drag_area == 2.5
    assert provenance["craft_source"] == "supplied_base_config"
    assert "drag_area" not in provenance["imported_fields"]


def test_position_density_is_explicit_and_does_not_change_pressure(capture_record):
    config, provenance = calibrate(capture_record, density_source="position")
    assert config.atmosphere[0] == (0., 1.3, 101325.)
    assert provenance["density_source"] == "position"
    capture_record["body"]["atmosphere"][1]["position_aware_density_kg_m3"] = None
    with pytest.raises(ValueError, match="finite"):
        calibrate(capture_record, density_source="position")


@pytest.mark.parametrize(("key", "value"), [
    ("gravitational_parameter_m3_s2", 0), ("equatorial_radius_m", -1),
    ("rotational_period_s", 0), ("atmosphere_depth_m", -1),
    ("gravitational_parameter_m3_s2", math.nan), ("equatorial_radius_m", math.inf),
    ("rotational_period_s", "20000"), ("atmosphere_depth_m", None),
])
def test_invalid_body_constants_are_rejected(capture_record, key, value):
    capture_record["body"][key] = value
    with pytest.raises(ValueError):
        calibrate(capture_record)


@pytest.mark.parametrize(("index", "key", "value"), [
    (1, "altitude_m", 0.), (1, "altitude_m", -1.), (2, "altitude_m", 49_999.),
    (0, "altitude_m", 1.), (1, "pressure_pa", math.nan),
    (1, "pressure_pa", -1.), (1, "pressure_pa", 200000.),
    (1, "density_kg_m3", math.inf), (1, "density_kg_m3", -1.),
    (1, "density_kg_m3", 2.),
])
def test_invalid_atmosphere_tables_are_rejected(capture_record, index, key, value):
    capture_record["body"]["atmosphere"][index][key] = value
    with pytest.raises(ValueError):
        calibrate(capture_record)


def test_rotation_direction_is_preserved_and_units_checked(capture_record):
    capture_record["body"]["rotational_speed_rad_s"] *= -1
    config, _ = calibrate(capture_record)
    assert config.rotation_period == -20_000.
    assert HeadlessOrbiterEnv(config).vy < 0
    capture_record["body"]["rotational_speed_rad_s"] *= 10
    with pytest.raises(ValueError, match="disagree"):
        calibrate(capture_record)
    capture_record["body"]["rotational_speed_rad_s"] = 0
    with pytest.raises(ValueError, match="nonrotating"):
        calibrate(capture_record)


def test_airless_body_gets_zero_atmosphere(capture_record):
    capture_record["body"].update(has_atmosphere=False, atmosphere_depth_m=0, atmosphere=[])
    config, _ = calibrate(capture_record)
    assert config.atmosphere == ((0., 0., 0.),)
    assert HeadlessOrbiterEnv(config).atmosphere_at(0) == (0., 0.)


def test_invalid_base_or_unreachable_goal_rejected(capture_record):
    with pytest.raises(ValueError, match="finite"):
        calibrate(capture_record, replace(EnvConfig(), mu=math.nan))
    capture_record["body"]["atmosphere_depth_m"] = 120_000.
    capture_record["body"]["atmosphere"][-1]["altitude_m"] = 120_000.
    with pytest.raises(ValueError, match="target orbit"):
        calibrate(capture_record)


def test_drag_estimates_are_diagnostics_until_explicitly_applied(capture_record):
    capture_record["flight_samples"] = [drag_sample(3.9), drag_sample(4.), drag_sample(4.1)]
    config, provenance = calibrate(capture_record)
    diagnostic = provenance["drag_diagnostics"]
    assert len(diagnostic["accepted_samples"]) == 3
    assert diagnostic["median_effective_drag_area_m2"] == pytest.approx(4.)
    assert diagnostic["eligible_for_constant_fit"]
    assert config.drag_area == EnvConfig().drag_area
    config, provenance = calibrate(capture_record, fit_drag=True)
    assert config.drag_area == pytest.approx(4.)
    assert provenance["calibration_scope"] == "body_and_local_effective_drag"
    assert "drag_area" in provenance["imported_fields"]


@pytest.mark.parametrize("sample", [
    drag_sample(velocity_body_fixed_m_s=[0, 0, 0]),
    drag_sample(drag_body_fixed_n=[5000, 0, 0]),
    drag_sample(drag_body_fixed_n=[0, 5000, 0]),
    drag_sample(dynamic_pressure_pa=100000),
    drag_sample(ut_end_s=11), drag_sample(angle_of_attack_deg=30),
    drag_sample(atmosphere_density_kg_m3=math.nan),
    drag_sample(velocity_body_fixed_m_s=[100, 0]),
])
def test_incoherent_drag_samples_are_rejected(sample):
    diagnostics = drag_diagnostics([sample])
    assert not diagnostics["accepted_samples"]
    assert diagnostics["rejected_counts"]
    assert not diagnostics["eligible_for_constant_fit"]


def test_drag_fit_needs_multiple_stable_samples(capture_record):
    for samples in ([], [drag_sample()], [drag_sample(1), drag_sample(4), drag_sample(10)]):
        capture_record["flight_samples"] = samples
        with pytest.raises(ValueError, match="at least three"):
            calibrate(capture_record, fit_drag=True)


def test_cli_emits_loadable_config_and_separate_honest_provenance(capture_record, tmp_path):
    source = tmp_path / "capture.json"
    source.write_text(json.dumps(capture_record))
    target = tmp_path / "nested" / "config.json"
    assert main(["--capture", str(source), "--output", str(target)]) == 0
    loaded = EnvConfig.load(target)
    assert loaded.mu == 4e12
    assert asdict(loaded) == asdict(calibrate(capture_record)[0])
    provenance = json.loads(target.with_suffix(".provenance.json").read_text())
    assert provenance["capture_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert not provenance["craft_imported"]
    assert "provenance" not in json.loads(target.read_text())


def test_invalid_capture_does_not_write_config(capture_record, tmp_path):
    capture_record["schema_version"] = 999
    source, target = tmp_path / "bad.json", tmp_path / "config.json"
    source.write_text(json.dumps(capture_record))
    with pytest.raises(SystemExit) as error:
        main(["--capture", str(source), "--output", str(target)])
    assert error.value.code == 1
    assert not target.exists()


@pytest.mark.parametrize('collision', ['capture', 'base', 'provenance_capture', 'provenance_base', 'symlink_capture'])
def test_calibration_outputs_cannot_overwrite_input_evidence(capture_record, tmp_path, collision):
    source = tmp_path / ('fit.provenance.json' if collision == 'provenance_capture' else 'capture.json')
    base = tmp_path / ('fit.provenance.json' if collision == 'provenance_base' else 'base.json')
    source.write_text(json.dumps(capture_record))
    EnvConfig().save(base)
    output = tmp_path / 'fit.json'
    if collision == 'capture':
        output = source
    elif collision == 'base':
        output = base
    elif collision == 'symlink_capture':
        output = tmp_path / 'capture_alias.json'
        output.symlink_to(source)
    originals = {path: path.read_bytes() for path in (source, base)}
    with pytest.raises(SystemExit) as error:
        main(['--capture', str(source), '--base-config', str(base), '--output', str(output)])
    assert error.value.code == 1
    assert all(path.read_bytes() == original for path, original in originals.items())


def test_calibration_config_and_provenance_cannot_share_symlink_target(capture_record, tmp_path):
    source, output = tmp_path/'capture.json', tmp_path/'fit.json'
    source.write_text(json.dumps(capture_record))
    output.with_suffix('.provenance.json').symlink_to(output)
    with pytest.raises(SystemExit):
        main(['--capture', str(source), '--output', str(output)])
    assert not output.exists()


def test_calibration_output_named_provenance_json_has_distinct_valid_sidecar(capture_record, tmp_path):
    source, output = tmp_path/'capture.json', tmp_path/'fit.provenance.json'
    source.write_text(json.dumps(capture_record))
    assert main(['--capture', str(source), '--output', str(output)]) == 0
    assert EnvConfig.load(output).radius == 700000.
    assert output.with_suffix('.provenance.json').name == 'fit.provenance.provenance.json'
