"""Actual-KSP gate contracts tested without launching or operating the game."""
from dataclasses import dataclass
import hashlib
import json
import math
from types import SimpleNamespace
from uuid import uuid4

import pytest

from orbiter.env import EnvConfig
from orbiter import validate


@pytest.mark.parametrize('verify_current', [False, True])
def test_validation_busy_endpoint_rejects_before_model_or_rpc(tmp_path, monkeypatch, verify_current):
    import krpc
    from uuid import uuid4
    from orbiter.ownership import FlightOwnership

    address = f'validation-lock-test-{uuid4().hex}.invalid'
    model = tmp_path/'model.zip'
    model.write_bytes(b'fake')
    monkeypatch.setattr(validate, 'load_controller', lambda *args: pytest.fail('Busy validation loaded model'))
    monkeypatch.setattr(krpc, 'connect', lambda **kwargs: pytest.fail('Busy validation contacted RPC'))
    args = ['--model', str(model), '--output-dir', str(tmp_path/'run'), '--address', address]
    args += ['--verify-current'] if verify_current else ['--stage-plan', '3,2', '--fuel-stages', '2,1']
    with FlightOwnership(address):
        assert validate.main(args) == 1
        result = json.loads((tmp_path/'run/validation.json').read_text())
        assert 'FlightOwnershipBusy' in result['error']


@pytest.mark.parametrize('outcome', ['success', 'interrupt', 'close_error', 'summary_error'])
def test_validation_lease_covers_models_flight_close_and_summary(tmp_path, monkeypatch, outcome):
    import krpc
    from pathlib import Path
    from uuid import uuid4
    from orbiter.ownership import flight_busy

    address = f'validation-lock-test-{uuid4().hex}.invalid'
    model_path = tmp_path/'model.zip'
    model_path.write_bytes(b'fake')
    output = tmp_path/'run/validation.json'
    observed = []
    original_write = Path.write_text

    def load(*args):
        observed.append(('model', flight_busy(address)))
        return SimpleNamespace(num_timesteps=1), EnvConfig()

    def close():
        observed.append(('close', flight_busy(address)))
        if outcome == 'close_error':
            raise RuntimeError('test validation close')

    def fly(*args, **kwargs):
        observed.append(('fly', flight_busy(address)))
        if outcome == 'interrupt':
            raise KeyboardInterrupt
        return dict(success=True)

    def write(path, *args, **kwargs):
        if path == output:
            observed.append(('summary', flight_busy(address)))
            if outcome == 'summary_error':
                raise OSError('test validation summary')
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(validate, 'load_controller', load)
    monkeypatch.setattr(validate, 'fly', fly)
    monkeypatch.setattr(validate, 'verify_coast', lambda *args, **kwargs: dict(success=True))
    monkeypatch.setattr(krpc, 'connect', lambda **kwargs: SimpleNamespace(close=close))
    monkeypatch.setattr(Path, 'write_text', write)
    args = ['--model', str(model_path), '--output-dir', str(output.parent),
            '--address', address, '--stage-plan', '3,2', '--fuel-stages', '2,1']
    if outcome == 'interrupt':
        with pytest.raises(KeyboardInterrupt):
            validate.main(args)
    elif outcome == 'summary_error':
        with pytest.raises(OSError, match='validation summary'):
            validate.main(args)
    else:
        assert validate.main(args) == 0
    assert observed == [('model', True), ('fly', True), ('close', True), ('summary', True)]
    assert not flight_busy(address)


@dataclass(frozen=True)
class ReadOnlyControl:
    throttle: float = 0.


class Connection:
    def __init__(self):
        vessel = SimpleNamespace(name='Test orbiter', situation='VesselSituation.orbiting',
                                 control=ReadOnlyControl(),
                                 orbit=SimpleNamespace(apoapsis_altitude=81000.,
                                                       periapsis_altitude=72000., eccentricity=.006))
        self.space_center = SimpleNamespace(active_vessel=vessel, ut=100.)
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def actual_sample():
    return dict(ut_begin=100., ut_end=100., apoapsis=81000., periapsis=72000.,
                eccentricity=.006, throttle=0., situation='VesselSituation.orbiting')


@pytest.mark.parametrize('updates,reason', [
    ({'periapsis': 70000.}, 'periapsis_below_target'),
    ({'apoapsis': 91000.}, 'apoapsis_above_target'),
    ({'eccentricity': 1.1}, 'not_a_bound_orbit'),
    ({'throttle': .1}, 'throttle_not_zero'),
    ({'situation': 'VesselSituation.sub_orbital'}, 'vessel_not_orbiting'),
    ({'ut_end': 99.}, 'universal_time_moved_backwards'),
    ({'apoapsis': math.nan}, 'nonfinite_or_missing_telemetry'),
    ({'periapsis': math.inf}, 'nonfinite_or_missing_telemetry'),
    ({'throttle': math.nan}, 'nonfinite_or_missing_telemetry'),
])
def test_gate_rejects_invalid_actual_orbit(actual_sample, updates, reason):
    assert reason in validate.orbit_failures({**actual_sample, **updates}, EnvConfig())


def test_gate_uses_checkpoint_target_and_atmosphere(actual_sample):
    assert not validate.orbit_failures(actual_sample, EnvConfig())
    assert 'periapsis_below_target' in validate.orbit_failures(
        actual_sample, EnvConfig(target_altitude=100000.))


def test_independent_coast_requires_elapsed_simulation_time(monkeypatch):
    connection = Connection()
    monkeypatch.setattr(validate, 'wait_for_ut',
                        lambda center, target, *args, **kwargs: setattr(center, 'ut', target))
    result = validate.verify_coast(connection, EnvConfig(), seconds=30)
    assert result['success'] and result['observed_coast_seconds'] == 30
    assert len(result['samples']) == 31
    assert connection.space_center.active_vessel.control.throttle == 0


def test_independent_coast_rejects_time_revert_between_probes(monkeypatch):
    connection = Connection()
    monkeypatch.setattr(validate, 'wait_for_ut',
                        lambda center, *args, **kwargs: setattr(center, 'ut', 10.))
    with pytest.raises(RuntimeError, match='backwards'):
        validate.verify_coast(connection, EnvConfig(), seconds=30)


@pytest.mark.parametrize('extra', [
    [], ['--stage-plan', '3,1', '--fuel-stages', '2,1'],
    ['--stage-plan', '3,2', '--fuel-stages', '2,1', '--trials', '2'],
    ['--verify-current', '--pad-save', 'Validation Pad'],
    ['--verify-current', '--coast-seconds', 'nan'],
])
def test_cli_rejects_missing_or_ambiguous_stage_and_load_options(extra):
    with pytest.raises(SystemExit):
        validate.parse_args(['--model', 'checkpoint.zip', '--output-dir', 'unused', *extra])


def configure_main(tmp_path, monkeypatch):
    import krpc
    connection = Connection()
    model = tmp_path/'checkpoint.zip'
    model.write_bytes(b'test checkpoint')
    monkeypatch.setattr(validate, 'load_controller',
                        lambda path: (SimpleNamespace(num_timesteps=4096), EnvConfig()))
    monkeypatch.setattr(krpc, 'connect', lambda **kwargs: connection)
    monkeypatch.setattr(validate, 'wait_for_ut',
                        lambda center, target, *args, **kwargs: setattr(center, 'ut', target))
    return connection, model


def test_verify_current_is_read_only_and_records_attribution_limit(tmp_path, monkeypatch):
    connection, model = configure_main(tmp_path, monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError('Read-only verification must not fly/load/write controls')
    connection.space_center.load = forbidden
    monkeypatch.setattr(validate, 'fly', forbidden)
    output = tmp_path/'verification'
    assert validate.main(['--model', str(model), '--output-dir', str(output),
                          '--verify-current', '--coast-seconds', '2',
                          '--address', f'validation-test-{uuid4().hex}.invalid']) == 0
    result = json.loads((output/'validation.json').read_text())
    assert result['source'] == 'actual_ksp_krpc'
    assert result['success'] and not result['controller_flight_executed']
    assert result['model_sha256'] == hashlib.sha256(model.read_bytes()).hexdigest()
    assert result['num_timesteps'] == 4096 and connection.closed
    assert 'controller attribution' in result['limitation']


@pytest.mark.parametrize('flight_success,expected_exit', [(True, 0), (False, 1)])
def test_flight_and_independent_coast_must_both_pass(tmp_path, monkeypatch, flight_success, expected_exit):
    connection, model = configure_main(tmp_path, monkeypatch)
    def fake_flight(conn, config, **kwargs):
        assert conn is connection and kwargs['model'].num_timesteps == 4096
        assert kwargs['plan'] == (3, 2) and kwargs['fuel_stages'] == (2, 1)
        assert not kwargs.get('scripted', False)
        return dict(success=flight_success)
    monkeypatch.setattr(validate, 'fly', fake_flight)
    output = tmp_path/'flight'
    assert validate.main(['--model', str(model), '--output-dir', str(output),
                          '--stage-plan', '3,2', '--fuel-stages', '2,1',
                          '--coast-seconds', '1',
                          '--address', f'validation-test-{uuid4().hex}.invalid']) == expected_exit
    result = json.loads((output/'validation.json').read_text())
    assert result['success'] is flight_success
    assert result['controller_flight_executed']


def test_existing_validation_evidence_is_never_overwritten(tmp_path, monkeypatch):
    connection, model = configure_main(tmp_path, monkeypatch)
    output = tmp_path/'record'; output.mkdir()
    target = output/'validation.json'; target.write_text('original evidence')
    assert validate.main(['--model', str(model), '--output-dir', str(output),
                          '--verify-current', '--address', f'validation-test-{uuid4().hex}.invalid']) == 1
    assert target.read_text() == 'original evidence'
    assert not connection.closed  # no connection was opened


def test_current_orbit_with_nonzero_throttle_returns_failure(tmp_path, monkeypatch):
    connection, model = configure_main(tmp_path, monkeypatch)
    connection.space_center.active_vessel.control = ReadOnlyControl(.2)
    output = tmp_path/'powered'
    assert validate.main(['--model', str(model), '--output-dir', str(output),
                          '--verify-current', '--address', f'validation-test-{uuid4().hex}.invalid']) == 1
    result = json.loads((output/'validation.json').read_text())
    assert not result['success']
    assert result['trials'][0]['coast']['failures'] == ['throttle_not_zero']
