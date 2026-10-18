"""Keep published checkpoint claims tied to their original flight evidence."""
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / 'validation/terrier/manifest.json').read_text())


@pytest.mark.parametrize('relative,expected', MANIFEST['files'].items())
def test_published_evidence_hashes(relative, expected):
    path = (ROOT / relative).resolve()
    assert path.is_relative_to(ROOT)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


@pytest.mark.parametrize('flight', MANIFEST['flights'], ids=['ppo', 'ppo_aero'])
def test_checkpoint_claims_match_flight_and_unseen_evaluation(flight):
    model = ROOT / flight['model']
    assert hashlib.sha256(model.read_bytes()).hexdigest() == flight['model_sha256']
    metadata = json.loads(model.with_suffix('.metadata.json').read_text())
    assert metadata['num_timesteps'] == flight['training_transitions']
    events = [json.loads(line) for line in (ROOT / flight['flight_log']).read_text().splitlines()]
    end = next(row for row in events if row['event'] == 'flight_end')
    release = next(row for row in events if row['event'] == 'controls_released')
    assert end['success'] is True
    assert end['termination_reason'] == 'orbit_achieved'
    assert release['errors'] == []
    telemetry = end['final_telemetry']
    assert telemetry['periapsis'] == flight['periapsis_m']
    assert telemetry['apoapsis'] == flight['apoapsis_m']
    assert telemetry['time'] == flight['insertion_seconds']
    assert telemetry['stage'] == 1
    evaluation = json.loads((ROOT / flight['simulator_test']).read_text())
    episodes = evaluation['episodes']
    assert len({row['seed'] for row in episodes}) == len(episodes) == flight['unseen_simulator_episodes']
    assert sum(row['success'] for row in episodes) == flight['unseen_simulator_successes']
    coast = json.loads((ROOT / flight['coast_record']).read_text())
    if 'trials' in coast:
        assert coast['source'] == 'actual_ksp_krpc'
        assert coast['success'] is True
        assert coast['model_sha256'] == flight['model_sha256']
        samples = coast['trials'][0]['coast']['samples']
    else:
        samples = [coast]
    config = metadata['environment']
    for sample in samples:
        assert sample['throttle'] == 0
        assert 'orbiting' in sample['situation']
        assert sample['periapsis'] >= max(config['atmosphere_depth'] + 1000,
                                         config['target_altitude'] - config['orbit_tolerance'])
        assert sample['apoapsis'] <= config['target_altitude'] + config['orbit_tolerance']
