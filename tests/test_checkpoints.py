"""Moved checkpoints retain their physics/action contract before any work runs."""
from dataclasses import asdict
import json

import numpy as np
import pytest
from gymnasium.spaces import Box
from stable_baselines3 import PPO

from orbiter.env import EnvConfig, HeadlessOrbiterEnv
from orbiter import live, train


@pytest.fixture
def checkpoint(tmp_path):
    config = EnvConfig(payload_mass=940., randomization=0.)
    env = HeadlessOrbiterEnv(config)
    model = PPO('MlpPolicy', env, gamma=config.gamma, n_steps=2, batch_size=2,
                policy_kwargs={'net_arch': {'pi': [8], 'vf': [8]}}, device='cpu')
    model.orbiter_config, model.orbiter_algorithm = asdict(config), 'ppo'
    path = tmp_path / 'policy.zip'
    train.atomic_save(model, path)
    env.close()
    return path, model, config


@pytest.mark.parametrize('field,value', [('residual', True), ('target_altitude', 85000.), ('gamma', .95)])
def test_sidecar_cannot_change_embedded_policy_contract(checkpoint, field, value):
    path, model, config = checkpoint
    metadata_path = path.with_suffix('.metadata.json')
    metadata = json.loads(metadata_path.read_text())
    metadata['environment'][field] = value
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='contract|metadata|gamma'):
        live.load_controller(path)


def test_loader_rejects_checkpoint_with_different_action_bounds(checkpoint):
    path, model, config = checkpoint
    model.action_space = Box(0., 1., (2,), dtype=np.float32)
    train.atomic_save(model, path)
    with pytest.raises(ValueError, match='contract|bounds|space'):
        live.load_controller(path)


def test_loader_rejects_embedded_algorithm_disagreement(checkpoint):
    path, model, config = checkpoint
    model.orbiter_algorithm = 'sac'
    model.save(path)
    with pytest.raises(ValueError, match='algorithm|contract|metadata'):
        live.load_controller(path)


def test_loader_rejects_learner_and_reward_discount_mismatch(checkpoint):
    path, model, config = checkpoint
    model.gamma = .95
    train.atomic_save(model, path)
    with pytest.raises(ValueError, match='gamma|discount'):
        live.load_controller(path)


def test_older_config_fields_normalize_to_current_zero_defaults(checkpoint):
    path, model, config = checkpoint
    del model.orbiter_config['drag_alpha_area'], model.orbiter_config['lift_area']
    train.atomic_save(model, path)
    restored, restored_config = live.load_controller(path)
    assert restored_config == config
    assert restored.gamma == config.gamma


def test_missing_resume_metadata_fails_before_overwriting_run(tmp_path, monkeypatch):
    model = tmp_path / 'model.zip'
    model.write_bytes(b'not loaded without a contract')
    output = tmp_path / 'existing-run'; output.mkdir()
    original_config = output / 'config.json'; original_config.write_text('existing experiment evidence')
    def forbidden(*args, **kwargs):
        raise AssertionError('Model loading must not precede contract availability')
    monkeypatch.setattr(PPO, 'load', forbidden)
    with pytest.raises(ValueError, match='metadata|contract'):
        train.main(['--resume', str(model), '--run-dir', str(output), '--steps', '1',
                    '--bc-episodes', '0', '--skip-baselines'])
    assert original_config.read_text() == 'existing experiment evidence'
    assert not (output / 'progress.jsonl').exists()


def test_resume_checks_embedded_contract_before_overwriting_run(checkpoint, tmp_path, monkeypatch):
    path, model, config = checkpoint
    metadata_path = path.with_suffix('.metadata.json')
    metadata = json.loads(metadata_path.read_text())
    metadata['environment']['residual'] = True
    metadata_path.write_text(json.dumps(metadata))
    output = tmp_path / 'existing'; output.mkdir()
    original = output / 'config.json'; original.write_text('original experiment evidence')
    def forbidden(*args, **kwargs):
        raise AssertionError('Baseline evaluation must not precede contract validation')
    monkeypatch.setattr(train, 'evaluate_policy', forbidden)
    with pytest.raises(ValueError, match='contract|metadata'):
        train.main(['--resume', str(path), '--run-dir', str(output), '--steps', '1', '--bc-episodes', '0'])
    assert original.read_text() == 'original experiment evidence'
