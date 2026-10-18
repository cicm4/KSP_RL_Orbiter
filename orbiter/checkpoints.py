"""Load Orbiter policies and verify their saved physics/action contracts."""
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np

from .env import EnvConfig, HeadlessOrbiterEnv


def validate_checkpoint_contract(model, metadata):
    """Check the external sidecar against the contract embedded in the model."""
    config = EnvConfig.from_dict(metadata['environment'])
    if not hasattr(model, 'orbiter_config') or not hasattr(model, 'orbiter_algorithm'):
        raise ValueError('Checkpoint lacks an embedded Orbiter environment/algorithm contract')
    embedded = EnvConfig.from_dict(model.orbiter_config)
    if asdict(embedded) != asdict(config):
        raise ValueError('Checkpoint metadata environment differs from its embedded policy contract')
    if str(model.orbiter_algorithm).lower() != str(metadata['algorithm']).lower():
        raise ValueError('Checkpoint metadata algorithm differs from its embedded algorithm')
    if not math.isclose(float(model.gamma), config.gamma, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError('Checkpoint learner gamma differs from the environment reward discount')
    expected = HeadlessOrbiterEnv(config)
    try:
        for name in ('observation_space', 'action_space'):
            actual, required = getattr(model, name), getattr(expected, name)
            if (type(actual) is not type(required) or actual.shape != required.shape
                    or actual.dtype != required.dtype
                    or not np.array_equal(actual.low, required.low)
                    or not np.array_equal(actual.high, required.high)):
                raise ValueError(f'Checkpoint {name} bounds/shape/dtype differ from the shared policy contract')
    finally:
        expected.close()
    return config


def load_controller(path):
    path = Path(path)
    metadata_path = path.with_suffix('.metadata.json')
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    else:
        source = json.loads((path.parent/'config.json').read_text())
        metadata = dict(environment=source['environment'], algorithm=source['arguments']['algo'])
    algorithm = metadata['algorithm'].lower()
    if algorithm not in {'ppo', 'sac', 'tqc'}:
        raise ValueError(f'Unsupported model algorithm {algorithm!r}')
    from stable_baselines3 import PPO, SAC
    from sb3_contrib import TQC
    model = {'ppo': PPO, 'sac': SAC, 'tqc': TQC}[algorithm].load(path, device='cpu')
    return model, validate_checkpoint_contract(model, metadata)
