"""Bitwise equivalence of the optimized hot paths with the frozen originals.

``_reference_hot_paths`` holds the pre-optimization implementations verbatim.
Every comparison here is exact (bit patterns, including NaN payload and the
sign of zero), and RNG streams are compared after each run.
"""
import copy
import types

import pytest
import torch

from _reference_hot_paths import (
    ReferenceAscent,
    ReferenceCapture,
    ReferenceHistory,
    reference_fit_teacher,
    reference_ppo_train,
    reference_ppo_update,
    reference_rollout,
)
from orbiter import train_architectures as ta
from orbiter import train_generalized as tg
from orbiter.batched_capture import BatchedCapture
from orbiter.generalized_aero_env import AeroVehicleEnvelope, BatchedAeroAscent
from orbiter.generalized_env import BatchedAscent
from orbiter.sequence_policy import History, SequencePolicy
from orbiter.tensor_memo import TensorMemo
from orbiter.train_generalized import experiment_contract


def bits(x):
    if isinstance(x, torch.Tensor) and x.is_floating_point():
        return x.contiguous().view({torch.float64: torch.int64, torch.float32: torch.int32,
                                    torch.float16: torch.int16}[x.dtype])
    return x


def assert_bitwise(a, b, where=''):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), where
        for key in a:
            assert_bitwise(a[key], b[key], f'{where}.{key}')
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            assert_bitwise(x, y, f'{where}[{i}]')
    elif isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape and a.device == b.device, where
        assert torch.equal(bits(a), bits(b)), where
    elif isinstance(a, float):
        assert torch.equal(bits(torch.tensor(a, dtype=torch.float64)),
                           bits(torch.tensor(b, dtype=torch.float64))), where
    else:
        assert a == b, where


def assert_same_state(new, old):
    for key, value in vars(old).items():
        if isinstance(value, torch.Tensor):
            assert_bitwise(getattr(new, key), value, key)
    assert torch.equal(new.generator.get_state(), old.generator.get_state())


# --------------------------------------------------------------------- memo

def test_tensor_memo_invalidates_on_reassignment_inplace_writes_and_output_mutation():
    memo, calls = TensorMemo(), []
    a = torch.arange(4.)

    def compute():
        calls.append(1)
        return dict(y=a * 2)

    first = memo.get([a], (1.0,), compute)
    assert memo.get([a], (1.0,), compute)['y'] is first['y'] and len(calls) == 1
    first['extra'] = 1  # Mutating the returned dict cannot alter the cache.
    assert 'extra' not in memo.get([a], (1.0,), compute)
    a[0] = 5  # In-place write bumps the version counter.
    assert memo.get([a], (1.0,), compute)['y'][0] == 10 and len(calls) == 2
    memo.get([a], (2.0,), compute)  # Changed scalar.
    assert len(calls) == 3
    memo.get([a], (2.0,), compute)['y'].add_(1)  # Caller mutates the output.
    assert memo.get([a], (2.0,), compute)['y'][0] == 10 and len(calls) == 4
    a = a.clone()  # Reassignment: a different tensor object.
    memo.get([a], (2.0,), compute)
    assert len(calls) == 5
    with torch.inference_mode():
        b = torch.ones(2)
        memo.get([b], (), lambda: dict(y=b + 1))
        memo.get([b], (), lambda: dict(y=b + 1))
    leaf = torch.ones(2, requires_grad=True)
    assert memo.get([leaf], (), lambda: leaf * 2).requires_grad


def _record_reads(env, method):
    """Names of the env attributes read while computing ``method`` uncached."""
    reads = []
    cls = type(env)
    original = cls.__getattribute__

    def recording(self, name):
        value = original(self, name)
        if self is env and not name.startswith('__') and not callable(value):
            reads.append(name)
        return value
    for memo in [v for v in vars(env).values() if isinstance(v, TensorMemo)]:
        memo.clear()
    cls.__getattribute__ = recording
    try:
        getattr(env, method)()
    finally:
        del cls.__getattribute__
    memos = {name for name, value in vars(env).items() if isinstance(value, TensorMemo)}
    tensors = {n for n in reads if isinstance(getattr(env, n), torch.Tensor)}
    others = set(reads) - tensors - memos - {
        'TELEMETRY_INPUTS', 'OBSERVATION_INPUTS', 'CONTEXT_INPUTS', 'ATMOSPHERE_TABLES',
        'dtype', 'device'}  # Fixed at construction.
    return tensors, others


@pytest.mark.parametrize('version', [1, 2])
def test_capture_memo_input_lists_cover_every_attribute_read(version):
    env = BatchedCapture(17, seed=1, version=version)
    tensors, others = _record_reads(env, '_telemetry')
    assert tensors <= set(env.TELEMETRY_INPUTS) and not others
    tensors, others = _record_reads(env, '_observation')
    assert tensors <= set(env.OBSERVATION_INPUTS) and others <= {'version'}


@pytest.mark.parametrize('cls', [BatchedAscent, BatchedAeroAscent])
def test_ascent_memo_input_lists_cover_every_attribute_read(cls):
    env = cls(8, seed=1)
    scalars = {'mu', 'radius', 'rotation_period', 'atmosphere_depth'}
    tensors, others = _record_reads(env, '_telemetry')
    assert tensors <= {'state', *env.ATMOSPHERE_TABLES} and others <= scalars
    tensors, others = _record_reads(env, '_context')
    assert tensors <= set(env.CONTEXT_INPUTS) and others <= {'mu', 'radius'}
    tensors, others = _record_reads(env, '_observation')
    assert tensors <= set(env.OBSERVATION_INPUTS) and others <= scalars | {'max_time'}


# ------------------------------------------------------------------ history

def test_history_windows_match_reference_and_earlier_windows_never_change():
    torch.manual_seed(4)
    count, length, obs, act = 9, 5, 6, 2
    new, old = History(count, length, obs, act, 'cpu'), ReferenceHistory(count, length, obs, act, 'cpu')
    windows = []
    for step in range(60):
        observation = torch.randn(count, obs, dtype=torch.float64 if step % 7 == 0 else torch.float32)
        observation[0, 0] = -0.0
        action = None if step % 11 == 0 else torch.randn(count, act)
        if step % 13 == 5:
            reset = torch.tensor([1, 4])  # Integer world indices.
        else:
            reset = None if step % 3 == 0 else torch.rand(count) < .3
        old_out = old.append(observation, action, reset)
        new_out = new.append(observation, action, reset)
        assert_bitwise(new_out, old_out, f'step {step}')
        assert new_out.is_contiguous()
        windows.append((new_out, old_out.clone()))
    for new_out, snapshot in windows:
        assert_bitwise(new_out, snapshot)


# ------------------------------------------------------------- environments

@pytest.mark.parametrize('version', [1, 2])
def test_orbital_env_is_bitwise_identical_over_long_rollout_with_resets(version):
    torch.set_num_threads(1)
    new, old = BatchedCapture(68, seed=13, version=version), ReferenceCapture(68, seed=13, version=version)
    assert_same_state(new, old)
    noise = torch.Generator().manual_seed(2)
    for step in range(260):
        assert_bitwise(new.teacher(), old.teacher(), f'teacher {step}')
        action = old.teacher() + .3 * torch.randn(68, 3, generator=noise)
        if step % 9 == 0:
            action[:5] = 0  # Coasting worlds and unit(0) fallbacks.
        active = None if step % 4 else torch.rand(68, generator=noise) > .2
        out_new = new.step(action, active)
        out_old = old.step(action, active)
        assert_bitwise(out_new, out_old, f'step {step}')
        done = out_old[2]
        assert_bitwise(new.reset(done), old.reset(done), f'reset {step}')
        assert_bitwise(new.observation(), old.observation())
        assert_same_state(new, old)
    assert_bitwise(new.telemetry(), old.telemetry())
    assert_bitwise(new.potential(), old.potential())


def _ascent_pair(count, seed, envelope=None):
    if envelope is None:
        envelope, _, _ = experiment_contract('stock', 'aero', deadline_mode='terminal', timeout_penalty=-30)
    kwargs = dict(device='cpu', seed=seed, envelope=envelope, deadline_mode='terminal', timeout_penalty=-30)
    return ta.AscentBenchmark(count, **kwargs), ReferenceAscent(count, **kwargs)


@pytest.mark.parametrize('envelope', [None, AeroVehicleEnvelope(
    drag_alpha_area_per_tonne_max=0., lift_area_per_tonne_max=0., minimum_ideal_delta_v=6500.)],
    ids=['stock_aero', 'zero_aero_frequent_rejection'])
def test_ascent_env_is_bitwise_identical_through_staging_and_resets(envelope):
    torch.set_num_threads(1)
    new, old = _ascent_pair(96, 21, envelope)
    assert_same_state(new, old)
    noise = torch.Generator().manual_seed(3)
    for step in range(420):
        action = (old.teacher() + .15 * torch.randn(96, 2, generator=noise)).clamp(-1, 1)
        out_new, out_old = new.step(action), old.step(action)
        assert_bitwise(out_new, out_old, f'step {step}')
        done = out_old[2]
        assert_bitwise(new.reset(done), old.reset(done), f'reset {step}')
        assert_same_state(new, old)
    assert_bitwise(new.telemetry(), old.telemetry())
    assert_bitwise(new.context(), old.context())
    assert_bitwise(new.available_thrust(), old.available_thrust())
    state = new.state + torch.randn(96, 4, dtype=torch.float64, generator=noise) * torch.tensor(
        [3e4, 3e4, 300., 300.], dtype=torch.float64)
    thrust = new.available_thrust() * .7
    assert_bitwise(new.derivative(state, new.mass(), new.pitch, thrust),
                   old.derivative(state, old.mass(), old.pitch, thrust))
    altitude = torch.linspace(-100, 9e4, 997, dtype=torch.float64)
    assert_bitwise(new.atmosphere_at(altitude), old.atmosphere_at(altitude))


def test_ascent_env_is_bitwise_identical_for_impacts_and_deadlines():
    torch.set_num_threads(1)
    kwargs = dict(seed=5, max_time=24., deadline_mode='terminal', timeout_penalty=-30)
    new, old = ta.AscentBenchmark(64, **kwargs), ReferenceAscent(64, **kwargs)
    noise = torch.Generator().manual_seed(9)
    reasons = set()
    for step in range(60):
        action = torch.rand(64, 2, generator=noise) * 2 - 1
        action[::3, 1] = -1  # Pitch down hard: early impacts.
        out_new, out_old = new.step(action), old.step(action)
        assert_bitwise(out_new, out_old, f'step {step}')
        reasons |= set(out_old[3]['reason'][out_old[2]].tolist())
        assert_bitwise(new.reset(out_old[2]), old.reset(out_old[2]))
        assert_same_state(new, old)
    assert {2, 5} <= reasons


def test_bitwise_comparison_detects_a_one_ulp_reassociation(monkeypatch):
    """Guard against vacuous equivalence: a harmless-looking regrouping fails."""
    import math
    new, old = _ascent_pair(16, 2)
    monkeypatch.setattr(BatchedAscent, '_attitude', staticmethod(
        lambda pitch: (lambda p: (p, p.sin(), p.cos()))(pitch * (math.pi / 180))))
    with pytest.raises(AssertionError):
        for _ in range(40):
            action = old.teacher()
            assert_bitwise(new.step(action), old.step(action))


# ---------------------------------------------------------------- training

def _args(**overrides):
    values = dict(device='cpu', epochs=2, batch_size=96, target_kl=.02, entropy=.001, critic_extra=3,
                  worlds=48, horizon=12, label_worlds=24, label_steps=20, seed=5, bc_updates=4,
                  orbital_version=2)
    values.update(overrides)
    return types.SimpleNamespace(**values)


@pytest.mark.parametrize('task,architecture,target_kl,anchor', [
    ('orbital', 'mlp256', .02, False), ('orbital', 'transformer', .02, True),
    ('ascent', 'gru', .02, False), ('ascent', 'mlp256', 1e-12, True)])
def test_rollout_and_ppo_update_match_reference(task, architecture, target_kl, anchor):
    torch.set_num_threads(1)
    args = _args(target_kl=target_kl)
    context = 1 if architecture.startswith('mlp') else 4
    runs = []
    for rollout, update in ((reference_rollout, reference_ppo_update), (ta.collect_rollout, ta.ppo_update)):
        torch.manual_seed(77)
        env = ta.make_env(task, args.worlds, 'cpu', 31)
        model = SequencePolicy(architecture, env.observations, env.actions, context,
                               action_transform='scaled_tanh', action_scale=1.05)
        history = (ReferenceHistory if rollout is reference_rollout else History)(
            args.worlds, context, env.observations, env.actions, 'cpu')
        history.append(env.observation())
        bank = None
        if anchor:
            bank = (torch.randn(40, context, env.observations + env.actions + 1),
                    torch.rand(40, env.actions) * 1.8 - .9)
        actor = torch.optim.Adam([*model.actor.parameters(), model.log_std], lr=1e-3)
        critic = torch.optim.Adam(model.critic.parameters(), lr=3e-4)
        results = []
        for update_index in range(2):
            data, stats = rollout(model, env, history, args)
            results.append((data, stats, update(model, data, actor, critic, args, anchor=bank,
                                                anchor_coef=.5 if anchor else 0., train_actor=update_index > 0)))
        runs.append((results, model.state_dict(), history.data, torch.get_rng_state(),
                     env.generator.get_state(), actor.state_dict()['state'], critic.state_dict()['state']))
    assert_bitwise(runs[1], runs[0])
    if target_kl < 1e-6:
        assert runs[1][0][1][2]['kl_early_stop']


@pytest.mark.parametrize('task', ['orbital', 'ascent'])
def test_fit_teacher_label_bank_matches_reference(task):
    torch.set_num_threads(1)
    args = _args()
    outputs = []
    for fit in (reference_fit_teacher, ta.fit_teacher):
        torch.manual_seed(8)
        env = ta.make_env(task, 4, 'cpu', 0)
        model = SequencePolicy('lstm', env.observations, env.actions, 3,
                               action_transform='scaled_tanh', action_scale=1.05)
        parent = None
        if task == 'ascent':
            parent = tg.VehiclePolicy(16, observations=56)
        events = []
        x, y = fit(model, task, parent, args, lambda event, **kw: events.append((event, kw)))
        for _, values in events:
            values.pop('elapsed_s', None)
        outputs.append((x, y, events, model.state_dict(), torch.get_rng_state()))
    assert_bitwise(outputs[1], outputs[0])


@pytest.mark.parametrize('target_kl,entropy', [(None, 0.), (1e-3, 0.), (1e-8, .01)])
def test_generalized_ppo_train_matches_reference(target_kl, entropy):
    torch.set_num_threads(1)
    runs = []
    for train in (reference_ppo_train, tg.ppo_train):
        torch.manual_seed(16)
        model = tg.VehiclePolicy(16)
        env = BatchedAscent(16, seed=16, max_time=40)
        events = []
        train(model, env, updates=2, horizon=6, batch_size=24, learning_rate=3e-3,
              target_kl=target_kl, entropy_coefficient=entropy,
              ppo_optimizer='separate' if entropy else 'joint', critic_extra_updates=2 if entropy else 0,
              selected_callback=lambda _: None,
              log=lambda event, **values: events.append((event, values)))
        for _, values in events:
            values.pop('elapsed_s', None)
        runs.append((events, model.state_dict(), torch.get_rng_state(), env.generator.get_state()))
    assert_bitwise(runs[1], runs[0])


def test_evaluate_reports_match_reference_environment_implementations(monkeypatch):
    torch.set_num_threads(1)
    model = SequencePolicy('gru', 56, 2, 3, action_transform='scaled_tanh', action_scale=1.05)
    reports = []
    for capture, ascent, history in ((ReferenceCapture, ReferenceAscent, ReferenceHistory),
                                     (BatchedCapture, ta.AscentBenchmark, History)):
        monkeypatch.setattr(ta, 'BatchedCapture', capture)
        monkeypatch.setattr(ta, 'AscentBenchmark', ascent)
        monkeypatch.setattr(ta, 'History', history)
        reports.append((ta.evaluate(None, 'orbital', 34, 'cpu', 3, teacher=True),
                        ta.evaluate(model, 'ascent', 6, 'cpu', 4)))
    assert reports[0] == reports[1]
    assert reports[1][0]['successes'] > 0
