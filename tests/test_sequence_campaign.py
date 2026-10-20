import json
import numpy as np
import pytest
import torch

from orbiter.batched_capture import BatchedCapture
from orbiter.capture_env import CaptureConfig, CaptureEnv
from orbiter.sequence_policy import ARCHITECTURES, History, SequencePolicy


@pytest.mark.parametrize('architecture', ARCHITECTURES)
def test_policy_history_likelihood_gradients_and_determinism(architecture):
    torch.set_num_threads(1)
    torch.manual_seed(9)
    policy = SequencePolicy(architecture, 32, 3, context=4)
    history = History(3, 4, 32, 3, 'cpu')
    history.append(torch.randn(3, 32))
    sample, latent, logp = policy.sample(history.data)
    assert sample.shape == (3, 3)
    assert torch.isfinite(logp).all()
    torch.testing.assert_close(logp, policy.log_probability(policy.distribution(history.data), latent))
    policy.log_probability(policy.distribution(history.data), latent).mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in policy.parameters())
    torch.testing.assert_close(policy(history.data), policy(history.data))
    # An episode reset must produce exactly a fresh history for this spacecraft.
    history.append(torch.randn(3, 32), sample)
    observation = torch.randn(3, 32)
    history.append(observation, sample, torch.tensor([True, False, True]))
    fresh = History(3, 4, 32, 3, 'cpu'); fresh.append(observation)
    torch.testing.assert_close(history.data[[0, 2]], fresh.data[[0, 2]])
    torch.testing.assert_close(policy(history.data)[[0, 2]], policy(fresh.data)[[0, 2]])


def test_all_stock_bodies_are_covered_with_valid_physical_goals():
    env = BatchedCapture(34, seed=3)
    assert len(env.names) == 17
    assert set(env.body.tolist()) == set(range(17))
    assert env.observation().shape == (34, 32)
    assert torch.isfinite(env.observation()).all()
    assert (env.goal-env.tolerance > env.safe).all()
    assert (env.goal+env.tolerance < env.soi).all()
    for i in range(17):
        case = env.export_case(i)
        CaptureConfig.from_dict(case['config'])
        assert (case['task'] == 'heliocentric_orbit_adjustment') == (env.names[i] == 'Sun')
    assert (env.telemetry()['energy'][env.body == env.names.index('Sun')] < 0).all()


@pytest.mark.parametrize('body', ['Kerbin', 'Mun', 'Gilly', 'Jool', 'Sun'])
def test_gpu_formulation_matches_independent_scalar_finite_thrust(body):
    env = BatchedCapture(1, bodies=[body], seed=11)
    case = env.export_case(0)
    scalar = CaptureEnv(CaptureConfig.from_dict(case['config']))
    scalar.reset(seed=3, options={k:case[k] for k in ['r', 'v', 'goal_normal']})
    scalar.direction = np.array(case['direction'])
    for _ in range(3):
        action = env.teacher()
        np.testing.assert_allclose(action[0].numpy(), scalar.teacher_action(), atol=2e-6)
        _, _, done, info = env.step(action)
        _, _, term, trunc, _ = scalar.step(action[0].numpy())
        np.testing.assert_allclose(env.r[0].numpy(), scalar.r, rtol=1e-11, atol=1e-5)
        # Independent acos/Rodrigues paths differ by ~1e-8 radians near
        # alignment; measured velocity differences stay below 5e-7 m/s.
        np.testing.assert_allclose(env.v[0].numpy(), scalar.v, rtol=1e-10, atol=1e-6)
        np.testing.assert_allclose(env.direction[0].numpy(), scalar.direction, atol=3e-8)
        np.testing.assert_allclose(float(env.mass[0]), scalar.mass, rtol=1e-11)
        assert bool(done[0]) == (term or trunc)
        if bool(done[0]): break


def test_inactive_world_and_body_selective_reset_do_not_corrupt_other_worlds():
    env = BatchedCapture(3, seed=7)
    r, v, mass = env.r.clone(), env.v.clone(), env.mass.clone()
    env.step(env.teacher(), active=torch.tensor([False, True, True]))
    torch.testing.assert_close(env.r[0], r[0]); torch.testing.assert_close(env.v[0], v[0])
    torch.testing.assert_close(env.mass[0], mass[0])
    r = env.r.clone(); env.reset(torch.tensor([False, True, False]))
    torch.testing.assert_close(env.r[[0, 2]], r[[0, 2]])


def test_teacher_can_capture_and_goals_are_not_initial_successes():
    env = BatchedCapture(34, seed=3)
    assert not env.telemetry()['in_band'].any()
    active = torch.ones(34, dtype=torch.bool)
    successes = 0
    for _ in range(402):
        _, _, done, info = env.step(env.teacher(), active)
        successes += int((info['success'] & active).sum())
        active &= ~done
        if not active.any(): break
    assert not active.any()
    assert successes > 10  # Real flight witnesses, not an assumed feasible bank.


def test_checkpoint_roundtrip_hash_guard_and_independent_coast(tmp_path):
    from orbiter.train_architectures import save, load, coast_numpy
    policy=SequencePolicy('gru',32,3,4)
    p=tmp_path/'best.pt';save(p,policy,{'scope':'test'},{'seed':1})
    restored,_=load(p)
    for a,b in zip(policy.parameters(),restored.parameters()):torch.testing.assert_close(a,b)
    p.write_bytes(p.read_bytes()+b'corrupt')
    with pytest.raises(ValueError,match='hash'):load(p)
    result=coast_numpy([1000000.,0,0],[0,1000.,0],1e12,1e6,10,9e5,[0,0,1])
    assert result['qualified']
    assert result['relative_energy_drift']<1e-9
    assert result['return_position_error_m']<.1


def test_benchmark_adapter_and_completed_report_body_coverage():
    from orbiter.train_architectures import make_env
    from orbiter.overnight_suite import validate_report
    env=make_env('ascent',2,'cpu',3)
    observation,reward,done,info=env.step(env.teacher())
    assert observation.shape==(2,56) and done.dtype==torch.bool
    report=dict(task='matched_history_architecture_research_v1',benchmark='orbital',
        comparison=dict(episodes=17,successes=0,success_rate=0.,per_body={}),
        interrupted=False,live_ksp_validated=False,final_tests_select_checkpoint=False)
    with pytest.raises(ValueError,match='every stock body'):
        validate_report('orbiter.train_architectures',report)


def test_kl_stop_keeps_actor_fixed_while_extra_value_steps_train():
    from types import SimpleNamespace
    from orbiter.train_architectures import ppo_update
    policy=SequencePolicy('mlp256',32,3,1)
    h=History(8,1,32,3,'cpu');h.append(torch.randn(8,32))
    with torch.no_grad():_,latent,logp=policy.sample(h.data)
    actor_before={k:v.clone() for k,v in policy.actor.state_dict().items()}
    critic_before={k:v.clone() for k,v in policy.critic.state_dict().items()}
    actor=torch.optim.Adam([*policy.actor.parameters(),policy.log_std],lr=1e-4)
    critic=torch.optim.Adam(policy.critic.parameters(),lr=1e-3)
    args=SimpleNamespace(device='cpu',epochs=2,batch_size=8,target_kl=.001,entropy=.001,critic_extra=3)
    report=ppo_update(policy,(h.data,latent,logp+2,torch.ones(8),torch.ones(8)),actor,critic,args)
    assert report['actor_steps']==0 and report['critic_steps']==3
    assert all(torch.equal(v,policy.actor.state_dict()[k]) for k,v in actor_before.items())
    assert any(not torch.equal(v,policy.critic.state_dict()[k]) for k,v in critic_before.items())


def test_real_trainer_sigterm_preserves_best_and_latest(tmp_path):
    import subprocess,sys,time,signal,hashlib
    root=tmp_path/'run'
    command=[sys.executable,'-m','orbiter.train_architectures','--run-dir',str(root),
        '--task','orbital','--architecture','mlp256','--device','cpu','--worlds','17',
        '--horizon','2','--updates','1','--label-worlds','17','--label-steps','4',
        '--bc-updates','100000','--batch-size','17','--cpu-threads','1',
        '--eval-worlds','17','--test-worlds','17']
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            event=root/'events.jsonl'
            if event.exists() and 'teacher_bank' in event.read_text():break
            if process.poll() is not None:raise AssertionError(process.communicate())
            time.sleep(.02)
        else:raise AssertionError('trainer did not reach bounded warmup')
        process.send_signal(signal.SIGTERM)
        out,error=process.communicate(timeout=10)
        assert process.returncode==0,error.decode()
        assert (root/'interrupted.json').exists()
        for name in ['best','latest']:
            meta=json.loads((root/(name+'.metadata.json')).read_text())
            assert hashlib.sha256((root/(name+'.pt')).read_bytes()).hexdigest()==meta['sha256']
        assert json.loads((root/'latest.metadata.json').read_text())['ancestry']['interrupted']
    finally:
        if process.poll() is None:process.kill();process.wait()


def test_scaled_actions_reach_bounds_at_finite_latent_and_invert_teacher_labels():
    torch.manual_seed(4)
    policy = SequencePolicy('mlp256', 35, 3, 1, action_transform='scaled_tanh',
                            action_scale=1.05, deadzone=.2)
    labels = torch.tensor([[1., 1., -1.], [0., 0., 0.], [.05, 0, 0], [.3, -.4, .1]])
    labels[0] /= labels[0].norm()
    target = policy.squashed_target(labels)
    assert target.abs().max() < 1/1.05+1e-6  # finite atanh: no unbounded BC pull
    torch.testing.assert_close(policy.execute(target), labels)
    # Exploration noise around a coasting mean executes exactly zero thrust.
    noise = .1*torch.randn(1000, 3)
    assert (policy.execute(noise.tanh()).norm(dim=-1) == 0).float().mean() > .5
    ascent = SequencePolicy('mlp256', 56, 2, 1, action_transform='scaled_tanh', action_scale=1.05)
    full = torch.tensor([[1., 1.]])
    torch.testing.assert_close(ascent.execute(ascent.squashed_target(full)), full)
    assert torch.equal(ascent.execute(torch.tensor([[.97, -.99]])), torch.tensor([[1., -1.]]))
    with pytest.raises(ValueError):
        SequencePolicy('mlp256', 56, 2, 1, deadzone=.2)  # v1 tanh mapping is unchanged


def test_v2_orbital_features_resolve_small_body_delta_v_and_sun_shaping():
    v1, v2 = BatchedCapture(136, seed=3), BatchedCapture(136, seed=3, version=2)
    o1, o2 = v1.observation(), v2.observation()
    assert o1.shape == (136, 32) and o2.shape == (136, 35) and v2.task.endswith('_v2')
    assert (o1[:, 15].abs() >= 10).any() and not (o2[:, 15].abs() >= 10).any()
    raw = v2.isp*9.80665*(v2.mass/v2.dry_mass).log()/v2.V
    torch.testing.assert_close(o2[:, 15], raw.log1p().float())
    sun = v2.body == v2.names.index('Sun')
    assert v1.potential()[sun].abs().max() < 1e-3 < v2.potential()[sun].abs().min()
    active = torch.ones(136, dtype=torch.bool); successes = 0
    for _ in range(402):
        _, _, done, info = v2.step(v2.teacher(), active)
        successes += int((info['success'] & active).sum()); active &= ~done
        if not active.any(): break
    assert not active.any() and successes > 40


def test_critic_warmup_freezes_actor_and_anchor_decays():
    from types import SimpleNamespace
    from orbiter.train_architectures import anchor_coefficient, ppo_update
    args = SimpleNamespace(device='cpu', epochs=2, batch_size=8, target_kl=10., entropy=.001,
                           critic_extra=1, bc_coef=1., bc_decay_updates=4)
    assert [anchor_coefficient(args, u) for u in (1, 3, 5)] == [1., .5, 0.]
    policy = SequencePolicy('mlp256', 32, 3, 1, action_transform='scaled_tanh', action_scale=1.05)
    h = History(8, 1, 32, 3, 'cpu'); h.append(torch.randn(8, 32))
    with torch.no_grad(): _, latent, logp = policy.sample(h.data)
    actor = torch.optim.Adam([*policy.actor.parameters(), policy.log_std], lr=1e-3)
    critic = torch.optim.Adam(policy.critic.parameters(), lr=1e-3)
    before = {k: v.clone() for k, v in policy.actor.state_dict().items()}
    data = (h.data, latent, logp, torch.randn(8), torch.ones(8))
    report = ppo_update(policy, data, actor, critic, args, train_actor=False)
    assert report['actor_steps'] == 0 and report['critic_steps'] == 3 and report['critic_warmup']
    assert all(torch.equal(v, policy.actor.state_dict()[k]) for k, v in before.items())
    anchor = (h.data, torch.zeros(8, 3))
    report = ppo_update(policy, data, actor, critic, args, anchor=anchor, anchor_coef=1.)
    assert report['actor_steps'] == 2 and report['anchor_mse'] is not None


def test_v2_trainer_completes_with_warmup_anchor_and_loads_selected_policy(tmp_path):
    from orbiter.train_architectures import load, main
    root = tmp_path/'run'
    main(['--run-dir', str(root), '--task', 'orbital', '--architecture', 'gru', '--device', 'cpu',
          '--worlds', '17', '--horizon', '3', '--updates', '2', '--label-worlds', '17',
          '--label-steps', '6', '--bc-updates', '3', '--batch-size', '17', '--cpu-threads', '1',
          '--eval-worlds', '17', '--test-worlds', '17', '--critic-warmup', '2', '--context', '4',
          '--validation-interval', '1', '--skip-qualification'])
    events = [json.loads(x) for x in (root/'events.jsonl').read_text().splitlines()]
    kinds = [e['event'] for e in events]
    assert kinds.count('critic_warmup') == 2 and kinds.count('ppo') == 2 and 'bc_latent' in kinds
    ppo = [e for e in events if e['event'] == 'ppo']
    assert ppo[0]['anchor_coef'] == 1. and ppo[1]['anchor_coef'] == 0.
    assert 'latent_saturation' in ppo[0] and 'explained_variance' in ppo[0]
    summary = json.loads((root/'summary.json').read_text())
    assert summary['task'] == 'matched_history_architecture_research_v2'
    from orbiter.overnight_suite import validate_report
    validate_report('orbiter.train_architectures', summary)
    model, metadata = load(root/'best.pt')
    assert model.spec['action_transform'] == 'scaled_tanh' and model.spec['deadzone'] == .2
    assert metadata['contract']['orbital_benchmark_version'] == 2
    assert metadata['contract']['policy_spec']['observations'] == 35
