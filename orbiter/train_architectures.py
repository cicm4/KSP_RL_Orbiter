"""Matched finite-history architecture research, never a live KSP controller.

Tasks are deliberately separate: Kerbin launch and a shared 17-body 3D local
orbital policy. Neither score measures the complete launch/transfer mission.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import signal
import time

import numpy as np
import torch

from .batched_capture import BatchedCapture
from .capture_env import CaptureConfig, CaptureEnv
from .generalized_aero_env import BatchedAeroAscent
from .sequence_policy import ARCHITECTURES, History, SequencePolicy
from .train_generalized import experiment_contract, load_policy, write_json

TASK = "matched_history_architecture_research_v2"
# v1 checkpoints (tanh actions, v1 orbital observations) remain loadable.
TASKS = ("matched_history_architecture_research_v1", TASK)
STOP = False


class AscentBenchmark(BatchedAeroAscent):
    """Adapt the established Gym-style termination pair to finite missions."""
    observations, actions = 56, 2

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        return observation, reward.float(), terminated | truncated, info


def request_stop(_signum, _frame):
    global STOP
    STOP = True


def check_stop():
    if STOP:
        raise InterruptedError("graceful stop requested")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_env(task, count, device, seed, *, four_stage=False, orbital_version=2):
    if task == 'orbital':
        return BatchedCapture(count, device=device, seed=seed, version=orbital_version)
    envelope, four, _ = experiment_contract('stock', 'aero', deadline_mode='terminal', timeout_penalty=-30)
    env = AscentBenchmark(count, device=device, seed=seed,
                           envelope=four if four_stage else envelope,
                           deadline_mode='terminal', timeout_penalty=-30)
    env.observations, env.actions = 56, 2
    return env


def save(path, model, contract, ancestry):
    temporary = path.with_suffix('.tmp.pt')
    torch.save(dict(task=TASK, spec=model.spec, state_dict=model.state_dict(),
                    contract=contract, ancestry=ancestry), temporary)
    temporary.replace(path)
    write_json(path.with_suffix('.metadata.json'), dict(task=TASK, contract=contract,
               ancestry=ancestry, sha256=sha(path), saved_at_unix=time.time(),
               live_ksp_validated=False, promoted=False))


def load(path, device='cpu'):
    metadata = json.loads(path.with_suffix('.metadata.json').read_text())
    if metadata['task'] not in TASKS or metadata['sha256'] != sha(path):
        raise ValueError('architecture checkpoint contract/hash mismatch')
    record = torch.load(path, map_location=device, weights_only=True)
    if record['task'] != metadata['task'] or record['contract'] != metadata['contract'] or record['ancestry'] != metadata['ancestry']:
        raise ValueError('architecture checkpoint metadata mismatch')
    model = SequencePolicy(**record['spec']).to(device)
    model.load_state_dict(record['state_dict'], strict=True)
    if any(not bool(torch.isfinite(x).all()) for x in model.state_dict().values()):
        raise ValueError('nonfinite architecture checkpoint')
    return model, metadata


def cpu_view(env):
    result = copy.copy(env)
    result.device = torch.device('cpu')
    for key, value in vars(env).items():
        if isinstance(value, torch.Tensor):
            setattr(result, key, value.detach().cpu())
    return result


def coast_numpy(r, v, mu, goal, tolerance, safe, normal):
    """Independent unpowered NumPy RK4 for one full osculating period."""
    r, v = np.array(r, dtype=float), np.array(v, dtype=float)
    energy = float(v@v)/2-mu/np.linalg.norm(r)
    if energy >= 0:
        return dict(qualified=False, reason='unbound')
    period = 2*math.pi*math.sqrt((-mu/(2*energy))**3/mu)
    dt = period/1024
    state = np.concatenate((r, v)); initial = state.copy()
    minimum, maximum = float('inf'), 0.
    def derivative(x):
        return np.concatenate((x[3:], -mu*x[:3]/np.linalg.norm(x[:3])**3))
    for _ in range(1024):
        k1=derivative(state);k2=derivative(state+dt*k1/2)
        k3=derivative(state+dt*k2/2);k4=derivative(state+dt*k3)
        state += dt*(k1+2*k2+2*k3+k4)/6
        radius=np.linalg.norm(state[:3]);minimum=min(minimum,radius);maximum=max(maximum,radius)
    final_energy=float(state[3:]@state[3:])/2-mu/np.linalg.norm(state[:3])
    qualified=(minimum>safe and minimum>=goal-tolerance-.01 and maximum<=goal+tolerance+.01)
    return dict(qualified=bool(qualified), period_s=period, minimum_radius_m=minimum,
                maximum_radius_m=maximum, return_position_error_m=float(np.linalg.norm(state[:3]-initial[:3])),
                relative_energy_drift=abs((final_energy-energy)/energy), independent_numpy=True)


def qualify_capture(case, actions, terminal):
    config = CaptureConfig.from_dict(case['config'])
    env = CaptureEnv(config)
    env.reset(seed=0, options={key:case[key] for key in ['r','v','goal_normal']})
    env.direction=np.array(case['direction'])
    info={}
    for action in actions:
        check_stop()
        _,_,terminated,truncated,info=env.step(action)
        if terminated or truncated:break
    position_error=float(np.linalg.norm(env.r-np.array(terminal[:3]))/config.goal_radius)
    velocity_error=float(np.linalg.norm(env.v-np.array(terminal[3:]))/config.velocity_scale)
    replay_ok=bool(info.get('success')) and position_error<5e-6 and velocity_error<5e-6
    coast=coast_numpy(env.r,env.v,config.mu,config.goal_radius,config.orbit_tolerance,
                      config.safe_radius,env.goal_normal) if replay_ok else dict(qualified=False)
    return dict(qualified=replay_ok and coast['qualified'], scalar_replay_success=bool(info.get('success')),
                normalized_position_error=position_error,normalized_velocity_error=velocity_error,
                coast=coast, live_ksp_validated=False)


@torch.no_grad()
def evaluate(model, task, count, device, seed, *, four_stage=False, qualify=False, teacher=False,
             orbital_version=2):
    env=make_env(task,count,device,seed,four_stage=four_stage,orbital_version=orbital_version)
    view=cpu_view(env)
    if task=='orbital':view._export_telemetry=view.telemetry()
    cases=[view.export_case(i) if task=='orbital' else asdict(view.export_config(i)) for i in range(count)]
    keys=[hashlib.sha256(json.dumps(c,sort_keys=True,allow_nan=False).encode()).hexdigest() for c in cases]
    history=History(count,model.spec['context'] if model is not None else 1,env.observations,env.actions,device)
    history.append(env.observation())
    active=torch.ones(count,device=device,dtype=torch.bool)
    success=active.clone()&False; reasons=torch.zeros(count,device=device,dtype=torch.long)
    ends=torch.zeros(count,device=device,dtype=torch.long)
    terminal=torch.zeros(count,6,device=device,dtype=torch.float64)
    orbital_extrema=torch.zeros(count,2,device=device,dtype=torch.float64)
    trajectories=[]
    max_steps=404 if task=='orbital' else 804
    for step in range(max_steps):
        check_stop()
        action=env.teacher() if teacher else model(history.data)
        trajectories.append(action.cpu()) if qualify else None
        obs,_,done,info=env.step(action,active) if task=='orbital' else env.step(action)
        new=active&done
        success|=new&info['success'];reasons=torch.where(new,info['reason'],reasons)
        ends=torch.where(new,step+1,ends)
        if task=='orbital':terminal=torch.where(new[:,None],torch.cat((env.r,env.v),-1),terminal)
        else:terminal[:,:4]=torch.where(new[:,None],env.state,terminal[:,:4])
        orbital_extrema=torch.where(new[:,None],torch.stack((info['periapsis'],info['apoapsis']),-1),orbital_extrema)
        active &= ~done
        if not bool(active.any()):break
        history.append(obs,action)
    if bool(active.any()):raise RuntimeError('bounded evaluation did not terminate')
    rows=[];counts=__import__('collections').Counter()
    success,reasons,ends=success.cpu().tolist(),reasons.cpu().tolist(),ends.cpu().tolist()
    terminal=terminal.cpu().tolist()
    orbital_extrema=orbital_extrema.cpu().tolist()
    actions=torch.stack(trajectories).numpy() if qualify else None
    for i in range(count):
        check_stop()
        body=cases[i]['config']['body_name'] if task=='orbital' else 'Kerbin'
        row=dict(case_sha256=keys[i],body=body,success=success[i],reason=reasons[i],steps=ends[i],case=cases[i])
        row['periapsis_m'],row['apoapsis_m']=orbital_extrema[i]
        row['terminal_state']=terminal[i] if task=='orbital' else terminal[i][:4]
        if qualify:
            row['actions']=actions[:ends[i],i].tolist()
            if success[i]:
                row['qualification']=qualify_capture(cases[i],row['actions'],terminal[i])
            else:row['qualification']=dict(qualified=False)
        rows.append(row)
    per_body={}
    for body in sorted(set(r['body'] for r in rows)):
        body_rows=[r for r in rows if r['body']==body]
        successes=sum(r['success'] for r in body_rows)
        per_body[body]=dict(episodes=len(body_rows),successes=successes,success_rate=successes/len(body_rows),
                 failure_counts=dict(__import__('collections').Counter(str(r['reason']) for r in body_rows if not r['success'])))
        if qualify:per_body[body]['qualified_successes']=sum(r['qualification']['qualified'] for r in body_rows)
    report=dict(task=task,seed=seed,episodes=count,successes=sum(success),success_rate=sum(success)/count,
                macro_body_success_rate=float(np.mean([x['success_rate'] for x in per_body.values()])),
                per_body=per_body,results=rows,live_ksp_validated=False,
                scope='local orbital control; no interbody transfer or KSP proof' if task=='orbital' else 'Kerbin planar launch surrogate')
    report['reason_codes']=({1:'success',2:'unsafe_altitude',3:'SOI_exit',4:'fuel',5:'timeout',6:'missed_capture'}
                             if task=='orbital' else {1:'success',2:'impact',3:'escape',4:'fuel',5:'timeout'})
    if qualify:report['qualified_successes']=sum(r['qualification']['qualified'] for r in rows)
    return report


def teacher_action(env, parent):
    """Teacher label at the env's current state: frozen parent or analytic."""
    return parent(env.observation()) if parent is not None else env.teacher()


def fit_teacher(model, task, parent, args, log):
    """Behavior-clone the teacher; return the label bank for the PPO anchor.

    Labels are fitted in tanh space through the policy's inverse action
    mapping. With ``scaled_tanh`` the targets lie strictly inside (-1, 1), so
    full-throttle/vertical labels no longer push latent means without bound.
    """
    env=make_env(task,args.label_worlds,args.device,args.seed+100000,
                 orbital_version=args.orbital_version)
    history=History(env.count,model.spec['context'],env.observations,env.actions,args.device)
    history.append(env.observation())
    observations,labels=[],[]
    started=time.monotonic()
    for _ in range(args.label_steps):
        check_stop()
        with torch.no_grad():
            action=parent(env.observation()) if parent is not None else env.teacher()
            # History.append returns a fresh window and never writes into an
            # earlier one, so the stored window needs no defensive clone.
            observations.append(history.data);labels.append(action)
            obs,_,done,_=env.step(action)
            # reset() returns exactly env.observation() of the post-reset state.
            obs=env.reset(done)
            history.append(obs,action,done)
    x=torch.cat(observations);y=model.squashed_target(torch.cat(labels))
    fingerprint=hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()
    observations_sha=hashlib.sha256(x[:,-1,:env.observations].contiguous().cpu().numpy().tobytes()).hexdigest()
    labels_sha=hashlib.sha256(y.contiguous().cpu().numpy().tobytes()).hexdigest()
    log('teacher_bank',samples=len(x),history_sha256=fingerprint,
        current_observation_sha256=observations_sha,labels_sha256=labels_sha,
        elapsed_s=time.monotonic()-started)
    optimizer=torch.optim.Adam(model.actor.parameters(),lr=3e-4)
    rng=torch.Generator(device=args.device).manual_seed(args.seed+200000)
    for update in range(args.bc_updates):
        check_stop()
        ids=torch.randint(len(x),(min(args.batch_size,len(x)),),generator=rng,device=args.device)
        loss=(model.actor(x[ids]).tanh()-y[ids]).square().mean()
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.actor.parameters(),.5);optimizer.step()
        if (update+1)%100==0:log('bc',update=update+1,action_mse=float(loss.detach()))
    for round_index in range(getattr(args,'dagger_rounds',0)):
        # DAgger (Ross et al. 2011): fly the current student, label every
        # visited state with the teacher, aggregate and refit. The teacher is
        # queried at the student's states, so covariate shift is corrected.
        check_stop();started=time.monotonic()
        env=make_env(task,args.label_worlds,args.device,args.seed+400000+round_index,
                     orbital_version=args.orbital_version)
        history=History(env.count,model.spec['context'],env.observations,env.actions,args.device)
        history.append(env.observation())
        xs,ys=[],[];successes=episodes=0
        with torch.no_grad():
            for _ in range(args.label_steps):
                check_stop()
                xs.append(history.data);ys.append(teacher_action(env,parent))
                action=model(history.data)
                _,_,done,info=env.step(action)
                successes+=int(info['success'].sum());episodes+=int(done.sum())
                history.append(env.reset(done),action,done)
        x=torch.cat((x,*xs));y=torch.cat((y,model.squashed_target(torch.cat(ys))))
        for update in range(args.bc_updates):
            check_stop()
            ids=torch.randint(len(x),(min(args.batch_size,len(x)),),generator=rng,device=args.device)
            loss=(model.actor(x[ids]).tanh()-y[ids]).square().mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.actor.parameters(),.5);optimizer.step()
        log('dagger_round',round=round_index+1,student_successes=successes,student_episodes=episodes,
            bank_samples=len(x),action_mse=float(loss.detach()),elapsed_s=time.monotonic()-started)
        del xs,ys,env,history
    with torch.no_grad():
        latent=torch.cat([model.actor(chunk) for chunk in x.split(8192)])
        log('bc_latent',mean_abs=float(latent.abs().mean()),max_abs=float(latent.abs().max()),
            saturated_fraction=float((latent.tanh().abs()>.99).float().mean()))
    del observations,labels
    return x,y


def anchor_coefficient(args, update):
    """Linearly decaying behavior-cloning anchor, never below ``bc_floor``."""
    floor=getattr(args,'bc_floor',0.)
    if not args.bc_coef or update > args.bc_decay_updates:
        return floor
    return max(floor,args.bc_coef*(1-(update-1)/max(1,args.bc_decay_updates)))


@torch.no_grad()
def collect_rollout(model, env, history, args, teacher=None):
    """One on-policy rollout of ``args.horizon`` steps plus GAE targets.

    Returns ``((histories, latents, log_probs, advantages, targets), stats)``.
    Host synchronization happens once, after the loop: episode/success
    counts accumulate on the device instead of ``int(...)`` every step.
    Stored history windows are not cloned because ``History.append`` always
    builds a new window, and reset() already returns the post-reset
    observation, so it is not rebuilt. Values equal the historical loop.
    With ``teacher`` (a callable on the env), the teacher's label at every
    visited state is appended as a sixth element for the on-policy anchor.
    """
    xs=[];zs=[];lps=[];values=[];rewards=[];dones=[];labels=[]
    successes=torch.zeros((),dtype=torch.long,device=args.device);episodes=torch.zeros_like(successes)
    for _ in range(args.horizon):
        check_stop();x=history.data;xs.append(x)
        if teacher is not None:labels.append(teacher(env))
        action,z,lp=model.sample(x);value=model.critic(x).squeeze(-1)
        _,reward,done,info=env.step(action)
        zs.append(z);lps.append(lp);values.append(value);rewards.append(reward);dones.append(done)
        successes+=info['success'].sum();episodes+=done.sum()
        history.append(env.reset(done),action,done)
    next_value=model.critic(history.data).squeeze(-1)
    # Per-step factors evaluated in two batched ops; the scan itself keeps the
    # historical expression order exactly.
    alive=(~torch.stack(dones)).float();carry=.999*.95*alive
    advantage=torch.zeros(args.worlds,device=args.device);advantages=[];returns=[]
    for t in reversed(range(args.horizon)):
        delta=rewards[t]+.999*next_value*alive[t]-values[t]
        advantage=delta+carry[t]*advantage
        advantages.append(advantage);returns.append(advantage+values[t]);next_value=values[t]
    advantages=torch.stack(advantages[::-1]).flatten()
    advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
    targets=torch.stack(returns[::-1]).flatten()
    latent_all=torch.cat(zs)
    stats=dict(successes=int(successes),episodes=int(episodes),
               latent_saturation=float((latent_all.tanh().abs()>.99).float().mean()),
               explained_variance=1-float((targets-torch.cat(values)).var()/(targets.var()+1e-8)))
    data=(torch.cat(xs),latent_all,torch.cat(lps),advantages,targets)
    if teacher is not None:data+=(model.squashed_target(torch.cat(labels)),)
    return data,stats


def ppo_update(model, data, actor_optimizer, critic_optimizer, args, *, anchor=None,
               anchor_coef=0., train_actor=True):
    histories,latents,old_logp,advantages,returns,*rollout_labels=data
    adaptive=getattr(args,'kl_control','early_stop')=='adaptive_lr'
    actor_steps=critic_steps=0;last_kl=0.;stopped=False;anchor_loss=None;kls=[]
    for _ in range(args.epochs if train_actor else 0):
        for indices in torch.randperm(len(histories),device=args.device).split(args.batch_size):
            check_stop()
            x=histories[indices];latent=latents[indices];target=returns[indices]
            dist=model.distribution(x);logp=model.log_probability(dist,latent)
            log_ratio=logp-old_logp[indices]
            ratio=log_ratio.exp()
            kl=(ratio-1-log_ratio).mean()
            # Enqueue the RNG-free, state-free work of this minibatch before
            # the KL check's host sync so a GPU overlaps it with the wait. The
            # critic forward is independent of the actor step that precedes
            # its optimizer step. Every RNG draw (rsample, anchor indices) and
            # every parameter update still happens only after the check.
            adv=advantages[indices]
            policy_loss=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()
            value_loss=(model.critic(x).squeeze(-1)-target).square().mean()
            last_kl=float(kl.detach());kls.append(last_kl)
            if adaptive:
                # Rudin et al. 2021 (rsl_rl): adapt the actor rate from the
                # measured KL instead of discarding the rest of the batch. A
                # hard stop remains only as a guard against runaway steps.
                for group in actor_optimizer.param_groups:
                    if last_kl>2*args.target_kl:group['lr']=max(args.min_actor_lr,group['lr']/1.5)
                    elif last_kl<args.target_kl/2:group['lr']=min(args.max_actor_lr,group['lr']*1.5)
                if last_kl>args.kl_guard*args.target_kl:stopped=True;break
            elif last_kl>args.target_kl:stopped=True;break
            sample=dist.rsample();entropy=-model.log_probability(dist,sample).mean()
            actor_loss=policy_loss-args.entropy*entropy
            if rollout_labels and anchor_coef>0:
                # On-policy anchor: teacher labels at the states this rollout
                # visited, reusing this minibatch's actor mean.
                anchor_loss=(dist.loc.tanh()-rollout_labels[0][indices]).square().mean()
                actor_loss=actor_loss+anchor_coef*anchor_loss
            elif anchor is not None and anchor_coef>0:
                bank_x,bank_y=anchor
                chosen=torch.randint(len(bank_x),(len(indices),),device=args.device)
                anchor_loss=(model.actor(bank_x[chosen]).tanh()-bank_y[chosen]).square().mean()
                actor_loss=actor_loss+anchor_coef*anchor_loss
            actor_optimizer.zero_grad();actor_loss.backward()
            torch.nn.utils.clip_grad_norm_([*model.actor.parameters(),model.log_std],.5)
            actor_optimizer.step();actor_steps+=1
            critic_optimizer.zero_grad();value_loss.backward();torch.nn.utils.clip_grad_norm_(model.critic.parameters(),.5)
            critic_optimizer.step();critic_steps+=1
        if stopped:break
    for _ in range(args.critic_extra if train_actor else args.critic_extra+args.epochs*math.ceil(len(histories)/args.batch_size)):
        check_stop()
        indices=torch.randint(len(histories),(min(args.batch_size,len(histories)),),device=args.device)
        loss=(model.critic(histories[indices]).squeeze(-1)-returns[indices]).square().mean()
        critic_optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.critic.parameters(),.5)
        critic_optimizer.step();critic_steps+=1
    report=dict(actor_steps=actor_steps,critic_steps=critic_steps,last_kl=last_kl,kl_early_stop=stopped,
                anchor_coef=anchor_coef,anchor_mse=None if anchor_loss is None else float(anchor_loss.detach()),
                critic_warmup=not train_actor)
    if adaptive:
        report.update(mean_kl=sum(kls)/len(kls) if kls else 0.,actor_lr=actor_optimizer.param_groups[0]['lr'])
    return report


def train(args):
    global STOP
    STOP=False
    torch.set_num_threads(args.cpu_threads);torch.manual_seed(args.seed)
    if args.device.startswith('cuda'):
        torch.cuda.set_device(torch.device(args.device))
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
    root=args.run_dir
    if root.exists() and any(root.iterdir()):raise ValueError('run directory must be empty')
    root.mkdir(parents=True,exist_ok=True)
    env=make_env(args.task,args.worlds,args.device,args.seed,orbital_version=args.orbital_version)
    context=args.context if not args.architecture.startswith('mlp') else 1
    policy_options=dict(initial_std=args.initial_std,action_transform=args.action_transform)
    if args.action_transform!='tanh':
        policy_options.update(action_scale=args.action_scale,deadzone=args.deadzone)
    model=SequencePolicy(args.architecture,env.observations,env.actions,context,**policy_options).to(args.device)
    parent=None
    if args.task=='ascent':
        if args.actor_from is None:raise ValueError('ascent benchmark requires frozen teacher')
        parent,_=load_policy(args.actor_from,device=args.device)
    elif args.actor_from is not None:raise ValueError('orbital teacher is analytical, not a launch checkpoint')
    contract=dict(task=TASK,benchmark=args.task,policy_spec=model.spec,
        architecture_parameters=sum(p.numel() for p in model.parameters()),
        body_names=list(env.names) if args.task=='orbital' else ['Kerbin'],
        body_catalog_sha256=getattr(env,'catalog_sha256',None),
        orbital_benchmark_version=args.orbital_version if args.task=='orbital' else None,
        orbital_benchmark_task=env.task if args.task=='orbital' else None,
        scope='17-body local orbital control; Sun bound-orbit adjustment; no interbody transfer' if args.task=='orbital' else 'Kerbin planar launch',
        history_semantics='fixed trailing observation/executed-action window, zero at each episode reset',
        training=dict(worlds=args.worlds,horizon=args.horizon,updates=args.updates,bc_updates=args.bc_updates,
            label_worlds=args.label_worlds,label_steps=args.label_steps,actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,entropy=args.entropy,critic_extra=args.critic_extra,
            batch_size=args.batch_size,epochs=args.epochs,target_kl=args.target_kl,
            bc_target='policy_inverse_action_mapping',bc_coef=args.bc_coef,
            bc_decay_updates=args.bc_decay_updates,critic_warmup=args.critic_warmup,
            minibatch_permutation='fresh_each_epoch',dagger_rounds=args.dagger_rounds,
            anchor_source=args.anchor_source,bc_floor=args.bc_floor,kl_control=args.kl_control,
            **(dict(min_actor_lr=args.min_actor_lr,max_actor_lr=args.max_actor_lr,kl_guard=args.kl_guard)
               if args.kl_control=='adaptive_lr' else {})),
        validation_seed=args.validation_seed,comparison_seed=args.test_seed,
        comparison_selects_weights=False,live_ksp_validated=False,
        limitations=['No actual KSP qualification','Local skill scores do not measure complete missions',
                     'Finite trailing-window memory, not unbounded recurrent state',
                     'Active liquid stage surrogate; no arbitrary propulsion/staging graphs',
                     'No heating, terrain, aerocapture, or structural dynamics'])
    ancestry=dict(seed=args.seed,phase='initial',update=0,selected_update=0)
    if args.actor_from:ancestry['upstream']=dict(sha256=sha(args.actor_from),usage='frozen_action_teacher')
    write_json(root/'config.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    write_json(root/'contract.json',contract)
    write_json(root/'runtime.json',dict(python=platform.python_version(),torch=torch.__version__,
         device=args.device,cuda_visible_devices=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
         matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
         cudnn_tf32=torch.backends.cudnn.allow_tf32,
         cuda_name=torch.cuda.get_device_name() if args.device.startswith('cuda') else None))
    events=(root/'events.jsonl').open('a',buffering=1)
    def log(event,**kw):events.write(json.dumps(dict(event=event,time_unix=time.time(),**kw),allow_nan=False)+'\n')
    save(root/'best.pt',model,contract,ancestry);save(root/'latest.pt',model,contract,ancestry)
    started=time.monotonic();best_score=-1.;selected_update=0
    previous_handlers={sig:signal.signal(sig,request_stop) for sig in [signal.SIGTERM,signal.SIGINT]}
    def validate(phase,update):
        nonlocal best_score,selected_update
        report=evaluate(model,args.task,args.eval_worlds,args.device,args.validation_seed,
                        orbital_version=args.orbital_version)
        write_json(root/f'validation_{phase}.json',report)
        score=report['macro_body_success_rate']
        if score>best_score:
            best_score=score;selected_update=update
            ancestry.update(phase=phase,update=update,selected_update=update)
            save(root/'best.pt',model,contract,ancestry)
        log('validation',phase=phase,update=update,episodes=report['episodes'],
            successes=report['successes'],macro_body_success_rate=score,best_score=best_score,
            selected_update=selected_update,per_body=report['per_body'])
        ancestry.update(phase=phase,update=update,selected_update=selected_update)
        save(root/'latest.pt',model,contract,ancestry)
    try:
        ancestry['phase']='teacher_warmup'
        anchor=fit_teacher(model,args.task,parent,args,log)
        if not args.bc_coef or args.anchor_source=='rollout':anchor=None
        rollout_teacher=(lambda env,parent=parent:teacher_action(env,parent)) if args.anchor_source=='rollout' else None
        del parent
        validate('teacher_warmup',0)
        env=make_env(args.task,args.worlds,args.device,args.seed+300000,orbital_version=args.orbital_version)
        history=History(args.worlds,context,env.observations,env.actions,args.device)
        history.append(env.observation())
        actor_optimizer=torch.optim.Adam([*model.actor.parameters(),model.log_std],lr=args.actor_lr)
        critic_optimizer=torch.optim.Adam(model.critic.parameters(),lr=args.critic_lr)
        # Negative updates fit only the fresh critic to the imitation actor's
        # returns; the actor and its selection ancestry are unchanged.
        for update in range(1-args.critic_warmup,args.updates+1):
            check_stop();tick=time.monotonic()
            data,rollout=collect_rollout(model,env,history,args,
                                         teacher=rollout_teacher if update>0 else None)
            rollout_successes,rollout_episodes=rollout['successes'],rollout['episodes']
            saturation,explained=rollout['latent_saturation'],rollout['explained_variance']
            stats=ppo_update(model,data,
                             actor_optimizer,critic_optimizer,args,anchor=anchor,
                             anchor_coef=anchor_coefficient(args,update) if update>0 else 0.,
                             train_actor=update>0)
            if update<=0:
                log('critic_warmup',update=update,elapsed_s=time.monotonic()-tick,
                    explained_variance=explained,**stats)
                continue
            ancestry.update(phase='ppo',update=update,selected_update=selected_update,
                            transitions=update*args.worlds*args.horizon)
            log('ppo',update=update,transitions=ancestry['transitions'],elapsed_s=time.monotonic()-tick,
                 rollout_episodes=rollout_episodes,rollout_successes=rollout_successes,
                 action_std=model.log_std.detach().exp().cpu().tolist(),
                 latent_saturation=saturation,explained_variance=explained,
                 gaussian_entropy=float((model.log_std.detach()+.5*math.log(2*math.pi*math.e)).sum()),**stats)
            if update%args.validation_interval==0 or update==args.updates:validate(f'ppo_{update}',update)
        selected,metadata=load(root/'best.pt',args.device)
        comparison=evaluate(selected,args.task,args.test_worlds,args.device,args.test_seed,
                            qualify=args.task=='orbital' and not args.skip_qualification,
                            orbital_version=args.orbital_version)
        write_json(root/'test_comparison.json',comparison)
        summary=dict(task=TASK,benchmark=args.task,comparison={k:v for k,v in comparison.items() if k!='results'},
                     selected_ancestry=metadata['ancestry'],interrupted=False,live_ksp_validated=False,
                     final_tests_select_checkpoint=False,elapsed_s=time.monotonic()-started)
        if args.task=='ascent':
            four=evaluate(selected,args.task,args.test_worlds,args.device,args.test_seed,four_stage=True)
            write_json(root/'test_four_stage.json',four)
            summary['four_stage']={k:v for k,v in four.items() if k!='results'}
            reference,_=load_policy(args.actor_from,device=args.device)
            class ParentView:
                spec={'context':1}
                def __call__(self,history):return reference(history[:,-1,:56])
            reference_report=evaluate(ParentView(),args.task,args.test_worlds,args.device,args.test_seed)
            write_json(root/'test_parent_reference.json',reference_report)
            summary['parent_reference']={k:v for k,v in reference_report.items() if k!='results'}
        else:
            reference_report=evaluate(None,args.task,args.test_worlds,args.device,args.test_seed,
                 teacher=True,qualify=not args.skip_qualification,orbital_version=args.orbital_version)
            write_json(root/'test_teacher_reference.json',reference_report)
            summary['teacher_reference']={k:v for k,v in reference_report.items() if k!='results'}
        write_json(root/'summary.json',summary);log('complete',summary=summary)
    except InterruptedError:
        ancestry.update(interrupted=True,resume_mode='warm_restart_fresh_optimizers_and_worlds')
        save(root/'latest.pt',model,contract,ancestry)
        write_json(root/'interrupted.json',dict(ancestry=ancestry,checkpoint='latest.pt',best_preserved=True))
        log('interrupted',ancestry=ancestry)
    finally:
        events.close()
        for sig,handler in previous_handlers.items():signal.signal(sig,handler)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--task',choices=['ascent','orbital'],required=True)
    p.add_argument('--architecture',choices=ARCHITECTURES,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--actor-from',type=Path)
    p.add_argument('--seed',type=int,default=901)
    p.add_argument('--validation-seed',type=int,default=4600505)
    p.add_argument('--test-seed',type=int,default=5600505)
    for flag,default in [('worlds',2048),('horizon',64),('updates',400),('batch-size',1024),
                         ('epochs',4),('context',16),('bc-updates',1000),('label-worlds',256),
                         ('label-steps',512),('critic-extra',32),('eval-worlds',544),
                         ('test-worlds',544),('validation-interval',50),('cpu-threads',2),
                         ('critic-warmup',16)]:
        p.add_argument('--'+flag,type=int,default=default)
    p.add_argument('--orbital-version',type=int,choices=[1,2],default=2)
    p.add_argument('--action-transform',choices=['tanh','scaled_tanh'],default='scaled_tanh')
    p.add_argument('--action-scale',type=float,default=1.05)
    # A norm deadzone silences coasting noise, but teacher labels jump from
    # exactly 0 in band to >= deadzone outside it; MSE averages the two into a
    # mean inside the deadzone, so the deterministic policy never thrusts.
    p.add_argument('--deadzone',type=float,default=0.)
    p.add_argument('--initial-std',type=float,default=.15)
    p.add_argument('--bc-coef',type=float,default=1.,help='initial teacher-anchor weight during PPO')
    p.add_argument('--bc-decay-updates',type=int,help='anchor decays linearly to zero (default updates/2)')
    p.add_argument('--bc-floor',type=float,default=0.,help='anchor weight never decays below this')
    p.add_argument('--anchor-source',choices=['bank','rollout'],default='bank',
                   help='bank: fixed teacher trajectories; rollout: teacher labels at PPO-visited states')
    p.add_argument('--dagger-rounds',type=int,default=0,help='student-rollout DAgger rounds after BC')
    p.add_argument('--kl-control',choices=['early_stop','adaptive_lr'],default='early_stop')
    p.add_argument('--min-actor-lr',type=float,default=1e-7)
    p.add_argument('--max-actor-lr',type=float,default=1e-3)
    p.add_argument('--kl-guard',type=float,default=8.,help='adaptive mode still stops above this multiple of target KL')
    p.add_argument('--actor-lr',type=float,default=1e-5)
    p.add_argument('--critic-lr',type=float,default=3e-4)
    p.add_argument('--entropy',type=float,default=.001)
    p.add_argument('--target-kl',type=float,default=.02)
    p.add_argument('--skip-qualification',action='store_true')
    args=p.parse_args(argv)
    integers=['worlds','horizon','updates','batch_size','epochs','context','label_worlds',
              'label_steps','eval_worlds','test_worlds','validation_interval','cpu_threads']
    if any(getattr(args,k)<1 for k in integers) or args.bc_updates<0 or args.critic_extra<0 or args.critic_warmup<0:
        p.error('counts must be positive; BC/extra/warm-up updates may be zero')
    if args.bc_decay_updates is None:args.bc_decay_updates=max(1,args.updates//2)
    if args.action_transform=='tanh' and (args.deadzone or args.action_scale!=1.05):
        p.error('the original tanh mapping takes no scale or deadzone')
    if args.action_transform=='tanh':args.action_scale=1.
    if not 1<=args.action_scale<=1.5 or not 0<=args.deadzone<.5 or not 0<args.initial_std<=2:
        p.error('invalid action mapping or initial std')
    if not math.isfinite(args.bc_coef) or args.bc_coef<0 or args.bc_decay_updates<1:
        p.error('invalid behavior-cloning anchor')
    if not math.isfinite(args.bc_floor) or not 0<=args.bc_floor<=args.bc_coef or args.dagger_rounds<0:
        p.error('invalid anchor floor or DAgger rounds')
    if args.anchor_source=='rollout' and not args.bc_coef:
        p.error('a rollout anchor needs a positive --bc-coef')
    if not (0<args.min_actor_lr<=args.actor_lr<=args.max_actor_lr) or args.kl_guard<=2:
        p.error('invalid adaptive learning-rate bounds')
    if args.task=='orbital' and (args.eval_worlds%17 or args.test_worlds%17):
        p.error('orbital evaluation banks must be balanced multiples of all 17 bodies')
    if not all(math.isfinite(x) and x>0 for x in [args.actor_lr,args.critic_lr,args.target_kl]) or not math.isfinite(args.entropy) or args.entropy<0:
        p.error('invalid optimization settings')
    train(args)


if __name__=='__main__':main()
