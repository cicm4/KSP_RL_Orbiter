"""Visited-state feedback, own-action history, and strict student provenance."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import pytest

torch = pytest.importorskip('torch')
from orbiter import train_dagger_student as d
from orbiter.ascent_planner import SearchBudget, DEFAULT, guidance_action, plan_ascent
from orbiter.distill_ascent import ResidualStudent, load_witnesses, student_contract
from orbiter.env import EnvConfig, Stage
from orbiter.generalized_aero_env import BatchedAeroAscent


@pytest.fixture(scope='module', autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope='module')
def banks(tmp_path_factory):
    config = replace(EnvConfig(), randomization=0, target_altitude=100000,
                     orbit_tolerance=20000, max_time=1600, physics_dt=1)
    three = replace(config, stages=(config.stages[0], Stage(600,600,120000,345,85),
                                   Stage(400,5400,120000,345,85)))
    r = plan_ascent([config,three], budget=SearchBudget(population=8,iterations=2,seed=91))
    assert r['qualified_successes'] == 2
    root = tmp_path_factory.mktemp('dagger-banks')
    paths=[]
    for i in range(2):
        one=deepcopy(r);one['results']=[one['results'][i]];one['craft_count']=1
        p=root/f'{i}.json';p.write_text(json.dumps(one));paths.append(p)
    return paths


def base(architecture='feedforward'):
    return student_contract(architecture,hidden=8,residual_bounds=(2,1),hide_aero=False,
                            bptt_steps=32,source_witness_sha256={'train':'a'*64,'validation':'b'*64},
                            train_crafts=['c'*64],validation_crafts=['d'*64])


def test_feedback_reconstructs_exact_qualified_initial_witness(banks):
    bank=load_witnesses(banks[0]);specs=d.feedback_specs(banks[0],bank)
    rows=d.collect_rollouts(None,specs,device='cpu',beta=1,seed=0,verify_witness=True)
    torch.testing.assert_close(rows[0].teacher_actions,bank.trajectories[0].teacher_actions)
    torch.testing.assert_close(rows[0].previous_actions[1:],rows[0].teacher_actions[:-1])
    assert not rows[0].qualification['qualified']


def test_tampered_feedback_parameters_fail_reconstruction(banks):
    bank=load_witnesses(banks[0]);spec=d.feedback_specs(banks[0],bank)[0]
    changed=replace(spec,parameters=(4000.,70000.,1.8,18.,10000.,1.1,.2,.6,65.,90000.),heuristic=False)
    with pytest.raises(ValueError,match='reproduce|terminal'):
        d.collect_rollouts(None,[changed],device='cpu',beta=1,seed=0,verify_witness=True)


@pytest.mark.parametrize('architecture',['feedforward','gru'])
def test_student_rollouts_use_own_executed_history_and_current_state_labels(banks,architecture):
    bank=load_witnesses(banks[0]);specs=d.feedback_specs(banks[0],bank)
    model=ResidualStudent(base(architecture))
    with torch.no_grad(): model.output.bias[1]=-.1
    row=d.collect_rollouts(model,specs,device='cpu',beta=0,seed=3)[0]
    with torch.no_grad():
        predicted,_=model(row.observations[None],row.previous_actions[None],row.baseline_actions[None])
    torch.testing.assert_close(row.previous_actions[0],torch.zeros(2))
    torch.testing.assert_close(row.previous_actions[1:],predicted[0,:-1],atol=1e-6,rtol=1e-6)
    assert not torch.equal(row.previous_actions[1:],row.teacher_actions[:-1])
    env=BatchedAeroAscent(1,seed=0);env.load_configs([row.config])
    parameters=torch.tensor([specs[0].parameters],dtype=env.dtype)
    for index in range(len(row.teacher_actions)):
        expected=env.teacher() if specs[0].heuristic else guidance_action(env,parameters)
        torch.testing.assert_close(row.teacher_actions[index],expected[0])
        env.step(predicted[:,index])


def test_aggregation_retains_original_and_recent_tail_with_fixed_capacity():
    assert d.update_dataset((1,2),(3,4),(5,6),mode='aggregate',capacity=5)==(1,2,4,5,6)
    assert d.update_dataset((1,2),(3,),(4,5),mode='recent',capacity=5)==(4,5)
    assert d.update_dataset((1,2),(3,),(4,5),mode='aggregate',capacity=2)==(1,2)
    with pytest.raises(ValueError):d.update_dataset((1,2),(),(3,),mode='aggregate',capacity=1)


@pytest.mark.parametrize('corruption',['hash','contract','bounds','overflow'])
def test_checkpoint_rejects_inconsistent_or_nonfinite_runtime_state(tmp_path,corruption):
    b=base();model=ResidualStudent(b);contract=d.make_contract(b,dataset_mode='recent',beta_start=0,beta_decay=.7,max_trajectories=8)
    path=tmp_path/'best.pt';d.save_checkpoint(path,model,contract,{'seed':1})
    loaded,_=d.load_checkpoint(path);torch.testing.assert_close(loaded.output.weight,model.output.weight)
    r=torch.load(path,weights_only=True)
    if corruption=='contract':r['contract']['previous_actions']='teacher'
    elif corruption=='bounds':r['state_dict']['bounds']*=.5
    elif corruption=='overflow':r['state_dict']['output.bias']=torch.full((2,),1e100,dtype=torch.float64)
    else:path.write_bytes(path.read_bytes()+b'wrong')
    if corruption!='hash':
        torch.save(r,path);m=json.loads(path.with_suffix('.metadata.json').read_text());m.update(contract=r['contract'],sha256=hashlib.sha256(path.read_bytes()).hexdigest());path.with_suffix('.metadata.json').write_text(json.dumps(m))
    with pytest.raises(ValueError):d.load_checkpoint(path)


def test_cpu_cli_records_closed_loop_selection_and_separate_contract(tmp_path,banks):
    root=tmp_path/'run'
    d.main(['--train-report',str(banks[0]),'--validation-report',str(banks[1]),
            '--run-dir',str(root),'--rounds','1','--warmup-epochs','1','--epochs-per-round','1',
            '--rollout-repeats','1','--max-trajectories','4','--hidden','8','--batch-size','1'])
    s=json.loads((root/'summary.json').read_text());model,r=d.load_checkpoint(root/'best.pt')
    assert s['task']==d.TASK and s['validation_selects_checkpoint']
    assert s['closed_loop_validation']['episodes']==1 and len(s['history'])==2
    assert not s['live_ksp_validated'] and not s['validation_is_final_unseen_test']
    assert r['ancestry']==s['selected_ancestry']


def test_contract_describes_visited_state_history_and_unqualified_feedback():
    c=d.make_contract(base(),dataset_mode='aggregate',beta_start=.5,beta_decay=.7,max_trajectories=8)
    assert c['student']['previous_action_training']=='actual_executed_mixture_after_teacher_warmup'
    assert c['student']['objective']=='current_state_feedback_action_mse'
    assert c['student']['task']==d.TASK


def test_stop_during_optimization_saves_latest_before_more_evaluation(tmp_path,banks,monkeypatch):
    monkeypatch.setattr(d,'_STOP',False)
    original=d._sequence_pass
    calls=[]
    evaluate=d.evaluate_controller
    def pass_with_stop(*args,**kwargs):
        if kwargs.get('optimizer') is not None:d._STOP=True
        return original(*args,**kwargs)
    def counted(*args,**kwargs):
        calls.append(1);return evaluate(*args,**kwargs)
    monkeypatch.setattr(d,'_sequence_pass',pass_with_stop)
    monkeypatch.setattr(d,'evaluate_controller',counted)
    root=tmp_path/'stopped'
    d.main(['--train-report',str(banks[0]),'--validation-report',str(banks[1]),'--run-dir',str(root),
            '--rounds','1','--warmup-epochs','2','--epochs-per-round','1','--hidden','8','--batch-size','1'])
    s=json.loads((root/'summary.json').read_text());_,latest=d.load_checkpoint(root/'latest.pt')
    assert s['interrupted'] and latest['ancestry']['interrupted']
    assert latest['ancestry']['validation_pending'] and len(calls)==1
    assert s['selected_ancestry']['phase']=='initialization'
