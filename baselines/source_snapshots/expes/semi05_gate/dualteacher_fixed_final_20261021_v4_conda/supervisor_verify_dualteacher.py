"""Independent stdlib-only Dualteacher acceptance; no model load or new test.

Candidate code: it must itself pass targeted adversarial checks before sealing.
"""
import argparse,datetime,hashlib,json,math,random,re,sys
from pathlib import Path
from types import SimpleNamespace
from supervisor_verify_common import ROOT,PYTHON,LOCK_ID,PROTOCOL,need,sha,digest,read_json,read_lines,number,integer,bound_data_functions,verify_runtime_quiescent
METHOD='dualteacher_sam'
DEPLOY=ROOT/'expes/semi05_gate/dualteacher_fixed_final_20261021_v4_conda'
BASE=ROOT/'expes/work_dir/semi05_gate/MAIN_TABLE_FIXED_FINAL_20261019'
STORE=ROOT/'expes/artifacts/FCD_A_MATRIX_20261006/MAIN_TABLE_FIXED_FINAL_20261019'
NATIVE_PROTOCOL='DUALTEACHER_BY_FIXED_FINAL_CONDA_V4_20261021'
METRICS=('dice','jaccard','hd95','asd')

def regular(p):
    p=Path(p);s=p.stat()
    need(p.is_file() and not any(x.is_symlink() for x in (p,*p.parents)) and s.st_nlink==1,'Missing/symlink/multilink file: '+str(p))
    return s

def binding(row):
    p=Path(row['path'])
    if p==Path('__RECORDED_HOME__/anaconda3/envs/sam3d/lib/python3.10/site-packages/numpy/__init__.py'):
        s=p.stat();need(p.is_file() and s.st_nlink==2 and not any(x.is_symlink() for x in (p,*p.parents)),'Pinned read-only Conda NumPy source differs')
    else:s=regular(p)
    need(p.is_absolute() and integer(row.get('bytes'),s.st_size) and digest(row.get('sha256')) and sha(p)==row['sha256'],'Changed source/provenance binding')

def source_inputs(c,source,job):
    need(c['method']==METHOD and c['dataset']=='BY' and c['ratio'] in (5,10),'Closed BY method differs')
    need(source==DEPLOY and c['launcher_protocol']==NATIVE_PROTOCOL and (c['gpu'],c['fold'],c['seed'])==(1,2,2027),'Protocol/GPU/fold/seed differs')
    need(c['owner_thread_id']=='01a153a0-a178-7002-bf0b-d1e33ccbd472' and re.fullmatch(r'r[1-9][0-9]*',c['tag']),'Owner/tag differs')
    rel=Path(METHOD)/'BY'/('semi%02d'%c['ratio'])/'fold2'/c['tag']
    need(job==BASE/rel and c['output']==str(job/'formal') and c['target']==str(STORE/rel/'final.pth'),'Exact job/output/target differs')
    need(c['selection']=='terminal_after_fixed_exposure' and c['test_evaluations']==1 and c['early_stop'] is False and c['exact_resume_supported'] is False,'Endpoint/model selection differs')
    need(c['checkpoint_bound']==256*1024**2,'Exact terminal bound differs')
    rows=c['sources'];need(rows and len({x['path'] for x in rows})==len(rows),'Missing/duplicate source inventory')
    for row in rows:binding(row)
    paths={x['path'] for x in rows}
    required={str(source/n) for n in ('run_dualteacher.py','dualteacher_backend.py','dualteacher_core.py','dualteacher_protocol.py','dualteacher_util.py','supervisor_verify_dualteacher.py','supervisor_verify_common.py','supervisor_observe_dualteacher.py','budget_protocol.py','fixed_endpoint.py','runtime_data.py','runtime_resources.py','dataset_pins.json','admission_review.json')}
    required|={str(p) for p in (source/'native').rglob('*') if p.is_file()}
    required|={str(ROOT/('expes/semi05_gate/public_hpq_epochdiag_20261019/'+n)) for n in ('hpq_contract.py','hpq_inputs.py','hpq_resources.py')}
    required|={str(ROOT/'expes/semi05_gate/public_sup_20261018/public_recovery_guard_20261019.py'),str(ROOT/'utils/cascade_train_utils.py'),str(ROOT/'expes/common/seg_metrics.py')}
    env=c['environment'];need(env.get('python_no_user_site')=='1','Isolated Conda environment proof absent');required|={x['path'] for x in env['files']};need(required.issubset(paths),'Source/dependency closure incomplete')
    need(env['torch_version']=='2.0.1+cu117' and env['optimizer_zero_grad_set_to_none_default'] is True,'Measured native optimizer semantics differ')
    for name in ('foundation_pin_document','environment_pin_document'):binding(c[name])
    need(read_json(Path(c['environment_pin_document']['path']))==env,'Environment document differs')
    f=c['foundation'];binding(f)
    need(f==read_json(Path(c['foundation_pin_document']['path'])) and f['path']==str(ROOT/'ckpt/sam_med3d_turbo.pth') and f['bytes']==402163626
         and f['sha256']=='899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c' and f['strict_cpu_verified'] is True,'Own Turbo pin differs')
    binding(c['launcher_source']);need(c['launcher_source']['path']==str(source/'run_dualteacher.py'),'Launcher binding differs')
    review=read_json(source/'admission_review.json')
    need(review['approved_for_prepare'] is True and review['SAM_equation5_autodiff_repair_disclosed'] is True and review['core_sha256']==sha(source/'dualteacher_core.py')
         and review['backend_sha256']==sha(source/'dualteacher_backend.py') and review['independent_verifier_sha256']==sha(source/'supervisor_verify_dualteacher.py'),'Admission/source review chain differs')
    nlab=c['ratio'];nunlab=100-nlab;cycle=5*((nunlab+4)//5);total=36*cycle
    budget=SimpleNamespace(dataset='BY',ratio=c['ratio'],labeled_cases=nlab,primary_pairs_per_cycle=cycle,replay_per_cycle=cycle-nunlab,cycles=36,primary_pairs=total)
    need(c['primary_pairs']==total,'Primary exposure budget differs')
    data=bound_data_functions(source,ROOT);splits=data['bound_splits']('BY',nlab)
    need(splits==c['splits'] and [len(splits[k]) for k in ('train_lab','train_unlab','test')]==[nlab,nunlab,25]
         and data['bind_case_files']('BY',splits)==c['case_files'],'Bound case/split identity differs')
    pairs=[]
    for epoch in range(36):
        rng=random.Random(2027+epoch*104729);u=list(splits['train_unlab']);l=list(splits['train_lab']);rng.shuffle(u);rng.shuffle(l)
        extra=[u[(epoch+i)%len(u)] for i in range(cycle-nunlab)]
        pairs.extend(dict(epoch=epoch,within=i,labeled=l[i%len(l)],unlabeled=case,unlabeled_replay=i>=len(u)) for i,case in enumerate(u+extra))
    need(len(pairs)==total and data['schedule'](budget,splits)==pairs,'Independent native schedule differs')
    p=c['dualteacher_recipe'];need(p['method']==METHOD and p['dataset']=='BY' and p['ratio']==nlab and p['official_commit']=='3710e3216db8bdb444296f0460eab9532b253848','Scientific source recipe identity differs')
    need(p['actual_native_batches']==dict(labeled=2,unlabeled=2) and p['native_iterations']==total//2 and p['prefix_training_updates']==0,'Native batching/prefix differs')
    need(p['sam']['paper_equations']==[5,10] and p['sam']['weight']==.1 and p['consistency']['weight']==.5 and p['prototype']['boundary_loss_weight']==.1,'Loss weights/autodiff repair disclosure differs')
    end=p['fixed_endpoint'];need(end['selection']=='fixed_terminal_EMA_plus_OP' and end['native_patch']==[96]*3 and end['native_stride']==[16]*3
         and end['post_processing'] is False and end['test_evaluations']==1 and end['intermediate_validation'] is False and end['best_checkpoint_selection'] is False,'Fixed native terminal inference differs')
    return budget,pairs

def clock(index,total):
    epoch=min(199,200*index//total)
    return dict(iteration=index,total_iterations=total,native_epoch=epoch,native_reference_iteration=40000*index/total,
                primary_progress=index/total,learning_rate=max(1e-6,.0025*(1-epoch/200)**.9),sam_enabled=epoch>5)

def validate_step(s,pairs,index,total,first_actual=None):
    if first_actual is None:first_actual=index==0
    flags=('finite_loss','finite_nonzero_gradients','teacher_no_grad','actual_case_ids_verified')
    need(all(s.get(k) is True for k in flags) and s.get('skip_count')==0 and s.get('unlabeled_gt_in_loss') is False,'Invalid actual native step flags')
    need(s['labeled_inputs']==s['unlabeled_inputs']==2 and s['successful_optimizer_updates']==1 and s['completed_primary_pairs']==2*(index+1),'Actual primary batch/update clock differs')
    actual=pairs[2*index:2*index+2];need(s['actual_pairs']==actual,'Actual native pair/case identity differs')
    need(number(s['gradient_norm']) and s['gradient_norm']>0,'Nonfinite/zero measured native gradient norm')
    r=s['native_receipt'];need(r['step']==index+1 and r['clock']==clock(index,total),'Native epoch/LR/SAM clock differs')
    need(r['case_ids']=={key:[p[key] for p in actual] for key in ('labeled','unlabeled')} and r['primary_labeled']==r['primary_unlabeled']==2,'Native receipt case identity differs')
    need(r['optimizer_updates']==2 and r['primary_optimizer_updates']==r['student_optimizer_calls']==r['effective_student_updates']==1
         and r['op_optimizer_calls']==2 and r['effective_op_updates']==1,'Native primary/aux optimizer count differs')
    events=r['op_step_events'];need(len(events)==2 and events[0]['gradient_l1']==0 and events[0]['effective_update'] is False
         and events[1]['effective_update'] is True and number(events[1]['gradient_l1']) and events[1]['gradient_l1']>0,'Actual native OP gradient/parameter deltas differ')
    for e in events:
        need(number(e['parameter_delta_l1']) and e['effective_update']==(e['parameter_delta_l1']>0),'OP effective-update/delta evidence differs')
    need(all(number(r[k]) and r[k]>0 for k in ('student_gradient_l1','op_gradient_l1')),'Native gradient evidence differs')
    need(r['student_case_forwards']==6 and r['teacher_case_forwards']==(4 if clock(index,total)['sam_enabled'] else 2) and r['op_case_forwards']==4,'Native student/teacher/fusion compute differs')
    valid=r['sam_valid_cases'];need(type(valid) is int and 0<=valid<=2,'Invalid SAM valid-case count')
    enabled=clock(index,total)['sam_enabled'];need(r['foundation_case_forwards']==(2 if enabled and valid else 0),'Native SAM batched compute differs')
    need(r['sam_loss_requires_grad']==bool(enabled and valid),'Student SAM loss detached or unexpected branch')
    need(r['unlab_gt_in_loss'] is False and r['test_evaluations']==0 and r['criterion_projection_optimizer_owned'] is False,'Native gradient/GT/test boundary differs')
    losses={k:r[k] for k in ('loss','sup_loss','unsup_loss','bc_loss','sam_loss','op_loss')}
    need(s['losses']==losses and all(number(x,-math.inf) for x in losses.values()),'Nonfinite/mismatched native loss components')
    need(math.isclose(r['loss'],r['sup_loss']+.5*r['unsup_loss']+.1*r['bc_loss']+.1*r['sam_loss'],rel_tol=1e-5,abs_tol=1e-6),'Native total loss composition differs')
    need(number(r['dynamic_threshold_fg'],.5,.95),'Dynamic threshold differs')
    return r

def verify_ledger(c,budget,pairs,rows,complete):
    total=budget.primary_pairs//2;need(0<=len(rows)<=total and (not complete or len(rows)==total),'Missing/extra native rows')
    for i,row in enumerate(rows):
        validate_step(row['step'],pairs,i,total)
        need(row['cumulative']==dict(labeled_inputs=2*(i+1),unlabeled_inputs=2*(i+1),iterations=i+1,successful_optimizer_updates=i+1),'Cumulative native input/update count differs')
    actual=[p for row in rows for p in row['step']['actual_pairs']];cycles=[];size=budget.primary_pairs_per_cycle
    for epoch in range(len(actual)//size):
        chunk=actual[epoch*size:(epoch+1)*size]
        need(set(p['labeled'] for p in chunk)==set(c['splits']['train_lab']) and set(p['unlabeled'] for p in chunk)==set(c['splits']['train_unlab'])
             and sum(p['unlabeled_replay'] for p in chunk)==budget.replay_per_cycle,'Actual native complete cycle coverage differs')
        cycles.append(dict(epoch=epoch,actual_case_coverage_verified=True,unlabeled_replay=budget.replay_per_cycle))
    return dict(labeled_inputs=2*len(rows),unlabeled_inputs=2*len(rows),iterations=len(rows),successful_optimizer_updates=len(rows)),cycles

def initialization_diagnostics(path,c):
    v=read_json(path);need(v['failed'] is False and v['stderr']=='' and v['stdout']=='creating model SAM-Med3D\n','Unexpected initialization diagnostic output')
    expected=dict(message='Importing from timm.models.layers is deprecated, please import via timm.layers',category='FutureWarning',
        filename='__RECORDED_HOME__/anaconda3/envs/sam3d/lib/python3.10/site-packages/timm/models/layers/__init__.py',lineno=48)
    need(len(v['warnings'])<=1 and all(w==expected for w in v['warnings']) and any(x['path']==expected['filename'] for x in c['environment']['files']),'Unreviewed initialization warning')

def verify(job,source,expected):
    need(digest(expected) and sha(job/'contract.json')==expected,'External contract SHA differs');c=read_json(job/'contract.json');budget,pairs=source_inputs(c,source,job)
    out=job/'formal';target=Path(c['target']);total=budget.primary_pairs//2
    lock=regular(ROOT/'expes/artifacts/FCD_A_MATRIX_20261006/save.lock');need((lock.st_dev,lock.st_ino,lock.st_nlink)==tuple(LOCK_ID[k] for k in ('device','inode','nlink')),'Original save lock replaced')
    reg=read_json(BASE/'active_targets.json');need(len({x['path'] for x in reg})==len(reg) and [x['bound'] for x in reg if x['path']==str(target)]==[256*1024**2],'Own shared resource target differs')
    for p in job.rglob('*'):need(not p.is_symlink() and 'fail' not in p.name.lower() and not p.name.endswith('.partial'),'Failure/partial/symlink preserved; not accepted')
    need({p.name for p in job.iterdir()}=={'prepare_intent.json','contract.json','preparation_bound.json','qualification_intent.json','qualification_backend','qualification.json','ready.json','launch_intent.json','dispatch.json','boot.log','started.json','formal','exited.json'},'Unexpected job control/evaluation evidence')
    names=('preparation_bound','qualification_intent','qualification','ready','launch_intent','dispatch','started','exited');control={n:read_json(job/(n+'.json')) for n in names}
    need(all(v['contract_sha256']==expected for v in control.values()),'Control contract chain differs')
    q,ready,launch,dispatch,started,exited=[control[n] for n in ('qualification','ready','launch_intent','dispatch','started','exited')]
    need(read_json(job/'prepare_intent.json')==dict(protocol=NATIVE_PROTOCOL,foundation_pin=c['foundation_pin_document']),'Unique prepare intent differs')
    need(control['preparation_bound']['actual_gpu_qualified'] is False and control['preparation_bound']['source_files']==len(c['sources']),'Prepared source phase differs')
    rh=sha(job/'ready.json');need(ready['status']=='READY' and ready['qualification_sha256']==sha(job/'qualification.json'),'Qualification ready chain differs')
    need(all(control[k]['gpu']==1 for k in ('qualification_intent','qualification','ready')) and all(control[k]['ready_sha256']==rh for k in ('launch_intent','dispatch','started')),'Control GPU/ready binding differs')
    pid=dispatch['pid'];need(type(pid) is int and pid>0 and started['pid']==exited['pid']==pid,'Actual PID chain differs')
    cmd=dispatch['command'];wanted=[PYTHON,str(source/'run_dualteacher.py'),'--action','worker','--method',METHOD,'--dataset','BY','--ratio',str(c['ratio']),'--tag',c['tag'],'--gpu','1','--contract-sha',expected,'--ready-sha',rh,'--lease-fd']
    need(len(cmd)==20 and cmd[:-1]==wanted and cmd[-1].isdigit() and started['command']==cmd,'Complete dispatched native command differs')
    lease=Path(c['native_gpu1_lease_path']);ls=regular(lease);need(lease==BASE/'native_gpu1_r4.lock' and lease.read_bytes()==b'NATIVE_MAIN_TABLE_R4_GPU1_EXCLUSIVE_OPERATION_V1\n' and dispatch['native_lease']==dict(device=ls.st_dev,inode=ls.st_ino),'Original native lease binding differs')
    need(exited['exit_code']==0 and exited['status']=='COMPLETE_VERIFIED' and exited['independent_final_acceptance_required'] is True and (job/'boot.log').read_bytes()==b'','Worker exit/boot differs')
    need(q['status']=='QUALIFIED' and q['actual_extra_engineering_updates']==2 and q['actual_extra_engineering_primary_inputs_per_stream']==4 and q['actual_extra_native_optimizer_updates']==4
         and q['new_test_evaluations']==0 and q['restored_model_optimizer_rng'] is True and q['actual_disk_roundtrip'] is True and q['foundation_unchanged'] is True,'Own complete qualification absent')
    active=math.ceil(6*total/200);need(q['qualification_clock_probe_indices']==[0,active] and len(q['receipts'])==2,'Qualification clock probes differ')
    for s,i,first in zip(q['receipts'],(0,active),(True,False)):validate_step(s,pairs,i,total,first)
    qp=q['probe_roundtrip'];need(qp['path']==str(target.parent/'own_dualteacher_update0_probe.pth.partial') and qp['bound']==512*1024**2 and 0<qp['bytes']<=qp['bound']
         and digest(qp['sha256']) and qp['deleted_exact_own_probe'] is True and qp['original_lock_verified'] is True and not Path(qp['path']).exists(),'Own complete disk probe/cleanup proof differs')
    need(all(qp['capacity_before']['save_lock'][k]==v for k,v in LOCK_ID.items()),'Qualification disk probe original lock differs')
    qfolder=job/'qualification_backend'
    need({p.name for p in qfolder.iterdir()}=={'native_initialization_diagnostics.json','qualification_cost_native_warmup.json','qualification_cost_native_SAM_start.json'},'Unexpected qualification output/evaluation evidence')
    for phase,i,receipt in zip(('native_warmup','native_SAM_start'),(0,active),q['receipts']):
        cost=read_json(qfolder/('qualification_cost_'+phase+'.json'));r=receipt['native_receipt']
        need(cost['phase']==phase and cost['native_index']==i and cost['op_events']==r['op_step_events'] and len(cost['student_events'])==1
             and cost['student_events'][0]['effective_update'] is True and cost['student_events'][0]['gradient_l1']==r['student_gradient_l1'],'Qualification partial native cost evidence differs')
    nr=q['receipts'][1]['native_receipt'];need(nr['sam_valid_cases']>0 and nr['foundation_case_forwards']==2 and nr['sam_loss_requires_grad'] is True and nr['sam_loss']>0,'Own SAM-active GPU witness absent')
    prep=read_json(out/'preparation_verified.json')
    for p,folder in ((prep,out),(q['preparation'],job/'qualification_backend')):
        need(all(p[k] is True for k in ('source_pins_verified','input_pins_verified','initialization_verified','resource_gate_passed','test_unopened','narrow_startup_diagnostics_verified')),'Preparation proof incomplete')
        initialization_diagnostics(folder/'native_initialization_diagnostics.json',c);need(p['initialization_diagnostics_sha256']==sha(folder/'native_initialization_diagnostics.json'),'Initialization diagnostics SHA differs')
    need(prep['initial_state_signatures']==q['native_state_signatures']==q['preparation']['initial_state_signatures'],'Formal initialization differs from own restored qualification')
    endpoint=dict(method=METHOD,labeled_inputs=budget.primary_pairs,unlabeled_inputs=budget.primary_pairs,optimizer_updates_per_iteration=1,test_case_ids=c['splits']['test'],protocol=PROTOCOL)
    need(read_json(out/'run_intent.json')==dict(endpoint=endpoint,status='INTENT',exact_resume_supported=False),'Formal endpoint intent differs')
    rows=read_lines(out/'steps.jsonl');counts,cycles=verify_ledger(c,budget,pairs,rows,True);need(len(cycles)==36,'Incomplete actual coverage cycles')
    coverage=read_json(out/'coverage_verified.json');need(coverage['actual_case_coverage_verified'] is True and coverage['cycles']==cycles and coverage['primary_labeled_inputs']==coverage['primary_unlabeled_inputs']==budget.primary_pairs
         and coverage['primary_student_optimizer_updates']==total and coverage['auxiliary_op_optimizer_updates']==total and coverage['total_native_optimizer_updates']==2*total
         and coverage['prefix_training_updates']==coverage['extra_donor_inputs']==0,'Independent native cost/coverage differs')
    for k in ('student_case_forwards','teacher_case_forwards','foundation_case_forwards','op_case_forwards'):need(coverage[k]==sum(x['step']['native_receipt'][k] for x in rows),'Auxiliary forward cost differs')
    final=read_json(out/'final_metrics.json');need(final['status']=='COMPLETE_VERIFIED' and final['counts']==counts and final['test_evaluations']==1,'Terminal completion counts differ')
    artifact=final['artifact'];info=regular(target);need(artifact['path']==str(target) and artifact['sha256']==sha(target) and artifact['bytes']==info.st_size and 0<info.st_size<=256*1024**2 and artifact['resource_lock_verified'] is True,'Fixed terminal artifact binding differs')
    sl=artifact['capacity_before']['save_lock'];need(all(sl[k]==v for k,v in LOCK_ID.items()) and sl['path']==str(ROOT/'expes/artifacts/FCD_A_MATRIX_20261006/save.lock'),'Saved artifact original lock differs')
    h=artifact['sha256'];permit=read_json(out/'final_test_intent.json');need(permit==dict(selection='terminal_after_fixed_exposure',final_sha256=h,test_case_ids=c['splits']['test'],test_evaluation=1),'Unique terminal test permit differs')
    need(read_json(out/'native_final_test_started.json')==dict(pid=pid,final_sha256=h,permit_sha256=sha(out/'final_test_intent.json'),test_evaluations=1),'Unique terminal test start differs')
    reload=read_json(out/'native_reload_verified.json');receipt=reload['receipt'];need(reload['artifact']==artifact and reload['pid']==pid and receipt['strict'] is True and receipt['sha256']==h
         and receipt['native_student_identity_verified'] is True and receipt['native_ema_op_identity_verified'] is True and read_json(out/'dualteacher_reload_verified.json')==receipt,'Composite terminal strict reload proof differs')
    need(set(receipt['signatures'])=={'student','ema','op','criterion'} and all(digest(x) for x in receipt['signatures'].values()),'Composite terminal signature set differs')
    start=read_json(out/'dualteacher_final_test_started.json');need(start==dict(final_sha256=h,permit_sha256=sha(out/'final_test_intent.json'),test_evaluations=1,selection='fixed_terminal_EMA_plus_OP'),'Backend unique final test marker differs')
    cases=read_lines(out/'final_test_cases.jsonl');need(len(cases)==25,'Missing/extra terminal test cases')
    for i,(case,row) in enumerate(zip(c['splits']['test'],cases)):
        need(row['case_id']==case and row['actual_index']==i and row['native_windows']==27 and row['selection']=='EMA+OP','Actual terminal case/window identity differs')
        need(all(number(row[k],0,1 if k in ('dice','jaccard') else None) for k in METRICS),'Invalid case metric')
    m=final['final_metrics'];aggregate={k:sum(r[k] for r in cases)/25 for k in METRICS}
    need(m['case_ids']==c['splits']['test'] and m['case_dice']==[r['dice'] for r in cases] and m['final_sha256']==h and m['test_evaluations']==1
         and m['selection']=='fixed_terminal_EMA_plus_OP' and m['native_patch_forwards']==675,'Terminal result/artifact/selection binding differs')
    need(all(number(m[k]) and math.isclose(m[k],v,rel_tol=0,abs_tol=1e-12) for k,v in aggregate.items()),'Independent case aggregation differs')
    need({p for p in target.parent.rglob('*') if p.is_file()}=={target},'Unexpected checkpoint/probe/partial residual')
    allowed={'run_intent.json','preparation_verified.json','native_initialization_diagnostics.json','steps.jsonl','coverage_verified.json','final_metrics.json','final_test_intent.json','final_test_cases.jsonl','native_reload_verified.json','native_final_test_started.json','dualteacher_reload_verified.json','dualteacher_final_test_started.json'}
    need({p.name for p in out.iterdir()}==allowed,'Unexpected formal output or evaluation evidence')
    runtime=verify_runtime_quiescent(c,pid)
    return dict(at=datetime.datetime.now().astimezone().isoformat(),status='COMPLETE_VERIFIED',method=METHOD,dataset='BY',ratio=c['ratio'],job=str(job),source_dir=str(source),contract_sha256=expected,
        counts=counts,complete_cycles=36,artifact=artifact,final_metrics=aggregate,test_evaluations=1,strict_reload_receipts_verified=True,models_loaded=0,tests_repeated=0,process_snapshot=runtime,
        native_auxiliary_op_updates=total,native_total_optimizer_updates=2*total,evidence={str(p.relative_to(job)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in job.rglob('*') if p.is_file()})

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--job',type=Path,required=True);a.add_argument('--source-dir',type=Path,required=True);a.add_argument('--contract-sha',required=True);a.add_argument('--output',type=Path,required=True);x=a.parse_args()
    need(not x.output.exists() and ROOT in x.output.parents and not any(p in x.output.parents for p in (x.job,x.source_dir,ROOT/'expes/artifacts')),'External acceptance output must be new and outside protected runtime trees')
    try:v=verify(x.job,x.source_dir,x.contract_sha);code=0
    except Exception as e:v=dict(status='NOT_VERIFIED',error=repr(e),models_loaded=0,tests_repeated=0);code=2
    with x.output.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
    print(json.dumps(dict(status=v['status'],output=str(x.output),sha256=sha(x.output))));raise SystemExit(code)

if __name__=='__main__':main()
