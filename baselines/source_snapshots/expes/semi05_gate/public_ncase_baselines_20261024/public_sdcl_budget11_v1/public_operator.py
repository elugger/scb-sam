"""Explicit SDCL own qualification, two fresh resume probes, admission, one start."""
import argparse,math,subprocess,traceback
from sdcl_protocol import *

def backend(c,out,engineering=False):
 import sdcl_runtime
 import sdcl_protocol
 from resumable_sdcl import make_backend
 return make_backend(sdcl_runtime,sdcl_protocol)(c,out,engineering=engineering)

def prepare(a):
 j,target=paths(a.labels,a.tag);assert not j.exists() and not target.parent.exists()
 template=read(HERE/'template.json');s=split(a.labels)
 src=template['sources']+[dict(path=str(p),sha256=sha(p)) for p in sorted(HERE.glob('*.py'))]+[dict(path=str(HERE/'template.json'),sha256=sha(HERE/'template.json'))]
 inputs=[dict(path=str(p),sha256=sha(p)) for p in sorted((SPLITS/'splits').glob('*.txt'))]+[dict(path=str(SPLITS/'ncase_protocol.py'),sha256=sha(SPLITS/'ncase_protocol.py'))]
 c=dict(method='sdcl',dataset='Public433',ratio=a.labels,labeled_cases=a.labels,labels_are_case_counts_not_percent=True,
  tag=a.tag,gpu=1,fold=2,seed=2027,target=str(target),output=str(j/'formal'),splits=s,case_files=bind_case_files('Public433',s),
  primary_pairs=plan(a.labels)['primary_inputs'],stage_recipe=plan(a.labels),sources=src,inputs=inputs,
  recovery_interval=100,recovery_bound=768*1024**2,checkpoint_bound=256*1024**2,engineering_probe_prefix_iterations=4,
  initializer='seed2027_random_native_VNet_ResVNet_no_SAM',test_evaluations=1,selection='terminal_after_fixed_exposure',
  budget_protocol='PUBLIC_PRIMARY_INPUT_11_CYCLE_CUTOFF_20261025',schedule_policy='retain_original_36_cycle_lr_ramp_clocks; cutoff_only',
  budget_manifest=dict(path=str(HERE/'budget11.json'),sha256=sha(HERE/'budget11.json')),cpu_proofs=template['cpu_proofs'],
  continuation_policy='saved_full_state_exact; native_FP32_continuation_numerical_rtol1e-4_atol1e-5; not_bitwise_guaranteed',
  standing_human_authorization='Run remaining baselines with the new standard; solve engineering/space problems and continue; CPAC held')
 verify(c);capacity(c,True);j.mkdir(parents=True,exist_ok=False)
 try:
  cap=enroll(c);exclusive_json(j/'contract.json',c);exclusive_json(j/'preparation.json',dict(status='BOUND_NOT_QUALIFIED',contract_sha256=contract_sha(c),capacity=cap))
 except BaseException as e:exclusive_json(j/'prepare_failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
 return dict(status='BOUND_NOT_QUALIFIED',job=str(j),contract_sha256=contract_sha(c),stage_recipe=c['stage_recipe'])

def qualify(c,j):
 assert not (j/'qualification_intent.json').exists() and not (j/'formal').exists()
 with leased(c):
  os.environ['CUDA_VISIBLE_DEVICES']=UUID
  import torch
  from recovery_state import tree_hash
  q=j/'qualification_backend';q.mkdir();exclusive_json(j/'qualification_intent.json',dict(contract_sha256=contract_sha(c),at=time.time()))
  b=None
  try:
   b=backend(c,q,True);preparation=b.prepare_initial();initial_state=b.capture();initial_hash=tree_hash(initial_state);refs={}
   for phase,slot in [('prefix',3),('ssl',4)]:
    step=b.prefix_iteration if phase=='prefix' else b.ssl_iteration
    for i in range(2):step(i)
    state=b.capture();artifact=write_tensor(c,rows(c)[slot]['path'],state,c['recovery_bound'],torch,False)
    loaded=torch.load(artifact['path'],map_location='cpu');assert tree_hash(loaded)==tree_hash(state)
    measured=[step(i) for i in range(2,4)]
    end=b.capture();ref=dict(phase=phase,contract_sha256=contract_sha(c),checkpoint=artifact,checkpoint_state_sha256=tree_hash(state),
     expected_state_sha256=tree_hash(end),expected_rng_sha256=tree_hash(end['rng']),steps=measured,
     actual_progress_at_checkpoint=state['progress'],actual_progress_after_reference=end['progress'],execution_role=state['execution_role'])
    exclusive_json(q/(phase+'_reference.json'),ref);refs[phase]=dict(path=str(q/(phase+'_reference.json')),sha256=sha(q/(phase+'_reference.json')))
    if phase=='prefix':b.enter_ssl(b.progress)
   b.restore(initial_state);assert tree_hash(b.capture())==initial_hash
   result=dict(status='OWN_TWO_PHASE_QUALIFIED_FRESH_PROBES_PENDING',contract_sha256=contract_sha(c),actual_extra_engineering_updates=16,
    engineering_labeled_inputs=32,engineering_unlabeled_inputs=16,new_test_evaluations=0,new_formal_updates=0,
    probe_prefix_iterations=4,not_the_formal_648_or_1620_prefix=True,initial_state_sha256=initial_hash,
    restored_initial_state_exact=True,actual_disk_roundtrips=True,preparation=preparation,references=refs,
    peak_cuda_bytes=torch.cuda.max_memory_allocated(),at=time.time())
   exclusive_json(j/'qualification.json',result);return dict(status=result['status'],extra_engineering_updates=16,peak_cuda_bytes=result['peak_cuda_bytes'])
  except BaseException as e:
   exclusive_json(j/'qualification_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),actual_progress=getattr(b,'progress',None),failed_partial_compute_not_assumed_zero=True));raise

def probe(c,j,phase):
 assert phase in ('prefix','ssl') and (j/'qualification.json').exists() and not (j/'formal').exists()
 intent=j/('resume_probe_'+phase+'_intent.json');assert not intent.exists()
 with leased(c):
  os.environ['CUDA_VISIBLE_DEVICES']=UUID
  import torch
  from recovery_state import tree_hash
  exclusive_json(intent,dict(phase=phase,contract_sha256=contract_sha(c),at=time.time()))
  try:
   ref=read(j/'qualification_backend'/(phase+'_reference.json'));cp=ref['checkpoint'];assert sha(cp['path'])==cp['sha256']
   b=backend(c,j/'qualification_backend',True);b.prepare_initial();state=torch.load(cp['path'],map_location='cpu');assert tree_hash(state)==ref['checkpoint_state_sha256'];b.restore(state)
   update=b.prefix_iteration if phase=='prefix' else b.ssl_iteration
   measured=[update(i) for i in range(2,4)];end=b.capture()
   index_key='actual_labeled_case_ids' if phase=='prefix' else 'actual_pairs'
   indices=all(x[index_key]==y[index_key] for x,y in zip(measured,ref['steps']))
   losses=all(math.isclose(x['losses'][k],y['losses'][k],rel_tol=1e-4,abs_tol=1e-5) for x,y in zip(measured,ref['steps']) for k in x['losses'])
   rng_equal=tree_hash(end['rng'])==ref['expected_rng_sha256'];assert indices and losses and rng_equal
   result=dict(status='PASS',phase=phase,contract_sha256=contract_sha(c),checkpoint_sha256=cp['sha256'],fresh_process=True,
    state_restoration_bitwise_equal=True,indices_equal=indices,losses_close=losses,rng_sequence_equal=rng_equal,
    continuation_state_bitwise_equal=tree_hash(end)==ref['expected_state_sha256'],
    numeric_tolerance=dict(rtol=1e-4,atol=1e-5),max_abs_loss_difference=max(abs(x['losses'][k]-y['losses'][k]) for x,y in zip(measured,ref['steps']) for k in x['losses']),
    actual_extra_engineering_updates=4,engineering_labeled_inputs=8,engineering_unlabeled_inputs=0 if phase=='prefix' else 8,
    new_formal_updates=0,new_test_evaluations=0,at=time.time())
   exclusive_json(j/('resume_qualification_'+phase+'.json'),result);return result
  except BaseException as e:exclusive_json(j/('resume_probe_'+phase+'_failure.json'),dict(error=repr(e),traceback=traceback.format_exc(),failed_partial_compute_not_assumed_zero=True));raise

def launch(c,j,a,resume=False):
 assert a.confirm==('RESUME_VALIDATED_PUBLIC_SDCL' if resume else 'START_PUBLIC_NCASE')
 from admission import inspect_ready
 admitted=inspect_ready(c,j,resume=resume);assert admitted['status']=='READY_INDEPENDENTLY_VERIFIED'
 assert not (j/'formal/final_test_intent.json').exists()
 with leased(c) as fd:
  cap=capacity(c,True)
  if resume:
   from resume_review import reconcile
   history=reconcile(c,j);suffix='_resume_'+history['event']
  else:
   suffix='';assert not (j/'formal').exists() and not (j/'launch_intent.json').exists()
  exclusive_json(j/('launch_intent'+suffix+'.json'),dict(contract_sha256=contract_sha(c),admission=admitted,capacity=cap,at=time.time()))
  cmd=[sys.executable,str(HERE/'public_operator.py'),'--action','worker','--job',str(j),'--contract-sha',contract_sha(c),'--fd',str(fd)]
  if resume:cmd.append('--resume-worker')
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUID,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
  log=j/('boot'+suffix+'.log')
  with log.open('xb') as f:p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=str(ROOT),env=env,pass_fds=(fd,),start_new_session=True)
  exclusive_json(j/('dispatch'+suffix+'.json'),dict(pid=p.pid,command=cmd,contract_sha256=contract_sha(c),boot=str(log),at=time.time()))
 return dict(status='DISPATCHED',pid=p.pid,job=str(j),contract_sha256=contract_sha(c))

def worker(c,j,a):
 import fcntl
 assert a.fd is not None;s=os.fstat(a.fd);p=LEASE.stat();assert (s.st_dev,s.st_ino)==(p.st_dev,p.st_ino)
 comp=lease_open()
 try:
  try:fcntl.flock(comp,fcntl.LOCK_EX|fcntl.LOCK_NB);raise AssertionError('Inherited lease lost')
  except BlockingIOError:pass
 finally:os.close(comp)
 os.environ['CUDA_VISIBLE_DEVICES']=UUID;sys.stdout.reconfigure(line_buffering=True)
 from two_stage_controller import TwoStageRun
 try:
  b=backend(c,Path(c['output']));v=TwoStageRun(c['stage_recipe'],c['output'],c['splits']['test'],resume=a.resume_worker).run(b)
  exclusive_json(j/('exit_%d.json'%os.getpid()),dict(exit_code=0,status=v['status'],at=time.time()))
 except BaseException as e:exclusive_json(j/('worker_error_%d.json'%os.getpid()),dict(error=repr(e),traceback=traceback.format_exc(),at=time.time()));raise

def main():
 p=argparse.ArgumentParser();p.add_argument('--action',required=True,choices=['prepare','qualify','resume-probe','admit','start','resume','worker'])
 p.add_argument('--labels',type=int,choices=[5,10]);p.add_argument('--tag',default='r1');p.add_argument('--phase',choices=['prefix','ssl'])
 p.add_argument('--job');p.add_argument('--contract-sha');p.add_argument('--confirm');p.add_argument('--fd',type=int);p.add_argument('--resume-worker',action='store_true')
 a=p.parse_args()
 if a.action=='prepare':v=prepare(a)
 else:
  j=Path(a.job);assert sha(j/'contract.json')==a.contract_sha;c=read(j/'contract.json');verify(c)
  if a.action=='qualify':v=qualify(c,j)
  elif a.action=='resume-probe':v=probe(c,j,a.phase)
  elif a.action=='admit':
   from admission import inspect_ready
   v=inspect_ready(c,j);exclusive_json(j/'admission.json',v)
  elif a.action in ('start','resume'):v=launch(c,j,a,resume=a.action=='resume')
  else:worker(c,j,a);return
 print(json.dumps(v),flush=True)
if __name__=='__main__':main()
