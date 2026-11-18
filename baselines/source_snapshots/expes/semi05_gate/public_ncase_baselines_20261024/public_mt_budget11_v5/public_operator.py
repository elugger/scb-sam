"""Explicit, independently admitted operations. No queue or automatic retry."""
import argparse,subprocess,traceback
from public_protocol import *

def prepare(a):
 job,target=paths(a.method,a.labels,a.tag)
 assert not job.exists() and not target.parent.exists(),'Existing attempt; no reuse'
 t=read(HERE/'template.json');s=split(a.labels)
 sources=list(t['sources'])+[{ 'path':str(p),'sha256':sha(p)} for p in sorted(HERE.glob('*.py'))]+[{'path':str(HERE/'template.json'),'sha256':sha(HERE/'template.json')}]
 inputs=[{'path':str(p),'sha256':sha(p)} for p in sorted((SPLITS/'splits').glob('*.txt'))]+[{'path':str(SPLITS/'ncase_protocol.py'),'sha256':sha(SPLITS/'ncase_protocol.py')}]
 c=dict(method=a.method,dataset='Public433',ratio=a.labels,labeled_cases=a.labels,labels_are_case_counts_not_percent=True,
  tag=a.tag,gpu=1,fold=2,seed=2027,target=str(target),output=str(job/'formal'),splits=s,
  case_files=bind_case_files('Public433',s),primary_pairs=budget(a.labels).primary_pairs,recipe=t['recipe'],sam_sha256=t['sam_sha256'],
  sources=sources,inputs=inputs,recovery_interval=100,recovery_bound=128*1024**2,checkpoint_bound=32*1024**2,
  continuation_policy='exact_saved_state_and_rng_restore_native_FP32_nondeterminism_disclosed',test_evaluations=1,exact_resume_supported='REQUIRES_OWN_FRESH_PROCESS_QUALIFICATION',selection='terminal_after_fixed_exposure',
  human_authorization='Use the new eleven-cycle primary-input standard for all remaining baselines and current MT five cases; solve engineering and space problems and continue',cpac_launch_authorized=False,
  budget_protocol='PUBLIC_PRIMARY_INPUT_11_CYCLE_CUTOFF_20261025',schedule_policy='retain_original_36_cycle_lr_ramp_clocks; cutoff_only',
  budget_manifest=dict(path=str(HERE/'budget11.json'),sha256=sha(HERE/'budget11.json')),click_patch=dict(path=str(HERE/'candidate_ritm.py'),sha256=sha(HERE/'candidate_ritm.py')))
 verify(c);cap=capacity(c,True)
 # All sources, data identity, and projected space pass before registration.
 job.mkdir(parents=True,exist_ok=False)
 try:
  cap=enroll(c);exclusive_json(job/'contract.json',c)
  exclusive_json(job/'preparation.json',dict(status='BOUND_NOT_QUALIFIED',contract_sha256=sha(job/'contract.json'),capacity=cap))
 except BaseException as e:
  exclusive_json(job/'prepare_failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
 return dict(status='BOUND_NOT_QUALIFIED',job=str(job),contract_sha256=sha(job/'contract.json'),capacity=cap)

def qualify(c,j):
 assert not (j/'qualification_intent.json').exists() and not (j/'formal').exists()
 with leased(c):
  os.environ['CUDA_VISIBLE_DEVICES']=UUID
  from resumable_mt import Backend,tree_hash
  import torch
  q=j/'qualification_backend';q.mkdir()
  exclusive_json(j/'qualification_intent.json',dict(contract_sha256=sha(j/'contract.json'),at=time.time()))
  try:
   b=Backend(c,q);r=b.qualify(endpoint(c));b.t.optimizer.norm=None
   initial=b.capture();remaining=dict(labeled_inputs=c['primary_pairs'],unlabeled_inputs=c['primary_pairs'])
   for _ in range(2):b.train_iteration(remaining)
   state=b.capture();p=Path(rows(c)[3]['path']);g=resource(c)
   with g.original_lock():
    g.capacity_snapshot(RAM);p.parent.mkdir(parents=True,exist_ok=False)
    with p.open('xb') as f:torch.save(state,BoundedWriter(f,c['recovery_bound']));f.flush();os.fsync(f.fileno())
    syncdir(p.parent)
    loaded=torch.load(p,map_location='cpu');assert tree_hash(loaded)==tree_hash(state)
   ref=[b.train_iteration(remaining) for _ in range(2)]
   reference=dict(contract_sha256=sha(j/'contract.json'),checkpoint_sha256=sha(p),checkpoint_path=str(p),
    expected_state_sha256=tree_hash(b.capture()),expected_rng_sha256=tree_hash(b.capture()['rng']),steps=ref,initial_state_sha256=tree_hash(initial),checkpoint_iteration=2)
   exclusive_json(q/'resume_reference.json',reference)
   b.restore(initial);assert tree_hash(b.capture())==tree_hash(initial)
   r.update(actual_extra_engineering_updates=6,actual_inputs_per_stream=6,new_test_evaluations=0,
    restored_after_all_probes=True,initial_state_sha256=tree_hash(initial),resume_checkpoint_bytes=p.stat().st_size,
    resume_reference_sha256=sha(q/'resume_reference.json'),peak_cuda_bytes=torch.cuda.max_memory_allocated(),
    contract_sha256=sha(j/'contract.json'),frozen_sam_signature=b.frozen_signature)
   exclusive_json(j/'qualification.json',r)
   return dict(status='OWN_QUALIFIED_RESUME_PROBE_PENDING',extra_engineering_updates=6,checkpoint_bytes=p.stat().st_size)
  except BaseException as e:
   exclusive_json(j/'qualification_failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise

def resume_probe(c,j):
 assert (j/'qualification.json').exists() and not (j/'resume_qualification.json').exists() and not (j/'formal').exists()
 with leased(c):
  os.environ['CUDA_VISIBLE_DEVICES']=UUID
  from resumable_mt import Backend,tree_hash
  import torch
  b=Backend(c,j/'qualification_backend');b.prepare(endpoint(c));r=read(j/'qualification_backend/resume_reference.json')
  assert sha(r['checkpoint_path'])==r['checkpoint_sha256']
  s=torch.load(r['checkpoint_path'],map_location='cpu');b.restore(s)
  steps=[b.train_iteration(dict(labeled_inputs=c['primary_pairs'],unlabeled_inputs=c['primary_pairs'])) for _ in range(2)]
  # restore() already proved every saved model/optimizer/RNG/counter tensor exact.
  # Native CUDA reductions are not deterministic on this pinned Torch runtime.
  import math
  indices=all(a['actual_pair']==v['actual_pair'] for a,v in zip(steps,r['steps']))
  losses_close=all(math.isclose(a['losses'][k],v['losses'][k],rel_tol=1e-4,abs_tol=1e-5) for a,v in zip(steps,r['steps']) for k in a['losses'])
  rng_equal=tree_hash(b.capture()['rng'])==r['expected_rng_sha256']
  assert indices and losses_close and rng_equal,'State/cursor/RNG/numerical continuation check failed'
  result=dict(status='PASS',fresh_process=True,state_restoration_bitwise_equal=True,
   continuation_state_bitwise_equal=tree_hash(b.capture())==r['expected_state_sha256'],
   metrics_equal=steps==r['steps'],losses_close=losses_close,indices_equal=indices,rng_sequence_equal=rng_equal,
   numeric_tolerance=dict(loss_rtol=1e-4,loss_atol=1e-5),
   max_abs_loss_difference=max(abs(a['losses'][k]-v['losses'][k]) for a,v in zip(steps,r['steps']) for k in a['losses']),
   differences=[{k:dict(reference=v[k],actual=a[k]) for k in a if a[k]!=v[k]} for a,v in zip(steps,r['steps'])],
   disclosure='Exact saved-state restoration; native CUDA continuation is numerically checked, not bitwise guaranteed',
   contract_sha256=sha(j/'contract.json'),checkpoint_sha256=r['checkpoint_sha256'],extra_engineering_updates=2,
   new_test_evaluations=0,at=time.time())
  exclusive_json(j/'resume_qualification.json',result)
  return result

def worker(c,j,a):
 import fcntl
 assert a.fd is not None
 f=os.fstat(a.fd);p=LEASE.stat();assert (f.st_dev,f.st_ino)==(p.st_dev,p.st_ino)
 competitor=lease_open()
 try:
  try:fcntl.flock(competitor,fcntl.LOCK_EX|fcntl.LOCK_NB);raise AssertionError('Inherited GPU lease lost')
  except BlockingIOError:pass
 finally:os.close(competitor)
 os.environ['CUDA_VISIBLE_DEVICES']=UUID
 from resumable_mt import Backend
 from controller import RecoveryRun
 b=Backend(c,Path(c['output']))
 try:
  result=RecoveryRun(endpoint(c),c['output'],resume=a.resume_worker).run(b)
  exclusive_json(j/('exit_%d.json'%os.getpid()),dict(exit_code=0,at=time.time(),status=result['status']))
 except BaseException as e:
  exclusive_json(j/('worker_error_%d.json'%os.getpid()),dict(error=repr(e),traceback=traceback.format_exc(),at=time.time()));raise

def launch(c,j,a,resume=False):
 transition=a.action=='start-transition'
 assert a.confirm==('START_BUDGET11_CONTINUATION' if transition else 'RESUME_VALIDATED_PUBLIC_MT' if resume else 'START_PUBLIC_NCASE')
 from admission import inspect_ready
 admitted=inspect_ready(c,j)
 assert admitted['status']=='READY_INDEPENDENTLY_VERIFIED'
 assert not any(j.glob('worker_error_*.json')) if not resume else True
 assert not (j/'formal/final_test_intent.json').exists(),'Final test already opened; no automatic replay'
 if not resume:
  assert not (j/'launch_intent.json').exists()
  if transition:
   assert c.get('budget_migration') and not checkpoints(c)
   assert ledger(c)==c['budget_migration']['checkpoint']['ledger']
  else:assert not (j/'formal').exists()
 with leased(c) as fd:
  cap=capacity(c,True)
  if resume:
   from resume_review import reconcile
   history=reconcile(c,j)
   intent=j/('resume_launch_%s.json'%history['event'])
  else:intent=j/'launch_intent.json'
  exclusive_json(intent,dict(contract_sha256=sha(j/'contract.json'),admission=admitted,at=time.time(),capacity=cap))
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=UUID,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
  cmd=[sys.executable,str(HERE/'public_operator.py'),'--action','worker','--job',str(j),'--contract-sha',sha(j/'contract.json'),'--fd',str(fd)]
  if resume or transition:cmd.append('--resume-worker')
  logpath=j/('boot_resume_%s.log'%history['event'] if resume else 'boot.log')
  with logpath.open('xb') as log:
   process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,pass_fds=(fd,),start_new_session=True,cwd=str(ROOT))
  dp=j/('dispatch_resume_%s.json'%history['event'] if resume else 'dispatch.json')
  exclusive_json(dp,dict(pid=process.pid,command=cmd,at=time.time(),contract_sha256=sha(j/'contract.json'),boot=str(logpath)))
 return dict(status='DISPATCHED',pid=process.pid,job=str(j))

def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=['prepare','qualify','resume-probe','admit','start','start-transition','resume','worker','status'],required=True)
 p.add_argument('--method',choices=['mt_semisam','uamt_semisam']);p.add_argument('--labels',type=int,choices=[5,10]);p.add_argument('--tag',default='r1')
 p.add_argument('--job');p.add_argument('--contract-sha');p.add_argument('--confirm');p.add_argument('--fd',type=int);p.add_argument('--resume-worker',action='store_true')
 a=p.parse_args()
 if a.action=='prepare':r=prepare(a)
 else:
  j=Path(a.job);assert sha(j/'contract.json')==a.contract_sha;c=read(j/'contract.json');verify(c)
  assert j==paths(c['method'],c['ratio'],c['tag'])[0]
  if a.action=='qualify':r=qualify(c,j)
  elif a.action=='resume-probe':r=resume_probe(c,j)
  elif a.action=='admit':
   from admission import inspect_ready
   r=inspect_ready(c,j);exclusive_json(j/'admission.json',r)
  elif a.action in ('start','start-transition','resume'):r=launch(c,j,a,resume=a.action=='resume')
  elif a.action=='worker':worker(c,j,a);return
  else:r=dict(job=str(j),formal_started=(j/'formal').exists(),checkpoints=checkpoints(c),capacity=capacity(c))
 print(json.dumps(r))
if __name__=='__main__':main()
