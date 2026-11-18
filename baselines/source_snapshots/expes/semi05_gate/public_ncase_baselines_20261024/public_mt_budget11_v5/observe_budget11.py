"""Independent live ledger/lease/source/recovery review; never imports torch."""
import argparse,hashlib,json,os,sys,subprocess,time
from pathlib import Path

def inspect(job,expected,previous=None):
 source=Path(__file__).resolve().parent
 sys.path.insert(0,str(source))
 import public_protocol as p
 assert p.sha(job/'contract.json')==expected
 c=p.read(job/'contract.json');p.verify(c)
 m=c.get('budget_migration')
 cp=m['checkpoint'] if m else None
 prep=job/'formal/preparation_verified.json'
 if prep.exists():
  actual=p.read(prep);qual=p.read(job/'qualification.json')
  assert actual['initial_student_sha']==qual['native_student_signature']
  assert actual['generalist_checkpoint_sha256']==c['sam_sha256']
 dispatches=[x for x in [job/'dispatch.json',*job.glob('dispatch_resume_*.json')] if x.exists()]
 assert dispatches
 dp=max(dispatches,key=lambda x:p.read(x)['at']);d=p.read(dp);pid=d['pid']
 proc=Path('/proc')/str(pid);alive=proc.exists()
 steps=Path(c['output'])/'steps.jsonl'
 if steps.exists():
  with steps.open('rb') as f:
   stat=os.fstat(f.fileno());raw=f.read()
  size=len(raw);end=raw.rfind(b'\n')+1;parsed=raw[:end]
  inode=stat.st_ino
 else:size=0;end=0;raw=b'';parsed=b'';inode=None
 from runtime_data import schedule
 pairs=schedule(p.budget(c['ratio']),c['splits'])
 totals=dict(iterations=0,labeled_inputs=0,unlabeled_inputs=0,successful_optimizer_updates=0)
 cases=set()
 for line in parsed.splitlines():
  r=json.loads(line);s=r['step'];i=totals['iterations']
  assert s['actual_pair']==pairs[i] and s['completed_primary_pairs']==i+1
  assert s['labeled_inputs']==s['unlabeled_inputs']==s['successful_optimizer_updates']==1
  assert all(s[k] for k in ['finite_loss','finite_nonzero_gradients','teacher_no_grad','actual_case_ids_verified'])
  assert s['skip_count']==0 and s['unlabeled_gt_in_loss'] is False
  assert all(__import__('math').isfinite(float(v)) for v in s['losses'].values())
  totals={k:v+1 for k,v in totals.items()};assert r['cumulative']==totals
  cases.update([s['actual_pair']['labeled'],s['actual_pair']['unlabeled']])
 assert totals['iterations']<=c['primary_pairs']
 if cp:assert totals['iterations']>=cp['iteration'] and hashlib.sha256(raw[:cp['ledger']['bytes']]).hexdigest()==cp['ledger']['sha256']
 restored=job/('restoration_verified_%d.json'%pid)
 if restored.exists():
  rv=p.read(restored);assert rv['loaded_state_exact'] and rv['contract_sha256']==expected
  assert rv['new_optimizer_updates']==rv['new_test_evaluations']==0
  if cp:assert rv['loaded_counts']['iterations']>=cp['iteration']
 if previous:
  old=p.read(previous)
  if old['dispatch_sha256']==p.sha(dp):
   assert old['ledger']['inode']==inode and len(raw)>=old['ledger']['read_bytes']
   assert hashlib.sha256(raw[:old['ledger']['read_bytes']]).hexdigest()==old['ledger']['parsed_prefix_sha256']
  else:
   assert list((job/'recovery_history').glob('*/reconciliation.json')),'Changed worker without reviewed recovery'
 boot=Path(d.get('boot',str(job/'boot.log')))
 assert not boot.read_bytes(),'Unexpected stdout/stderr: independently review exact source and content'
 errors=list(job.glob('worker_error_*.json'))+list((job/'formal').glob('failure.json'))
 assert not errors,errors
 commits=p.checkpoints(c)
 if commits and cp:assert restored.exists(),'Migrated checkpoint without exact restoration evidence'
 gpu=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid,pci.bus_id,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
 assert gpu.split(',')[0].strip()==p.UUID and gpu.split(',')[1].strip().endswith('B3:00.0')
 if alive:
  cmd=(proc/'cmdline').read_bytes().decode().rstrip('\0').split('\0');assert cmd==d['command']
  env=(proc/'environ').read_bytes().split(b'\0')
  assert b'CUDA_VISIBLE_DEVICES='+p.UUID.encode() in env
  fd=d['command'][d['command'].index('--fd')+1];s=(proc/'fd'/fd).stat();l=p.LEASE.stat()
  assert (s.st_dev,s.st_ino)==(l.st_dev,l.st_ino)
  import fcntl
  competitor=p.lease_open()
  try:
   try:fcntl.flock(competitor,fcntl.LOCK_EX|fcntl.LOCK_NB);raise AssertionError('Lease not held')
   except BlockingIOError:pass
  finally:os.close(competitor)
  status='RUNNING' if commits else 'INITIALIZING'
 else:
  assert (job/'formal/final_metrics.json').exists(),'PID absent without complete result'
  result=p.read(job/'formal/final_metrics.json')
  assert totals['iterations']==c['primary_pairs'] and result['counts']==totals
  assert result['final_metrics']['case_ids']==c['splits']['test'] and result['test_evaluations']==1
  assert p.sha(result['artifact']['path'])==result['artifact']['sha256']
  reload=result['strict_reload'];assert reload['strict'] and reload['native_student_identity_verified'] and reload['sha256']==result['artifact']['sha256']
  permit=p.read(job/'formal/final_test_intent.json');assert permit['final_sha256']==result['artifact']['sha256'] and permit['test_evaluation']==1
  assert len(result['final_metrics']['case_dice'])==87 and len(set(result['final_metrics']['case_ids']))==87
  import math
  values=result['final_metrics']['case_dice'];assert all(math.isfinite(v) and 0<=v<=1 for v in values)
  assert math.isclose(sum(values)/87,result['final_metrics']['dice'],abs_tol=1e-12,rel_tol=0)
  exits=[p.read(x) for x in job.glob('exit_*.json')];assert exits and exits[-1]['exit_code']==0
  p.gpu_idle()
  import fcntl
  fd=p.lease_open()
  try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  finally:os.close(fd)
  status='COMPLETE_NEEDS_INDEPENDENT_FINAL_ACCEPTANCE'
 try:cap=p.capacity(c)
 except (ValueError,RuntimeError,AssertionError) as e:
  assert str(e).startswith(('Insufficient filesystem capacity:','Insufficient RAM including cgroup ancestors'))
  waiting=p.read(job/'resource_status.json');assert waiting['state']=='WAIT_CAPACITY' and time.time()-waiting['at']<180 and alive
  cap=dict(resource_wait=waiting,error=str(e));status='WAIT_CAPACITY'
 if commits:assert commits[0]['iteration']<=totals['iterations'] and totals['iterations']-commits[0]['iteration']<=100
 return dict(status=status,at=time.time(),job=str(job),contract_sha256=expected,pid=pid,pid_alive=alive,
  dispatch_sha256=p.sha(dp),counts=totals,budget_per_stream=c['primary_pairs'],actual_training_cases=len(cases),
  ledger=dict(read_bytes=end,pending_bytes=size-end,parsed_prefix_sha256=hashlib.sha256(parsed).hexdigest(),inode=inode),
  latest_recovery=commits[0] if commits else None,gpu1=gpu,capacity=cap,test_started=(job/'formal/final_test_intent.json').exists(),
  frozen_sources_verified=True,lease_verified=alive,lease_released=not alive,
  migration_origin_checkpoint=cp['sha256'] if cp else None,imported_committed_iterations=cp['iteration'] if cp else 0,
  new_segment_updates=totals['iterations']-(cp['iteration'] if cp else 0),
  preserved_original_extra_updates=98 if cp else 0,unmeasured_inflight_cost_disclosed=bool(cp),
  continuation_bitwise_guaranteed=False)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--job',type=Path,required=True);a.add_argument('--contract-sha',required=True);a.add_argument('--previous');a.add_argument('--out',type=Path,required=True)
 v=a.parse_args();r=inspect(v.job,v.contract_sha,v.previous)
 with v.out.open('x') as f:json.dump(r,f,indent=2);f.flush();os.fsync(f.fileno())
 print(json.dumps({k:r[k] for k in ['status','counts','actual_training_cases','pid','latest_recovery']}))
