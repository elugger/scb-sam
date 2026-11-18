"""Independent SFR phase/identity/resource observer and terminal acceptance."""
import argparse,ast,fcntl,math,subprocess
from sfr_protocol import *

def boot_check(c,raw,finished):
 # SFR has no approved stdout/stderr diagnostics. Preserve and independently
 # inspect any nonempty stream instead of inheriting another method's allowlist.
 assert raw==b'', 'Unknown SFR boot stdout/stderr'

def check_prefix(c):
 f=Path(c['output']);p=f/'pretrain_terminal.json'
 if not p.exists():return None
 r=read(p);b=r['boundary'];a=b['prefix_artifact']
 assert b['strict_prefix_disk_reload'] and b['frozen_sam'] and b['fresh_stage_seed']==2027
 assert b['native_teacher_initialization']=='independent'
 assert a['path']==str(prefix_path(c,'formal')) and file_pin(a['path'])=={k:a[k] for k in ('bytes','sha256')}
 assert r['prefix']==dict(iterations=6000,labeled_inputs=c['stage_recipe']['prefix_labeled_inputs'],unlabeled_inputs=0,successful_optimizer_updates=6000)
 return r

def check_recovery_link(c,j,old,dp,pid,snaps):
 d=read(dp);event=dp.stem.removeprefix('dispatch_resume_');assert event.isdigit()
 hist=j/'recovery_history'/event;rc=read(hist/'reconciliation.json');rv=read(hist/'review.json')
 assert rv['contract_sha256']==contract_sha(c) and rc['checkpoint']==rv['checkpoint']
 assert old['dispatch_sha256'] in [p['sha256'] for p in rv['old_dispatches'].values()]
 cp=rc['checkpoint']
 for name,v in old['ledgers'].items():
  pin=cp['ledgers'][name];prefix=snaps[name]['raw'][:pin['bytes']]
  tail=(hist/('uncommitted_'+name)).read_bytes();tailpin=rc['uncommitted_ledgers'][name]
  assert hashlib.sha256(tail).hexdigest()==tailpin['sha256'] and len(tail)==tailpin['bytes']
  original=prefix+tail
  assert len(original)>=v['read_bytes'] and hashlib.sha256(original[:v['read_bytes']]).hexdigest()==v['parsed_prefix_sha256']
  assert hashlib.sha256(original).hexdigest()==tailpin['original_full_pin']['sha256']
 restored=read(j/('restoration_verified_%d.json'%pid))
 assert restored['loaded_state_exact'] and restored['contract_sha256']==contract_sha(c)
 assert restored['checkpoint_sha256']==cp['sha256'] and restored['ledgers']==cp['ledgers']
 return dict(event=event,checkpoint_sha256=cp['sha256'],preserved_tails=rc['uncommitted_ledgers'])

def inspect(j,expected,previous=None,final=False):
 assert sha(j/'contract.json')==expected;c=read(j/'contract.json');verify(c)
 dp=max(j.glob('dispatch*.json'),key=lambda p:read(p)['at']);d=read(dp);pid=d['pid'];proc=Path('/proc')/str(pid);alive=proc.exists()
 review=inspect_ledgers(c);progress=review['progress'];snaps=snapshot_ledgers(c);prefix=check_prefix(c)
 if progress['phase']=='ssl':assert prefix is not None
 if previous:
  old=read(previous)
  if old['dispatch_sha256']==sha(dp):
   for name,v in old['ledgers'].items():
    raw=snaps[name]['raw'];assert snaps[name]['inode']==v['inode'] or v['inode'] is None and v['read_bytes']==0
    assert len(raw)>=v['read_bytes'] and hashlib.sha256(raw[:v['read_bytes']]).hexdigest()==v['parsed_prefix_sha256']
  else:
   check_recovery_link(c,j,old,dp,pid,snaps)
 boot_check(c,Path(d['boot']).read_bytes(),not alive)
 assert not list(j.glob('worker_error_*.json')) and not (Path(c['output'])/'failure.json').exists()
 cps=checkpoints(c)
 for cp in cps:
  for name,v in cp['ledgers'].items():
   raw=snaps[name]['raw'];assert len(raw)>=v['bytes'] and hashlib.sha256(raw[:v['bytes']]).hexdigest()==v['sha256']
 if cps:
  cp=cps[0];total=lambda p:p['prefix']['iterations']+p['ssl']['iterations']
  assert 0<=total(progress)-total(cp['progress'])<=100
 gpu=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid,pci.bus_id,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
 assert gpu.split(',')[0].strip()==UUID and gpu.split(',')[1].strip().endswith('B3:00.0')
 if alive:
  assert (proc/'cmdline').read_bytes().decode().rstrip('\0').split('\0')==d['command']
  assert b'CUDA_VISIBLE_DEVICES='+UUID.encode() in (proc/'environ').read_bytes().split(b'\0')
  fd=d['command'][d['command'].index('--fd')+1];s=(proc/'fd'/fd).stat();l=LEASE.stat();assert (s.st_dev,s.st_ino)==(l.st_dev,l.st_ino)
  comp=lease_open()
  try:
   try:fcntl.flock(comp,fcntl.LOCK_EX|fcntl.LOCK_NB);raise AssertionError('GPU1 lease free while alive')
   except BlockingIOError:pass
  finally:os.close(comp)
  status='RUNNING_'+progress['phase'].upper()
 else:
  f=Path(c['output']);m=read(f/'final_metrics.json');assert m['status']=='COMPLETE_NEEDS_INDEPENDENT_ACCEPTANCE'
  assert progress==m['counts'] and progress['ssl']['iterations']==c['stage_recipe']['ssl_iterations']
  assert progress['prefix']['iterations']==c['stage_recipe']['prefix_iterations'] and all(x['pending_bytes']==0 for x in snaps.values())
  assert m['strict_reload']['strict'] and m['strict_reload']['native_student_identity_verified']
  assert m['strict_reload']['sha256']==m['artifact']['sha256']==sha(c['target'])
  assert read(j/('exit_%d.json'%pid))['exit_code']==0;gpu_idle();fd=lease_open()
  try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  finally:os.close(fd)
  status='COMPLETE_NEEDS_INDEPENDENT_ACCEPTANCE'
 try:cap=capacity(c)
 except (AssertionError,RuntimeError,ValueError) as e:
  assert str(e).startswith(('Insufficient filesystem capacity:','Insufficient RAM including cgroup ancestors'))
  wait=read(j/'resource_status.json');assert alive and wait['state']=='WAIT_CAPACITY' and time.time()-wait['at']<180
  cap=dict(wait=wait,error=str(e));status='WAIT_CAPACITY'
 out=dict(status=status,at=time.time(),job=str(j),contract_sha256=expected,pid=pid,pid_alive=alive,dispatch_sha256=sha(dp),
  progress=progress,actual_training_cases=review['actual_training_cases'],ledgers=review['ledgers'],latest_recovery=cps[0] if cps else None,
  gpu1=gpu,capacity=cap,source_inputs_verified=True,lease_verified=alive,lease_released=not alive,test_started=(Path(c['output'])/'final_test_intent.json').exists())
 if final:
  assert not alive and status=='COMPLETE_NEEDS_INDEPENDENT_ACCEPTANCE'
  f=Path(c['output']);cov=read(f/'coverage_verified.json');intent=read(f/'final_test_intent.json');rows_test=[json.loads(x) for x in (f/'final_test_cases.jsonl').read_text().splitlines()]
  assert cov['actual_case_coverage_verified'] and cov['prefix']==progress['prefix'] and cov['ssl']==progress['ssl']
  assert prefix is not None and cov['prefix_artifact']==prefix['boundary']['prefix_artifact']
  assert m['final_metrics']['prefix_artifact']==cov['prefix_artifact']
  assert cov['original_clock_inputs']==c['stage_recipe']['original_clock_inputs'] and len(cov['complete_cycles'])==11
  assert cov['rounding_extra_inputs']==c['stage_recipe']['rounding_extra_inputs']
  started=read(f/'sfr_test_started.json');assert started['test_evaluations']==1 and started['final_sha256']==m['artifact']['sha256'] and started['pid']==pid
  assert cov['actual_inputs']==c['primary_pairs'] and cov['reference_inputs']==c['stage_recipe']['reference_inputs']
  assert len(rows_test)==87 and [x['case_id'] for x in rows_test]==c['splits']['test']==intent['test_case_ids']
  assert len({x['case_id'] for x in rows_test})==87 and [x['actual_index'] for x in rows_test]==list(range(87))
  assert m['test_evaluations']==m['final_metrics']['test_evaluations']==intent['test_evaluation']==1
  assert intent['final_sha256']==m['artifact']['sha256'] and intent['selection']=='terminal_after_fixed_exposure'
  metrics={}
  for key in ('dice','jaccard','hd95','asd'):
   vals=[x[key] for x in rows_test];assert all(math.isfinite(x) and x>=0 for x in vals)
   if key in ('dice','jaccard'):assert all(x<=1 for x in vals)
   metrics[key]=sum(vals)/87;assert math.isclose(metrics[key],m['final_metrics'][key],rel_tol=0,abs_tol=1e-12)
  assert m['final_metrics']['case_dice']==[x['dice'] for x in rows_test]
  assert len(cps)==2 and review['actual_training_cases']==346
  assert cps[0]['progress']==progress and cps[0]['ledgers']==ledgers(c)
  out.update(status='ACCEPTED_FIXED11_SFR_TERMINAL',metrics=metrics,final_artifact=m['artifact'],strict_reload=m['strict_reload'],
   evidence={str(p):file_pin(p) for p in [f/'final_metrics.json',f/'pretrain_terminal.json',f/'coverage_verified.json',f/'final_test_intent.json',f/'final_test_cases.jsonl',f/'sfr_test_started.json',j/('exit_%d.json'%pid)]},
   test_cases=87,test_evaluations=1,distance_unit='voxel',new_model_loads=0,new_optimization_updates=0,new_test_evaluations=0)
 return out

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--contract-sha',required=True);p.add_argument('--previous');p.add_argument('--out',type=Path,required=True);p.add_argument('--final',action='store_true');a=p.parse_args()
 r=inspect(a.job,a.contract_sha,a.previous,a.final);exclusive_json(a.out,r)
 print(json.dumps({k:r[k] for k in ('status','progress','actual_training_cases','pid')}))
