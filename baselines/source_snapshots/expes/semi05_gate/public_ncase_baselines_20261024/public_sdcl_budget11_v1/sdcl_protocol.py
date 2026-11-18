"""Pinned SDCL Public case-budget protocol and bounded two-ledger storage."""
import contextlib,hashlib,json,os,random,sys,time
from pathlib import Path
from stage_state import plan,initial,accept,validate
from two_stage_controller import exclusive as exclusive_json
ROOT=Path('__RECORDED_PROJECT_ROOT__');HERE=Path(__file__).resolve().parent
OLD=ROOT/'expes/semi05_gate/main_table_20261019_r3'
SHARED=ROOT/'expes/work_dir/semi05_gate/MAIN_TABLE_FIXED_FINAL_20261019'
BASE=SHARED/'public_ncase_baselines_20261024'
STORE=ROOT/'expes/artifacts/FCD_A_MATRIX_20261006/MAIN_TABLE_FIXED_FINAL_20261019/public_ncase_baselines_20261024'
SPLITS=ROOT/'expes/semi05_gate/public_ncase_20261019'
LEASE=SHARED/'native_gpu1_r4.lock';UUID='GPU_CONFIGURED_AT_RUNTIME';RAM=12*1024**3
sys.path.insert(0,str(OLD))
from runtime_resources import resource_backend,BoundedWriter,REGISTRY,sha
from runtime_data import bind_case_files,schedule
from budget_protocol import Budget
from fixed_endpoint import Endpoint

def read(p):return json.loads(Path(p).read_text())
def syncdir(p):
 fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def atomic(p,v):
 p=Path(p);t=p.with_name(p.name+'.tmp');assert not p.is_symlink() and not t.exists()
 with t.open('x') as f:json.dump(v,f,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(t,p);syncdir(p.parent)
def split(n):
 assert n in (5,10);sys.path.insert(0,str(SPLITS));from ncase_protocol import validate_splits
 v=validate_splits(SPLITS/'splits')[n]
 return dict(train_lab=v['lab'],train_unlab=v['unlab'],test=v['test'])
def bound_splits(dataset,n):
 assert dataset=='Public433';return split(n)
def clock_budget(n):return Budget('Public433',n,n,346-n,cycles=36)
def paths(n,tag):
 import re
 assert n in (5,10) and re.fullmatch(r'r[1-9][0-9]*',tag)
 rel=Path('sdcl')/'Public433'/('lab%03d'%n)/'fold2'/tag
 return BASE/rel,STORE/rel/'final.pth'
def job(c):return Path(c['output']).parent
def contract_sha(c):return sha(job(c)/'contract.json')
def endpoint(c):return Endpoint('sdcl',c['primary_pairs'],c['primary_pairs'],2,tuple(c['splits']['test']))
def rows(c):
 base=Path(c['target']).parent
 return [dict(path=c['target'],bound=256*1024**2)]+[dict(path=str(base/n/'final.pth'),bound=768*1024**2) for n in ('recovery_a','recovery_b','qualification_prefix','qualification_ssl')]+[dict(path=str(base/'ledger_stdout_capacity/final.pth'),bound=64*1024**2)]
def resource(c):
 g=resource_backend(Path(c['target']),256*1024**2);old=g.targets
 def targets():
  out=old();seen={str(p):b for p,b in out}
  ap=SHARED/'cpac_auxiliary_targets_20261022.json'
  for r in read(ap):
   p=Path(r['path']);assert SHARED/'cpac_paper_20261022' in p.parents and p.name in ('steps.jsonl','epochs.jsonl','roundtrip.pth')
   assert type(r['bound']) is int and 0<r['bound']<=10*1024**3 and str(p) not in seen
   out.append((p,r['bound']));seen[str(p)]=r['bound']
  for r in rows(c):
   if r['path'] in seen:assert seen[r['path']]==r['bound']
   else:out.append((Path(r['path']),r['bound']))
  return out
 g.targets=targets;return g
def capacity(c,start=False):
 g=resource(c)
 with g.original_lock():
  cap=g.capacity_snapshot(RAM)
  if start:assert all(x['margin_bytes']>=2*1024**3 for x in cap['filesystems']),'SDCL launch requires 2GiB after all growth'
 return cap
def enroll(c):
 g=resource(c)
 with g.original_lock():
  cap=g.capacity_snapshot(RAM);assert all(x['margin_bytes']>=2*1024**3 for x in cap['filesystems'])
  rr=read(REGISTRY)
  for r in rows(c):
   matches=[x for x in rr if x['path']==r['path']]
   if matches:assert matches==[r]
   else:rr.append(r)
  atomic(REGISTRY,rr)
 return cap
def verify(c):
 assert c['method']=='sdcl' and c['dataset']=='Public433' and c['gpu']==1 and c['fold']==2 and c['seed']==2027
 assert c['labeled_cases']==c['ratio'] in (5,10) and c['labels_are_case_counts_not_percent']
 assert c['stage_recipe']==plan(c['ratio']) and c['primary_pairs']==plan(c['ratio'])['primary_inputs']
 assert c['splits']==split(c['ratio']) and c['case_files']==bind_case_files('Public433',c['splits'])
 assert paths(c['ratio'],c['tag'])==(job(c),Path(c['target']))
 assert c['recovery_interval']==100 and c['recovery_bound']==768*1024**2 and c['checkpoint_bound']==256*1024**2
 assert c['test_evaluations']==1 and c['selection']=='terminal_after_fixed_exposure'
 assert c['budget_protocol']=='PUBLIC_PRIMARY_INPUT_11_CYCLE_CUTOFF_20261025'
 assert c['schedule_policy']=='retain_original_36_cycle_lr_ramp_clocks; cutoff_only'
 assert c['engineering_probe_prefix_iterations']==4
 for r in c['sources']+c['inputs']:
  p=Path(r['path']);assert not p.is_symlink() and sha(p)==r['sha256'],str(p)
 assert sha(HERE/'budget11.json')==c['budget_manifest']['sha256']=='9c79216dafe33283c4e49bd64c40e4dbe9c9ff92cff8f0a3b3af02c6c844677b'
 for r in c['cpu_proofs']:
  assert sha(r['path'])==r['sha256'];v=read(r['path']);assert v['passed'] is True
def resource_status(c,v):atomic(job(c)/'resource_status.json',v)
def gpu_idle():
 import subprocess
 s=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid,pci.bus_id','--format=csv,noheader'],text=True).strip()
 assert s.split(',')[0].strip()==UUID and s.split(',')[1].strip().endswith('B3:00.0')
 s=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 assert not any(x.split(',')[0].strip()==UUID for x in s.splitlines() if x.strip()),'GPU1 busy'
def lease_open():
 assert not LEASE.is_symlink() and LEASE.read_bytes()==b'NATIVE_MAIN_TABLE_R4_GPU1_EXCLUSIVE_OPERATION_V1\n'
 fd=os.open(LEASE,os.O_RDONLY|os.O_NOFOLLOW);s=os.fstat(fd);p=LEASE.stat();assert (s.st_dev,s.st_ino)==(p.st_dev,p.st_ino);return fd
@contextlib.contextmanager
def leased(c):
 import fcntl
 fd=None
 try:
  g=resource(c)
  with g.original_lock():
   fd=lease_open();fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);g.capacity_snapshot(RAM);gpu_idle()
  yield fd
 finally:
  if fd is not None:os.close(fd)
def file_pin(p):
 p=Path(p)
 if not p.exists():return dict(bytes=0,sha256=hashlib.sha256(b'').hexdigest())
 assert p.is_file() and not p.is_symlink() and p.stat().st_nlink==1
 return dict(bytes=p.stat().st_size,sha256=sha(p))
def ledgers(c):return {n:file_pin(Path(c['output'])/n) for n in ('pretrain_steps.jsonl','steps.jsonl')}
def snapshot_ledgers(c):
 out={}
 for n in ('pretrain_steps.jsonl','steps.jsonl'):
  p=Path(c['output'])/n
  if p.exists():
   assert not p.is_symlink()
   with p.open('rb') as f:s=os.fstat(f.fileno());raw=f.read()
   end=raw.rfind(b'\n')+1
   out[n]=dict(raw=raw[:end],inode=s.st_ino,pending_bytes=len(raw)-end,read_bytes=end,parsed_prefix_sha256=hashlib.sha256(raw[:end]).hexdigest())
  else:out[n]=dict(raw=b'',inode=None,pending_bytes=0,read_bytes=0,parsed_prefix_sha256=hashlib.sha256(b'').hexdigest())
 return out
def inspect_ledgers(c):
 snaps=snapshot_ledgers(c);progress=initial();recipe=plan(c['ratio']);pairs=schedule(clock_budget(c['ratio']),c['splits']);case_ids=set()
 for phase,name in [('prefix','pretrain_steps.jsonl'),('ssl','steps.jsonl')]:
  raw=snaps[name]['raw']
  if phase=='ssl' and raw:
   assert progress['prefix']['iterations']==recipe['prefix_iterations'];progress['phase']='ssl'
  for line in raw.splitlines():
   row=json.loads(line);step=row['step'];idx=progress[phase]['iterations']
   if phase=='prefix':
    expected=[];ids=c['splits']['train_lab']
    for k in range(4):
     ep,within=divmod(idx*4+k,len(ids));order=list(ids);random.Random(2027+ep*104729).shuffle(order);expected.append(order[within])
    assert step['actual_labeled_case_ids']==expected;case_ids.update(expected)
   else:
    expected=pairs[idx*4:idx*4+4];assert step['actual_pairs']==expected and step['completed_primary_pairs']==4*(idx+1)
    for pair in expected:case_ids.update([pair['labeled'],pair['unlabeled']])
   progress=accept(progress,step,recipe);assert row['cumulative']==progress[phase]
 return dict(progress=progress,actual_training_cases=len(case_ids),ledgers={n:{k:v for k,v in d.items() if k!='raw'} for n,d in snaps.items()})
def verify_actual_ledgers(c,pre_rows,pairs):
 review=inspect_ledgers(c);assert review['progress']['prefix']['iterations']==len(pre_rows)
 assert review['progress']['ssl']['labeled_inputs']==len(pairs)==c['primary_pairs']
 assert pairs==schedule(clock_budget(c['ratio']),c['splits'])[:c['primary_pairs']]
def checkpoints(c):
 out=[]
 for p in sorted((job(c)/'recovery_commits').glob('commit_*.json'),reverse=True):
  r=read(p);assert r['contract_sha256']==contract_sha(c) and r['path'] in [x['path'] for x in rows(c)[1:3]]
  validate(r['progress'],c['stage_recipe'])
  if file_pin(r['path'])==dict(bytes=r['bytes'],sha256=r['sha256']):out.append(r)
 return out
def write_tensor(c,p,payload,bound,torch,replace):
 p=Path(p);t=Path(str(p)+'.partial');g=resource(c)
 with g.original_lock():
  cap=g.capacity_snapshot(RAM);assert not any(x.is_symlink() for x in [p,t,*p.parents]) and not t.exists()
  if not replace:assert not p.exists()
  p.parent.mkdir(parents=True,exist_ok=True)
  with t.open('xb') as f:torch.save(payload,BoundedWriter(f,bound));f.flush();os.fsync(f.fileno())
  digest=sha(t);size=t.stat().st_size
  if replace:os.replace(t,p)
  else:
   os.link(t,p);assert p.stat().st_ino==t.stat().st_ino and p.stat().st_nlink==2;t.unlink()
  syncdir(p.parent)
 return dict(path=str(p),sha256=digest,bytes=size,resource_lock_verified=True,capacity_before=cap)
def save_checkpoint(c,state):
 import torch
 assert state['execution_role']=='formal' and state['recipe']==c['stage_recipe'] and state['contract_sha256']==contract_sha(c)
 validate(state['progress'],c['stage_recipe']);assert state['ledgers']==ledgers(c)
 d=job(c)/'recovery_commits';d.mkdir(exist_ok=True);gen=max([int(p.stem.split('_')[1]) for p in d.glob('commit_*.json')],default=-1)+1
 r=write_tensor(c,rows(c)[1+gen%2]['path'],state,c['recovery_bound'],torch,True)
 r={k:r[k] for k in ('path','sha256','bytes')};r.update(generation=gen,progress=state['progress'],contract_sha256=contract_sha(c),ledgers=state['ledgers'],at=time.time())
 exclusive_json(d/('commit_%06d.json'%gen),r);syncdir(d);return r
def guard_test_lineage(c):
 assert not (Path(c['output'])/'final_test_intent.json').exists()
 for p in (job(c)/'recovery_history').glob('*/preserved_final_test_intent.json'):raise ValueError('Previously opened test: '+str(p))
def load_reviewed_checkpoint(c,torch):
 # No automatic fallback/reconciliation on unknown failure; external review pins exact generation and both prefixes.
 r=read(job(c)/'resume_review.json');assert r['contract_sha256']==contract_sha(c) and r['engineering_review_passed'] and r['standing_human_authorization']
 cp=next(x for x in checkpoints(c) if x['sha256']==r['checkpoint_sha256']);assert cp['ledgers']==ledgers(c)
 return cp,torch.load(cp['path'],map_location='cpu')
def record_restoration(c,commit,state_sha,progress):
 exclusive_json(job(c)/('restoration_verified_%d.json'%os.getpid()),dict(contract_sha256=contract_sha(c),checkpoint_sha256=commit['sha256'],loaded_state_exact=True,state_sha256=state_sha,progress=progress,ledgers=commit['ledgers'],new_optimization_updates=0,new_test_evaluations=0))
def save_terminal(c,payload,torch,tree_hash):
 p=Path(c['target'])
 if p.exists():
  guard_test_lineage(c);old=torch.load(p,map_location='cpu');assert tree_hash(old)==tree_hash(payload)
  return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,resource_lock_verified=True,reused_fixed_terminal=True)
 return write_tensor(c,p,payload,c['checkpoint_bound'],torch,False)
