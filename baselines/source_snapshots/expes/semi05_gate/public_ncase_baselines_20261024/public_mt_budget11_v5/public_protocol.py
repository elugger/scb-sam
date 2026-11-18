"""Public433 five/ten CASES; isolated controller over immutable BY MT kernels."""
import contextlib, hashlib, json, os, sys, time
from pathlib import Path

ROOT=Path('__RECORDED_PROJECT_ROOT__')
HERE=Path(__file__).resolve().parent
OLD=ROOT/'expes/semi05_gate/main_table_20261019_r2'
SHARED=ROOT/'expes/work_dir/semi05_gate/MAIN_TABLE_FIXED_FINAL_20261019'
BASE=SHARED/'public_ncase_baselines_20261024'
STORE=ROOT/'expes/artifacts/FCD_A_MATRIX_20261006/MAIN_TABLE_FIXED_FINAL_20261019/public_ncase_baselines_20261024'
SPLITS=ROOT/'expes/semi05_gate/public_ncase_20261019'
LEASE=SHARED/'native_gpu1_r4.lock'
UUID='GPU_CONFIGURED_AT_RUNTIME'
RAM=10*1024**3
sys.path.insert(0,str(OLD))
from runtime_resources import sha, BoundedWriter, REGISTRY, resource_backend
from runtime_data import bind_case_files
from fixed_endpoint import exclusive_json, Endpoint
from budget_protocol import Budget

def read(p): return json.loads(Path(p).read_text())
def syncdir(p):
 fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def atomic(p,value):
 p=Path(p);t=p.with_name(p.name+'.tmp')
 assert not p.is_symlink() and not t.is_symlink() and not t.exists()
 with t.open('x') as f:json.dump(value,f,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(t,p);syncdir(p.parent)
def split(n):
 assert n in (5,10)
 sys.path.insert(0,str(SPLITS))
 from ncase_protocol import validate_splits
 s=validate_splits(SPLITS/'splits')[n]
 return dict(train_lab=s['lab'],train_unlab=s['unlab'],test=s['test'])
def paths(method,n,tag):
 import re
 assert method in ('mt_semisam','uamt_semisam') and n in (5,10) and re.fullmatch(r'r[1-9][0-9]*',tag)
 rel=Path(method)/'Public433'/('lab%03d'%n)/'fold2'/tag
 return BASE/rel,STORE/rel/'final.pth'
def budget(n):return Budget('Public433',n,n,346-n,cycles=11)
def clock_budget(n):return Budget('Public433',n,n,346-n,cycles=36)
def endpoint(c):return Endpoint(c['method'],c['primary_pairs'],c['primary_pairs'],1,tuple(c['splits']['test']))
def rows(c):
 base=Path(c['target']).parent
 return [dict(path=c['target'],bound=32*1024**2)]+[
  dict(path=str(base/x/'final.pth'),bound=128*1024**2) for x in ('recovery_a','recovery_b','qualification_resume')]+[
  dict(path=str(base/'ledger_stdout_capacity/final.pth'),bound=64*1024**2)]
def resource(c):
 g=resource_backend(Path(c['target']));old=g.targets
 def all_targets():
  targets=old();seen={str(p):b for p,b in targets}
  # Include ALL existing CPAC auxiliary growth while CPAC remains on human hold.
  ap=SHARED/'cpac_auxiliary_targets_20261022.json'
  for r in (read(ap) if ap.exists() else []):
   p=Path(r['path'])
   assert SHARED/'cpac_paper_20261022' in p.parents and p.name in ('steps.jsonl','epochs.jsonl','roundtrip.pth')
   assert type(r['bound']) is int and 0<r['bound']<=10*1024**3
   assert str(p) not in seen
   targets.append((p,r['bound']));seen[str(p)]=r['bound']
  for r in rows(c):
   if r['path'] in seen:assert seen[r['path']]==r['bound']
   else:targets.append((Path(r['path']),r['bound']))
  return targets
 g.targets=all_targets
 return g
def capacity(c,start=False):
 g=resource(c)
 with g.original_lock():
  cap=g.capacity_snapshot(RAM)
  if start:assert all(r['margin_bytes']>=2*1024**3 for r in cap['filesystems']), 'Public launch requires 2GiB headroom after all growth reservations'
 return cap
def enroll(c):
 g=resource(c)
 with g.original_lock():
  cap=g.capacity_snapshot(RAM)
  assert all(r['margin_bytes']>=2*1024**3 for r in cap['filesystems']), 'Insufficient projected launch headroom'
  registered=read(REGISTRY)
  for r in rows(c):
   matched=[v for v in registered if v['path']==r['path']]
   if matched:assert matched==[r]
   else:registered.append(r)
  atomic(REGISTRY,registered)
 return cap
def verify(c):
 assert c['dataset']=='Public433' and c['gpu']==1 and c['fold']==2 and c['seed']==2027
 assert c['labeled_cases']==c['ratio'] in (5,10) and c['labels_are_case_counts_not_percent']
 assert c['splits']==split(c['ratio']) and c['case_files']==bind_case_files('Public433',c['splits'])
 assert c['primary_pairs']==budget(c['ratio']).primary_pairs
 assert str(paths(c['method'],c['ratio'],c['tag'])[1])==c['target']
 for r in c['sources']+c['inputs']:
  p=Path(r['path']);assert not p.is_symlink() and sha(p)==r['sha256'], str(p)
 assert sha(ROOT/'ckpt/sam_med3d_turbo.pth')==c['sam_sha256']
 assert c['recovery_interval']==100 and c['recovery_bound']==128*1024**2 and c['test_evaluations']==1
 assert c['budget_protocol']=='PUBLIC_PRIMARY_INPUT_11_CYCLE_CUTOFF_20261025'
 assert c['schedule_policy']=='retain_original_36_cycle_lr_ramp_clocks; cutoff_only'
 from budget_lineage import verify_lineage
 verify_lineage(c)
def gpu_idle():
 import subprocess
 r=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid,pci.bus_id','--format=csv,noheader'],text=True).strip()
 assert r.split(',')[0].strip()==UUID and r.split(',')[1].strip().endswith('B3:00.0'),r
 p=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 assert not any(x.split(',')[0].strip()==UUID for x in p.splitlines() if x.strip()),'GPU1 busy'
def lease_open():
 assert not LEASE.is_symlink() and LEASE.read_bytes()==b'NATIVE_MAIN_TABLE_R4_GPU1_EXCLUSIVE_OPERATION_V1\n'
 fd=os.open(LEASE,os.O_RDONLY|os.O_NOFOLLOW)
 assert (os.fstat(fd).st_dev,os.fstat(fd).st_ino)==(LEASE.stat().st_dev,LEASE.stat().st_ino)
 return fd
@contextlib.contextmanager
def leased(c):
 import fcntl
 fd=None
 try:
  g=resource(c)
  with g.original_lock():
   fd=lease_open();fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
   g.capacity_snapshot(RAM);gpu_idle()
  yield fd
 finally:
  if fd is not None:os.close(fd)
def file_pin(p):
 p=Path(p)
 if not p.exists():return dict(bytes=0,sha256=hashlib.sha256(b'').hexdigest())
 assert p.is_file() and not p.is_symlink() and p.stat().st_nlink==1
 return dict(bytes=p.stat().st_size,sha256=sha(p))
def ledger(c):return file_pin(Path(c['output'])/'steps.jsonl')
def checkpoints(c):
 d=Path(c['output']).parent/'recovery_commits';out=[]
 for p in sorted(d.glob('commit_*.json'),reverse=True):
  r=read(p);assert r['contract_sha256']==sha(Path(c['output']).parent/'contract.json')
  assert r['path'] in [x['path'] for x in rows(c)[1:3]]
  if file_pin(r['path'])==dict(bytes=r['bytes'],sha256=r['sha256']):out.append(r)
 return out
def save_checkpoint(c,payload):
 import torch
 d=Path(c['output']).parent/'recovery_commits';d.mkdir(exist_ok=True)
 gen=max([int(p.stem.split('_')[1]) for p in d.glob('commit_*.json')],default=-1)+1
 p=Path(rows(c)[1+gen%2]['path']);t=Path(str(p)+'.partial');g=resource(c)
 with g.original_lock():
  g.capacity_snapshot(RAM)
  assert not any(x.is_symlink() for x in [p,t,*p.parents]) and not t.exists()
  p.parent.mkdir(parents=True,exist_ok=True)
  with t.open('xb') as f:torch.save(payload,BoundedWriter(f,c['recovery_bound']));f.flush();os.fsync(f.fileno())
  digest=sha(t);size=t.stat().st_size
  os.replace(t,p);syncdir(p.parent)
  r=dict(generation=gen,iteration=payload['position'],path=str(p),bytes=size,sha256=digest,
   contract_sha256=payload['contract_sha256'],ledger=payload['ledger'],at=time.time())
  exclusive_json(d/('commit_%06d.json'%gen),r);syncdir(d)
 return r
