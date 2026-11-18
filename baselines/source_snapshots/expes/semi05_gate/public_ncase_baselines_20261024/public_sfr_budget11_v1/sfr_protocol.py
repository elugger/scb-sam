"""Pinned SFR Public case-budget protocol and bounded two-ledger storage."""
import contextlib,hashlib,json,os,random,sys,time
from pathlib import Path
from sfr_state import plan,initial,accept,validate
from sfr_controller import exclusive as exclusive_json
ROOT=Path('__RECORDED_PROJECT_ROOT__');HERE=Path(__file__).resolve().parent
OLD=ROOT/'expes/semi05_gate/main_table_20261019_r4'
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
 rel=Path('sfr')/'Public433'/('lab%03d'%n)/'fold2'/tag
 return BASE/rel,STORE/rel/'final.pth'
def job(c):return Path(c['output']).parent
def contract_sha(c):return sha(job(c)/'contract.json')
def endpoint(c):return Endpoint('sfr',c['primary_pairs'],c['primary_pairs'],1,tuple(c['splits']['test']))
def rows(c):
 base=Path(c['target']).parent
 return [dict(path=c['target'],bound=64*1024**2)]+[dict(path=str(base/n/'final.pth'),bound=768*1024**2) for n in ('recovery_a','recovery_b','qualification_prefix','qualification_ssl')]+[dict(path=str(base/'ledger_stdout_capacity/final.pth'),bound=128*1024**2)]+[dict(path=str(base/n/'final.pth'),bound=64*1024**2) for n in ('sam_prefix','engineering_sam_prefix')]
def resource(c):
 g=resource_backend(Path(c['target']),64*1024**2);old=g.targets
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
  if start:assert all(x['margin_bytes']>=2*1024**3 for x in cap['filesystems']),'SFR launch requires 2GiB after all growth'
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
 assert c['method']=='sfr' and c['dataset']=='Public433' and c['gpu']==1 and c['fold']==2 and c['seed']==2027
 assert c['labeled_cases']==c['ratio'] in (5,10) and c['labels_are_case_counts_not_percent']
 assert c['stage_recipe']==plan(c['ratio']) and c['primary_pairs']==plan(c['ratio'])['primary_inputs']
 assert c['splits']==split(c['ratio']) and c['case_files']==bind_case_files('Public433',c['splits'])
 assert paths(c['ratio'],c['tag'])==(job(c),Path(c['target']))
 assert c['recovery_interval']==100 and c['recovery_bound']==768*1024**2 and c['checkpoint_bound']==64*1024**2
 assert c['test_evaluations']==1 and c['selection']=='terminal_after_fixed_exposure' and c['early_stop'] is False
 assert c['budget_protocol']=='PUBLIC_PRIMARY_INPUT_11_CYCLE_CUTOFF_20261025'
 assert c['schedule_policy']=='retain_original_36_cycle_lr_ramp_clocks; cutoff_only'
 assert c['engineering_probe_prefix_iterations']==4
 assert c['sfr_targets']=={'prefix':dict(path=rows(c)[6]['path'],bound=64*1024**2),'final':dict(path=c['target'],bound=64*1024**2)}
 for r in c['sources']+c['inputs']:
  p=Path(r['path']);assert not any(x.is_symlink() for x in [p,*p.parents]) and sha(p)==r['sha256'],str(p)
 bound={r['path'] for r in c['sources']}
 assert {str(p) for p in HERE.glob('*.py')}.issubset(bound)
 origin=read(HERE/'template.json')['origin_contract'];assert sha(origin['path'])==origin['sha256'];old=read(origin['path'])
 assert old['method']=='sfr' and all(r in c['sources'] for r in old['sources'])
 assert c['official_code_root']==old['official_code_root'] and c['official_commit']==old['official_commit']
 assert c['foundation']==old['foundation']
 init=c['foundation'];p=Path(init['path'])
 assert init['family']=='original SAM 2D ViT-B' and init['bytes']==375042383 and p.stat().st_size==init['bytes']
 assert not any(x.is_symlink() for x in [p,*p.parents]) and sha(p)==init['sha256']=='ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912'
 from sfr_runtime_components import verify_native_sources
 verify_native_sources(c['official_code_root'])
 assert {str(p) for p in Path(c['official_code_root']).rglob('*.py')}==set(old['official_source_paths'])
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
 from sfr_stage_protocol import finetune_batches
 snaps=snapshot_ledgers(c);progress=initial();recipe=plan(c['ratio']);pairs=schedule(clock_budget(c['ratio']),c['splits']);case_ids=set()
 prefixes=list(finetune_batches(c['splits']['train_lab']))
 for phase,name in [('prefix','pretrain_steps.jsonl'),('ssl','steps.jsonl')]:
  raw=snaps[name]['raw']
  if phase=='ssl' and raw:
   assert progress['prefix']['iterations']==recipe['prefix_iterations'];progress['phase']='ssl'
  for line in raw.splitlines():
   row=json.loads(line);step=row['step'];idx=progress[phase]['iterations']
   if phase=='prefix':
    expected=list(prefixes[idx]);assert step['actual_labeled_case_ids']==expected;case_ids.update(expected)
   else:
    expected=pairs[idx*2:idx*2+2];assert step['actual_pairs']==expected and step['completed_primary_pairs']==2*(idx+1)
    for pair in expected:case_ids.update([pair['labeled'],pair['unlabeled']])
   progress=accept(progress,step,recipe);assert row['cumulative']==progress[phase]
 # An empty SSL ledger may follow the committed fresh-seed stage transition.
 if not snaps['steps.jsonl']['read_bytes'] and (Path(c['output'])/'pretrain_terminal.json').exists():
  assert progress['prefix']['iterations']==recipe['prefix_iterations'];progress['phase']='ssl'
 return dict(progress=progress,actual_training_cases=len(case_ids),ledgers={n:{k:v for k,v in d.items() if k!='raw'} for n,d in snaps.items()})
def verify_actual_ledgers(c,pre_rows,ssl_rows):
 review=inspect_ledgers(c);assert review['progress']['prefix']['iterations']==len(pre_rows)
 actual=[p for r in ssl_rows for p in r['actual_pairs']]
 assert review['progress']['ssl']['labeled_inputs']==len(actual)==c['primary_pairs']
 assert actual==schedule(clock_budget(c['ratio']),c['splits'])[:c['primary_pairs']]
 for name,rows_native in [('pretrain_steps.jsonl',pre_rows),('steps.jsonl',ssl_rows)]:
  recorded=[json.loads(x)['step'] for x in (Path(c['output'])/name).read_text().splitlines()]
  assert len(recorded)==len(rows_native)
  for outer,inner in zip(recorded,rows_native):assert all(outer[k]==v for k,v in inner.items())
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
def save_checkpoint(c,state,torch,execution_role):
 assert execution_role==state['execution_role']=='formal' and state['recipe']==c['stage_recipe'] and state['contract_sha256']==contract_sha(c)
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
def save_terminal(c,payload,torch):
 from recovery_state import tree_hash
 assert payload['execution_role']=='formal' and payload['contract_sha256']==contract_sha(c)
 p=Path(c['target']);g=resource(c)
 with g.original_lock():
  g.capacity_snapshot(RAM)
  if p.exists():
   guard_test_lineage(c);old=torch.load(p,map_location='cpu');assert tree_hash(old)==tree_hash(payload)
   return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,resource_lock_verified=True,reused_fixed_terminal=True)
 return write_tensor(c,p,payload,c['checkpoint_bound'],torch,False)

def configure_native(native,c):
 import budget_protocol,sfr_stage_protocol
 budgets=tuple(clock_budget(n) for n in (5,10))
 budget_protocol.BUDGETS=budgets;sfr_stage_protocol.BUDGETS=budgets;native.BUDGETS=budgets
 native.bound_splits=bound_splits

def capacity_locked(c):return resource(c).capacity_snapshot(RAM)

def assert_worker_lease(c):
 import fcntl
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==UUID
 expected=LEASE.stat();matches=[]
 for p in Path('/proc/self/fd').iterdir():
  try:s=p.stat()
  except FileNotFoundError:continue
  if (s.st_dev,s.st_ino)==(expected.st_dev,expected.st_ino):matches.append(p)
 assert matches
 comp=lease_open()
 try:
  try:fcntl.flock(comp,fcntl.LOCK_EX|fcntl.LOCK_NB);raise AssertionError('GPU1 lease not held')
  except BlockingIOError:pass
 finally:os.close(comp)

def prefix_path(c,execution_role):
 assert execution_role in ('formal','engineering_two_phase_probe')
 return Path(rows(c)[6 if execution_role=='formal' else 7]['path'])

def prefix_identity(c,payload,execution_role):
 from sfr_state import prefix_inputs
 assert payload['execution_role']==execution_role and payload['contract_sha256']==contract_sha(c)
 assert payload['method']=='sfr' and payload['stage']=='sam_finetuning_terminal'
 n=6000 if execution_role=='formal' else 4
 assert payload['successful_optimizer_updates']==payload['plan']['prefix_iterations']==n
 assert payload['actual_labeled_inputs']==payload['plan']['prefix_labeled_inputs']==prefix_inputs(n,c['labeled_cases'])
 assert payload['plan']['ssl_iterations']==c['stage_recipe']['original_clock_ssl_iterations']
 assert payload['foundation']['sha256']==c['foundation']['sha256']

def persist_prefix_and_reload(c,payload,torch,execution_role):
 from recovery_state import tree_hash
 prefix_identity(c,payload,execution_role);p=prefix_path(c,execution_role);g=resource(c)
 # A stopped pre-transition update may have already committed this exact prefix.
 with g.original_lock():
  g.capacity_snapshot(RAM)
  if p.exists():
   assert not p.is_symlink() and p.stat().st_nlink==1
   old=torch.load(p,map_location='cpu');assert tree_hash(old)==tree_hash(payload)
   artifact=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,resource_lock_verified=True)
  else:artifact=None
 if artifact is None:artifact=write_tensor(c,p,payload,64*1024**2,torch,False)
 artifact={k:artifact[k] for k in ('path','sha256','bytes','resource_lock_verified')}
 return artifact,load_bound_prefix(c,artifact,torch,execution_role)

def load_bound_prefix(c,artifact,torch,execution_role):
 p=prefix_path(c,execution_role);g=resource(c)
 with g.original_lock():
  g.capacity_snapshot(RAM)
  assert artifact['path']==str(p) and not any(x.is_symlink() for x in [p,*p.parents])
  assert p.stat().st_nlink==1 and file_pin(p)=={k:artifact[k] for k in ('bytes','sha256')}
  loaded=torch.load(p,map_location='cpu');prefix_identity(c,loaded,execution_role)
  assert file_pin(p)=={k:artifact[k] for k in ('bytes','sha256')}
 return loaded

def reload_artifact(c,role,artifact,torch,verifier):
 assert role=='final';p=Path(c['target']);g=resource(c)
 with g.original_lock():
  g.capacity_snapshot(RAM)
  assert artifact['path']==str(p) and not any(x.is_symlink() for x in [p,*p.parents])
  assert p.stat().st_nlink==1 and file_pin(p)=={k:artifact[k] for k in ('bytes','sha256')}
  loaded=torch.load(p,map_location='cpu')
  assert loaded['execution_role']=='formal' and loaded['contract_sha256']==contract_sha(c)
  assert loaded['cutoff_recipe']==c['stage_recipe']
  verifier(loaded);assert file_pin(p)=={k:artifact[k] for k in ('bytes','sha256')}
 return loaded
