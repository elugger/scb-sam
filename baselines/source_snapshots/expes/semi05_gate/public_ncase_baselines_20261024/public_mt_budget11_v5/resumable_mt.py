"""Full-state recovery wrapper; original MT/UA kernels and data transforms unchanged."""
import copy,hashlib,json,os,random,time
from pathlib import Path
from public_protocol import *
import mt_runtime as native
import numpy as np
import torch

def cpu(v):
 if torch.is_tensor(v):return v.detach().cpu().clone()
 if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
 if isinstance(v,list):return [cpu(x) for x in v]
 if isinstance(v,tuple):return tuple(cpu(x) for x in v)
 return copy.deepcopy(v)
def tree_hash(v):
 h=hashlib.sha256()
 def visit(x):
  if torch.is_tensor(x):h.update(b'tensor'+str(x.dtype).encode()+str(tuple(x.shape)).encode()+x.detach().cpu().contiguous().numpy().tobytes())
  elif isinstance(x,np.ndarray):h.update(b'array'+str(x.dtype).encode()+str(x.shape).encode()+x.tobytes())
  elif isinstance(x,dict):
   h.update(b'dict')
   for k in sorted(x,key=repr):visit(k);visit(x[k])
  elif isinstance(x,(tuple,list)):
   h.update(type(x).__name__.encode())
   for y in x:visit(y)
  else:h.update(type(x).__name__.encode()+repr(x).encode()+b'\0')
 visit(v);return h.hexdigest()
def rng():return cpu((random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all()))
def restore_rng(r):
 random.setstate(r[0]);np.random.set_state(r[1]);torch.set_rng_state(r[2]);torch.cuda.set_rng_state_all(r[3])

class Backend(native.SemiSAMBackend):
 def __init__(self,c,out):
  native.BUDGETS=(clock_budget(5),clock_budget(10))
  native.bound_splits=lambda dataset,n:split(n) if dataset=='Public433' else (_ for _ in ()).throw(ValueError('Public only'))
  native.resource_backend=lambda target,*args:resource(c)
  super().__init__(c,out,verify)
  self.c=c;self.allow_wait=False
 def gate(self):
  while True:
   try:
    cap=capacity(self.c)
    if self.allow_wait:atomic(Path(self.c['output']).parent/'resource_status.json',dict(state='PASS',at=time.time(),iteration=self.position))
    return cap
   except (ValueError,RuntimeError,AssertionError) as e:
    if not self.allow_wait or not str(e).startswith(('Insufficient filesystem capacity:','Insufficient RAM including cgroup ancestors')):raise
    atomic(Path(self.c['output']).parent/'resource_status.json',dict(state='WAIT_CAPACITY',at=time.time(),iteration=self.position,error=str(e)))
    time.sleep(60)
 def prepare(self,e):
  self.gate();p=super().prepare(e)
  self.frozen_signature=native.signature(torch,self.t.generalist.sam)
  from budget_lineage import install_patch
  install_patch(self)
  return p
 def train_iteration(self,remaining):
  self.gate()
  return super().train_iteration(remaining)
 def capture(self):
  assert self.position==self.t.global_step==self.t.optimizer.calls
  assert len(self.actual_rows)==self.position and self.actual_rows==self.pairs[:self.position]
  assert native.signature(torch,self.t.generalist.sam)==self.frozen_signature
  return dict(format='PUBLIC_MT_EXACT_RECOVERY_V1',contract_sha256=sha(Path(self.c['output']).parent/'contract.json'),
   student=cpu(self.t.student.state_dict()),teacher=cpu(self.t.teacher.state_dict()),optimizer=cpu(self.t.optimizer.inner.state_dict()),
   rng=rng(),position=self.position,global_step=self.t.global_step,optimizer_calls=self.t.optimizer.calls,
   optimizer_norm=self.t.optimizer.norm,actual_rows=cpu(self.actual_rows),cycle_rows=cpu(self.cycle_rows),
   initial_signature=self.initial_signature,frozen_signature=self.frozen_signature)
 def restore(self,s):
  assert s['format']=='PUBLIC_MT_EXACT_RECOVERY_V1' and s['contract_sha256']==sha(Path(self.c['output']).parent/'contract.json')
  assert self.initial_signature==s['initial_signature'] and self.frozen_signature==s['frozen_signature']
  assert 0<=s['position']<=self.c['primary_pairs'] and s['actual_rows']==self.pairs[:s['position']]
  self.t.student.load_state_dict(s['student'],strict=True);self.t.teacher.load_state_dict(s['teacher'],strict=True)
  self.t.optimizer.inner.load_state_dict(s['optimizer'])
  self.position=s['position'];self.t.global_step=s['global_step'];self.t.optimizer.calls=s['optimizer_calls'];self.t.optimizer.norm=s['optimizer_norm']
  self.actual_rows=cpu(s['actual_rows']);self.cycle_rows=cpu(s['cycle_rows'])
  for m in (self.t.student,self.t.teacher,self.t.generalist.sam):
   for p in m.parameters():p.grad=None
  restore_rng(s['rng'])
  assert tree_hash(self.capture())==tree_hash({k:v for k,v in s.items() if k not in ('counts','ledger')})
 def checkpoint(self,counts):
  assert counts==dict(iterations=self.position,successful_optimizer_updates=self.position,labeled_inputs=self.position,unlabeled_inputs=self.position)
  self.gate();s=self.capture();s.update(counts=counts,ledger=ledger(self.c))
  while True:
   try:return save_checkpoint(self.c,s)
   except (ValueError,RuntimeError,AssertionError) as e:
    if not str(e).startswith(('Insufficient filesystem capacity:','Insufficient RAM including cgroup ancestors')):raise
    self.gate()
 def verify_training_coverage(self):
  assert self.position==self.c['primary_pairs']==budget(self.c['ratio']).primary_pairs
  assert len(self.cycle_rows)==11 and self.actual_rows==self.pairs[:self.position]
  assert all(x['actual_case_coverage_verified'] for x in self.cycle_rows)
  return dict(actual_case_coverage_verified=True,cycles=self.cycle_rows,primary_labeled_inputs=self.position,primary_unlabeled_inputs=self.position)
 def guard_test_lineage(self):
  from budget_lineage import guard_test_lineage
  guard_test_lineage(self.c)
 def prepare_resume(self,e):
  p=self.prepare(e);records=checkpoints(self.c)
  if records:
   r=records[0];s=torch.load(r['path'],map_location='cpu')
  else:
   from budget_lineage import load_state
   r=self.c['budget_migration']['checkpoint']
   s=load_state(self.c,sha(Path(self.c['output']).parent/'contract.json'),torch.load,tree_hash)
  assert ledger(self.c)==r['ledger'],'Unreconciled ledger'
  self.restore(s);assert s['ledger']==r['ledger']
  exclusive_json(Path(self.c['output']).parent/('restoration_verified_%d.json'%os.getpid()),dict(contract_sha256=sha(Path(self.c['output']).parent/'contract.json'),checkpoint_sha256=r['sha256'],loaded_counts=s['counts'],loaded_state_exact=True,loaded_state_sha256=tree_hash(self.capture()),new_optimizer_updates=0,new_test_evaluations=0,at=time.time()))
  return dict(p,counts=s['counts'],recovery=r)
 def save_terminal(self,counts):
  assert native.signature(torch,self.t.student)!=self.initial_signature
  self.final_signature=native.signature(torch,self.t.student)
  p=Path(self.c['target']);tmp=Path(str(p)+'.partial')
  if p.exists():
   # A committed terminal can only be recovered before any final test intent.
   assert not (Path(self.c['output'])/'final_test_intent.json').exists()
   state=torch.load(p,map_location='cpu')
   assert state['counts']==counts and state['student_signature']==self.final_signature
   assert tree_hash(state['student'])==tree_hash(cpu(self.t.student.state_dict()))
   return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,resource_lock_verified=True,reused_fixed_terminal=True)
  while True:
   self.gate();g=resource(self.c)
   try:
    with g.original_lock():
     cap=g.capacity_snapshot(RAM)
     assert not p.exists() and not tmp.exists() and not any(x.is_symlink() for x in [p,tmp,*p.parents])
     p.parent.mkdir(parents=True,exist_ok=True)
     with tmp.open('xb') as f:
      torch.save(dict(student=self.t.student.state_dict(),student_signature=self.final_signature,counts=counts,
       contract_sha256=sha(Path(self.c['output']).parent/'contract.json'),selection='fixed_exposure_terminal'),BoundedWriter(f,32*1024**2));f.flush();os.fsync(f.fileno())
     digest=sha(tmp);size=tmp.stat().st_size
     os.link(tmp,p);assert p.stat().st_ino==tmp.stat().st_ino and p.stat().st_nlink==2
     tmp.unlink();syncdir(p.parent)
    return dict(path=str(p),sha256=digest,bytes=size,resource_lock_verified=True,capacity_before=cap)
   except (ValueError,RuntimeError,AssertionError) as e:
    if not str(e).startswith(('Insufficient filesystem capacity:','Insufficient RAM including cgroup ancestors')):raise
