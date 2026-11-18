"""New SFR storage/phase-artifact tests using pickle scalars and a mock gate."""
import argparse,contextlib,json,os,pickle,sys,tempfile,types
from pathlib import Path
import sfr_protocol as p
from resumable_sfr import make_backend
from sfr_state import initial,plan,prefix_inputs

class FakeTorch:
 @staticmethod
 def save(v,f):f.write(pickle.dumps(v))
 @staticmethod
 def load(path,map_location=None):return pickle.loads(Path(path).read_bytes())
class InterruptedTorch:
 @staticmethod
 def save(v,f):f.write(b'partial');raise RuntimeError('synthetic interrupted write')
class Guard:
 @contextlib.contextmanager
 def original_lock(self):yield
 def capacity_snapshot(self,ram):return dict(synthetic_gate_only=True)

def main():
 a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);args=a.parse_args()
 root=Path(tempfile.mkdtemp(prefix='sfr_storage_cpu_',dir=p.HERE.parent));assert root.resolve().parent==p.HERE.parent.resolve()
 c=dict(target=str(root/'artifact/final.pth'),output=str(root/'formal'),stage_recipe=plan(5),recovery_bound=768*1024**2,checkpoint_bound=64*1024**2,labeled_cases=5,ratio=5,foundation=dict(sha256='synthetic-foundation'))
 p.exclusive_json(root/'contract.json',dict(synthetic_test_only=True));original_resource=p.resource;p.resource=lambda c:Guard()
 passed=[]
 def good(name):passed.append(name)
 def rejects(name,fn):
  try:fn()
  except (AssertionError,RuntimeError,ValueError):good(name)
  else:raise AssertionError('Not rejected: '+name)
 path=root/'bounded/final.pth';p.write_tensor(c,path,{'x':1},128,FakeTorch,False);before=path.read_bytes()
 rejects('partial_write_preserves_old_committed_file',lambda:p.write_tensor(c,path,{'x':2},128,InterruptedTorch,True))
 assert path.read_bytes()==before and Path(str(path)+'.partial').read_bytes()==b'partial'
 Path(str(path)+'.partial').rename(root/'preserved_write_partial')
 rejects('bounded_write_refuses_oversize',lambda:p.write_tensor(c,path,{'x':'z'*300},128,FakeTorch,True))
 assert path.read_bytes()==before;Path(str(path)+'.partial').rename(root/'preserved_bound_partial')
 result=p.write_tensor(c,path,{'x':2},128,FakeTorch,True);assert p.sha(path)==result['sha256'];good('atomic_replace_SHA_commit')
 rejects('no_replace_preserves_existing',lambda:p.write_tensor(c,path,{},128,FakeTorch,False))
 # Storage tests use only fake scalars. Do not import torch via tree_hash.
 fake_recovery=types.ModuleType('recovery_state');fake_recovery.tree_hash=lambda v:repr(v)
 prior=sys.modules.get('recovery_state');sys.modules['recovery_state']=fake_recovery
 try:
  ch=p.contract_sha(c)
  def payload(role):
   n=6000 if role=='formal' else 4
   return dict(execution_role=role,contract_sha256=ch,method='sfr',stage='sam_finetuning_terminal',successful_optimizer_updates=n,
     actual_labeled_inputs=prefix_inputs(n,5),plan=dict(prefix_iterations=n,prefix_labeled_inputs=prefix_inputs(n,5),ssl_iterations=6210),
     foundation=dict(sha256='synthetic-foundation'),synthetic_state={'x':1})
  pp=payload('formal');ar,loaded=p.persist_prefix_and_reload(c,pp,FakeTorch,'formal');assert loaded==pp
  before=Path(ar['path']).read_bytes();ar2,loaded2=p.persist_prefix_and_reload(c,pp,FakeTorch,'formal')
  assert ar==ar2 and loaded2==pp and Path(ar['path']).read_bytes()==before;good('prefix_commit_crash_window_idempotent_exact_state')
  rejects('different_prefix_state_never_overwrites',lambda:p.persist_prefix_and_reload(c,dict(pp,synthetic_state={'x':2}),FakeTorch,'formal'))
  rejects('engineering_prefix_not_formal',lambda:p.persist_prefix_and_reload(c,payload('engineering_two_phase_probe'),FakeTorch,'formal'))
  ep=payload('engineering_two_phase_probe');ea,_=p.persist_prefix_and_reload(c,ep,FakeTorch,'engineering_two_phase_probe')
  rejects('engineering_artifact_path_not_formal',lambda:p.load_bound_prefix(c,ea,FakeTorch,'formal'))
  bad=dict(pp,plan=dict(pp['plan'],ssl_iterations=1898))
  rejects('compressed_clock_rejected_at_prefix_boundary',lambda:p.prefix_identity(c,bad,'formal'))
  terminal=dict(execution_role='formal',contract_sha256=ch,cutoff_recipe=plan(5),synthetic_state={'x':1})
  ta=p.save_terminal(c,terminal,FakeTorch);assert p.save_terminal(c,terminal,FakeTorch)['sha256']==ta['sha256'];good('terminal_existing_exact_payload_is_reused')
  Path(c['output']).mkdir();p.exclusive_json(Path(c['output'])/'final_test_intent.json',{'synthetic_test_opened':True})
  rejects('terminal_after_test_intent_is_never_replayed',lambda:p.save_terminal(c,terminal,FakeTorch))
 finally:
  if prior is None:sys.modules.pop('recovery_state',None)
  else:sys.modules['recovery_state']=prior
 # Check alternating committed generation identity without any model data.
 for gen in (0,1):
  loc=Path(p.rows(c)[1+gen]['path']);r=p.write_tensor(c,loc,{'gen':gen},128,FakeTorch,False)
  d=root/'recovery_commits';d.mkdir(exist_ok=True)
  p.exclusive_json(d/('commit_%06d.json'%gen),dict(generation=gen,contract_sha256=p.contract_sha(c),progress=initial(),path=r['path'],bytes=r['bytes'],sha256=r['sha256'],ledgers={}))
 assert len(p.checkpoints(c))==2
 Path(p.rows(c)[2]['path']).write_bytes(b'bad newest state')
 assert [x['generation'] for x in p.checkpoints(c)]==[0];good('older_SHA_verified_generation_survives_corrupt_newest')
 p.resource=original_resource
 proto=types.SimpleNamespace();B=make_backend(types.SimpleNamespace(SFRBackend=object),proto);b=object.__new__(B)
 b.c={};b.allow_wait=True;b.progress=initial();proto.resource_status=lambda c,v:None
 proto.capacity=lambda c:(_ for _ in ()).throw(ValueError('unknown capacity backend error'))
 rejects('unknown_resource_error_not_retried',lambda:b.gate())
 import resumable_sfr
 sleep=resumable_sfr.time.sleep;resumable_sfr.time.sleep=lambda n:None
 try:
  calls=[];events=[]
  def cap(c):
   calls.append(1)
   if len(calls)==1:raise ValueError('Insufficient filesystem capacity: synthetic')
   return {'PASS':True}
  proto.capacity=cap;proto.resource_status=lambda c,v:events.append(v['state'])
  assert b.gate()=={'PASS':True} and events==['WAIT_CAPACITY','PASS'];good('precise_capacity_gate_waits_before_updates')
  writes=[]
  def io():
   writes.append(1)
   if len(writes)==1:raise ValueError('Insufficient RAM including cgroup ancestors: synthetic')
   return 'saved'
  assert b.io_with_wait(io)=='saved' and len(writes)==2;good('IO_capacity_race_reuses_state_without_training_update')
  rejects('unknown_partial_write_escapes_IO_retry',lambda:b.io_with_wait(lambda:(_ for _ in ()).throw(RuntimeError('unknown partial write'))))
 finally:resumable_sfr.time.sleep=sleep
 for f in p.HERE.glob('*.py'):compile(f.read_bytes(),str(f),'exec')
 assert 'torch' not in sys.modules;good('zero_torch_CUDA_models_patient_inputs')
 result=dict(passed=True,status='SFR_NEW_STORAGE_PHASE_CPU_PASS_NOT_GPU_QUALIFIED',test_count=len(passed),checks=passed,
  new_optimization_updates=0,new_test_evaluations=0,new_model_loads=0,new_cuda_loads=0,new_patient_inputs=0,
  source_files={str(f):p.sha(f) for f in p.HERE.glob('*.py')},synthetic_files=str(root),
  caveat='Tiny pickle scalar payloads and a mock resource gate; not proof of real model state, GPU continuation, actual capacity or launch readiness.')
 p.exclusive_json(args.out,result);print(json.dumps(dict(passed=True,test_count=len(passed))))
if __name__=='__main__':main()
