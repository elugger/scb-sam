"""New storage and execution-role checks with tiny fake payloads, no torch/CUDA."""
import argparse,ast,contextlib,copy,hashlib,json,os,pickle,sys,tempfile,types
from pathlib import Path
import sdcl_protocol as p
from resumable_sdcl import make_backend
from stage_state import initial,plan

class FakeTorch:
 @staticmethod
 def save(v,f):f.write(pickle.dumps(v))
class InterruptedTorch:
 @staticmethod
 def save(v,f):f.write(b'partial');raise RuntimeError('simulated write interruption')
class Guard:
 @contextlib.contextmanager
 def original_lock(self):yield
 def capacity_snapshot(self,ram):return dict(synthetic_unit_test=True)

def main():
 a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);args=a.parse_args()
 root=Path(tempfile.mkdtemp(prefix='sdcl_storage_cpu_',dir=p.HERE.parent));assert root.resolve().parent==p.HERE.parent.resolve()
 c=dict(target=str(root/'artifact/final.pth'),output=str(root/'formal'),stage_recipe=plan(5),recovery_bound=768*1024**2)
 p.exclusive_json(root/'contract.json',dict(test='synthetic_only'))
 original_resource=p.resource;p.resource=lambda c:Guard();passed=[]
 def good(n):passed.append(n)
 def rejected(n,fn):
  try:fn()
  except (AssertionError,RuntimeError,ValueError):good(n)
  else:raise AssertionError('Not rejected: '+n)
 target=root/'bounded/final.pth';old=p.write_tensor(c,target,{'v':1},128,FakeTorch,False);before=target.read_bytes()
 rejected('interrupted_replace_preserves_committed_bytes',lambda:p.write_tensor(c,target,{'v':2},128,InterruptedTorch,True))
 assert target.read_bytes()==before and Path(str(target)+'.partial').read_bytes()==b'partial'
 # Own synthetic partial is preserved under another name, never discarded.
 part=Path(str(target)+'.partial');part.rename(root/'preserved_synthetic_write_partial')
 rejected('bounded_write_refuses_growth',lambda:p.write_tensor(c,target,{'v':'x'*200},128,FakeTorch,True));assert target.read_bytes()==before
 Path(str(target)+'.partial').rename(root/'preserved_synthetic_bound_partial')
 new=p.write_tensor(c,target,{'v':2},128,FakeTorch,True);assert target.read_bytes()!=before and p.sha(target)==new['sha256'];good('atomic_replace_and_SHA')
 rejected('no_replace_refuses_existing_final',lambda:p.write_tensor(c,target,{'v':3},128,FakeTorch,False))
 # Both latest and older committed recovery generations exist; corrupt latest.
 d=root/'recovery_commits';d.mkdir()
 for gen in (0,1):
  path=Path(p.rows(c)[1+gen]['path']);r=p.write_tensor(c,path,{'gen':gen},128,FakeTorch,False)
  p.exclusive_json(d/('commit_%06d.json'%gen),dict(generation=gen,contract_sha256=p.contract_sha(c),progress=initial(),path=r['path'],bytes=r['bytes'],sha256=r['sha256'],ledgers={}))
 assert len(p.checkpoints(c))==2
 Path(p.rows(c)[2]['path']).write_bytes(b'corrupted-new-generation')
 assert [x['generation'] for x in p.checkpoints(c)]==[0];good('corrupt_new_generation_falls_back_to_old_SHA_verified_generation')
 # A complete fake class allows the actual restore role guard to run before
 # any tensors, models, optimizer objects, or RNG are referenced.
 proto=types.SimpleNamespace(contract_sha=lambda c:'pin')
 cls=make_backend(types.SimpleNamespace(SDCLBackend=object),proto);obj=object.__new__(cls)
 obj.c={};obj.execution_role='formal';obj.recipe=plan(5)
 state=dict(format='PUBLIC_SDCL_BOTH_PHASE_RECOVERY_V1',contract_sha256='pin',execution_role='engineering_two_phase_probe',recipe=plan(5))
 rejected('formal_restore_refuses_engineering_role',lambda:obj.restore(state))
 state['execution_role']='formal';state['recipe']=dict(plan(5),prefix_iterations=4)
 rejected('formal_restore_refuses_short_engineering_prefix',lambda:obj.restore(state))
 state['recipe']=plan(5);state['contract_sha256']='other'
 rejected('restore_refuses_wrong_contract',lambda:obj.restore(state))
 # Precise gate errors wait; unknown failures escape without sleeping/retrying.
 proto.capacity=lambda c:(_ for _ in ()).throw(ValueError('unknown injected error'))
 obj.allow_wait=True;obj.progress=initial()
 rejected('unknown_resource_error_is_not_swallowed',lambda:obj.gate())
 calls=[]
 def capacity(c):
  calls.append(1)
  if len(calls)==1:raise ValueError('Insufficient filesystem capacity: synthetic')
  return {'PASS':True}
 proto.capacity=capacity;events=[];proto.resource_status=lambda c,v:events.append(v['state'])
 import resumable_sdcl
 old_sleep=resumable_sdcl.time.sleep;resumable_sdcl.time.sleep=lambda n:None
 try:assert obj.gate()=={'PASS':True} and events==['WAIT_CAPACITY','PASS'] and len(calls)==2
 finally:resumable_sdcl.time.sleep=old_sleep
 good('known_capacity_error_waits_before_next_update')
 p.resource=original_resource
 # Source-scoped stdout whitelist proves exact constructor print only.
 source=p.ROOT/'SDCL-master/code/networks/ResVNet.py'
 from observe_sdcl import boot_check
 cc=dict(sources=[dict(path=str(source),sha256=p.sha(source))])
 boot_check(cc,b'new res\n',True);good('exact_source_AST_constructor_output')
 rejected('unknown_stdout_is_rejected',lambda:boot_check(cc,b'new res\nwarning\n',False))
 for f in p.HERE.glob('*.py'):compile(f.read_bytes(),str(f),'exec')
 assert 'torch' not in sys.modules;good('no_model_torch_or_CUDA_import')
 result=dict(passed=True,status='NEW_STORAGE_ROLE_INTEGRATION_CPU_PASS',test_count=len(passed),checks=passed,
  new_optimization_updates=0,new_test_evaluations=0,new_model_loads=0,new_cuda_loads=0,
  source_files={str(f):p.sha(f) for f in p.HERE.glob('*.py')},synthetic_files=str(root),
  caveat='Storage uses tiny fake payloads and a mock guard; actual all-growth gate and model recovery require independent real qualification/admission.')
 p.exclusive_json(args.out,result);print(json.dumps(dict(passed=True,test_count=len(passed),new_optimization_updates=0)))
if __name__=='__main__':main()
