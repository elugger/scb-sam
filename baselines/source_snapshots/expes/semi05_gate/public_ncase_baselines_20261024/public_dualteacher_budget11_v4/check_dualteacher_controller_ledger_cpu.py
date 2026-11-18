"""New controller/ledger checks: real small bindings, synthetic IO and receipts.
No native model/tensor/CUDA/optimizer/patient/test/job/reservation is executed.
"""
import argparse,copy,hashlib,json,sys,tempfile,time
from pathlib import Path
from dualteacher_binding import DualTeacherBinding
from dualteacher_budget import counts
from dualteacher_state import accept
from dualteacher_ledger import scan,project
from dualteacher_controller import FixedCutoffRun,check_metrics
from check_dualteacher_recovery_cpu import receipt
checks=[]
def yes(name,value):
 if not value:raise AssertionError(name)
 checks.append(name)
def reject(name,fn):
 try:fn()
 except (ValueError,KeyError,TypeError):checks.append(name);return
 raise AssertionError('Invalid state accepted: '+name)
def encode(v):return (json.dumps(v,sort_keys=True,allow_nan=False)+'\n').encode()
class MockBackend:
 def __init__(self,b,out):
  self.binding=b;self.recipe=b.recipe;self.pairs=b.pairs;self.output=out;self.execution_role='formal';self.progress=counts(0);self.checkpoints=[];self.saved=False;self.opened=0
 def guard_run_lineage(self,resume):
  if resume or (self.output/'run_intent.json').exists():raise ValueError('Wrong or repeated mock run')
 def record(self,name,v,consistent=False):
  p=self.output/name
  if p.exists():raise ValueError('Repeated mock record')
  p.write_bytes(encode(v))
 def prepare_initial(self):return dict(source_pins_verified=True,input_pins_verified=True,initialization_verified=True,resource_gate_passed=True,test_unopened=True)
 def checkpoint(self,c):
  assert c==self.progress;self.checkpoints.append(c['iterations'])
 def train_iteration(self,i):
  assert i==self.progress['iterations'];r=receipt(i,self.recipe,self.pairs);self.progress=accept(self.progress,r,self.recipe,'formal',self.binding.pair_batch(i),self.original_clock(i));return r
 def original_clock(self,i):return self.binding.original_clock(i)
 def append_step(self,r):
  with (self.output/'steps.jsonl').open('ab') as f:f.write(encode(r))
 def verify_training_coverage(self,c):
  v=scan((self.output/'steps.jsonl').read_bytes(),self.binding);assert v['counts']==c and v['complete_reference_cycles']==11
  return dict(actual_case_coverage_verified=True,complete_reference_cycles=11,primary_labeled_inputs=c['labeled_inputs'],primary_unlabeled_inputs=c['unlabeled_inputs'],successful_optimizer_updates=c['successful_optimizer_updates'])
 def save_terminal(self,c):
  assert c==self.progress;self.saved=True;p=self.output/'mock_terminal';p.write_bytes(b'synthetic not a model')
  return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),resource_lock_verified=True)
 def strict_reload(self,a):return dict(strict=True,native_student_identity_verified=True,native_ema_op_identity_verified=True,sha256=a['sha256'])
 def verify_test_prerequisites(self):pass
 def guard_test_lineage(self):assert self.opened==0
 def test(self,a,permit):
  assert self.saved and Path(permit).exists();self.opened+=1;ids=self.binding.splits['test']
  rows=[dict(case_id=c,actual_index=i,native_windows=27,selection='EMA+OP',dice=.5,jaccard=.25,hd95=2.,asd=1.) for i,c in enumerate(ids)]
  (self.output/'final_test_cases.jsonl').write_bytes(b''.join(encode(r) for r in rows))
  return dict(case_ids=ids,case_dice=[.5]*87,dice=.5,jaccard=.25,hd95=2.,asd=1.,final_sha256=a['sha256'],test_evaluations=1,distance_unit='voxel',native_patch_forwards=27*87,selection='fixed_terminal_EMA_plus_OP')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--native-dir',required=True);ap.add_argument('--by-contract',required=True);ap.add_argument('--budget-file',required=True);ap.add_argument('--split-dir',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();records=[]
 for labels in (5,10):
  b=DualTeacherBinding(labels,native_dir=a.native_dir,by_contract=a.by_contract,budget_file=a.budget_file,split_dir=a.split_dir)
  with tempfile.TemporaryDirectory(prefix='dual_controller_mock_') as tmp:
   out=Path(tmp);m=MockBackend(b,out);v=FixedCutoffRun(b.recipe,out,b.splits['test']).run(m);raw=(out/'steps.jsonl').read_bytes();s=scan(raw,b)
   yes(f'{labels}_full_native_rows',s['counts']==counts(b.recipe['ssl_iterations']))
   yes(f'{labels}_11cycles_346cases',s['complete_reference_cycles']==11 and s['actual_train_cases']==346)
   yes(f'{labels}_initial_every100_terminal_checkpoints',m.checkpoints==[0]+list(range(100,b.recipe['ssl_iterations']+1,100))+[b.recipe['ssl_iterations']])
   yes(f'{labels}_2effective_updates',v['counts']['successful_optimizer_updates']==2*b.recipe['ssl_iterations'])
   yes(f'{labels}_one_mock_evaluation',m.opened==1 and v['test_evaluations']==1)
   reject(f'{labels}_repeat_controller_closed',lambda:FixedCutoffRun(b.recipe,out,b.splits['test']).run(m))
   n=100;cut=s['line_end_offsets'][n];pin=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),inode=17,device=2)
   commit=dict(contract_sha256='a'*64,execution_role='formal',generation=1,sha256='b'*64,counts=counts(n),ledger=dict(bytes=cut,sha256=hashlib.sha256(raw[:cut]).hexdigest(),inode=17,device=2))
   projected=project(raw,pin,commit,b,'a'*64);tail=projected['complete_tail'];diff=b.recipe['ssl_iterations']-n
   yes(f'{labels}_tail_two_updates_two_inputs',tail['successful_optimizer_updates']==tail['labeled_inputs']==tail['unlabeled_inputs']==2*diff and tail['iterations']==diff)
   yes(f'{labels}_not_resume_authority',projected['independent_resume_review_passed'] is False and projected['ledger_modified'] is False)
   partial=raw+b'{"opaque_partial"';pp=dict(pin,bytes=len(partial),sha256=hashlib.sha256(partial).hexdigest());pv=project(partial,pp,commit,b,'a'*64)
   yes(f'{labels}_opaque_partial_preserved',pv['partial_tail']['bytes']==len(partial)-len(raw) and pv['failed_or_inflight_compute_must_not_be_assumed_zero'])
   bad=copy.deepcopy(commit);bad['ledger']['inode']=18;reject(f'{labels}_inode_replacement',lambda:project(raw,pin,bad,b,'a'*64))
   bad=copy.deepcopy(commit);bad['ledger']['sha256']='0'*64;reject(f'{labels}_prefix_modified',lambda:project(raw,pin,bad,b,'a'*64))
   bad=copy.deepcopy(commit);bad['execution_role']='engineering_sam';reject(f'{labels}_engineering_checkpoint',lambda:project(raw,pin,bad,b,'a'*64))
   first=json.loads(raw.splitlines()[0]);first['cumulative']['successful_optimizer_updates']=1;reject(f'{labels}_hidden_OP_update',lambda:scan(encode(first),b))
   first=json.loads(raw.splitlines()[0]);first['step']['actual_pairs'].reverse();reject(f'{labels}_changed_sampling_order',lambda:scan(encode(first),b))
   reject(f'{labels}_duplicate_json_key',lambda:scan(b'{"step":{},"step":{}}\n',b))
   reject(f'{labels}_unknown_complete_line',lambda:scan(b'unknown\n',b))
   reject(f'{labels}_blank_complete_line',lambda:scan(b'\n',b))
   metrics=v['final_metrics'];artifact=v['artifact'];rows=[json.loads(x) for x in (out/'final_test_cases.jsonl').read_bytes().splitlines()];rows[0]['native_windows']=26;(out/'final_test_cases.jsonl').write_bytes(b''.join(encode(r) for r in rows))
   reject(f'{labels}_wrong_native_windows',lambda:check_metrics(out,metrics,b.splits['test'],artifact))
   records.append(dict(labels=labels,counts=v['counts'],complete_reference_cycles=s['complete_reference_cycles'],actual_train_cases=s['actual_train_cases'],checkpoint_positions=m.checkpoints))
 yes('no_heavy_modules',all(n not in sys.modules for n in ('torch','numpy','SimpleITK')))
 result=dict(status='PASS_CONTROLLER_LEDGER_SYNTHETIC_IO_ONLY',checks=len(checks),check_names=checks,at=time.time(),records=records,model_loads=0,cuda_loads=0,patient_loads=0,actual_optimization_updates=0,actual_test_evaluations=0,new_jobs=0,new_reservations=0,
 source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},prior_34_91_78_checks_reused=True,
 scope='Pinned real small source/splits; synthetic complete ledgers and mock native/controller lifecycle. No real model, tensor restoration, shared gate/storage, optimizer or evaluation.')
 Path(a.out).open('x').write(json.dumps(result,indent=2));print(json.dumps(dict(status=result['status'],checks=len(checks))))
if __name__=='__main__':main()
