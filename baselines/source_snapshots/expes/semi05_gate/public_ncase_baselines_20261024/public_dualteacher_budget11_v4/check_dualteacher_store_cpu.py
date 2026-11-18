"""Method-specific formal storage/ledger integration: synthetic small pickle IO.
Mock capacity/lock only; no real tensors, GPU, patients, optimization or tests.
"""
import argparse,contextlib,copy,hashlib,json,os,pickle,sys,tempfile,time
from pathlib import Path
from dualteacher_binding import DualTeacherBinding
from dualteacher_store_binding import DualTeacherStore
from dualteacher_storage import read_pin,safe_file
from dualteacher_budget import counts
from check_dualteacher_recovery_cpu import payload,receipt
checks=[]
def yes(n,v):
 if not v:raise AssertionError(n)
 checks.append(n)
def reject(n,fn):
 try:fn()
 except (ValueError,KeyError,TypeError,RuntimeError):checks.append(n);return
 raise AssertionError('Accepted invalid IO: '+n)
class Gate:
 def __init__(self):self.held=False;self.calls=0;self.fail=False
 @contextlib.contextmanager
 def original_lock(self):
  assert not self.held;self.held=True
  try:yield
  finally:self.held=False
 def capacity_snapshot(self):
  assert self.held;self.calls+=1
  if self.fail:raise ValueError('Insufficient filesystem capacity: synthetic')
  return dict(mock_only=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--native-dir',required=True);ap.add_argument('--by-contract',required=True);ap.add_argument('--budget-file',required=True);ap.add_argument('--split-dir',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
 b=DualTeacherBinding(5,native_dir=a.native_dir,by_contract=a.by_contract,budget_file=a.budget_file,split_dir=a.split_dir);pairs=b.pairs;identity='a'*64;foundation=payload(b.recipe,'formal',pairs,0)['foundation'];loads=[]
 def load(f):loads.append(1);return pickle.load(f)
 with tempfile.TemporaryDirectory(prefix='dual_store_cpu_') as tmp:
  root=Path(tmp);job=root/'job';art=root/'art';job.mkdir();art.mkdir();g=Gate();slots=[art/'a/final.pth',art/'b/final.pth']
  st=DualTeacherStore(job,art,slots,32*1024**2,32*1024**2,identity,g,pickle.dump,load,b,foundation,{'mock':'native_recipe'},art/'terminal/final.pth',1024**2)
  commits=[]
  for n in range(3):
   if n:
    row=receipt(n-1,b.recipe,pairs);st.append_step(dict(step=row,cumulative=counts(n)))
   s=payload(b.recipe,'formal',pairs,n);s['ledger']=st.ledger_pin();c=st.checkpoint(s);commits.append(c)
   yes('generation_'+str(n),c['generation']==n and c['counts']==counts(n) and c['resource_lock_verified'])
  yes('two_alternating_generations',[x['generation'] for x in st.available()]==[2,1])
  yes('native_effective_counts',commits[-1]['counts']['successful_optimizer_updates']==4)
  yes('ledger_inode_stable',commits[1]['ledger']['inode']==commits[2]['ledger']['inode'])
  c=st.available()[0];review=dict(contract_sha256=identity,checkpoint_sha256=c['sha256'],generation=c['generation'],independent_resume_review_passed=True,consumed_by_pid=os.getpid())
  s=st.load_reviewed(c,review);yes('exact_synthetic_payload_restored',s['counts']==counts(2) and s['actual_pairs']==pairs[:4] and len(loads)==1)
  for field,value in [('contract_sha256','b'*64),('checkpoint_sha256','c'*64),('generation',1),('independent_resume_review_passed',False),('consumed_by_pid',0)]:
   bad=dict(review);bad[field]=value;reject('review_'+field,lambda:st.load_reviewed(c,bad))
  yes('unreviewed_no_deserialize',len(loads)==1)
  bad=copy.deepcopy(s);bad['execution_role']='engineering_warmup';reject('engineering_in_formal_slots',lambda:st.checkpoint(bad))
  bad=copy.deepcopy(s);bad['foundation']['sha256']='0'*64;reject('foreign_foundation',lambda:st.checkpoint(bad))
  bad=copy.deepcopy(s);bad['ledger']['sha256']='0'*64;reject('foreign_ledger_prefix',lambda:st.checkpoint(bad))
  g.fail=True;reject('capacity_failure_before_serialization',lambda:st.checkpoint(s));g.fail=False
  yes('capacity_failure_preserves_two_generations',[x['generation'] for x in st.available()]==[2,1])
  st.record('small_evidence.json',{'key':1});st.record('small_evidence.json',{'key':1},consistent=True)
  reject('changed_evidence_not_overwritten',lambda:st.record('small_evidence.json',{'key':2},consistent=True))
  reject('duplicate_evidence_not_overwritten',lambda:st.record('small_evidence.json',{'key':1}))
  reject('storage_escape',lambda:safe_file(root/'outside',job))
  link=job/'linked';link.symlink_to(job/'small_evidence.json');reject('symlink',lambda:read_pin(link,job))
  hard=job/'hard';os.link(job/'small_evidence.json',hard);reject('hardlink',lambda:read_pin(hard,job));hard.unlink();link.unlink()
  # Unknown trailing bytes refuse checkpoint/reload; nothing is truncated here.
  ledger=job/'formal/steps.jsonl';original=ledger.read_bytes();ledger.write_bytes(original+b'{partial')
  reject('opaque_partial_checkpoint_refused',lambda:st.checkpoint(s));yes('partial_bytes_preserved',ledger.read_bytes().endswith(b'{partial'))
  # Synthetic test fixture reset only, not production reconciliation.
  ledger.write_bytes(original)
  row=receipt(2,b.recipe,pairs);st.append_step(dict(step=row,cumulative=counts(3)))
  reject('older_commit_ledger_requires_reconciliation',lambda:st.load_reviewed(c,review))
  # Complete synthetic 11-cycle ledger to exercise terminal identity checks.
  for i in range(3,b.recipe['ssl_iterations']):
   r=receipt(i,b.recipe,pairs)
   with ledger.open('ab') as f:f.write((json.dumps(dict(step=r,cumulative=counts(i+1)))+'\n').encode())
  terminal=dict(format='public_dualteacher_fixed11_terminal_v1',method='dualteacher_sam',execution_role='formal',selection='fixed_terminal_EMA_plus_OP',contract_sha256=identity,recipe=b.recipe,counts=counts(b.recipe['ssl_iterations']),signatures={k:'b'*64 for k in ('student','ema','op','criterion')},foundation=foundation,native_recipe={'mock':'native_recipe'},student={},ema={},op={},criterion={})
  for key,value in [('selection','student'),('execution_role','engineering_sam'),('counts',counts(1)),('foundation',{}),('signatures',{}),('native_recipe',{})]:
   bad=copy.deepcopy(terminal);bad[key]=value;reject('terminal_'+key,lambda:st.save_terminal(bad))
  yes('invalid_terminal_absent',not st.target.exists())
  artifact=st.save_terminal(terminal);loaded=st.load_terminal(artifact);yes('strict_small_terminal_metadata',loaded==terminal and artifact['resource_lock_verified'])
  reject('terminal_never_overwritten',lambda:st.save_terminal(terminal));bad=dict(artifact,sha256='0'*64);oldloads=len(loads);reject('terminal_SHA_before_deserialization',lambda:st.load_terminal(bad));yes('bad_terminal_no_load',len(loads)==oldloads)
  # Overflow keeps a partial for review rather than rewriting an old payload.
  st2=DualTeacherStore(job,art,[art/'c/final.pth',art/'d/final.pth'],16,32*1024**2,identity,g,pickle.dump,load,b,foundation,{'mock':'native_recipe'},art/'terminal2/final.pth',16)
  reject('bounded_terminal_overflow',lambda:st2.save_terminal(terminal));yes('overflow_partial_preserved',Path(str(st2.target)+'.partial').exists() and not st2.target.exists())
  yes('original_generations_not_deleted',all(p.exists() for p in slots))
  yes('mock_shared_gate_used',g.calls>10 and not g.held)
 yes('no_heavy_modules',all(n not in sys.modules for n in ('torch','numpy','SimpleITK')))
 result=dict(status='PASS_METHOD_STORAGE_SYNTHETIC_PICKLE_MOCK_GATE_ONLY',checks=len(checks),check_names=checks,at=time.time(),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},model_loads=0,cuda_loads=0,patient_loads=0,actual_optimization_updates=0,actual_test_evaluations=0,new_jobs=0,new_reservations=0,prior_34_91_78_37_checks_reused=True,scope='Own full-state metadata/actual source-bound sampling ledger plus small synthetic pickle and mock lock/capacity. Not real tensors, torch serialization, GPU or original full shared gate admission.')
 Path(a.out).open('x').write(json.dumps(result,indent=2));print(json.dumps(dict(status=result['status'],checks=len(checks))))
if __name__=='__main__':main()
