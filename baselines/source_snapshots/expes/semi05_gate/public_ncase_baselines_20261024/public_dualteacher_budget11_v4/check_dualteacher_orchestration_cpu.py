"""Own admission/resume and dispatch guards with synthetic local evidence.
All lease/capacity/checkpoint providers are mock; no native backend is loaded.
"""
import argparse,contextlib,copy,hashlib,json,os,sys,tempfile,time
from pathlib import Path
import admission as ad
import resume_review as rr
import public_operator as op
from dualteacher_budget import plan,counts
from dualteacher_ledger import project
from check_dualteacher_recovery_cpu import receipt
from dualteacher_binding import DualTeacherBinding
checks=[]
def yes(n,v):
 if not v:raise AssertionError(n)
 checks.append(n)
def reject(n,fn):
 try:fn()
 except (ValueError,KeyError,TypeError,FileNotFoundError):checks.append(n);return
 raise AssertionError('Accepted invalid integration: '+n)
def put(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
@contextlib.contextmanager
def lease(c):yield 123
class Guard:
 @contextlib.contextmanager
 def original_lock(self):yield

def main(out):
 import public_dualteacher_protocol as proto
 b=proto.binding(dict(labeled_cases=5));identity='a'*64
 with tempfile.TemporaryDirectory(prefix='dt_integration_cpu_') as tmp:
  root=Path(tmp);j=root/'job';j.mkdir();target=root/'artifact/final.pth';target.parent.mkdir();c=dict(output=str(j/'formal'),target=str(target),recipe=b.recipe);registry=root/'registry';rows=[dict(path=str(root/('artifact/'+str(i))),bound=4096) for i in range(6)];put(registry,rows)
  ad.verify=lambda c:None;ad.contract_sha=lambda c:identity;ad.REGISTRY=registry;ad.rows=lambda c:rows;ad.leased=lease;ad.capacity=lambda *a:dict(mock=True)
  q=dict(status='OWN_WINDOWS_QUALIFIED_FRESH_PROBES_PENDING',contract_sha256=identity,actual_extra_engineering_iterations=8,actual_extra_engineering_updates=16,engineering_labeled_inputs=16,engineering_unlabeled_inputs=16,new_formal_updates=0,new_test_evaluations=0,actual_disk_roundtrips=True,references={},pid=10)
  for k,(window,slot) in enumerate((('warmup',3),('sam',4))):
   role='engineering_'+window;start=0 if k==0 else b.recipe['original_sam_start_index'];path=Path(rows[slot]['path']);path.write_bytes(b'mock complete state, not tensors');cp=dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path));ref=dict(execution_role=role,contract_sha256=identity,checkpoint=cp,expected_counts=counts(4),steps=[dict(completed_primary_pairs=2*(start+3)),dict(completed_primary_pairs=2*(start+4))],active_sam_actual_gradient_verified=k==1)
   rp=j/('qualification_'+role)/'reference.json';put(rp,ref);q['references'][role]=dict(path=str(rp),sha256=sha(rp))
   r=dict(status='PASS',role=role,contract_sha256=identity,fresh_process=True,state_restoration_bitwise_equal=True,indices_equal=True,losses_close=True,rng_sequence_equal=True,numeric_tolerance=dict(rtol=1e-4,atol=1e-5),actual_extra_engineering_iterations=2,actual_extra_engineering_updates=4,engineering_labeled_inputs=4,engineering_unlabeled_inputs=4,new_formal_updates=0,new_test_evaluations=0,pid=11+k,checkpoint_sha256=cp['sha256'],max_abs_loss_difference=1e-6,continuation_state_bitwise_equal=False);put(j/('resume_qualification_'+window+'.json'),r)
  put(j/'qualification.json',q)
  for name in ('qualify.log','probe_warmup.log','probe_sam.log'):(j/name).write_bytes(b'')
  admitted=ad.inspect_ready(c,j);yes('own_counts_24_updates_24L24U',admitted['total_engineering_updates']==admitted['engineering_labeled_inputs']==admitted['engineering_unlabeled_inputs']==24 and admitted['total_engineering_iterations']==12)
  yes('nonbitwise_continuation_allowed',all(not v['continuation_state_bitwise_equal'] for v in admitted['probes'].values()))
  p=j/'resume_qualification_sam.json';saved=p.read_bytes();v=json.loads(saved)
  for name,key,value in [('same_qualification_pid','pid',10),('same_probe_pid','pid',11),('missing_exact_state','state_restoration_bitwise_equal',False),('wrong_tolerance','numeric_tolerance',dict(rtol=1e-3,atol=1e-5)),('old_one_update_accounting','actual_extra_engineering_updates',2),('wrong_role','role','engineering_warmup')]:
   bad=copy.deepcopy(v);bad[key]=value;put(p,bad);reject(name,lambda:ad.inspect_ready(c,j));p.write_bytes(saved)
  log=j/'probe_sam.log';log.write_bytes(b'unknown');reject('unknown_qualification_log',lambda:ad.inspect_ready(c,j));log.write_bytes(b'')
  put(j/'resume_probe_sam_failure.json',{});reject('failure_cannot_admit',lambda:ad.inspect_ready(c,j));(j/'resume_probe_sam_failure.json').unlink()
  put(j/'dispatch.json',{});reject('duplicate_start_lineage',lambda:ad.inspect_ready(c,j));(j/'dispatch.json').unlink()
  Path(c['output']).mkdir();reject('existing_formal_lineage',lambda:ad.inspect_ready(c,j))
  # Real byte-prefix projection and history/inode preservation in a temp tree.
  ledger=Path(c['output'])/'steps.jsonl';raw=b'';pairs=b.pairs
  for i in range(4):raw+=(json.dumps(dict(step=receipt(i,b.recipe,pairs),cumulative=counts(i+1)))+'\n').encode()
  ledger.write_bytes(raw+b'{partial');full=ledger.read_bytes();pin=rr.read_pin(ledger,j);prefix=b''.join(full.splitlines(keepends=True)[:2]);commit=dict(contract_sha256=identity,execution_role='formal',generation=0,sha256='b'*64,counts=counts(2),ledger=dict(bytes=len(prefix),sha256=hashlib.sha256(prefix).hexdigest(),inode=pin['inode'],device=pin['device']));projection=project(full,pin,commit,b,identity)
  rr.verify=lambda c:None;rr.contract_sha=lambda c:identity;rr.binding=lambda c:b;rr.capacity=lambda *a:dict(mock=True);rr.leased=lease;rr.resource=lambda c:Guard();rr.capacity_locked=lambda c:dict(mock=True);rr.checkpoint_review=lambda *a:(commit,projection)
  put(Path(c['output'])/'failure.json',dict(error='synthetic'));rr.create_review(c,j,'b'*64,'Synthetic resource failure after two complete uncommitted iterations')
  yes('review_no_model_or_test',rr.read(j/'resume_review.json')['new_model_loads']==rr.read(j/'resume_review.json')['new_test_evaluations']==0)
  event=rr.reconcile(c,j);history=j/'recovery_history'/event['event'];yes('full_and_tail_preserved',(history/'original_steps.jsonl').read_bytes()==full and (history/'uncommitted_steps.jsonl').read_bytes()==full[len(prefix):])
  yes('same_inode_exact_rollback',rr.read_pin(ledger,j)==commit['ledger'])
  yes('true_tail_cost_two_updates_per_iter',event['projection']['complete_tail']['successful_optimizer_updates']==4 and event['projection']['partial_tail']['bytes']>0)
  yes('failure_preserved_not_deleted',len(event['preserved_failures'])==1 and not (Path(c['output'])/'failure.json').exists())
  reject('review_one_use',lambda:rr.reconcile(c,j));rr.bind_worker_pid(c,os.getpid());consumed=rr.require_consumed_review(c);yes('review_bound_current_worker',consumed['consumed_by_pid']==os.getpid())
  bad=rr.read(history/'consumed.json');bad['consumed_by_pid']=1;put(history/'consumed.json',bad);reject('other_worker_refused',lambda:rr.require_consumed_review(c))
  put(Path(c['output'])/'dualteacher_final_test_started.json',{});reject('original_native_test_prevents_resume',lambda:rr.no_test_or_worker(c,j));(Path(c['output'])/'dualteacher_final_test_started.json').unlink()
  put(history/'final_test_cases.jsonl',{});reject('preserved_test_history_prevents_resume',lambda:rr.no_test_or_worker(c,j))
  reject('launch_confirmation_required',lambda:op.launch(c,j,type('A',(),{'confirm':'bad'})()))
  yes('no_actual_backend_import',all(n not in sys.modules for n in ('torch','numpy','SimpleITK','_public_dt_pinned_backend')))
 result=dict(status='PASS_OWN_ORCHESTRATION_SYNTHETIC_GUARDS_ONLY',checks=len(checks),check_names=checks,at=time.time(),model_loads=0,cuda_loads=0,patient_loads=0,actual_optimization_updates=0,actual_test_evaluations=0,new_jobs=0,new_reservations=0,actual_formal_dispatch=False,source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')},scope='Synthetic admission receipts and temporary ledger/history with mock lease/capacity/contract providers; no native tensor/GPU or actual shared admission proof.',prior_checks_reused=True)
 Path(out).open('x').write(json.dumps(result,indent=2));print(json.dumps(dict(status=result['status'],checks=len(checks))))
if __name__=='__main__':main(sys.argv[1])
