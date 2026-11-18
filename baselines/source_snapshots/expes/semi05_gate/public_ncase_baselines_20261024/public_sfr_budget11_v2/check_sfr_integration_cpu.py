"""Metadata-only entry/admission/recovery/observer fault checks; no model import."""
import argparse,ast,contextlib,copy,json,sys,tempfile
from pathlib import Path
import sfr_protocol as p
import admission as a
import resume_review as r
import observe_sfr as o
from sfr_state import plan,prefix_inputs,counts

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
 root=Path(tempfile.mkdtemp(prefix='sfr_integration_cpu_',dir=p.HERE.parent));passed=[]
 def good(name):passed.append(name)
 def reject(name,fn):
  try:fn()
  except (AssertionError,ValueError,RuntimeError,KeyError,StopIteration):good(name)
  else:raise AssertionError('Missing rejection '+name)
 @contextlib.contextmanager
 def fake_lease(c):yield 7
 # Independent admission actually reads synthetic phase references and SHA pins.
 c=dict(ratio=5,labeled_cases=5,target=str(root/'weights/final.pth'),output=str(root/'formal'),recovery_bound=768*1024**2,
        stage_recipe=plan(5),cpu_proofs=[])
 p.exclusive_json(root/'contract.json',dict(synthetic=True));ch=p.contract_sha(c)
 refs={}
 for phase,slot in [('prefix',3),('ssl',4)]:
  path=Path(p.rows(c)[slot]['path']);path.parent.mkdir(parents=True);path.write_bytes(b'synthetic-state-'+phase.encode())
  cp=dict(path=str(path),**p.file_pin(path))
  before=dict(phase=phase,prefix=counts(2 if phase=='prefix' else 4,4 if phase=='prefix' else 7,0),ssl=counts(0 if phase=='prefix' else 2,0 if phase=='prefix' else 4,0 if phase=='prefix' else 4))
  after=copy.deepcopy(before);after[phase]=counts(4,7 if phase=='prefix' else 8,0 if phase=='prefix' else 8)
  ref=dict(phase=phase,contract_sha256=ch,checkpoint=cp,execution_role='engineering_two_phase_probe',actual_progress_at_checkpoint=before,actual_progress_after_reference=after)
  path=root/(phase+'_reference.json');p.exclusive_json(path,ref);refs[phase]=dict(path=str(path),sha256=p.sha(path))
  probe=dict(status='PASS',phase=phase,contract_sha256=ch,checkpoint_sha256=cp['sha256'],fresh_process=True,state_restoration_bitwise_equal=True,
   indices_equal=True,losses_close=True,rng_sequence_equal=True,numeric_tolerance=dict(rtol=1e-4,atol=1e-5),
   actual_extra_engineering_updates=2,engineering_labeled_inputs=3 if phase=='prefix' else 4,engineering_unlabeled_inputs=0 if phase=='prefix' else 4,
   new_formal_updates=0,new_test_evaluations=0,max_abs_loss_difference=0.0)
  p.exclusive_json(root/('resume_qualification_'+phase+'.json'),probe)
 q=dict(status='OWN_TWO_PHASE_QUALIFIED_FRESH_PROBES_PENDING',contract_sha256=ch,actual_extra_engineering_updates=8,
  engineering_labeled_inputs=15,engineering_unlabeled_inputs=8,initial_state_pinned_without_inplace_rewind=True,actual_disk_roundtrips=True,
  probe_prefix_iterations=4,not_the_formal_6000_prefix=True,new_formal_updates=0,new_test_evaluations=0,references=refs)
 p.exclusive_json(root/'qualification.json',q)
 for name in ('qualify.log','probe_prefix.log','probe_ssl.log'):(root/name).write_bytes(b'')
 reg=root/'registry.json';p.exclusive_json(reg,p.rows(c));a.REGISTRY=reg
 a.verify=lambda c:None;a.leased=fake_lease;a.capacity=lambda c,start=False:dict(mock_capacity=True,start=start)
 v=a.inspect_ready(c,root);assert v['total_engineering_updates']==12 and v['engineering_labeled_inputs']==22 and v['engineering_unlabeled_inputs']==12
 good('SFR_5case_12_updates_22L12U_admission_counts')
 for key,value in [('actual_extra_engineering_updates',24),('engineering_labeled_inputs',48),('probe_prefix_iterations',6000)]:
  bad=dict(q,**{key:value});p.atomic(root/'qualification.json',bad)
  reject('reject_wrong_'+key,lambda:a.inspect_ready(c,root))
 p.atomic(root/'qualification.json',q)
 path=root/'resume_qualification_ssl.json';base=p.read(path)
 for key,value in [('fresh_process',False),('rng_sequence_equal',False),('state_restoration_bitwise_equal',False),('losses_close',False),('new_test_evaluations',1)]:
  p.atomic(path,dict(base,**{key:value}));reject('reject_missing_'+key,lambda:a.inspect_ready(c,root))
 p.atomic(path,base)
 (root/'probe_prefix.log').write_bytes(b'unrecognized diagnostic\n');reject('unknown_engineering_stderr_rejects_admission',lambda:a.inspect_ready(c,root));(root/'probe_prefix.log').write_bytes(b'')
 (root/'formal').mkdir();reject('existing_formal_directory_cannot_restart',lambda:a.inspect_ready(c,root))
 reject('observer_unknown_stdout_is_not_suppressed',lambda:o.boot_check(c,b'unknown\n',False));o.boot_check(c,b'',False)
 good('observer_empty_diagnostics_allowed')
 # Real tiny files exercise reviewed truncation and retained tail accounting.
 out=Path(c['output']);lines=[]
 for size in (2,2,1):lines.append((json.dumps(dict(step=dict(labeled_inputs=size,unlabeled_inputs=0)))+'\n').encode())
 pre=out/'pretrain_steps.jsonl';pre.write_bytes(b''.join(lines));pin=dict(bytes=len(lines[0]),sha256=p.hashlib.sha256(lines[0]).hexdigest())
 empty=dict(bytes=0,sha256=p.hashlib.sha256(b'').hexdigest())
 cp=dict(contract_sha256=ch,sha256='a'*64,progress=dict(phase='prefix',prefix=counts(1,2,0),ssl=counts(0,0,0)),ledgers={'pretrain_steps.jsonl':pin,'steps.jsonl':empty})
 r.verify=lambda c:None;r.leased=fake_lease;r.capacity=lambda c,start=False:dict(mock_capacity=True)
 r.checkpoints=lambda c:[cp]
 p.exclusive_json(out/'failure.json',dict(error='synthetic failure, no model'))
 review=r.create_review(c,root,'a'*64,'Synthetic metadata-only boundary failure review for testing')
 assert r.validate_review(c,root)==review;good('review_binds_full_old_ledger_and_latest_commit')
 pre.write_bytes(pre.read_bytes()+b'changed');reject('changed_ledger_after_review_rejected',lambda:r.validate_review(c,root));pre.write_bytes(b''.join(lines))
 result=r.reconcile(c,root);assert pre.read_bytes()==lines[0]
 tail=result['uncommitted_ledgers']['pretrain_steps.jsonl'];assert tail['extra_complete_updates']==2 and tail['extra_complete_labeled_inputs']==3 and tail['extra_complete_unlabeled_inputs']==0
 assert (root/'recovery_history'/result['event']/'uncommitted_pretrain_steps.jsonl').read_bytes()==b''.join(lines[1:])
 assert not (out/'failure.json').exists();good('preserved_odd_prefix_tail_counts_one_update_per_row')
 reject('consumed_review_cannot_reconcile_twice',lambda:r.validate_review(c,root))
 p.exclusive_json(out/'sfr_test_started.json',dict(synthetic=True));reject('native_test_intent_blocks_recovery',lambda:r.no_test(c,root))
 # AST checks bind entry to distinct fresh phase probes without loading torch.
 source=(p.HERE/'public_operator.py').read_text();tree=ast.parse(source)
 qualify=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='qualify')
 assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='restore' for n in ast.walk(qualify))
 good('qualifier_never_rewinds_used_SAM_backend_in_place')
 assert 'from sfr_controller import TwoStageRun' in source and "origin[k] for k in ('official_code_root','official_commit','official_source_paths','foundation')" in source
 good('entry_binds_original_SFR_foundation_and_own_controller')
 assert 'torch' not in sys.modules;good('integration_checks_never_import_torch')
 v=dict(passed=True,test_count=len(passed),checks=passed,synthetic_only=True,mock_resource_gate=True,
  new_model_loads=0,new_cuda_loads=0,new_patient_inputs=0,new_optimization_updates=0,new_test_evaluations=0,new_jobs=0,new_reservations=0,new_dispatches=0,
  source_files={str(x):p.sha(x) for x in sorted(p.HERE.glob('*.py'))},scratch=str(root))
 p.exclusive_json(args.out,v);print(json.dumps(dict(passed=True,test_count=len(passed),out=str(args.out))))
if __name__=='__main__':main()
