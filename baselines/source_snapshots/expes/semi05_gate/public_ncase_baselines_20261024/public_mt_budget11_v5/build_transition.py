"""One explicit human-authorized MT budget migration, CPU-only preparation."""
import io,copy,json,time,ast
from pathlib import Path
import public_protocol as p
from budget_lineage import verify_lineage,translate,OLD_CONTRACT

def build():
 base=p.HERE.parent;origin=p.BASE/'mt_semisam/Public433/lab005/fold2/r4'
 stop_path=base/'mt5_r4_budget_stop_audit_20261025_1407.json';stop=p.read(stop_path)
 old=p.read(origin/'contract.json');assert p.sha(origin/'contract.json')==OLD_CONTRACT
 j,t=p.paths('mt_semisam',5,'r5');assert not j.exists() and not t.parent.exists()
 c=copy.deepcopy(old);c.pop('migration')
 c.update(tag='r5',target=str(t),output=str(j/'formal'),primary_pairs=3795,
  human_authorization='Current MT five cases also change to new budget; solve engineering and space problems and continue',
  budget_protocol='PUBLIC_PRIMARY_INPUT_11_CYCLE_CUTOFF_20261025',schedule_policy='retain_original_36_cycle_lr_ramp_clocks; cutoff_only',
  budget_manifest=dict(path=str(p.HERE/'budget11.json'),sha256=p.sha(p.HERE/'budget11.json')),
  click_patch=dict(path=str(p.HERE/'candidate_ritm.py'),sha256=p.sha(p.HERE/'candidate_ritm.py')),
  budget_migration=dict(origin_job=str(origin),origin_contract_sha256=OLD_CONTRACT,checkpoint=stop['checkpoint'],stop_audit=dict(path=str(stop_path),sha256=p.sha(stop_path)),
   original_complete_iterations=3132,replayed_tail_iterations=32,previous_origin_tail_iterations=66,previous_failure_extra_inputs_per_stream=1,inflight_step_extra_compute='not fully measured; not zero',
   preserves_optimizer_rng_and_36_cycle_schedule=True))
 c['sources'] += [dict(path=str(x),sha256=p.sha(x)) for x in sorted(p.HERE.glob('*.py'))]
 c['sources'] += [dict(path=str(p.HERE/'template.json'),sha256=p.sha(p.HERE/'template.json'))]
 p.verify(c);p.gpu_idle();cap=p.capacity(c,True)
 j.mkdir(parents=True);p.exclusive_json(j/'contract.json',c);ch=p.sha(j/'contract.json')
 import torch
 from resumable_mt import tree_hash,Backend
 from runtime_data import schedule
 # Actual new checkpoint and new identity conversion: no models, CUDA or training.
 state=torch.load(stop['checkpoint']['path'],map_location='cpu');converted=translate(state,c['budget_migration'],ch,tree_hash)
 stream=io.BytesIO();torch.save(converted,stream);stream.seek(0);loaded=torch.load(stream,map_location='cpu')
 assert tree_hash(loaded)==tree_hash(converted)
 short=schedule(p.budget(5),c['splits']);long=schedule(p.clock_budget(5),c['splits'])
 assert short==long[:3795] and state['actual_rows']==long[:3100]
 assert p.clock_budget(5).primary_pairs==old['primary_pairs']==12420
 # Existing train_iteration, native_step and original denominator stay identical.
 new_ast=ast.parse((p.HERE/'resumable_mt.py').read_text());old_ast=ast.parse((p.HERE.parent/'public_mt_v3/resumable_mt.py').read_text())
 def method(tree,name):return next(x for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Backend' for x in n.body if isinstance(x,ast.FunctionDef) and x.name==name)
 for n in ('train_iteration','capture','restore','checkpoint','save_terminal','gate'):
  assert ast.dump(method(new_ast,n),include_attributes=False)==ast.dump(method(old_ast,n),include_attributes=False),n
 assert (p.HERE/'controller.py').read_bytes()==(p.HERE.parent/'public_mt_click_migration_v4/controller.py').read_bytes()
 checks=['identity_only_actual_checkpoint_translation','strict_cpu_serialization_roundtrip','3100_counts_preserved','original_sampling_prefix_preserved','original_12420_lr_ramp_clock_preserved','training_capture_restore_checkpoint_save_AST_unchanged','controller_byte_identical']
 fake=Backend.__new__(Backend);fake.c=c;fake.position=3795;fake.pairs=long;fake.actual_rows=short;fake.cycle_rows=[dict(actual_case_coverage_verified=True) for _ in range(11)]
 assert fake.verify_training_coverage()['primary_labeled_inputs']==3795
 fake.position=3794
 try:fake.verify_training_coverage();raise RuntimeError('Early endpoint accepted')
 except AssertionError:checks.append('incomplete_new_endpoint_rejected')
 for n,E in ((5,3795),(10,3740)):
  assert p.budget(n).primary_pairs==E
  for b in (1,2,4):assert 0<=((E+b-1)//b)*b-E<b
 checks.append('two_case_budgets_and_native_rounding')
 review=dict(status='PASS',contract_sha256=ch,checkpoint_sha256=stop['checkpoint']['sha256'],loaded_state_unchanged_except_contract=True,prefix_sampling_and_clock_unchanged=True,checks=checks,loaded_state_hash=tree_hash(loaded),bytes_in_memory=stream.getbuffer().nbytes,new_engineering_updates=0,new_test_evaluations=0,models_built=0,cuda_loaded=False,at=time.time())
 p.exclusive_json(j/'budget_migration_cpu_review.json',review)
 with p.leased(c):
  p.verify(c);cap=p.enroll(c);out=Path(c['output']);out.mkdir()
  raw=(origin/'formal/steps.jsonl').read_bytes()[:stop['checkpoint']['ledger']['bytes']]
  with (out/'steps.jsonl').open('xb') as f:f.write(raw);f.flush();p.os.fsync(f.fileno())
  assert p.ledger(c)==stop['checkpoint']['ledger']
  p.exclusive_json(j/'migration_import.json',dict(new_contract_sha256=ch,origin_contract_sha256=OLD_CONTRACT,origin_checkpoint=stop['checkpoint'],imported_counts=state['counts'],original_tail_preserved=True,original_tail_complete_updates=32,previous_original_tail_updates=66,formal_new_updates=0,test_evaluations=0,at=time.time()))
  p.exclusive_json(j/'human_authorization.json',dict(instruction=c['human_authorization'],action='START_BUDGET11_CONTINUATION',contract_sha256=ch,standing_engineering_recovery_authorized=True,cpac_launch_authorized=False,at=time.time()))
 from admission import inspect_ready
 admit=inspect_ready(c,j);p.exclusive_json(j/'admission.json',admit)
 return dict(status='READY_INDEPENDENTLY_VERIFIED',job=str(j),contract_sha256=ch,imported=3100,endpoint=3795,remaining=695,margin_bytes=min(x['margin_bytes'] for x in cap['filesystems']))
if __name__=='__main__':print(json.dumps(build()))
