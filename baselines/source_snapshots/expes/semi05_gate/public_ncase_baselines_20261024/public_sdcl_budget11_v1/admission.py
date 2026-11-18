"""Independent metadata-only SDCL admission; both stages need their own probes."""
from sdcl_protocol import *
def inspect_ready(c,j,resume=False):
 verify(c)
 if not resume:assert not list(j.glob('*failure*')) and not list(j.glob('worker_error_*.json'))
 q=read(j/'qualification.json');ch=contract_sha(c)
 assert q['status']=='OWN_TWO_PHASE_QUALIFIED_FRESH_PROBES_PENDING' and q['contract_sha256']==ch
 assert q['actual_extra_engineering_updates']==16 and q['engineering_labeled_inputs']==32 and q['engineering_unlabeled_inputs']==16
 assert q['restored_initial_state_exact'] and q['actual_disk_roundtrips'] and q['probe_prefix_iterations']==4 and q['not_the_formal_648_or_1620_prefix']
 assert q['new_formal_updates']==q['new_test_evaluations']==0
 probes={}
 for phase,slot in [('prefix',3),('ssl',4)]:
  r=read(j/('resume_qualification_'+phase+'.json'));ref=read(q['references'][phase]['path'])
  assert sha(q['references'][phase]['path'])==q['references'][phase]['sha256']
  assert r['status']=='PASS' and r['contract_sha256']==ref['contract_sha256']==ch and r['phase']==ref['phase']==phase
  assert all(r[k] for k in ('fresh_process','state_restoration_bitwise_equal','indices_equal','losses_close','rng_sequence_equal'))
  assert r['numeric_tolerance']==dict(rtol=1e-4,atol=1e-5)
  cp=ref['checkpoint'];assert cp['path']==rows(c)[slot]['path'] and sha(cp['path'])==cp['sha256']==r['checkpoint_sha256']
  assert Path(cp['path']).stat().st_size==cp['bytes']<=c['recovery_bound']
  assert ref['execution_role']=='engineering_two_phase_probe'
  before=ref['actual_progress_at_checkpoint'];after=ref['actual_progress_after_reference']
  assert before[phase]['iterations']==2 and after[phase]['iterations']==4
  assert before['prefix']['iterations']==(2 if phase=='prefix' else 4)
  assert before['ssl']['iterations']==(0 if phase=='prefix' else 2)
  assert r['actual_extra_engineering_updates']==4 and r['engineering_labeled_inputs']==8 and r['engineering_unlabeled_inputs']==(0 if phase=='prefix' else 8)
  assert r['new_formal_updates']==r['new_test_evaluations']==0;probes[phase]=dict(sha256=sha(j/('resume_qualification_'+phase+'.json')),max_abs_loss_difference=r['max_abs_loss_difference'])
 for item in c['cpu_proofs']:
  assert sha(item['path'])==item['sha256'];v=read(item['path']);assert v['passed'] is True and v['new_optimization_updates']==v['new_test_evaluations']==0
 registered=read(REGISTRY)
 for row in rows(c):assert [x for x in registered if x['path']==row['path']]==[row]
 return dict(status='READY_INDEPENDENTLY_VERIFIED',contract_sha256=ch,qualification_sha256=sha(j/'qualification.json'),probes=probes,
  total_engineering_updates=24,engineering_labeled_inputs=48,engineering_unlabeled_inputs=24,new_formal_updates=0,new_test_evaluations=0,
  role_separation_verified=True,capacity=capacity(c,True),at=time.time())
