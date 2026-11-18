"""Read-only independent admission. No CUDA/model import."""
from public_protocol import *
def inspect_ready(c,j):
 verify(c)
 assert not list(j.glob('*failure*'))
 if c.get('budget_migration'):
  from budget_lineage import inspect_migration_ready
  return inspect_migration_ready(c,j)
 q=read(j/'qualification.json');r=read(j/'resume_qualification.json')
 ref=read(j/'qualification_backend/resume_reference.json');ch=sha(j/'contract.json')
 assert q['status']=='QUALIFIED' and q['actual_extra_engineering_updates']==6 and q['new_test_evaluations']==0
 assert q['restored_model_optimizer_rng'] and q['actual_disk_roundtrip'] and q['restored_after_all_probes']
 assert q['contract_sha256']==r['contract_sha256']==ref['contract_sha256']==ch
 assert q['resume_reference_sha256']==sha(j/'qualification_backend/resume_reference.json')
 assert r['status']=='PASS' and all(r[k] for k in ['fresh_process','state_restoration_bitwise_equal','losses_close','indices_equal','rng_sequence_equal'])
 assert r['extra_engineering_updates']==2 and r['new_test_evaluations']==0
 assert r['checkpoint_sha256']==ref['checkpoint_sha256']==sha(ref['checkpoint_path'])
 assert Path(ref['checkpoint_path']).stat().st_size<=c['recovery_bound']
 assert q['initial_state_sha256']==ref['initial_state_sha256']
 registered=read(REGISTRY)
 for row in rows(c):assert [x for x in registered if x['path']==row['path']]==[row]
 return dict(status='READY_INDEPENDENTLY_VERIFIED',contract_sha256=ch,qualification_sha256=sha(j/'qualification.json'),
  resume_qualification_sha256=sha(j/'resume_qualification.json'),engineering_updates=8,formal_updates=0,new_test_evaluations=0,
  gpu=1,capacity=capacity(c,True),source_files=len(c['sources']),at=time.time())
