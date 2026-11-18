"""Budget-cutoff lineage, reviewed empty-click patch, independent migration admission."""
import hashlib,json,os
from pathlib import Path
PATCH_SHA='fa8585987c45b98b450a1aef869be849b50452113551845e5870560ce8b2ea96'
AUTHOR_SHA='a7e07b3033a7f6ebbea753a94d0ef45cb5c037a2fefdaf8a1412acc9e41936e5'
OLD_CONTRACT='a49109bec263f67b9b9ca65c51369caae4b93e9c4cc8faad5e0dd84ac70f0dbd'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def pin(p):
 p=Path(p);s=p.stat();assert p.is_file() and not p.is_symlink() and s.st_nlink==1
 return dict(bytes=s.st_size,sha256=sha(p))
def guard_test_lineage(c):
 if not c.get('budget_migration'):return
 m=c['budget_migration'];old=read(Path(m['origin_job'])/'contract.json')
 for j,t in [(m['origin_job'],old['target']),(old['migration']['origin_job'],old['migration']['origin_target'])]:
  assert not Path(t).exists()
  for n in ('final_test_intent.json','final_metrics.json'):assert not (Path(j)/'formal'/n).exists()
def verify_lineage(c):
 assert sha(c['click_patch']['path'])==c['click_patch']['sha256']==PATCH_SHA
 b=c['budget_manifest'];assert sha(b['path'])==b['sha256']
 manifest=read(b['path']);assert manifest['reference_cycles']==11 and not manifest['performance_early_stop']
 if not c.get('budget_migration'):return
 m=c['budget_migration'];j=Path(m['origin_job']);old=read(j/'contract.json')
 assert sha(j/'contract.json')==m['origin_contract_sha256']==OLD_CONTRACT
 assert c['tag']=='r5' and c['method']=='mt_semisam' and c['labeled_cases']==5 and c['primary_pairs']==3795
 allowed={'tag','target','output','sources','migration','primary_pairs','human_authorization'}
 for k in set(old)-allowed:assert c[k]==old[k],k
 assert c['sources'][:len(old['sources'])]==old['sources']
 a=m['stop_audit'];assert sha(a['path'])==a['sha256'];stop=read(a['path'])
 assert stop['checkpoint']==m['checkpoint'] and stop['complete_counts']['iterations']==3132
 assert stop['uncommitted_complete_iterations']==32 and stop['partial_ledger_bytes']==0
 assert not Path('/proc/4082').exists(),'Old worker has not exited'
 assert sorted(str(x) for x in j.rglob('*') if x.is_file())==sorted(stop['original_files'])
 for path,expected in stop['original_files'].items():assert pin(path)==expected,path
 for row in old['sources']+old['inputs']:assert sha(row['path'])==row['sha256']
 for path,expected in old['migration']['protected_files'].items():
  actual=pin(path);actual['inode']=Path(path).stat().st_ino;assert actual==expected,path
 cp=m['checkpoint'];assert cp['iteration']==3100 and cp['sha256']==sha(cp['path'])
 raw=(j/'formal/steps.jsonl').read_bytes();n=cp['ledger']['bytes']
 assert hashlib.sha256(raw[:n]).hexdigest()==cp['ledger']['sha256'] and len(raw[:n].splitlines())==3100
 assert (j/'formal/steps.jsonl').stat().st_ino==stop['ledger']['inode']
 assert len(raw[n:].splitlines())==32 and len(raw)==stop['ledger']['bytes'] and sha(j/'formal/steps.jsonl')==stop['ledger']['sha256']
 guard_test_lineage(c)
 return m
def translate(s,m,new_sha,tree_hash):
 assert s['contract_sha256']==OLD_CONTRACT and s['position']==s['global_step']==s['optimizer_calls']==3100
 assert s['counts']==dict(iterations=3100,successful_optimizer_updates=3100,labeled_inputs=3100,unlabeled_inputs=3100)
 assert s['ledger']==m['checkpoint']['ledger'] and len(s['actual_rows'])==3100
 assert len(new_sha)==64 and new_sha!=OLD_CONTRACT
 result=dict(s,contract_sha256=new_sha)
 assert tree_hash({k:v for k,v in result.items() if k!='contract_sha256'})==tree_hash({k:v for k,v in s.items() if k!='contract_sha256'})
 return result
def load_state(c,new_sha,loader,tree_hash):
 m=verify_lineage(c);s=loader(m['checkpoint']['path'],map_location='cpu')
 return translate(s,m,new_sha,tree_hash)
def install_patch(b):
 path=b.c['click_patch']['path'];assert sha(path)==PATCH_SHA
 def unwrap(f):
  while hasattr(f,'__wrapped__'):f=f.__wrapped__
  return f
 g=unwrap(b.t.generalist.__call__).__globals__;pg=unwrap(g['predict3d_unc']).__globals__
 original=pg['get_next_click3D_torch_ritm'];assert sha(original.__code__.co_filename)==AUTHOR_SHA
 scope=dict(original.__globals__);exec(compile(Path(path).read_text(),path,'exec'),scope)
 pg['get_next_click3D_torch_ritm']=scope['get_next_click3D_torch_ritm']
def inspect_migration_ready(c,j):
 from public_protocol import verify,capacity,read as rd,REGISTRY,rows
 verify(c);m=c['budget_migration'];audit=rd(j/'budget_migration_cpu_review.json')
 assert audit['status']=='PASS' and audit['contract_sha256']==sha(j/'contract.json')
 assert audit['loaded_state_unchanged_except_contract'] and audit['prefix_sampling_and_clock_unchanged']
 assert audit['new_engineering_updates']==0 and audit['new_test_evaluations']==0
 old=rd(Path(m['origin_job'])/'contract.json');old3=Path(old['migration']['origin_job'])
 q=rd(old3/'qualification.json');r=rd(old3/'resume_qualification.json')
 assert q['status']=='QUALIFIED' and q['restored_after_all_probes'] and q['actual_disk_roundtrip']
 assert r['status']=='PASS' and all(r[k] for k in ('fresh_process','state_restoration_bitwise_equal','losses_close','indices_equal','rng_sequence_equal'))
 for row in rows(c):assert [x for x in rd(REGISTRY) if x['path']==row['path']]==[row]
 return dict(status='READY_INDEPENDENTLY_VERIFIED',contract_sha256=sha(j/'contract.json'),migration_cpu_review_sha256=sha(j/'budget_migration_cpu_review.json'),origin_checkpoint_sha256=m['checkpoint']['sha256'],reused_origin_qualification_updates=8,reused_click_repair_updates=69,new_engineering_updates=0,capacity=capacity(c,True),continuation_bitwise_guaranteed=False)
