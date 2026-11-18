"""SFR explicit metadata review and preserved two-phase rollback; never implicit."""
from sfr_protocol import *

def evidence_paths(c,j):
 out=Path(c['output'])
 return sorted(set([*j.glob('dispatch*.json'),*j.glob('boot*.log'),*j.glob('worker_error_*.json'),
  *[p for p in (out/'failure.json',out/'pretrain_steps.jsonl',out/'steps.jsonl',out/'pretrain_terminal.json',out/'coverage_verified.json') if p.exists()],
  *[Path(r['path']+'.partial') for r in rows(c) if Path(r['path']+'.partial').exists()]]))

def no_test(c,j):
 guard_test_lineage(c)
 for name in ('final_test_intent.json','sfr_test_started.json','final_test_cases.jsonl','final_metrics.json'):
  assert not (Path(c['output'])/name).exists(), 'Opened terminal evaluation cannot be replayed'
 for dp in j.glob('dispatch*.json'):assert not (Path('/proc')/str(read(dp)['pid'])).exists(), 'Prior worker remains alive'

def validate_checkpoint(c,cp):
 assert cp==checkpoints(c)[0], 'Only latest valid committed generation; older fallback needs separate review'
 current=ledgers(c)
 for name,pin in cp['ledgers'].items():
  p=Path(c['output'])/name;raw=p.read_bytes() if p.exists() else b''
  assert len(raw)>=pin['bytes'] and hashlib.sha256(raw[:pin['bytes']]).hexdigest()==pin['sha256']
  assert pin['bytes']==0 or raw[:pin['bytes']].endswith(b'\n')
 if prefix_path(c,'formal').exists() and cp['progress']['phase']=='prefix':
  assert cp['progress']['prefix']['iterations']==6000, 'Existing later prefix requires separately bounded artifact migration'
 return current

def create_review(c,j,checkpoint_sha,reason):
 assert isinstance(reason,str) and len(reason.strip())>=20
 assert isinstance(checkpoint_sha,str) and len(checkpoint_sha)==64
 verify(c);no_test(c,j)
 # No automatic error suppression. Unknown diagnostics must first receive their
 # own bounded review; a free-text reason cannot whitelist stderr.
 for p in j.glob('boot*.log'):assert p.read_bytes()==b'', 'Nonempty boot requires source-bound diagnostic review'
 cp=next(x for x in checkpoints(c) if x['sha256']==checkpoint_sha)
 current=validate_checkpoint(c,cp);event=str(time.time_ns())
 with leased(c):cap=capacity(c,True)
 v=dict(event=event,contract_sha256=contract_sha(c),checkpoint_sha256=checkpoint_sha,checkpoint=cp,
  engineering_review_passed=True,standing_human_authorization=True,reason=reason,
  original_ledgers=current,evidence={str(p):file_pin(p) for p in evidence_paths(c,j)},
  old_dispatches={str(p):file_pin(p) for p in j.glob('dispatch*.json')},
  capacity=cap,at=time.time(),new_model_loads=0,new_optimization_updates=0,new_test_evaluations=0)
 prior=j/'resume_review.json'
 if prior.exists():
  old=read(prior);history=j/'recovery_history'/old['event']/'reconciliation.json';assert history.exists()
  exclusive_json(j/('consumed_resume_review_'+old['event']+'.json'),old)
  atomic(prior,v)
 else:exclusive_json(prior,v)
 return v

def validate_review(c,j):
 r=read(j/'resume_review.json');assert r['contract_sha256']==contract_sha(c)
 assert r['engineering_review_passed'] and r['standing_human_authorization']
 assert not (j/'recovery_history'/r['event']).exists()
 no_test(c,j);cp=next(x for x in checkpoints(c) if x['sha256']==r['checkpoint_sha256'])
 assert cp==r['checkpoint'] and validate_checkpoint(c,cp)==r['original_ledgers']
 assert {str(p):file_pin(p) for p in evidence_paths(c,j)}==r['evidence']
 assert {str(p):file_pin(p) for p in j.glob('dispatch*.json')}==r['old_dispatches']
 return r

def reconcile(c,j):
 r=validate_review(c,j);cp=r['checkpoint'];d=j/'recovery_history'/r['event'];d.mkdir(parents=True,exist_ok=False)
 # Preserve all tail bytes and evidence before touching a live ledger path.
 preserved={}
 for name,pin in cp['ledgers'].items():
  p=Path(c['output'])/name;raw=p.read_bytes() if p.exists() else b'';tail=raw[pin['bytes']:]
  dest=d/('uncommitted_'+name)
  with dest.open('xb') as f:f.write(tail);f.flush();os.fsync(f.fileno())
  complete=[json.loads(x) for x in tail[:tail.rfind(b'\n')+1].splitlines()]
  preserved[name]=dict(bytes=len(tail),sha256=sha(dest),complete_rows=len(complete),
   extra_complete_updates=tail.count(b'\n'),partial_compute_not_assumed_zero=True,original_full_pin=r['original_ledgers'][name])
  preserved[name]['extra_complete_labeled_inputs']=sum(x['step']['labeled_inputs'] for x in complete)
  preserved[name]['extra_complete_unlabeled_inputs']=sum(x['step']['unlabeled_inputs'] for x in complete)
 moved=[]
 for p in [Path(c['output'])/'failure.json',*j.glob('worker_error_*.json'),*[Path(x['path']+'.partial') for x in rows(c)]]:
  if p.exists():
   assert not p.is_symlink();dest=d/(p.parent.name+'_'+p.name);assert not dest.exists()
   pin=file_pin(p);p.rename(dest);assert file_pin(dest)==pin;moved.append(dict(original=str(p),preserved=str(dest),pin=pin))
 exclusive_json(d/'review.json',r)
 result=dict(event=r['event'],checkpoint=cp,uncommitted_ledgers=preserved,preserved_failures=moved)
 exclusive_json(d/'reconciliation.json',result);syncdir(d)
 for name,pin in cp['ledgers'].items():
  p=Path(c['output'])/name
  if p.exists():
   assert not p.is_symlink()
   with p.open('r+b') as f:f.truncate(pin['bytes']);f.flush();os.fsync(f.fileno())
 assert ledgers(c)==cp['ledgers'];return result
