"""Explicitly reviewed restart only; preserve every uncommitted tail before rollback."""
from public_protocol import *

def validate_prefix(raw,pin):
 assert len(raw)>=pin['bytes'] and hashlib.sha256(raw[:pin['bytes']]).hexdigest()==pin['sha256'],'Recovery prefix mismatch'
 return raw[pin['bytes']:]

def reconcile(c,j):
 review=read(j/'resume_review.json')
 assert review['contract_sha256']==sha(j/'contract.json')
 assert review['human_resume_authorized'] is True and review['unknown_errors_reviewed'] is True
 assert isinstance(review['reason'],str) and len(review['reason'])>10
 event=review['event'];assert event.isdigit()
 d=j/'recovery_history'/event
 assert not d.exists() and not (j/'formal/final_test_intent.json').exists()
 for dp in [j/'dispatch.json',*j.glob('dispatch_resume_*.json')]:
  if dp.exists():assert not Path('/proc') .joinpath(str(read(dp)['pid'])).exists(),'Previous worker still exists'
 record=checkpoints(c)[0]
 assert review['checkpoint_sha256']==record['sha256'] and review['checkpoint_iteration']==record['iteration']
 p=Path(c['output'])/'steps.jsonl';raw=p.read_bytes() if p.exists() else b''
 tail=validate_prefix(raw,record['ledger']);d.mkdir(parents=True)
 with (d/'uncommitted_steps.jsonl').open('xb') as f:f.write(tail);f.flush();os.fsync(f.fileno())
 moved=[]
 for path in [Path(c['output'])/'failure.json',*j.glob('worker_error_*.json'),*[Path(x['path']+'.partial') for x in rows(c)[:3]]]:
  if path.exists():
   assert not path.is_symlink();dest=d/(path.parent.name+'_'+path.name)
   assert not dest.exists();pin=file_pin(path);path.rename(dest);assert file_pin(dest)==pin
   moved.append(dict(path=str(path),preserved=str(dest),pin=pin))
 exclusive_json(d/'review.json',review)
 result=dict(event=event,checkpoint=record,uncommitted_bytes=len(tail),uncommitted_complete_rows=len(tail.splitlines()),
  uncommitted_sha256=hashlib.sha256(tail).hexdigest(),extra_compute='Preserved uncommitted complete/partial work; not zero',preserved=moved)
 exclusive_json(d/'reconciliation.json',result);syncdir(d)
 if p.exists():
  assert not p.is_symlink()
  with p.open('r+b') as f:f.truncate(record['ledger']['bytes']);f.flush();os.fsync(f.fileno())
 assert ledger(c)==record['ledger']
 return result
