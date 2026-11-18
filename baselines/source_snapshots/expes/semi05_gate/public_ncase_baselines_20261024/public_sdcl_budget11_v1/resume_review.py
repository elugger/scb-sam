"""Explicit audited recovery; preserve both uncommitted ledgers and all failures."""
from sdcl_protocol import *
def reconcile(c,j):
 review=read(j/'resume_review.json');assert review['contract_sha256']==contract_sha(c)
 assert review['engineering_review_passed'] and review['standing_human_authorization'] and len(review['reason'])>10
 event=review['event'];assert event.isdigit();d=j/'recovery_history'/event;assert not d.exists()
 guard_test_lineage(c)
 for dp in j.glob('dispatch*.json'):assert not (Path('/proc')/str(read(dp)['pid'])).exists()
 cp=next(x for x in checkpoints(c) if x['sha256']==review['checkpoint_sha256']);d.mkdir(parents=True)
 preserved={}
 for name,pin in cp['ledgers'].items():
  p=Path(c['output'])/name;raw=p.read_bytes() if p.exists() else b''
  assert len(raw)>=pin['bytes'] and hashlib.sha256(raw[:pin['bytes']]).hexdigest()==pin['sha256']
  tail=raw[pin['bytes']:];dest=d/('uncommitted_'+name)
  with dest.open('xb') as f:f.write(tail);f.flush();os.fsync(f.fileno())
  preserved[name]=dict(bytes=len(tail),sha256=sha(dest),complete_rows=tail.count(b'\n'),extra_complete_updates=2*tail.count(b'\n'),partial_compute_not_assumed_zero=True)
 moved=[]
 for p in [Path(c['output'])/'failure.json',*j.glob('worker_error_*.json'),*[Path(x['path']+'.partial') for x in rows(c)[:3]]]:
  if p.exists():
   assert not p.is_symlink();dest=d/(p.parent.name+'_'+p.name);assert not dest.exists();pin=file_pin(p);p.rename(dest);assert file_pin(dest)==pin;moved.append(dict(original=str(p),preserved=str(dest),pin=pin))
 exclusive_json(d/'review.json',review)
 result=dict(event=event,checkpoint=cp,uncommitted_ledgers=preserved,preserved_failures=moved)
 exclusive_json(d/'reconciliation.json',result);syncdir(d)
 for name,pin in cp['ledgers'].items():
  p=Path(c['output'])/name
  if p.exists():
   assert not p.is_symlink()
   with p.open('r+b') as f:f.truncate(pin['bytes']);f.flush();os.fsync(f.fileno())
 assert ledgers(c)==cp['ledgers'];return result
