"""Explicit one-use review of a dead ReLiF worker and preserved ledger tail."""
import time
from relif_protocol import *
from relif_ledger import project_file


def no_test_or_worker(c,j):
    guard_test_lineage(c)
    require(not Path(c['target']).exists(), 'Existing terminal artifact requires a separate bounded terminal repair')
    for p in j.glob('dispatch*.json'):
        require(not (Path('/proc')/str(read(p)['pid'])).exists(), 'Previous PID still present')


def evidence(c,j):
    paths=sorted(set([*j.glob('dispatch*.json'),*j.glob('boot*.log'),*j.glob('worker_error_*.json'),
        *Path(c['output']).glob('*.json'),Path(c['output'])/'steps.jsonl',
        *[Path(r['path']+'.partial') for r in rows(c) if Path(r['path']+'.partial').exists()]]))
    return {str(p):read_pin(p,STORE if STORE in p.parents else j) for p in paths if p.exists()}


def checkpoint_review(c,j,checkpoint_sha):
    cps=store(c).available()
    require(cps and cps[0]['sha256']==checkpoint_sha, 'Only newest valid generation may be resumed by this protocol')
    require(len(cps)==min(2,len(store(c)._commits())), 'A required committed generation is invalid')
    cp=cps[0]
    projection=project_file(Path(c['output'])/'steps.jsonl',j,cp,binding(c),contract_sha(c))
    return cp,projection


def create_review(c,j,checkpoint_sha,reason):
    require(isinstance(reason,str) and len(reason.strip())>=20,'Concrete technical review reason required')
    verify(c);no_test_or_worker(c,j)
    for p in j.glob('boot*.log'):
        check_logs(p.read_bytes())
    cp,projection=checkpoint_review(c,j,checkpoint_sha)
    with leased(c):cap=capacity(c,True)
    r=dict(event=str(time.time_ns()),contract_sha256=contract_sha(c),checkpoint_sha256=checkpoint_sha,
        generation=cp['generation'],checkpoint=cp,projection=projection,evidence=evidence(c,j),
        independent_resume_review_passed=True,reason=reason,capacity=cap,at=time.time(),
        new_model_loads=0,new_optimization_updates=0,new_test_evaluations=0)
    path=j/'resume_review.json'
    if path.exists():
        old=read(path);require((j/'recovery_history'/old['event']/'consumed.json').exists(), 'Previous review unconsumed')
        exclusive(j/('old_resume_review_'+old['event']+'.json'),old);atomic(path,r)
    else:exclusive(path,r)
    return r


def validate_review(c,j):
    r=read(j/'resume_review.json')
    require(r['contract_sha256']==contract_sha(c) and r['independent_resume_review_passed'] is True,
            'Resume review identity differs')
    require(not (j/'recovery_history'/r['event']).exists(),'Resume review already consumed')
    no_test_or_worker(c,j)
    cp,projection=checkpoint_review(c,j,r['checkpoint_sha256'])
    require(cp==r['checkpoint'] and projection==r['projection'] and evidence(c,j)==r['evidence'],
            'Review evidence changed')
    return r


def reconcile(c,j):
    r=validate_review(c,j);cp=r['checkpoint'];d=j/'recovery_history'/r['event']
    with resource(c).original_lock():
        capacity_locked(c);require(evidence(c,j)==r['evidence'],'Evidence raced review')
        d.mkdir(parents=True,exist_ok=False)
        ledger=Path(c['output'])/'steps.jsonl';pin=read_pin(ledger,j,True);raw=pin.pop('raw')
        require(pin==r['projection']['original_ledger_pin'],'Ledger changed before preservation')
        for name,data in (('original_steps.jsonl',raw),('uncommitted_steps.jsonl',raw[cp['ledger']['bytes']:])):
            with (d/name).open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
        require(sha(d/'original_steps.jsonl')==pin['sha256']
                and sha(d/'uncommitted_steps.jsonl')==r['projection']['entire_uncommitted_tail']['sha256'],
                'Preserved tail/full ledger differ')
        moved=[]
        for p in [Path(c['output'])/'failure.json',*j.glob('worker_error_*.json'),
                  *[Path(x['path']+'.partial') for x in rows(c)]]:
            if p.exists():
                root=STORE if STORE in p.parents else j;old=read_pin(p,root)
                dest=d/(p.parent.name+'_'+p.name);require(not dest.exists(),'Preserved path collision')
                p.rename(dest);require(read_pin(dest,j)==old,'Moved evidence identity changed')
                moved.append(dict(original=str(p),preserved=str(dest),pin=old))
        exclusive(d/'review.json',r)
        event=dict(event=r['event'],checkpoint=cp,projection=r['projection'],preserved_failures=moved,
                   original_ledger_sha256=sha(d/'original_steps.jsonl'))
        exclusive(d/'reconciliation.json',event);syncdir(d)
        with ledger.open('r+b') as stream:
            opened=os.fstat(stream.fileno())
            require((opened.st_dev,opened.st_ino)==(pin['device'],pin['inode']),'Ledger inode changed')
            stream.truncate(cp['ledger']['bytes']);stream.flush();os.fsync(stream.fileno())
        require(read_pin(ledger,j)==cp['ledger'],'Reconciled ledger does not match committed identity')
    return event


def bind_worker_pid(c,pid):
    r=read(job(c)/'resume_review.json');d=job(c)/'recovery_history'/r['event']
    require(read(d/'reconciliation.json')['checkpoint']==r['checkpoint'],'Missing reconciliation')
    exclusive(d/'consumed.json',dict(r,consumed_by_pid=pid))


def require_consumed_review(c):
    r=read(job(c)/'resume_review.json');d=job(c)/'recovery_history'/r['event']
    consumed=read(d/'consumed.json')
    require(consumed['consumed_by_pid']==os.getpid() and consumed['contract_sha256']==contract_sha(c)
            and {k:v for k,v in consumed.items() if k!='consumed_by_pid'}==r,'Resume review bound to another worker')
    require(read(d/'reconciliation.json')['checkpoint']==r['checkpoint'],'Reconciliation changed')
    return consumed


def wait_worker_binding(c):
    r=read(job(c)/'resume_review.json');p=job(c)/'recovery_history'/r['event']/'consumed.json'
    for _ in range(100):
        if p.exists():require_consumed_review(c);return
        time.sleep(.1)
    raise ValueError('Parent did not bind the one-use resume review within 10 seconds')
