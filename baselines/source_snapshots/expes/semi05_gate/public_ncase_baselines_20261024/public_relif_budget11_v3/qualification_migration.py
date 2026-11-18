"""Bounded correction of a qualification comparison predicate, not training.

The original r1 own 8-update qualification is retained. Its first 2-update
fresh probe used math.isclose's max(atol,rtol*scale), which was stricter than
the declared tensor/numpy additive rtol/atol rule. A new isolated contract may
reuse the exact original engineering tensors with only their contract context
rebound, then obtain its own fresh-process continuation evidence. All old files
and failed-update costs remain preserved. No formal state can use this route.
"""
import copy,math,time
from relif_protocol import *
from relif_state import validate_payload

SCIENCE=('method','dataset','fold','seed','gpu','labeled_cases','ratio','labels_are_case_counts_not_percent',
    'splits','case_files','recipe','primary_pairs','relif_recipe','recovery_interval','recovery_bound',
    'checkpoint_bound','log_bound','schedule_policy','selection','test_evaluations','early_stop','qualification_windows')
REVISED={'public_operator.py','admission.py','relif_protocol.py'}


def numeric_close(actual,reference):
    return math.isfinite(actual) and math.isfinite(reference) and abs(actual-reference)<=1e-5+1e-4*abs(reference)


def attach_origin(c,origin):
    old=read(origin['contract']['path'])
    if old['labeled_cases']==c['labeled_cases']:
        c['qualification_origin']=copy.deepcopy(origin)
        verify_origin(c)


def verify_origin(c):
    origin=c['qualification_origin'];old=read(origin['contract']['path']);j=Path(old['output']).parent
    for item in origin['evidence']+[origin['contract']]:
        require(sha(item['path'])==item['sha256'],'Preserved qualification evidence changed')
    require(all(c[k]==old[k] for k in SCIENCE),'Scientific contract differs from own qualification')
    require(old['method']=='relif3d' and old['labeled_cases']==5 and old['tag']=='r1' and c['tag']=='r2',
            'This bounded engineering migration only covers the inspected ReLiF5 r1 to r2')
    require(not Path(old['output']).exists() and not list(j.glob('dispatch*.json'))
            and not list(j.glob('launch_intent*.json')),'Origin has formal execution')
    for item in old['sources']+old['inputs']:require(sha(item['path'])==item['sha256'],'Origin bound source/input changed')
    original_source=Path(origin['source'])
    for p in original_source.glob('*.py'):
        if p.name not in REVISED:require(sha(p)==sha(HERE/p.name),'Unreviewed source change outside comparison/admission binding')
    q=read(j/'qualification.json')
    require(q['contract_sha256']==origin['contract']['sha256'] and q['actual_extra_engineering_updates']==8
        and q['new_formal_updates']==q['new_test_evaluations']==0,'Original own GPU qualification differs')
    for role,item in q['references'].items():
        require(sha(item['path'])==item['sha256'],'Original reference changed');ref=read(item['path']);cp=ref['checkpoint']
        require(ref['execution_role']==role and ref['contract_sha256']==origin['contract']['sha256']
            and sha(cp['path'])==cp['sha256'] and Path(cp['path']).stat().st_size==cp['bytes'],'Origin tensor/ref identity differs')
    failure=read(j/'resume_probe_warmup_failure.json')
    require(failure['error']=="ValueError('Fresh native continuation differs')" and failure['actual_progress']==counts(4),
            'Origin failure differs from inspected predicate rejection')
    require(not (j/'resume_probe_larc_intent.json').exists(),'Original LARC probe already attempted')
    ref=read(j/'qualification_engineering_warmup/reference.json')
    actual=[json.loads(x)['step'] for x in (j/'probe_engineering_warmup/steps.jsonl').read_bytes().splitlines()]
    require(len(actual)==2 and all(x['actual_pair']==y['actual_pair'] and x['native_receipt']['clock']==y['native_receipt']['clock']
        and set(x['losses'])==set(y['losses']) and all(numeric_close(x['losses'][k],y['losses'][k]) for k in x['losses'])
        for x,y in zip(actual,ref['steps'])),'Original measured continuation does not meet declared additive tolerance')
    require(any(not math.isclose(x['losses'][k],y['losses'][k],rel_tol=1e-4,abs_tol=1e-5)
        for x,y in zip(actual,ref['steps']) for k in x['losses']),'Not the inspected max-versus-sum predicate failure')
    for name in ('qualify.log','probe_warmup.log'):check_logs((j/name).read_bytes())
    return old,q


def migrate_engineering_identity(c,state,ref,pairs):
    origin=c.get('qualification_origin');require(origin,'No bound engineering migration')
    identity=origin['contract']['sha256'];role=ref['execution_role']
    require(role in ('engineering_warmup','engineering_larc') and ref['origin_contract_sha256']==identity,
            'Formal/foreign checkpoint cannot enter engineering migration')
    validate_payload(state,identity,c['recipe'],role,pairs)
    require(state['counts']==counts(2),'Only own two-update engineering snapshots may be migrated')
    migrated=copy.deepcopy(state);migrated['contract_sha256']=contract_sha(c)
    validate_payload(migrated,contract_sha(c),c['recipe'],role,pairs)
    return migrated


def import_qualification(c,j):
    verify(c);old,q=verify_origin(c)
    require(not (j/'qualification.json').exists() and not (j/'qualification_import_intent.json').exists()
            and not Path(c['output']).exists(),'Qualification import repeated or formal started')
    with leased(c):
        exclusive(j/'qualification_import_intent.json',dict(contract_sha256=contract_sha(c),at=time.time(),origin=c['qualification_origin']))
        imported={}
        for role,slot in [('engineering_warmup',3),('engineering_larc',4)]:
            reference=q['references'][role];ref=read(reference['path']);source=Path(ref['checkpoint']['path'])
            target=Path(rows(c)[slot]['path']);partial=Path(str(target)+'.partial')
            with resource(c).original_lock():
                capacity_locked(c);safe_file(target,STORE);safe_file(partial,STORE)
                require(not target.exists() and not partial.exists(),'Engineering target already exists')
                target.parent.mkdir(parents=True,exist_ok=True)
                before=read_pin(source,STORE)
                require(before['sha256']==ref['checkpoint']['sha256'] and before['bytes']==ref['checkpoint']['bytes']<=RECOVERY_BOUND,
                        'Origin engineering checkpoint changed before copy')
                with source.open('rb') as src,partial.open('xb') as dst:
                    sink=BoundedSink(dst,RECOVERY_BOUND)
                    for chunk in iter(lambda:src.read(8*1024**2),b''):sink.write(chunk)
                    dst.flush();os.fsync(dst.fileno())
                require(read_pin(source,STORE)==before and sha(partial)==before['sha256'],'Engineering copy/source changed')
                os.link(partial,target);partial.unlink();syncdir(target.parent)
            ref['origin_contract_sha256']=ref['contract_sha256'];ref['contract_sha256']=contract_sha(c)
            ref['checkpoint']=dict(ref['checkpoint'],path=str(target))
            d=j/('qualification_'+role);d.mkdir(exist_ok=False);exclusive(d/'reference.json',ref)
            imported[role]=dict(path=str(d/'reference.json'),sha256=sha(d/'reference.json'))
        result=dict(q,contract_sha256=contract_sha(c),references=imported,imported_own_qualification=True,
            qualification_origin=c['qualification_origin'],preserved_failed_engineering_updates=2,
            engineering_updates_performed_by_import=0,new_test_evaluations=0,at=time.time())
        exclusive(j/'qualification.json',result)
        source=Path(old['output']).parent/'qualify.log';require(not (j/'qualify.log').exists(),'Import log exists')
        (j/'qualify.log').write_bytes(source.read_bytes())
    return dict(status='OWN_QUALIFICATION_IMPORTED_FRESH_PROBES_PENDING',new_engineering_updates=0,
        reused_own_engineering_updates=8,preserved_failed_engineering_updates=2,contract_sha256=contract_sha(c))
