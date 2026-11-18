"""Explicit Public ReLiF contract, shared resource and recovery IO bindings."""
import contextlib
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from relif_binding import ReLiFBinding, BY_CONTRACT_SHA, BUDGET_SHA
from relif_state import counts, plan, require, validate_payload
from relif_storage import RecoveryStore, BoundedSink, read_pin, safe_file, syncdir

ROOT = Path('__RECORDED_PROJECT_ROOT__')
HERE = Path(__file__).resolve().parent
OLD = ROOT/'expes/semi05_gate/main_table_20261019_r4'
SHARED = ROOT/'expes/work_dir/semi05_gate/MAIN_TABLE_FIXED_FINAL_20261019'
BASE = SHARED/'public_ncase_baselines_20261024'
STORE = ROOT/'expes/artifacts/FCD_A_MATRIX_20261006/MAIN_TABLE_FIXED_FINAL_20261019/public_ncase_baselines_20261024'
SPLITS = ROOT/'expes/semi05_gate/public_ncase_20261019/splits'
LEASE = SHARED/'native_gpu1_r4.lock'
UUID = 'GPU_CONFIGURED_AT_RUNTIME'
RAM = 12*1024**3
FINAL_BOUND = 32*1024**2
RECOVERY_BOUND = 512*1024**2
LOG_BOUND = 128*1024**2
TORCH_ROOT=Path('__RECORDED_HOME__/anaconda3/envs/sam3d/lib/python3.10/site-packages/torch')
WARNING_SOURCES=[dict(path=str(TORCH_ROOT/'nn/modules/lazy.py'),sha256='6ffedd0102a0d4a1f93d0e2f58fa880379f79ae70c01dd9ccdaf388acd3bed4f'),
    dict(path=str(TORCH_ROOT/'nn/functional.py'),sha256='f66a758d3f97f47b74291f3bdd945b6f9ce9be43b4e79a3bf0891f266abaea6c')]
sys.path.insert(0, str(OLD))
from budget_protocol import Budget
from runtime_data import bind_case_files, schedule
from runtime_resources import REGISTRY, resource_backend, sha


def read(path):
    return json.loads(Path(path).read_bytes())


def known_warning_records():
    # Original LazyLinear and KL(mean) warnings only, bound to installed library
    # bytes and the exact AST warning sites. No global or local warning filter.
    prefixes=('Lazy modules are a new feature under heavy development ',"reduction: 'mean' divides the total loss ")
    records=[]
    for source,prefix in zip(WARNING_SOURCES,prefixes):
        path=Path(source['path']);raw=path.read_bytes()
        require(hashlib.sha256(raw).hexdigest()==source['sha256'],'Warning source changed')
        nodes=[n for n in ast.walk(ast.parse(raw)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
            and isinstance(n.func.value,ast.Name) and n.func.value.id=='warnings' and n.func.attr=='warn'
            and n.args and isinstance(n.args[0],ast.Constant) and isinstance(n.args[0].value,str)
            and n.args[0].value.startswith(prefix)]
        require(len(nodes)==1,'Warning AST site differs');node=nodes[0]
        records.append(('%s:%d: UserWarning: %s\n  %s\n'%(path,node.lineno,node.args[0].value,
            raw.decode().splitlines()[node.lineno-1].strip())).encode())
    return records


def check_log_bytes(raw,records,allow_partial=False):
    require(type(raw) is bytes,'Log must be raw bytes')
    remaining=raw;used=[]
    while remaining:
        match=next((i for i,r in enumerate(records) if i not in used and remaining.startswith(r)),None)
        if match is not None:
            remaining=remaining[len(records[match]):];used.append(match);continue
        require(allow_partial and any(i not in used and r.startswith(remaining) for i,r in enumerate(records)),
                'Unknown stdout/stderr; exact pinned native warnings only')
        remaining=b''
    return dict(approved_warning_records=len(used),empty=not raw,partial_allowed=allow_partial)


def check_logs(raw,allow_partial=False):
    return check_log_bytes(raw,known_warning_records(),allow_partial)


def atomic(path, value):
    path = Path(path)
    partial = path.with_name(path.name+'.partial')
    require(not path.is_symlink() and not partial.exists(), 'Unsafe/existing atomic evidence temporary')
    with partial.open('x') as stream:
        json.dump(value, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    syncdir(path.parent)


def exclusive(path, value):
    path = Path(path)
    require(not path.exists() and not any(p.is_symlink() for p in (path,*path.parents)), 'Evidence exists/symlink')
    with path.open('x') as stream:
        json.dump(value, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    syncdir(path.parent)


def paths(labels, tag):
    plan(labels)
    require(isinstance(tag,str) and re.fullmatch(r'r[1-9][0-9]*',tag), 'Invalid job tag')
    relative = Path('relif3d')/'Public433'/('lab%03d'%labels)/'fold2'/tag
    return BASE/relative, STORE/relative/'final.pth'


def job(c):
    return Path(c['output']).parent


def contract_sha(c):
    return sha(job(c)/'contract.json')


def binding(c):
    template = read(HERE/'template.json')
    require(template['origin_contract']['sha256'] == BY_CONTRACT_SHA, 'BY origin differs')
    return ReLiFBinding(c['labeled_cases'], native_dir=OLD,
        by_contract=template['origin_contract']['path'], budget_file=HERE/'budget11.json', split_dir=SPLITS)


def clock_budget(labels):
    plan(labels)
    return Budget('Public433',labels,labels,346-labels,cycles=36)


def rows(c):
    base = Path(c['target']).parent
    return [dict(path=str(base/'final.pth'),bound=FINAL_BOUND)]+[
        dict(path=str(base/name/'final.pth'),bound=RECOVERY_BOUND)
        for name in ('recovery_a','recovery_b','qualification_warmup','qualification_larc')]+[
        dict(path=str(base/'ledger_stdout_capacity/final.pth'),bound=LOG_BOUND)]


def resource(c):
    guard = resource_backend(Path(c['target']), FINAL_BOUND)
    prior = guard.targets
    def targets():
        result = prior()
        seen = {str(p):b for p,b in result}
        require(len(result)==len(seen), 'Duplicate upstream resource reservation')
        for row in read(SHARED/'cpac_auxiliary_targets_20261022.json'):
            path = Path(row['path'])
            require(SHARED/'cpac_paper_20261022' in path.parents
                and path.name in ('steps.jsonl','epochs.jsonl','roundtrip.pth')
                and type(row['bound']) is int and 0<row['bound']<=10*1024**3
                and str(path) not in seen, 'Invalid/duplicate CPAC auxiliary reservation')
            result.append((path,row['bound']))
            seen[str(path)] = row['bound']
        for row in rows(c):
            if row['path'] in seen:
                require(seen[row['path']] == row['bound'], 'Own resource reservation differs')
            else:
                result.append((Path(row['path']),row['bound']))
        return result
    guard.targets = targets
    return guard


def capacity(c, start=False):
    guard = resource(c)
    with guard.original_lock():
        result = guard.capacity_snapshot(RAM)
        if start:
            require(all(row['margin_bytes']>=2*1024**3 for row in result['filesystems']),
                    'ReLiF startup requires extra 2GiB after all reserved growth')
    return result


def capacity_locked(c):
    return resource(c).capacity_snapshot(RAM)


def enroll(c):
    guard = resource(c)
    with guard.original_lock():
        result = guard.capacity_snapshot(RAM)
        require(all(r['margin_bytes']>=2*1024**3 for r in result['filesystems']), 'Startup margin insufficient')
        registry = read(REGISTRY)
        for row in rows(c):
            existing = [r for r in registry if r['path']==row['path']]
            if existing:
                require(existing==[row], 'Conflicting registered target')
            else:
                registry.append(row)
        atomic(REGISTRY, registry)
    return result


def verify(c):
    require((c['method'],c['dataset'],c['fold'],c['seed'],c['gpu'])==('relif3d','Public433',2,2027,1),
            'Wrong scientific job identity')
    require(c['labels_are_case_counts_not_percent'] is True and c['labeled_cases']==c['ratio'], 'Not case counts')
    bound = binding(c)
    require(c['recipe']==bound.recipe and c['splits']==bound.splits
            and c['primary_pairs']==bound.recipe['ssl_iterations'], 'Cutoff/splits differ')
    require(paths(c['labeled_cases'],c['tag'])==(job(c),Path(c['target'])), 'Wrong job/artifact paths')
    require(c['early_stop'] is False and c['test_evaluations']==1
            and c['selection']=='terminal_after_fixed_exposure', 'Selection changed')
    require(c['recovery_interval']==100 and c['recovery_bound']==RECOVERY_BOUND
            and c['checkpoint_bound']==FINAL_BOUND and c['log_bound']==LOG_BOUND, 'Storage bounds changed')
    require(c['schedule_policy']=='retain_original_36_cycle_lr_ramp_clocks; cutoff_only', 'Clock policy changed')
    require(c['case_files']==bind_case_files('Public433',c['splits']), 'Patient file identities changed')
    source_paths = [row['path'] for row in c['sources']]
    require(len(source_paths)==len(set(source_paths)), 'Duplicate source')
    required = {str(p) for p in HERE.glob('*.py')} | {str(HERE/'template.json'),str(HERE/'budget11.json')}
    require(required<=set(source_paths), 'Unbound new source file')
    for row in c['sources']+c['inputs']:
        path=Path(row['path'])
        require(not any(p.is_symlink() for p in (path,*path.parents)) and sha(path)==row['sha256'],
                'Bound source/input changed: '+str(path))
    template=read(HERE/'template.json')
    origin=template['origin_contract']
    require(sha(origin['path'])==origin['sha256']==BY_CONTRACT_SHA, 'Original BY contract changed')
    old=read(origin['path'])
    require(c['relif_recipe']==old['relif_recipe'] and all(r in c['sources'] for r in old['sources']+WARNING_SOURCES),
            'Original ReLiF scientific source/recipe replaced')
    foundation=c['relif_recipe']['original_sam'];path=Path(foundation['path'])
    require(foundation['variant']=='vit_b_ori' and foundation['model_family']=='SAM-Med3D_original'
            and path.stat().st_size==foundation['bytes'] and sha(path)==foundation['sha256'], 'Original SAM differs')
    require(sha(HERE/'budget11.json')==BUDGET_SHA, 'Frozen exposure budget changed')
    from relif_native import pinned_sources
    pinned_sources(c['relif_recipe']['official_code_root'])
    for proof in template['cpu_proofs']:
        require(sha(proof['path'])==proof['sha256'] and read(proof['path']).get(proof.get('result_field','passed'))==proof.get('expected_result',True),
                'CPU preparation evidence missing/changed')
    if c.get('qualification_origin'):
        from qualification_migration import verify_origin
        verify_origin(c)


def gpu_idle():
    fields=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid,pci.bus_id',
                                   '--format=csv,noheader'],universal_newlines=True).strip().split(',')
    require(fields[0].strip()==UUID and fields[1].strip().endswith('B3:00.0'), 'GPU1 identity changed')
    processes=subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid',
                                       '--format=csv,noheader'],universal_newlines=True).strip()
    require(not processes,'GPU1 busy')


def lease_open():
    require(not LEASE.is_symlink() and LEASE.read_bytes()==b'NATIVE_MAIN_TABLE_R4_GPU1_EXCLUSIVE_OPERATION_V1\n',
            'GPU1 lease identity changed')
    fd=os.open(LEASE,os.O_RDONLY|os.O_NOFOLLOW)
    a,b=os.fstat(fd),LEASE.stat()
    require((a.st_dev,a.st_ino)==(b.st_dev,b.st_ino),'Lease path replaced')
    return fd


@contextlib.contextmanager
def leased(c):
    import fcntl
    fd=None
    try:
        with resource(c).original_lock():
            fd=lease_open();fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            capacity_locked(c);gpu_idle()
        yield fd
    finally:
        if fd is not None:os.close(fd)


def assert_worker_lease(c):
    import fcntl
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==UUID,'Wrong CUDA-visible GPU')
    require(os.environ.get('PYTHONWARNINGS','')=='','Unreviewed warning suppression environment')
    expected=LEASE.stat();found=False
    for path in Path('/proc/self/fd').iterdir():
        try:s=path.stat()
        except FileNotFoundError:continue
        found |= (s.st_dev,s.st_ino)==(expected.st_dev,expected.st_ino)
    require(found,'Inherited lease FD missing')
    competitor=lease_open()
    try:
        try:fcntl.flock(competitor,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        raise ValueError('GPU1 lease competition unexpectedly succeeded')
    finally:os.close(competitor)


def resource_status(c,value):
    atomic(job(c)/'resource_status.json',value)


def store(c,torch=None):
    original=resource(c)
    class Guard:
        original_lock=staticmethod(original.original_lock)
        capacity_snapshot=staticmethod(lambda: original.capacity_snapshot(RAM))
    pairs=schedule(clock_budget(c['labeled_cases']),c['splits'])
    return RecoveryStore(job(c),STORE,[Path(r['path']) for r in rows(c)[1:3]],RECOVERY_BOUND,LOG_BOUND,
        contract_sha(c),Guard(),lambda state:validate_payload(state,contract_sha(c),c['recipe'],'formal',pairs),
        lambda state,stream:torch.save(state,stream),lambda stream:torch.load(stream,map_location='cpu'))


def output_check(c,output,role):
    expected=[Path(c['output'])] if role=='formal' else [job(c)/('qualification_'+role),job(c)/('probe_'+role)]
    require(role in ('formal','engineering_warmup','engineering_larc') and Path(output) in expected,
            'Engineering/formal output isolation differs')


def record(c,output,name,value,role,consistent=False):
    output_check(c,output,role)
    relative=(Path(output)/name).relative_to(job(c))
    return store(c).record(str(relative),value,consistent)


def append_step(c,output,value,role):
    output_check(c,output,role)
    if role=='formal':return store(c).append_step(value)
    path=safe_file(Path(output)/'steps.jsonl',job(c))
    raw=(json.dumps(value,allow_nan=False)+'\n').encode()
    require(len(raw)<=8192,'Engineering receipt exceeds bound')
    with resource(c).original_lock():
        capacity_locked(c);before=read_pin(path,job(c),True)
        require(not before['raw'] or before['raw'].endswith(b'\n'),'Partial engineering ledger')
        require(before['bytes']+len(raw)<=LOG_BOUND,'Engineering log bound exceeded')
        with path.open('ab') as stream:
            s=os.fstat(stream.fileno())
            if before['inode'] is not None:
                require((s.st_dev,s.st_ino,s.st_size)==(before['device'],before['inode'],before['bytes']),
                        'Engineering ledger replaced')
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
        syncdir(path.parent)


def ledger_pin(c):
    path=Path(c['output'])/'steps.jsonl'
    if not path.exists():
        with resource(c).original_lock():
            capacity_locked(c)
            require(not path.exists() and not store(c)._commits(),'Missing ledger after a committed generation')
            with path.open('xb') as stream:stream.flush();os.fsync(stream.fileno())
            syncdir(path.parent)
    return store(c).ledger_pin()


def save_checkpoint(c,state,torch):
    return store(c,torch).checkpoint(state)


def verify_actual_ledger(c,actual_rows,progress):
    from relif_ledger import scan
    pin=read_pin(Path(c['output'])/'steps.jsonl',job(c),True)
    result=scan(pin['raw'],binding(c))
    require(result['counts']==progress and result['partial_bytes']==0,'Durable training ledger differs')
    rows=[json.loads(line)['step']['actual_pair'] for line in pin['raw'].splitlines()]
    require(rows==actual_rows,'Durable actual pairs differ from native execution')


def write_tensor(c,target,payload,bound,torch):
    target=safe_file(target,STORE);partial=safe_file(Path(str(target)+'.partial'),STORE)
    require(dict(path=str(target),bound=bound) in rows(c),'Unreserved tensor output')
    with resource(c).original_lock():
        cap=capacity_locked(c)
        require(not target.exists() and not partial.exists(),'Tensor artifact already exists')
        target.parent.mkdir(parents=True,exist_ok=True)
        with partial.open('xb') as stream:
            torch.save(payload,BoundedSink(stream,bound));stream.flush();os.fsync(stream.fileno())
        pin=read_pin(partial,STORE)
        os.link(partial,target)
        require(target.stat().st_ino==partial.stat().st_ino and target.stat().st_nlink==2,'Tensor link differs')
        partial.unlink();syncdir(target.parent)
    return dict(path=str(target),bytes=pin['bytes'],sha256=pin['sha256'],resource_lock_verified=True,capacity_before=cap)


def load_tensor(c,artifact,torch):
    target=safe_file(artifact['path'],STORE)
    require(str(target) in [r['path'] for r in rows(c)],'Unbound tensor source')
    with resource(c).original_lock():
        capacity_locked(c)
        with target.open('rb') as stream:
            before=os.fstat(stream.fileno());digest=hashlib.sha256()
            for chunk in iter(lambda:stream.read(8*1024**2),b''):digest.update(chunk)
            require(before.st_size==artifact['bytes'] and digest.hexdigest()==artifact['sha256'],'Tensor SHA differs')
            stream.seek(0);value=torch.load(stream,map_location='cpu');after=os.fstat(stream.fileno());named=target.stat()
            identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
            require(identity(before)==identity(after)==identity(named),'Tensor changed during same-FD load')
    return value


def save_terminal(c,payload,torch):
    require(payload['execution_role']=='formal' and payload['contract_sha256']==contract_sha(c)
            and payload['counts']==counts(c['recipe']['ssl_iterations']),'Nonterminal/formal payload')
    return write_tensor(c,c['target'],payload,FINAL_BOUND,torch)


def load_terminal(c,artifact,torch):
    require(artifact['path']==c['target'],'Wrong terminal artifact')
    return load_tensor(c,artifact,torch)


TEST_NAMES=('final_test_intent.json','native_test_started.json','final_test_cases.jsonl','final_metrics.json')


def guard_native_test_lineage(c):
    for root in (Path(c['output']),job(c)/'recovery_history'):
        if root.exists():
            for path in root.rglob('*'):
                require(not any(name in path.name for name in TEST_NAMES[1:]),'Previous native test evidence: '+str(path))


def guard_test_lineage(c):
    guard_native_test_lineage(c)
    for root in (Path(c['output']),job(c)/'recovery_history'):
        if root.exists():
            require(not any('final_test_intent.json' in p.name for p in root.rglob('*')),'Previous test intent')


def guard_run_lineage(c,resume):
    guard_test_lineage(c)
    if resume:
        from resume_review import require_consumed_review
        require_consumed_review(c)
    else:
        require(not Path(c['output']).exists() and not (job(c)/'recovery_history').exists(),'Existing formal lineage')


def verify_test_prerequisites(c):
    sys.path.insert(0,str(ROOT))
    import expes.common.seg_metrics as metrics
    expected=[r for r in c['sources'] if r['path']==str(ROOT/'expes/common/seg_metrics.py')]
    require(len(expected)==1 and Path(metrics.__file__).resolve()==ROOT/'expes/common/seg_metrics.py'
            and sha(metrics.__file__)==expected[0]['sha256'],'Metrics module import differs')


def resume_intent_name(c):
    from resume_review import require_consumed_review
    return 'resume_intent_'+require_consumed_review(c)['event']+'.json'


def load_reviewed_checkpoint(c,torch):
    from resume_review import require_consumed_review
    review=require_consumed_review(c)
    candidates=store(c).available()
    matches=[x for x in candidates if x['sha256']==review['checkpoint_sha256']]
    require(len(matches)==1,'Reviewed checkpoint no longer valid')
    state=store(c,torch).load_reviewed(matches[0],review)
    return matches[0],state,dict(independent_resume_review_consumed=True)


def record_restoration(c,commit,state,progress):
    from recovery_state import tree_hash
    exclusive(job(c)/('restoration_verified_%d.json'%os.getpid()),dict(contract_sha256=contract_sha(c),
        checkpoint_sha256=commit['sha256'],loaded_state_exact=True,state_sha256=tree_hash(state),counts=progress,
        ledger=commit['ledger'],new_optimization_updates=0,new_test_evaluations=0))
