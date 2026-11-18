"""Explicit Public DualTeacher contract, shared resource and recovery IO bindings."""
import contextlib
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from dualteacher_binding import DualTeacherBinding, BY_CONTRACT_SHA, BUDGET_SHA
from dualteacher_budget import counts, plan, require
from dualteacher_state import validate_payload
from dualteacher_storage import RecoveryStore, BoundedSink, read_pin, safe_file, syncdir

ROOT = Path('__RECORDED_PROJECT_ROOT__')
HERE = Path(__file__).resolve().parent
OLD = ROOT/'expes/semi05_gate/dualteacher_fixed_final_20261021_v4_conda'
SHARED = ROOT/'expes/work_dir/semi05_gate/MAIN_TABLE_FIXED_FINAL_20261019'
BASE = SHARED/'public_ncase_baselines_20261024'
STORE = ROOT/'expes/artifacts/FCD_A_MATRIX_20261006/MAIN_TABLE_FIXED_FINAL_20261019/public_ncase_baselines_20261024'
SPLITS = ROOT/'expes/semi05_gate/public_ncase_20261019/splits'
LEASE = SHARED/'native_gpu1_r4.lock'
UUID = 'GPU_CONFIGURED_AT_RUNTIME'
RAM = 12*1024**3
FINAL_BOUND = 256*1024**2
RECOVERY_BOUND = 768*1024**2
LOG_BOUND = 128*1024**2
sys.path.insert(0, str(OLD))
from budget_protocol import Budget
from runtime_data import bind_case_files, schedule
from runtime_resources import REGISTRY, resource_backend, sha
from dualteacher_store_binding import DualTeacherStore
from dualteacher_native import public_core_ast



def read(path):
    return json.loads(Path(path).read_bytes())


def known_warning_records():
    # Startup output/warnings are captured into structured evidence and reviewed
    # separately. Worker stdout/stderr must be empty unless separately audited.
    return []


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
    relative = Path('dualteacher_sam')/'Public433'/('lab%03d'%labels)/'fold2'/tag
    return BASE/relative, STORE/relative/'final.pth'


def job(c):
    return Path(c['output']).parent


def contract_sha(c):
    return sha(job(c)/'contract.json')


def binding(c):
    template=read(HERE/'template.json');origin=template['origin_contract']
    require(origin['sha256']==BY_CONTRACT_SHA,'Own BY origin differs')
    return DualTeacherBinding(c['labeled_cases'],native_dir=OLD,by_contract=origin['path'],budget_file=HERE/'budget11.json',split_dir=SPLITS)


def clock_budget(labels):
    plan(labels)
    return Budget('Public433',labels,labels,346-labels,cycles=36)


def rows(c):
    base = Path(c['target']).parent
    return [dict(path=str(base/'final.pth'),bound=FINAL_BOUND)]+[
        dict(path=str(base/name/'final.pth'),bound=RECOVERY_BOUND)
        for name in ('recovery_a','recovery_b','qualification_warmup','qualification_sam')]+[
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
                    'DualTeacher startup requires extra 2GiB after all reserved growth')
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
    require((c['method'],c['dataset'],c['fold'],c['seed'],c['gpu'])==('dualteacher_sam','Public433',2,2027,1),'Wrong own scientific job')
    require(c['labels_are_case_counts_not_percent'] is True and c['labeled_cases']==c['ratio'],'Not case counts')
    bound=binding(c);require(c['recipe']==bound.recipe and c['splits']==bound.splits and c['primary_pairs']==bound.recipe['primary_inputs_per_stream'],'Cutoff/splits differ')
    require(paths(c['labeled_cases'],c['tag'])==(job(c),Path(c['target'])),'Wrong job/artifact paths')
    require(c.get('numerical_runtime_policy')==__import__('dualteacher_numerics').POLICY,'Numeric policy not bound')
    template=read(HERE/'template.json');repair=template['repair_lineage']
    require(c.get('repair_lineage')==(repair if c['labeled_cases']==5 else None),'Historical repair accounting differs')
    require(sha(repair['path'])==repair['sha256'] and read(repair['path'])['total_engineering_effective_updates']==20,'Prior failure evidence changed')
    numeric=template['numerical_operator_proof'];require(sha(numeric['path'])==numeric['sha256'],'Numeric proof changed')
    require(read(numeric['path'])['source_sha256']['dualteacher_numerics.py']==sha(HERE/'dualteacher_numerics.py'),'Numeric policy is not the tested source')
    for old_failure in template.get('additional_failed_attempts',[]):
        require(sha(old_failure['failure_path'])==old_failure['failure_sha256'],'Prior partial failure changed')
        if old_failure.get('completed_ledger_path'):
            require(sha(old_failure['completed_ledger_path'])==old_failure['completed_ledger_sha256'],'Prior completed engineering ledger changed')
    require(c['early_stop'] is False and c['test_evaluations']==1 and c['selection']=='terminal_after_fixed_exposure','Selection changed')
    require((c['recovery_interval'],c['recovery_bound'],c['checkpoint_bound'],c['log_bound'])==(100,RECOVERY_BOUND,FINAL_BOUND,LOG_BOUND),'Storage bounds changed')
    require(c['schedule_policy']=='retain_original_36_cycle_lr_ramp_clocks; cutoff_only','Clock policy changed')
    require(c['case_files']==bind_case_files('Public433',c['splits']),'Patient file bindings changed')
    sources=[r['path'] for r in c['sources']];require(len(sources)==len(set(sources)),'Duplicate source')
    required={str(p) for p in HERE.glob('*.py')}|{str(HERE/'template.json'),str(HERE/'budget11.json')}
    require(required<=set(sources),'Unbound new source')
    for row in c['sources']+c['inputs']:
        path=Path(row['path']);require(not any(p.is_symlink() for p in (path,*path.parents)) and sha(path)==row['sha256'],'Bound source/input changed: '+str(path))
    origin=read(HERE/'template.json')['origin_contract'];require(sha(origin['path'])==origin['sha256']==BY_CONTRACT_SHA,'Original own BY contract changed')
    old=read(origin['path']);require(c['foundation']==old['foundation'] and c['official_code_root']==old['official_code_root'] and c['native_validate_sha256']==old['native_validate_sha256'],'Original initialization/code differs')
    require(all(any(x['path']==row['path'] and x['sha256']==row['sha256'] for x in c['sources']) for row in old['sources']+old['environment']['files']),'Missing original source/environment closure')
    f=c['foundation'];path=Path(f['path']);require(f['model_family']=='SAM-Med3D_turbo' and f['bytes']==402163626 and path.stat().st_size==f['bytes'] and sha(path)==f['sha256']=='899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c','Original Turbo differs')
    require(c['dualteacher_recipe']==public_recipe(c),'Public native metadata differs')
    public_core_ast(OLD/'dualteacher_core.py');require(sha(HERE/'budget11.json')==BUDGET_SHA,'Frozen budget changed')
    for proof in read(HERE/'template.json')['cpu_proofs']:
        require(sha(proof['path'])==proof['sha256'] and read(proof['path']).get(proof['result_field'])==proof['expected_result'],'Own CPU evidence changed')


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
    return DualTeacherStore(job(c),STORE,[Path(r['path']) for r in rows(c)[1:3]],RECOVERY_BOUND,LOG_BOUND,contract_sha(c),Guard(),
        lambda state,stream:torch.save(state,stream),lambda stream:torch.load(stream,map_location='cpu'),
        binding(c),c['foundation'],c['dualteacher_recipe'],c['target'],FINAL_BOUND)


def output_check(c,output,role):
    expected=[Path(c['output'])] if role=='formal' else [job(c)/('qualification_'+role),job(c)/('probe_'+role)]
    require(role in ('formal','engineering_warmup','engineering_sam') and Path(output) in expected,
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
    from dualteacher_ledger import scan
    pin=read_pin(Path(c['output'])/'steps.jsonl',job(c),True);result=scan(pin['raw'],binding(c))
    require(result['counts']==progress and result['partial_bytes']==0,'Durable complete native ledger differs')
    pairs=[pair for line in pin['raw'].splitlines() for pair in json.loads(line)['step']['actual_pairs']]
    require(pairs==actual_rows,'Durable actual native pairs differ')


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
    return store(c,torch).save_terminal(payload)


def load_terminal(c,artifact,torch):
    return store(c,torch).load_terminal(artifact)


TEST_NAMES=('final_test_intent.json','native_test_started.json','final_test_cases.jsonl','final_metrics.json','dualteacher_final_test_started.json')


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


def native_recipe_module():
    import importlib.util
    path=OLD/'dualteacher_protocol.py'
    # SHA is rechecked against the own immutable BY contract before loading.
    origin=read(read(HERE/'template.json')['origin_contract']['path'])
    matches=[r for r in origin['sources'] if r['path']==str(path)]
    require(len(matches)==1 and sha(path)==matches[0]['sha256'],'Native recipe module differs')
    spec=importlib.util.spec_from_file_location('_public_dt_pinned_recipe',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def public_recipe(c):
    import copy
    native=native_recipe_module().recipe(c['official_code_root'],c['foundation'],c['labeled_cases'],c['native_validate_sha256'])
    r=plan(c['labeled_cases']);v=copy.deepcopy(native)
    v.update(dataset='Public433',ratio=c['labeled_cases'],labels_are_case_counts_not_percent=True,
        primary_pairs=r['primary_inputs_per_stream'],native_iterations=r['ssl_iterations'],original_clock_iterations=r['original_clock_iterations'],
        clock_policy='retain_original_36_cycle_lr_ramp_clocks; cutoff_only',exact_resume_supported=True)
    v['optimizer'].update(primary_student_updates=r['ssl_iterations'],op_step_calls=2*r['ssl_iterations'],op_effective_updates_expected=r['ssl_iterations'],all_native_effective_optimizer_updates_expected=2*r['ssl_iterations'])
    v['fixed_endpoint']['test_case_count']=87
    v['source_native_BY_recipe']=native
    return v


def load_native_backend(c):
    # Called explicitly only inside a GPU-leased, independently admitted worker.
    verify(c);assert_worker_lease(c);capacity(c)
    import importlib.util
    path=OLD/'dualteacher_backend.py';origin=read(read(HERE/'template.json')['origin_contract']['path'])
    matches=[r for r in origin['sources'] if r['path']==str(path)]
    require(len(matches)==1 and sha(path)==matches[0]['sha256'],'Native backend bytes changed')
    spec=importlib.util.spec_from_file_location('_public_dt_pinned_backend',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    require(Path(module.__file__)==path,'Different native backend loaded')
    from resumable_dualteacher import make_backend
    return make_backend(module,sys.modules[__name__])


def check_initialization_diagnostics(c,value):
    require(value.get('failed') is False and value.get('stderr')=='' and value.get('stdout')=='creating model SAM-Med3D\n','Unknown initialization output')
    path=Path('__RECORDED_HOME__/anaconda3/envs/sam3d/lib/python3.10/site-packages/timm/models/layers/__init__.py')
    expected=dict(message='Importing from timm.models.layers is deprecated, please import via timm.layers',category='FutureWarning',filename=str(path),lineno=48)
    matches=[r for r in c['sources'] if r['path']==str(path)];require(len(matches)==1 and sha(path)==matches[0]['sha256'],'Warning source not pinned')
    raw=path.read_bytes();sites=[n for n in ast.walk(ast.parse(raw)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='warnings' and n.func.attr=='warn' and n.lineno==48 and ast.dump(n)==ast.dump(ast.parse('warnings.warn(f\"Importing from {__name__} is deprecated, please import via timm.layers\", FutureWarning)',mode='eval').body)]
    require(len(sites)==1 and len(value['warnings'])<=1 and all(w==expected for w in value['warnings']),'Unknown initialization warning/site')
