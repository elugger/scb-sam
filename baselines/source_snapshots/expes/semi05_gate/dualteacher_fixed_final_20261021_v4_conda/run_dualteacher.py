"""Closed BY-only Dualteacher candidate. No admission before independent review.

Reuses source-derived original save lock and GPU1 lease controls. Existing
original methods cannot be selected. Static code is not actual qualification.
"""
import argparse,contextlib,json,os,re,stat,subprocess,sys,traceback
from pathlib import Path
sys.dont_write_bytecode=True
from budget_protocol import BUDGETS
from fixed_endpoint import Endpoint,FixedFinalRun,exclusive_json
from runtime_data import ROOT,bind_case_files,bound_splits
from runtime_resources import REGISTRY,RAM_REQUIRED,enroll_target,resource_backend,sha
from dualteacher_protocol import METHOD,FINAL_BOUND,recipe
HERE=Path(__file__).resolve().parent
DEPLOY=ROOT/'expes/semi05_gate/dualteacher_fixed_final_20261021_v4_conda'
BASE=ROOT/'expes/work_dir/semi05_gate/MAIN_TABLE_FIXED_FINAL_20261019'
STORE=ROOT/'expes/artifacts/FCD_A_MATRIX_20261006/MAIN_TABLE_FIXED_FINAL_20261019'
# Intentionally same original lease: never create a competing ownership domain.
LEASE=BASE/'native_gpu1_r4.lock'
LEASE_BYTES=b'NATIVE_MAIN_TABLE_R4_GPU1_EXCLUSIVE_OPERATION_V1\n'
PROTOCOL='DUALTEACHER_BY_FIXED_FINAL_CONDA_V4_20261021'
COMMON=('run_dualteacher.py','budget_protocol.py','fixed_endpoint.py','runtime_data.py','runtime_resources.py','dataset_pins.json','dualteacher_backend.py','dualteacher_protocol.py','dualteacher_core.py','dualteacher_util.py','supervisor_verify_dualteacher.py','supervisor_verify_common.py','supervisor_observe_dualteacher.py','admission_review.json')
RESOURCE_DEPENDENCIES=tuple(ROOT/('expes/semi05_gate/public_hpq_epochdiag_20261019/'+n) for n in ('hpq_contract.py','hpq_inputs.py','hpq_resources.py'))+(ROOT/'expes/semi05_gate/public_sup_20261018/public_recovery_guard_20261019.py',)

def need(ok, message):
    if not ok:
        raise ValueError(message)

def plain(path):
    path = Path(path)
    need(not any(p.is_symlink() for p in (path, *path.parents)), 'Symlink path rejected: ' + str(path))
    info = path.stat()
    need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'Not a single-link regular file: ' + str(path))
    return info

def unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result

def read_json(path):
    plain(path)
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + x)))

def binding(path):
    path = Path(path).absolute()
    if path==Path('__RECORDED_HOME__/anaconda3/envs/sam3d/lib/python3.10/site-packages/numpy/__init__.py'):
        need(not any(x.is_symlink() for x in (path,*path.parents)),'Framework source symlink rejected')
        info=path.stat();need(stat.S_ISREG(info.st_mode) and info.st_nlink==2,'Pinned read-only Conda NumPy source identity differs')
    else:info = plain(path)
    return dict(path=str(path), bytes=info.st_size, sha256=sha(path))

def source_rows(paths):
    return [binding(path) for path in sorted(set(Path(p).absolute() for p in paths))]

def resource(c):
    return resource_backend(Path(c['target']), targets(c)['final']['bound'])

def assert_registered(c):
    rows = read_json(REGISTRY)
    need(len({r['path'] for r in rows}) == len(rows), 'Duplicate shared reservation')
    for row in targets(c).values():
        need([r['bound'] for r in rows if r['path'] == row['path']] == [row['bound']], 'Exact native target not enrolled')

def assert_no_artifacts(c):
    parent = Path(c['target']).parent
    if parent.exists():
        need(not parent.is_symlink() and not any(parent.rglob('*')), 'Existing native artifact/residual; preserve')

def assert_job_files(job, allowed):
    for path in job.iterdir():
        need(path.name in allowed and not path.is_symlink(), 'Existing unexpected job control/residual: ' + str(path))
    for path in job.rglob('*'):
        need(not path.is_symlink() and 'fail' not in path.name.lower() and not path.name.endswith('.partial'),
             'Existing failure/partial; preserve: ' + str(path))

def lease_open(create=False):
    """Called under the original save lock; lock content is never rewritten."""
    if create and not os.path.lexists(LEASE):
        with LEASE.open('xb') as stream:
            stream.write(LEASE_BYTES)
            stream.flush()
            os.fsync(stream.fileno())
    info = plain(LEASE)
    need(LEASE.read_bytes() == LEASE_BYTES, 'Native GPU1 lease marker changed')
    fd = os.open(LEASE, os.O_RDONLY | os.O_NOFOLLOW)
    opened = os.fstat(fd)
    if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
        os.close(fd)
        raise ValueError('Lease file replaced while opening')
    return fd

def gpu_idle():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True).splitlines()
    ids = [r.split(',')[1].strip() for r in rows if int(r.split(',')[0]) == 1]
    need(len(ids) == 1, 'Physical GPU1 not uniquely identified')
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    need(not any(row.split(',')[0].strip() == ids[0] for row in processes.splitlines() if row.strip()), 'GPU1 is busy')

@contextlib.contextmanager
def qualification_lease(c):
    import fcntl
    fd = None
    try:
        guard = resource(c)
        with guard.original_lock():
            fd = lease_open()
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert_registered(c)
            guard.capacity_snapshot(RAM_REQUIRED)
            gpu_idle()
        yield
    finally:
        if fd is not None:
            os.close(fd)

def endpoint(c):
    return Endpoint(c['method'], c['primary_pairs'], c['primary_pairs'], 1, tuple(c['splits']['test']))

class PersistedFinalBackend:
    """Keep unchanged adapters; add durable reload and unique-test start evidence."""
    def __init__(self, backend, output):
        self.backend, self.output = backend, Path(output)
    def __getattr__(self, name):
        return getattr(self.backend, name)
    def strict_reload(self, artifact):
        receipt = self.backend.strict_reload(artifact)
        need(receipt.get('strict') is True and receipt.get('sha256') == artifact['sha256']
             and receipt.get('native_student_identity_verified') is True, 'Strict native reload failed')
        exclusive_json(self.output / 'native_reload_verified.json', dict(receipt=receipt, artifact=artifact, pid=os.getpid()))
        return receipt
    def test(self, artifact, permit):
        exclusive_json(self.output / 'native_final_test_started.json', dict(pid=os.getpid(),
                       final_sha256=artifact['sha256'], permit_sha256=sha(permit), test_evaluations=1))
        return self.backend.test(artifact, permit)

def qualify(job, c):
    allowed = {'prepare_intent.json', 'contract.json', 'preparation_bound.json'}
    assert_job_files(job, allowed)
    assert_no_artifacts(c)
    with qualification_lease(c):
        assert_job_files(job, allowed)
        exclusive_json(job / 'qualification_intent.json', dict(contract_sha256=sha(job / 'contract.json'), gpu=1))
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        backend = None
        try:
            (job / 'qualification_backend').mkdir(exist_ok=False)
            backend = backend_for(c, job / 'qualification_backend')
            result = backend.qualify(endpoint(c))
            need(result.get('status') == 'QUALIFIED' and result.get('actual_extra_engineering_updates') == 2
                 and result.get('new_test_evaluations') == 0 and result.get('restored_model_optimizer_rng') is True
                 and result.get('actual_disk_roundtrip') is True, 'Own native qualification incomplete')
            assert_no_artifacts_after_probe(c)
            result.update(contract_sha256=sha(job / 'contract.json'), gpu=1)
            exclusive_json(job / 'qualification.json', result)
            exclusive_json(job / 'ready.json', dict(status='READY', contract_sha256=sha(job / 'contract.json'),
                           qualification_sha256=sha(job / 'qualification.json'), gpu=1))
            return dict(status='READY', ready_sha256=sha(job / 'ready.json'), job=str(job))
        except BaseException as error:
            exclusive_json(job / 'qualification_failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
                           measured_extra_engineering_updates=getattr(backend, 'measured_extra_engineering_optimizer_updates', None),
                           formal_updates=0, formal_test_evaluations=0, failed_attempt_preserved=True))
            raise

def assert_no_artifacts_after_probe(c):
    # Successful qualification may leave empty stage directories, never files.
    parent = Path(c['target']).parent
    if parent.exists():
        need(all(p.is_dir() and not p.is_symlink() for p in parent.rglob('*')), 'Native probe/final/partial residual remains')

def ready_binding(job, expected):
    need(re.fullmatch(r'[0-9a-f]{64}', expected or '') and sha(job / 'ready.json') == expected, 'Ready SHA differs')
    ready = read_json(job / 'ready.json')
    need(ready.get('status') == 'READY' and ready.get('gpu') == 1
         and ready.get('contract_sha256') == sha(job / 'contract.json')
         and ready.get('qualification_sha256') == sha(job / 'qualification.json'), 'Ready chain differs')

def launch(job, c, contract_sha, ready_sha):
    import fcntl
    ready_binding(job, ready_sha)
    allowed = {'prepare_intent.json', 'contract.json', 'preparation_bound.json', 'qualification_intent.json',
               'qualification_backend', 'qualification.json', 'ready.json'}
    fd = None
    try:
        guard = resource(c)
        with guard.original_lock():
            assert_job_files(job, allowed)
            assert_no_artifacts_after_probe(c)
            assert_registered(c)
            guard.capacity_snapshot(RAM_REQUIRED)
            fd = lease_open()
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            gpu_idle()
            exclusive_json(job / 'launch_intent.json', dict(ready_sha256=ready_sha, contract_sha256=contract_sha))
            command = [sys.executable, str(HERE / 'run_dualteacher.py'), '--action', 'worker',
                       '--method', c['method'], '--dataset', c['dataset'], '--ratio', str(c['ratio']),
                       '--tag', c['tag'], '--gpu', '1', '--contract-sha', contract_sha,
                       '--ready-sha', ready_sha, '--lease-fd', str(fd)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT))
            with (job / 'boot.log').open('xb') as stream:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=stream,
                                           start_new_session=True, pass_fds=(fd,))
            exclusive_json(job / 'dispatch.json', dict(pid=process.pid, command=command, ready_sha256=ready_sha,
                           contract_sha256=contract_sha, native_lease=dict(device=os.fstat(fd).st_dev, inode=os.fstat(fd).st_ino)))
        return dict(status='DISPATCHED_ONCE', pid=process.pid, job=str(job))
    except BaseException as error:
        if (job / 'launch_intent.json').exists():
            exclusive_json(job / 'launch_failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
                           failed_attempt_preserved=True, do_not_relaunch=True))
        raise
    finally:
        if fd is not None:
            os.close(fd)

def worker(job, c, contract_sha, ready_sha, fd):
    import fcntl
    need(type(fd) is int and fd >= 0, 'Worker requires inherited native GPU1 lease')
    try:
        guard = resource(c)
        with guard.original_lock():
            info = plain(LEASE)
            need(LEASE.read_bytes() == LEASE_BYTES and (os.fstat(fd).st_dev, os.fstat(fd).st_ino) == (info.st_dev, info.st_ino),
                 'Worker inherited lease identity differs')
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            dispatch = read_json(job / 'dispatch.json')
            need(dispatch.get('pid') == os.getpid() and dispatch.get('command') == [sys.executable] + sys.argv
                 and dispatch.get('contract_sha256') == contract_sha and dispatch.get('ready_sha256') == ready_sha,
                 'Worker not the uniquely dispatched process')
            need(dispatch.get('native_lease') == dict(device=info.st_dev, inode=info.st_ino)
                 and read_json(job / 'launch_intent.json') == dict(ready_sha256=ready_sha, contract_sha256=contract_sha),
                 'Dispatch lease/unique launch intent binding differs')
            ready_binding(job, ready_sha)
            assert_job_files(job, {'prepare_intent.json', 'contract.json', 'preparation_bound.json',
                'qualification_intent.json', 'qualification_backend', 'qualification.json', 'ready.json',
                'launch_intent.json', 'dispatch.json', 'boot.log'})
            assert_no_artifacts_after_probe(c)
            assert_registered(c)
            guard.capacity_snapshot(RAM_REQUIRED)
            gpu_idle()
            need((job / 'boot.log').read_bytes() == b'', 'Nonempty boot before worker start; preserve')
            exclusive_json(job / 'started.json', dict(pid=os.getpid(), command=[sys.executable] + sys.argv,
                           contract_sha256=contract_sha, ready_sha256=ready_sha))
        backend = PersistedFinalBackend(backend_for(c, job / 'formal'), job / 'formal')
        result = FixedFinalRun(endpoint(c), job / 'formal').run(backend)
        exclusive_json(job / 'exited.json', dict(pid=os.getpid(), exit_code=0, status=result['status'],
                       contract_sha256=contract_sha, independent_final_acceptance_required=True))
    except BaseException as error:
        exclusive_json(job / 'worker_failure.json', dict(pid=os.getpid(), error=repr(error),
                       traceback=traceback.format_exc(), failed_attempt_preserved=True))
        raise
    finally:
        os.close(fd)

def job_paths(method,dataset,ratio,tag):
    need(method==METHOD and dataset=='BY' and ratio in (5,10),'Closed authorized BY-only scope differs')
    need(isinstance(tag,str) and re.fullmatch(r'r[1-9][0-9]*',tag),'Invalid independent attempt tag')
    rel=Path(METHOD)/'BY'/('semi%02d'%ratio)/'fold2'/tag
    return BASE/rel,STORE/rel/'final.pth'

def targets(c):return {'final':dict(path=c['target'],bound=FINAL_BOUND)}

def admission_review():
    a=read_json(HERE/'admission_review.json')
    need(a.get('approved_for_prepare') is True and a.get('method')==METHOD and a.get('dataset')=='BY','Candidate not admitted for prepare')
    need(a.get('official_commit')=='3710e3216db8bdb444296f0460eab9532b253848' and a.get('SAM_equation5_autodiff_repair_disclosed') is True,'Scientific source review incomplete')
    need(a.get('independent_verifier_sha256')==sha(HERE/'supervisor_verify_dualteacher.py'),'Own independent verifier not sealed')
    need(a.get('core_sha256')==sha(HERE/'dualteacher_core.py') and a.get('backend_sha256')==sha(HERE/'dualteacher_backend.py'),'Own source review stale')
    return a

def foundation_binding(path,expected):
    need(re.fullmatch(r'[0-9a-f]{64}',expected or '') and sha(path)==expected,'Independent Turbo pin SHA differs')
    v=read_json(path);p=Path(v['path'])
    need(p==ROOT/'ckpt/sam_med3d_turbo.pth' and v.get('method')==METHOD and v.get('model_family')=='SAM-Med3D_turbo','Own Turbo identity differs')
    need(v.get('bytes')==402163626 and v.get('sha256')=='899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c' and v.get('strict_cpu_verified') is True,'Own Turbo strict/source pin absent')
    need(binding(p)=={k:v[k] for k in ('path','bytes','sha256')},'Own Turbo file changed')
    return v

def contract_candidate(args):
    admission_review();job,target=job_paths(args.method,args.dataset,args.ratio,args.tag)
    need(Path(args.official_code_root).absolute()==HERE/'native','Must use own frozen complete source root')
    foundation=foundation_binding(args.foundation_pin,args.foundation_pin_sha)
    ep=Path(args.environment_pin).absolute();need(ROOT in ep.parents and sha(ep)==args.environment_pin_sha,'Independent environment pin SHA differs')
    environment=read_json(ep);need(environment.get('method')==METHOD and environment.get('torch_version')=='2.0.1+cu117','Own environment identity differs')
    need(environment.get('optimizer_zero_grad_set_to_none_default') is True and environment.get('python_no_user_site')=='1' and environment.get('torchvision_version')=='0.15.2+cu117','Coherent Conda runtime/OP semantics differ')
    budget=next(b for b in BUDGETS if b.dataset=='BY' and b.ratio==args.ratio);splits=bound_splits('BY',args.ratio)
    code=HERE/'native';vs=sha(code/'validate.py');need(vs=='38552d6d94d8d129e211aaeeddf4791827ece79f60cf2f631e3f1c4c04eb3650','Native inference source changed')
    source_paths=[HERE/n for n in COMMON]+[p for p in code.rglob('*') if p.is_file()]+list(RESOURCE_DEPENDENCIES)
    source_paths += [ROOT/'utils/cascade_train_utils.py',ROOT/'expes/common/seg_metrics.py']+[Path(x['path']) for x in environment['files']]
    c=dict(method=METHOD,dataset='BY',ratio=args.ratio,tag=args.tag,gpu=1,fold=2,seed=2027,target=str(target),output=str(job/'formal'),primary_pairs=budget.primary_pairs,
        checkpoint_bound=FINAL_BOUND,splits=splits,case_files=bind_case_files('BY',splits),foundation=foundation,foundation_pin_document=binding(args.foundation_pin),
        environment_pin_document=binding(ep),environment=environment,official_code_root=str(code),native_validate_sha256=vs,dualteacher_recipe=recipe(code,foundation,args.ratio,vs),
        native_gpu1_lease_path=str(LEASE),launcher_protocol=PROTOCOL,launcher_source=binding(HERE/'run_dualteacher.py'),sources=source_rows(source_paths),
        selection='terminal_after_fixed_exposure',test_evaluations=1,early_stop=False,exact_resume_supported=False,owner_thread_id='01a153a0-a178-7002-bf0b-d1e33ccbd472')
    verify_contract(c);return job,c

def verify_contract(c):
    admission_review();job,target=job_paths(c['method'],c['dataset'],c['ratio'],c['tag'])
    need(c['launcher_protocol']==PROTOCOL and c['launcher_source']==binding(HERE/'run_dualteacher.py'),'Launcher source/protocol differs')
    need(c['gpu']==1 and c['fold']==2 and c['seed']==2027 and c['owner_thread_id']=='01a153a0-a178-7002-bf0b-d1e33ccbd472','Closed owner/GPU/fold/seed differs')
    need(c['target']==str(target) and c['output']==str(job/'formal') and c['checkpoint_bound']==FINAL_BOUND and c['native_gpu1_lease_path']==str(LEASE),'Own target/output/lease differs')
    need(c['selection']=='terminal_after_fixed_exposure' and c['test_evaluations']==1 and c['early_stop'] is False and c['exact_resume_supported'] is False,'Fixed terminal policy differs')
    rows=c['sources'];need(len({x['path'] for x in rows})==len(rows) and source_rows(x['path'] for x in rows)==rows,'Source closure changed')
    required={str(HERE/n) for n in COMMON}|{str(p) for p in (HERE/'native').rglob('*') if p.is_file()}|{str(p) for p in RESOURCE_DEPENDENCIES}
    required|={str(ROOT/'utils/cascade_train_utils.py'),str(ROOT/'expes/common/seg_metrics.py')}
    for key in ('foundation_pin_document','environment_pin_document'):need(binding(c[key]['path'])==c[key],'Provenance document changed')
    need(c['environment']==read_json(c['environment_pin_document']['path']),'Environment pin changed')
    required|={x['path'] for x in c['environment']['files']};need(required.issubset({x['path'] for x in rows}),'Native dependency closure incomplete')
    need(foundation_binding(c['foundation_pin_document']['path'],c['foundation_pin_document']['sha256'])==c['foundation'],'Own Turbo binding differs')
    code=HERE/'native';need(c['official_code_root']==str(code) and c['native_validate_sha256']==sha(code/'validate.py'),'Own native source root differs')
    need(c['dualteacher_recipe']==recipe(code,c['foundation'],c['ratio'],c['native_validate_sha256']),'Prospective paper/source recipe differs')
    b=next(b for b in BUDGETS if b.dataset=='BY' and b.ratio==c['ratio']);need(c['primary_pairs']==b.primary_pairs,'Primary budget differs')
    splits=bound_splits('BY',c['ratio']);need(c['splits']==splits and bind_case_files('BY',splits)==c['case_files'],'Source case/input identity differs')

def backend_for(c,output):
    from dualteacher_backend import DualteacherBackend
    return DualteacherBackend(c,output,verify_contract)

def prepare(args):
    job,c=contract_candidate(args);need(not os.path.lexists(job),'Existing attempt, preserve');assert_no_artifacts(c)
    guard=resource(c)
    with guard.original_lock():
        guard.capacity_snapshot(RAM_REQUIRED);gpu_idle();fd=lease_open();os.close(fd)
        job.mkdir(parents=True,exist_ok=False);exclusive_json(job/'prepare_intent.json',dict(protocol=PROTOCOL,foundation_pin=c['foundation_pin_document']))
    try:
        capacities={'final':enroll_target(Path(c['target']),FINAL_BOUND)}
        with guard.original_lock():
            assert_registered(c);assert_no_artifacts(c);capacity=guard.capacity_snapshot(RAM_REQUIRED)
            exclusive_json(job/'contract.json',c)
            exclusive_json(job/'preparation_bound.json',dict(contract_sha256=sha(job/'contract.json'),source_files=len(c['sources']),capacity=capacity,enrollment=capacities,actual_gpu_qualified=False))
        return dict(status='BOUND_NEEDS_OWN_GPU_QUALIFICATION',job=str(job),contract_sha256=sha(job/'contract.json'))
    except BaseException as error:
        exclusive_json(job/'prepare_failure.json',dict(error=repr(error),traceback=traceback.format_exc(),failed_attempt_preserved=True));raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('prepare','qualify','launch','worker'),required=True)
    parser.add_argument('--method',choices=(METHOD,),required=True);parser.add_argument('--dataset',choices=('BY',),required=True)
    parser.add_argument('--ratio',type=int,choices=(5,10),required=True);parser.add_argument('--gpu',type=int,choices=(1,),default=1);parser.add_argument('--tag',default='r1')
    for n in ('official-code-root','foundation-pin','foundation-pin-sha','environment-pin','environment-pin-sha','contract-sha','ready-sha'):parser.add_argument('--'+n)
    parser.add_argument('--lease-fd',type=int);args=parser.parse_args();need(HERE==DEPLOY,'Only isolated own deployment accepted');need(os.environ.get('PYTHONNOUSERSITE')=='1','Isolated existing Conda dependencies required')
    job,_=job_paths(args.method,args.dataset,args.ratio,args.tag)
    if args.action=='prepare':
        need(all((args.official_code_root,args.foundation_pin,args.foundation_pin_sha,args.environment_pin,args.environment_pin_sha)),'Prepare needs own source/Turbo/environment documents')
        result=prepare(args)
    else:
        need(re.fullmatch(r'[0-9a-f]{64}',args.contract_sha or '') and sha(job/'contract.json')==args.contract_sha,'Independent contract SHA required')
        c=read_json(job/'contract.json');need((c['method'],c['dataset'],c['ratio'],c['tag'],c['gpu'])==(args.method,args.dataset,args.ratio,args.tag,1),'CLI/contract differs');verify_contract(c)
        if args.action=='worker':worker(job,c,args.contract_sha,args.ready_sha,args.lease_fd);return
        need(args.lease_fd is None,'Only unique worker may inherit lease')
        result=qualify(job,c) if args.action=='qualify' else launch(job,c,args.contract_sha,args.ready_sha)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)

if __name__=='__main__':main()
