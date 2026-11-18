"""Independent r4 candidate: closed SFR/ReLiF prepare/qualify/launch/worker.

Never imports or edits r2/r3's launcher and never substitutes Turbo. Preparation
requires an independently SHA-pinned original-foundation binding JSON and all
actual source/input files. Existing controls/artifacts are never resumed. This
candidate still requires deployment review, its own GPU qualification, and a
method-specific independent final verifier; CPU checks are not launch evidence.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import traceback

sys.dont_write_bytecode = True
from budget_protocol import BUDGETS
from fixed_endpoint import Endpoint, FixedFinalRun, exclusive_json
from runtime_data import ROOT, bind_case_files, bound_splits
from runtime_resources import REGISTRY, RAM_REQUIRED, enroll_target, resource_backend, sha

HERE = Path(__file__).resolve().parent
BASE = ROOT / 'expes/work_dir/semi05_gate/MAIN_TABLE_FIXED_FINAL_20261019'
STORE = ROOT / 'expes/artifacts/FCD_A_MATRIX_20261006/MAIN_TABLE_FIXED_FINAL_20261019'
LEASE = BASE / 'native_gpu1_r4.lock'
LEASE_BYTES = b'NATIVE_MAIN_TABLE_R4_GPU1_EXCLUSIVE_OPERATION_V1\n'
PROTOCOL = 'NATIVE_MAIN_TABLE_R4_V1_20261019'
COMMON = ('run_native_baseline.py', 'budget_protocol.py', 'fixed_endpoint.py', 'runtime_data.py',
          'runtime_resources.py', 'dataset_pins.json')
RELIF = ('relif_backend.py', 'relif_runtime.py', 'relif_native.py', 'relif_components.py', 'relif_official_pin.json')
RESOURCE_DEPENDENCIES = tuple(ROOT / ('expes/semi05_gate/public_hpq_epochdiag_20261019/' + name)
    for name in ('hpq_contract.py', 'hpq_inputs.py', 'hpq_resources.py')) + (
    ROOT / 'expes/semi05_gate/public_sup_20261018/public_recovery_guard_20261019.py',)


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
    info = plain(path)
    return dict(path=str(path), bytes=info.st_size, sha256=sha(path))


def job_paths(method, dataset, ratio, tag):
    need(method in ('sfr', 'relif3d') and dataset in ('BY', 'Public433') and ratio in (5, 10), 'Closed native scope differs')
    need(isinstance(tag, str) and re.fullmatch(r'r[1-9][0-9]*', tag), 'Invalid independent attempt tag')
    rel = Path(method) / dataset / ('semi%02d' % ratio) / 'fold2' / tag
    return BASE / rel, STORE / rel / 'final.pth'


def original_binding(path, expected_sha, method):
    need(isinstance(expected_sha, str) and re.fullmatch(r'[0-9a-f]{64}', expected_sha), 'Original pin document SHA required')
    need(sha(path) == expected_sha, 'Original pin document changed')
    value = read_json(path)
    weight = Path(value['path'])
    need(weight.is_absolute() and ROOT in weight.parents, 'Original weight must be in the authorized project')
    need(binding(weight) == {k: value[k] for k in ('path', 'bytes', 'sha256')}, 'Original weight absent/changed')
    if method == 'sfr':
        need(value.get('family') == 'original SAM 2D ViT-B' and value['bytes'] == 375042383
             and weight.name == 'sam_vit_b_01ec64.pth'
             and value.get('source_url') == 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
             'Original SFR SAM provenance differs')
    else:
        need(value.get('model_family') == 'SAM-Med3D_original' and value.get('variant') == 'vit_b_ori'
             and isinstance(value.get('provenance'), str) and value['provenance']
             and 'turbo' not in weight.name.lower(), 'Original SAM-Med3D provenance missing; no Turbo substitution')
    return value


def source_rows(paths):
    return [binding(path) for path in sorted(set(Path(p).absolute() for p in paths))]


def contract_candidate(args):
    """Read-only; missing original weights fail before job/target/lease creation."""
    job, target = job_paths(args.method, args.dataset, args.ratio, args.tag)
    code = Path(args.official_code_root).absolute()
    need(code.is_dir() and ROOT in code.parents and not code.is_symlink(), 'Official checkout missing/outside project')
    init = original_binding(args.foundation_pin, args.foundation_pin_sha, args.method)
    if args.method == 'sfr':
        from sfr_backend import prospective_contract
        c = prospective_contract(args.dataset, args.ratio, args.tag, code, init)
    else:
        from relif_backend import recipe, FINAL_BOUND
        from relif_native import pinned_sources
        pinned_sources(code)
        plan = next(b for b in BUDGETS if b.dataset == args.dataset and b.ratio == args.ratio)
        splits = bound_splits(args.dataset, args.ratio)
        c = dict(method='relif3d', dataset=args.dataset, ratio=args.ratio, tag=args.tag, gpu=1, fold=2, seed=2027,
                 target=str(target), output=str(job / 'formal'), primary_pairs=plan.primary_pairs,
                 splits=splits, case_files=bind_case_files(args.dataset, splits), checkpoint_bound=FINAL_BOUND,
                 relif_recipe=recipe(code, init), official_code_root=str(code),
                 official_source_paths=[str(p) for p in sorted(code.rglob('*.py'))],
                 selection='terminal_after_fixed_exposure', test_evaluations=1, early_stop=False,
                 exact_resume_supported=False,
                 comparison_scope='primary SSL 3D inputs only; auxiliary foundation/donor compute disclosed')
        paths = [HERE / name for name in COMMON + RELIF] + list(code.rglob('*.py'))
        paths += [ROOT / 'utils/cascade_train_utils.py', ROOT / 'expes/common/seg_metrics.py']
        c['sources'] = source_rows(paths)
    c['launcher_protocol'] = PROTOCOL
    c['launcher_source'] = binding(HERE / 'run_native_baseline.py')
    c['foundation_pin_document'] = binding(args.foundation_pin)
    c['native_gpu1_lease_path'] = str(LEASE)
    c['sources'] = source_rows([row['path'] for row in c['sources']] + [HERE / name for name in COMMON]
                               + list(RESOURCE_DEPENDENCIES))
    verify_contract(c)
    return job, c


def verify_contract(c):
    need(c.get('launcher_protocol') == PROTOCOL and c.get('launcher_source') == binding(HERE / 'run_native_baseline.py'),
         'Independent launcher source/protocol differs')
    job, target = job_paths(c['method'], c['dataset'], c['ratio'], c['tag'])
    need(c.get('gpu') == 1 and c.get('fold') == 2 and c.get('seed') == 2027 and c.get('target') == str(target)
         and c.get('output') == str(job / 'formal') and c.get('native_gpu1_lease_path') == str(LEASE), 'Native job identity differs')
    need(c.get('selection') == 'terminal_after_fixed_exposure' and c.get('test_evaluations') == 1
         and c.get('early_stop') is False and c.get('exact_resume_supported') is False, 'Terminal-only policy differs')
    plan = next(b for b in BUDGETS if b.dataset == c['dataset'] and b.ratio == c['ratio'])
    need(c['primary_pairs'] == plan.primary_pairs, 'Primary exposure budget differs')
    rows = c['sources']
    need(len({r['path'] for r in rows}) == len(rows) and source_rows(r['path'] for r in rows) == rows, 'Source inventory changed')
    need({str(HERE / name) for name in COMMON}.issubset({r['path'] for r in rows}), 'Shared source binding incomplete')
    need({str(p) for p in RESOURCE_DEPENDENCIES}.issubset({r['path'] for r in rows}), 'Original resource chain not source-bound')
    pin = c['foundation_pin_document']
    need(binding(pin['path']) == pin, 'Original provenance document changed')
    init = original_binding(pin['path'], pin['sha256'], c['method'])
    splits = bound_splits(c['dataset'], c['ratio'])
    need(splits == c['splits'] and bind_case_files(c['dataset'], splits) == c['case_files'], 'Actual split/case identity changed')
    if c['method'] == 'sfr':
        from sfr_backend import verify_sfr_contract
        need(c['foundation'] == init, 'SFR foundation binding differs')
        verify_sfr_contract(c)
    else:
        from relif_backend import recipe, FINAL_BOUND
        from relif_native import pinned_sources
        code = Path(c['official_code_root'])
        pinned_sources(code)
        need(c['relif_recipe'] == recipe(code, init) and c['checkpoint_bound'] == FINAL_BOUND, 'ReLiF original recipe differs')
        official = sorted(str(p) for p in code.rglob('*.py'))
        need(c['official_source_paths'] == official, 'ReLiF official source set changed')
        required = {str(HERE / name) for name in RELIF} | set(official)
        required |= {str(ROOT / 'utils/cascade_train_utils.py'), str(ROOT / 'expes/common/seg_metrics.py')}
        need(required.issubset({r['path'] for r in rows}), 'ReLiF source dependency missing')


def targets(c):
    return c['sfr_targets'] if c['method'] == 'sfr' else {'final': dict(path=c['target'], bound=c['checkpoint_bound'])}


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


def backend_for(c, output):
    if c['method'] == 'sfr':
        from sfr_backend import SFRBackend
        return SFRBackend(c, output, verify_contract)
    from relif_backend import ReLiFBackend
    return ReLiFBackend(c, output, verify_contract)


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


def prepare(args):
    job, c = contract_candidate(args)
    need(not os.path.lexists(job), 'Existing attempt; prepare is never repeated')
    assert_no_artifacts(c)
    job.mkdir(parents=True, exist_ok=False)
    try:
        exclusive_json(job / 'prepare_intent.json', dict(protocol=PROTOCOL, foundation_pin=c['foundation_pin_document']))
        if c['method'] == 'sfr':
            from sfr_backend import enroll_contract_targets
            capacities = enroll_contract_targets(c)
        else:
            capacities = {'final': enroll_target(Path(c['target']), c['checkpoint_bound'])}
        guard = resource(c)
        with guard.original_lock():
            assert_registered(c)
            assert_no_artifacts(c)
            fd = lease_open(create=True)
            os.close(fd)
            capacity = guard.capacity_snapshot(RAM_REQUIRED)
        exclusive_json(job / 'contract.json', c)
        exclusive_json(job / 'preparation_bound.json', dict(contract_sha256=sha(job / 'contract.json'),
                       source_files=len(c['sources']), capacity=capacity, enrollment=capacities, actual_gpu_qualified=False))
        return dict(status='BOUND_NEEDS_OWN_GPU_QUALIFICATION', job=str(job), contract_sha256=sha(job / 'contract.json'))
    except BaseException as error:
        exclusive_json(job / 'prepare_failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
                       failed_attempt_preserved=True, reservations_not_removed=True))
        raise


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
            if c['method'] == 'relif3d':
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
            command = [sys.executable, str(HERE / 'run_native_baseline.py'), '--action', 'worker',
                       '--method', c['method'], '--dataset', c['dataset'], '--ratio', str(c['ratio']),
                       '--tag', c['tag'], '--gpu', '1', '--contract-sha', contract_sha,
                       '--ready-sha', ready_sha, '--lease-fd', str(fd)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('prepare', 'qualify', 'launch', 'worker'), required=True)
    parser.add_argument('--method', choices=('sfr', 'relif3d'), required=True)
    parser.add_argument('--dataset', choices=('BY', 'Public433'), required=True)
    parser.add_argument('--ratio', type=int, choices=(5, 10), required=True)
    parser.add_argument('--gpu', type=int, choices=(1,), default=1)
    parser.add_argument('--tag', default='r1')
    parser.add_argument('--official-code-root')
    parser.add_argument('--foundation-pin')
    parser.add_argument('--foundation-pin-sha')
    parser.add_argument('--contract-sha')
    parser.add_argument('--ready-sha')
    parser.add_argument('--lease-fd', type=int)
    args = parser.parse_args()
    need(HERE == ROOT / 'expes/semi05_gate/main_table_20261019_r4', 'Deploy only to an independent frozen r4 source directory')
    job, _ = job_paths(args.method, args.dataset, args.ratio, args.tag)
    if args.action == 'prepare':
        need(args.official_code_root and args.foundation_pin and args.foundation_pin_sha, 'Prepare requires explicit original source/pin inputs')
        result = prepare(args)
    else:
        need(re.fullmatch(r'[0-9a-f]{64}', args.contract_sha or '')
             and sha(job / 'contract.json') == args.contract_sha, 'Independently pinned contract SHA required')
        c = read_json(job / 'contract.json')
        need((c['method'], c['dataset'], c['ratio'], c['tag'], c['gpu']) ==
             (args.method, args.dataset, args.ratio, args.tag, 1), 'CLI and contract identity differ')
        verify_contract(c)
        if args.action == 'worker':
            worker(job, c, args.contract_sha, args.ready_sha, args.lease_fd)
            return  # Successful worker boot remains empty; all evidence is on disk.
        need(args.lease_fd is None, 'Only the launched worker may receive an inherited lease')
        result = qualify(job, c) if args.action == 'qualify' else launch(job, c, args.contract_sha, args.ready_sha)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
