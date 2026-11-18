"""Independent, one-shot Public05 fold dispatcher. No queue, retry, or resume.

Default is read-only check. Source/input preparation and each fold's own bridge
must be bound. The same original save.lock protects all five folds' ten targets.
"""
import argparse
import datetime
import importlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys

try:
    from expes.semi05_gate import fcd_public05_contract_20261015 as c
except ModuleNotFoundError:
    import fcd_public05_contract_20261015 as c

NAME = 'launch_fcd_public05_a1b_20261015.py'
PYTHON = '__RECORDED_PYTHON__'
READY = Path(c.BASE)/c.PREPARATION_NAME


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def regular(path):
    path = Path(path)
    c.require(not path.is_symlink() and not any(x.is_symlink() for x in path.parents), 'Symlink evidence')
    c.require(path.is_file() and path.stat().st_nlink == 1, 'Missing/nonregular/multilink evidence: '+str(path))
    return path


def read(path):
    return json.loads(regular(path).read_text(encoding='utf-8'))


def exclusive_json(path, value):
    path = Path(path)
    c.require(not any(x.is_symlink() for x in path.parents), 'Symlink destination')
    with path.open('x', encoding='utf8') as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())


def validate_preparation(value, hashes):
    c.require(value.get('schema') == 'FCD_PUBLIC05_PREPARATION_V1'
              and value.get('status') == 'REVIEWED_READY_FOR_OWN_BRIDGE', 'Wrong/unreviewed preparation')
    c.require(c.readiness(value.get('checks', {}))['status'] == 'READY_FOR_OWN_BRIDGE', 'Preparation incomplete')
    c.require(value.get('folds') == list(range(5)) and value.get('seed') == 2027
              and value.get('group') == 'A1B' and value.get('ratio') == '05', 'Wrong scope')
    c.require(value.get('source_hashes') == hashes and set(c.ENTRY_SOURCES) <= set(hashes), 'Source identity changed')
    c.require(value.get('runner_sha256') == hashes['run_fcd_public05_a1b_20261015.py'], 'Runner not bound')
    c.require(c.is_sha(value.get('input_registry_sha256')), 'Input registry not bound')
    c.require(value.get('protocol') == c.PROTOCOL and value.get('clean_confirmation') is False, 'Missing development disclosure')
    c.require(value.get('tail_policy') == c.TAIL_POLICY and value.get('budget') == dict(epochs=36, updates=2376, microbatches=11880), 'Wrong data schedule/budget')
    tests = value.get('cpu_tests', {})
    c.require(tests.get('status') == 'PASS' and tests.get('full_revision_reviewed') is True, 'Draft CPU test record')
    c.require(value.get('source_provenance', {}).get('historical_test_selected') is True, 'Historical init provenance missing')
    c.validate_save_lock(value.get('save_lock'))
    return True


def preparation():
    value = read(READY)
    names = value.get('source_hashes', {})
    c.require(isinstance(names, dict) and names and all(Path(n).name == n and n.endswith('.py') for n in names), 'Unsafe source name')
    hashes = {n: c.sha(regular(Path(c.ROOT)/'expes/semi05_gate'/n)) for n in names}
    validate_preparation(value, hashes)
    c.load_registry(value['input_registry_sha256'])
    return value


def validate_bridge_sources(bridge, current):
    c.require(bridge == current, 'Public bridge/formal sources must be identical; no silent compatibility exception')
    return True


def formal_bridge(runner, spec):
    fold = spec['fold']
    proof_path = Path(spec['bridge_report'])
    proof, manifest = read(proof_path), read(proof_path.parent/'manifest.json')
    runner.validate_manifest(manifest, runner.build_spec('A1B', manifest['gpu'], 'bridge', fold))
    runner.validate_bridge_report(proof, manifest)
    c.require(proof['manifest_sha256'] == c.sha(proof_path.parent/'manifest.json'), 'Bridge manifest changed')
    c.require(not (proof_path.parent/'failure.json').exists(), 'Bridge failure exists')
    mirror = read(Path(c.BASE)/'mirrors'/('fold'+str(fold)+'_'+c.TAGS['bridge']+'.json'))
    c.require(mirror.get('status') == 'VERIFIED' and mirror.get('phase') == 'bridge'
              and mirror.get('fold') == fold and mirror.get('group') == 'A1B', 'Own fold mirror missing')
    c.require(mirror.get('bridge_report_sha256') == c.sha(proof_path), 'Wrong mirror bridge')
    c.require(mirror.get('exact_file_set_verified') is True and mirror.get('all_member_sizes_and_sha256_verified') is True, 'Incomplete mirror verification')
    directory = Path(c.BASE)/('fold'+str(fold))/'bridge_evidence_20261015'
    inventory = read(directory/'inventory.json')
    c.require(mirror.get('inventory_sha256') == c.sha(directory/'inventory.json'), 'Bridge inventory changed')
    collector = importlib.import_module('expes.semi05_gate.collect_fcd_public05_a1b_20261015')
    c.require({str(p.relative_to(directory)).replace('\\', '/') for p in directory.rglob('*') if p.is_file()}
              == set(inventory['files']) | {'inventory.json', 'report.json'}, 'Bridge exact set changed')
    for name, item in inventory['files'].items():
        collector.safe_member(name)
        raw = collector.read_small(directory/name)
        c.require(len(raw) == item['bytes'] and collector.digest(raw) == item['sha256'], 'Bridge evidence changed')
    c.require(c.sha(regular(directory/'report.json')) == inventory['report_sha256'], 'Bridge report changed')
    return True


def check_gpu_reservations(spec):
    # A dispatched-but-not-started child must reserve its GPU as well. Hardware
    # emptiness alone has a race before the child's first CUDA allocation.
    own = c.paths(spec['phase'], spec['fold'], spec['gpu'])['dispatch']
    for path in Path(c.BASE).glob('fold*/A1B/*.dispatch.json'):
        if str(path) == own:
            continue
        item = read(path); other = item.get('scope', {})
        c.validate_scope(**{k: other[k] for k in ('group', 'ratio', 'fold', 'seed', 'gpu', 'phase')})
        if other['gpu'] != spec['gpu']:
            continue
        out = Path(c.paths(other['phase'], other['fold'], other['gpu'])['output'])
        complete = out/('bridge_report.json' if other['phase'] == 'bridge' else 'final_metrics.json')
        c.require(complete.is_file() and not (out/'failure.json').exists(), 'Unfinished/failed GPU dispatch reservation: '+str(path))
        row = read(complete)
        if other['phase'] == 'bridge':
            c.require(row.get('status') == 'BRIDGE_PASS_FORMAL_DISK_GATE_PENDING' and row.get('successful_updates') == 9, 'Incomplete bridge reservation')
        else:
            c.validate_final(row)
    return True


def resource_check(runner, spec):
    from expes.semi05_gate.fcd_matrix_storage_gate_20261006 import Target, inspect_capacity
    result = inspect_capacity(str(c.STORE), [Target(**r) for r in c.capacity_targets()])
    c.require(result['passes'], 'Capacity insufficient: '+json.dumps(result))
    memory = runner._memory_gate(); runner._gpu_empty(spec['gpu']); check_gpu_reservations(spec)
    return dict(capacity=result, memory=memory)


def verify_child(spec, dispatch_sha):
    c.require(sys.executable == PYTHON and __debug__, 'Wrong interpreter or disabled assertions')
    value = preparation(); paths = c.paths(spec['phase'], spec['fold'], spec['gpu'])
    dispatch = read(paths['dispatch'])
    c.require(c.sha(paths['dispatch']) == dispatch_sha, 'Dispatch changed')
    expected = c.validate_scope(fold=spec['fold'], gpu=spec['gpu'], phase=spec['phase'])
    c.require(dispatch.get('schema') == 'FCD_PUBLIC05_DISPATCH_V1' and dispatch.get('scope') == expected, 'Wrong dispatch scope')
    c.require(dispatch.get('preparation_sha256') == c.sha(READY) and dispatch.get('source_hashes') == value['source_hashes'], 'Dispatch provenance changed')
    c.require(dispatch.get('launcher_sha256') == c.sha(__file__), 'Launcher changed')
    return dispatch


def require_claim(spec):
    paths = c.paths(spec['phase'], spec['fold'], spec['gpu']); started = read(paths['started'])
    c.require(started.get('pid') == os.getpid() and started.get('phase') == spec['phase']
              and started.get('fold') == spec['fold'] and started.get('gpu') == spec['gpu'], 'Missing own process claim')
    verify_child(spec, started['dispatch_sha256'])
    return dict(dispatch_sha256=started['dispatch_sha256'], preparation_sha256=c.sha(READY), started_sha256=c.sha(paths['started']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('check', 'launch', 'child'), default='check')
    parser.add_argument('--phase', choices=('bridge', 'formal'), required=True)
    parser.add_argument('--fold', type=int, choices=range(5), required=True)
    parser.add_argument('--gpu', choices=('0', '1'), required=True)
    parser.add_argument('--preparation-sha'); parser.add_argument('--dispatch-sha')
    args = parser.parse_args()
    c.require(Path(__file__).resolve() == Path(c.ROOT)/'expes/semi05_gate'/NAME and sys.executable == PYTHON and __debug__, 'Registered runtime only')
    import fcntl
    runner = importlib.import_module('expes.semi05_gate.run_fcd_public05_a1b_20261015')
    spec = runner.build_spec('A1B', args.gpu, args.phase, args.fold)
    paths = c.paths(args.phase, args.fold, args.gpu)
    value = preparation()
    with c.locked_save_lock(value['save_lock']) as lock:
        fcntl.flock(lock, fcntl.LOCK_EX); value = preparation()
        if args.action == 'child':
            verify_child(spec, args.dispatch_sha); runner.assert_fresh(spec); resource_check(runner, spec)
            if args.phase == 'formal': formal_bridge(runner, spec)
            exclusive_json(paths['started'], dict(pid=os.getpid(), phase=args.phase, fold=args.fold, gpu=args.gpu,
                                                  time=now(), dispatch_sha256=args.dispatch_sha))
        else:
            c.assert_fresh(args.phase, args.fold, args.gpu); resources = resource_check(runner, spec)
            if args.phase == 'formal': formal_bridge(runner, spec)
            if args.action == 'check':
                print(json.dumps(dict(status='CHECK_PASS_NOT_LAUNCHED', phase=args.phase, fold=args.fold, gpu=args.gpu, **resources))); return
            c.require(args.preparation_sha == c.sha(READY), 'Explicit preparation SHA required')
            Path(paths['dispatch']).parent.mkdir(parents=True, exist_ok=True)
            dispatch = dict(schema='FCD_PUBLIC05_DISPATCH_V1', time=now(), scope=c.validate_scope(fold=args.fold, gpu=args.gpu, phase=args.phase),
                            preparation_sha256=args.preparation_sha, source_hashes=value['source_hashes'], launcher_sha256=c.sha(__file__),
                            nonce=secrets.token_hex(16), **resources)
            exclusive_json(paths['dispatch'], dispatch)
            with Path(paths['boot']).open('x') as boot:
                env = dict(os.environ, PYTHONPATH=str(c.ROOT), PYTHONUNBUFFERED='1')
                child = subprocess.Popen([PYTHON, str(Path(__file__).resolve()), '--action', 'child', '--phase', args.phase,
                                          '--fold', str(args.fold), '--gpu', args.gpu, '--dispatch-sha', c.sha(paths['dispatch'])],
                                         cwd=str(c.ROOT), env=env, stdin=subprocess.DEVNULL, stdout=boot,
                                         stderr=subprocess.STDOUT, start_new_session=True)
            print(json.dumps(dict(status='DISPATCHED_NOT_YET_STARTED', pid=child.pid, dispatch=paths['dispatch']))); return
    runner._execute(spec)


if __name__ == '__main__':
    main()
