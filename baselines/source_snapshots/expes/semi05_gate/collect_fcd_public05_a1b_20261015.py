"""Own PUBLIC05 A1B small-evidence collector. CPU only; never launches a model.

Bridge and completion are separate once-only destinations. No old group package,
checkpoint copying, cleanup, test inference, audit invocation or retry.
"""
import argparse
import ctypes
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import stat
import sys

ROOT = Path('__RECORDED_PROJECT_ROOT__')
BASE = ROOT/'expes/work_dir/semi05_gate/FCD_PUBLIC05_A1B_20261015'
HERE = Path(__file__).resolve().parent
NAME = 'collect_fcd_public05_a1b_20261015.py'
LIMIT = 32*1024**2
AUDIT_KEYS = {'roots_optimizer_scaler', 'rng', 'modes', 'gradients', 'updates'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_small(path):
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'Symlink evidence')
    before = path.stat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= LIMIT, 'Not bounded regular small evidence')
    raw = path.read_bytes()
    after = path.stat()
    require(identity(before) == identity(after) and len(raw) == before.st_size, 'Evidence changed during read')
    return raw


def identity(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def safe_member(name):
    p = PurePosixPath(name)
    require(isinstance(name, str) and name and not p.is_absolute() and '\\' not in name
            and ':' not in name and all(x not in ('', '.', '..') for x in name.split('/')), 'Unsafe member')
    require(p.suffix not in ('.pth', '.pt', '.ckpt'), 'No model archive copying')
    return p


def output_files(output, phase):
    common = {'config.yaml', 'manifest.json', 'log.txt', 'initialization_memory.jsonl', 'cross_style_steps.jsonl'}
    allowed = common | ({'qualification_steps.jsonl', 'context_vb_components.jsonl', 'overflow.jsonl', 'bridge_report.json'}
                        if phase == 'bridge' else {'matrix_steps.jsonl', 'train_epochs.jsonl', 'metrics.jsonl',
                        'pair_trace.jsonl', 'startup_disk_gate.json', 'final_metrics.json', 'test_cases.jsonl',
                        'gate_epochs.jsonl', 'final_train_label_audit/report.json',
                        'final_train_label_audit/runtime/config.yaml', 'final_train_label_audit/runtime/log.txt'})
    directories = {'runtime_checkpoints'} if phase == 'bridge' else {'final_train_label_audit', 'final_train_label_audit/runtime'}
    actual = set()
    for path in output.rglob('*'):
        require(not path.is_symlink(), 'Symlink output')
        rel = path.relative_to(output).as_posix()
        if path.is_dir():
            require(rel in directories, 'Unexpected output directory: '+rel)
        else:
            require(rel in allowed, 'Unexpected output file: '+rel)
            actual.add(rel)
    require(actual == allowed, 'Missing own evidence: '+repr(sorted(allowed-actual)))
    return actual


def validate_audit(report, data, names):
    fold = data['manifest']['fold']
    final, manifest = data['final'], data['manifest']
    require(report.get('schema') == 'FCD_PUBLIC05_A1B_FINAL_AUDIT_V1' and report.get('status') == 'COMPLETE', 'Own audit not COMPLETE')
    for key, value in dict(group='A1B', ratio='05', fold=fold, seed=2027, completed_epochs=36,
                           total_updates=2376, context_microbatches=11880, context_applied_microbatches=11880,
                           data_trace_rows=11880, cross_style_rows=11880, new_test_forwards=0, test_forwards=0,
                           optimizer_updates=0, clean_confirmation=False).items():
        require(type(report.get(key)) is type(value) and report[key] == value, 'Audit field mismatch: '+key)
    for key in ('source_hashes', 'init_from', 'init_sha256', 'split_hashes', 'purpose', 'protocol', 'primary_metric',
                'input_registry', 'budget', 'tail_policy'):
        require(report.get(key) == manifest[key], 'Audit provenance mismatch: '+key)
    require(report.get('evidence_sha256') == data['evidence_sha256'] and report.get('bridge_sha256') == data['bridge_sha256'], 'Audited evidence changed')
    require(report.get('natural_unlab_count') == (329 if fold < 3 else 330)
            and report.get('tail_padding_microbatches') == (36 if fold < 3 else 0), 'Wrong tail exposure audit')
    require(report.get('final_dice') == final['final_dice'] and report.get('epoch35') == data['epochs'][-1], 'Wrong endpoint')
    require(set(report.get('cases', {})) == set(report.get('weights', {})) == {'best.pth', 'final.pth'}, 'Missing archives')
    require(len(names) == len(set(names)) == 17, 'Wrong train_lab identity')
    for archive in ('best.pth', 'final.pth'):
        weight, rows = report['weights'][archive], report['cases'][archive]
        require([row['case'] for row in rows] == names, 'Case identity/order changed')
        for key, value in final['weights'][archive].items():
            require(weight.get(key) == value, 'Actual archive identity changed')
        epoch = 35 if archive == 'final.pth' else final['best_epoch']
        require(weight.get('saved_epoch') == epoch and weight.get('updates') == (epoch+1)*66, 'Wrong archive budget')
        for key in ('all_six_roots_loaded', 'all_six_root_values_exact', 'zero_update_state_unchanged',
                    'requested_case_identity_preserved', 'declared_forward_transaction_restored'):
            require(weight.get(key) is True, 'Incomplete audit: '+key)
        before, after = weight.get('full_state_before'), weight.get('full_state_after')
        require(isinstance(before, dict) and set(before) == AUDIT_KEYS and before == after and before['updates'] == 0, 'Audit state changed')
        for row in rows:
            require(all(type(row[k]) is int and row[k] >= 0 for k in ('tp', 'fp', 'fn')), 'Invalid case counts')
            require(abs(row['dice'] - 2*row['tp']/max(1, 2*row['tp']+row['fp']+row['fn'])) < 1e-12, 'Case Dice mismatch')
        require(weight['totals'] == {k: sum(row[k] for row in rows) for k in ('tp', 'fp', 'fn')}, 'Case totals mismatch')
    return True


def rename_new(source, destination):
    if os.name == 'nt':
        os.rename(source, destination)
        return
    require(platform.system() == 'Linux' and platform.machine() == 'x86_64' and ctypes.sizeof(ctypes.c_void_p) == 8, 'Unverified atomic publication ABI')
    libc = ctypes.CDLL(None, use_errno=True)
    if hasattr(libc, 'renameat2'):
        fn = libc.renameat2
        fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        fn.restype = ctypes.c_int
        result = fn(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    else:
        fn = libc.syscall
        fn.argtypes = [ctypes.c_long, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        fn.restype = ctypes.c_long
        result = fn(316, -100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def publish(destination, files, report):
    destination = Path(destination)
    partial = destination.with_name(destination.name+'.partial')
    require(not os.path.lexists(destination) and not os.path.lexists(partial), 'Existing collection; investigate, never retry')
    for name in files:
        safe_member(name)
    require('report.json' not in files and 'inventory.json' not in files, 'Reserved output name')
    members = {name: dict(bytes=len(raw), sha256=digest(raw)) for name, raw in files.items()}
    report_raw = encoded(report)
    inventory = dict(schema='FCD_PUBLIC05_A1B_SMALL_INVENTORY_V1', status='COLLECTED_NOT_MIRRORED', phase=report['phase'],
                     files=members, file_count=len(members), total_bytes=sum(m['bytes'] for m in members.values()), report_sha256=digest(report_raw))
    partial.mkdir(parents=True, exist_ok=False)
    for name, raw in {**files, 'report.json': report_raw, 'inventory.json': encoded(inventory)}.items():
        target = partial.joinpath(*PurePosixPath(name).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as handle:
            handle.write(raw); handle.flush(); os.fsync(handle.fileno())
        require(read_small(target) == raw, 'Copied evidence differs')
    rename_new(partial, destination)
    return dict(destination=str(destination), files=len(members), bytes=inventory['total_bytes'],
                report_sha256=digest(report_raw), inventory_sha256=digest(encoded(inventory)))


def collection_report(phase, fold, collector_sha):
    require(phase in ('bridge', 'completion') and type(fold) is int and fold in range(5), 'Wrong collection scope')
    return dict(schema='FCD_PUBLIC05_A1B_COLLECTION_V1', status='COMPLETE', phase=phase, group='A1B', ratio='05', fold=fold,
                seed=2027, contains_weights=False, new_model_forwards=0, new_test_forwards=0, optimizer_updates=0,
                clean_confirmation=False, collector_sha256=collector_sha)


def collect(phase, fold, gpu=None):
    require(phase in ('bridge', 'completion'), 'Wrong collection phase')
    require(type(fold) is int and fold in range(5), 'Explicit fold0-4 required')
    require(Path(__file__).resolve() == ROOT/'expes/semi05_gate'/NAME, 'Registered deployment only')
    require('torch' not in sys.modules, 'CPU collector cannot run with model imports')
    runner = importlib.import_module('expes.semi05_gate.run_fcd_public05_a1b_20261015')
    audit = importlib.import_module('expes.semi05_gate.audit_fcd_public05_a1b_final_20261015')
    spec = runner.build_spec('A1B', phase='bridge' if phase == 'bridge' else 'formal',
                             fold=fold, gpu='0' if gpu is None else str(gpu))
    output = Path(spec['output'])
    destination = BASE/('fold'+str(fold))/(phase+'_evidence_20261015')
    require(not os.path.lexists(destination) and not os.path.lexists(str(destination)+'.partial'), 'Existing collection evidence')
    manifest = audit.read_json(output/'manifest.json')
    require(gpu is None or str(gpu) == manifest['gpu'], 'Collector GPU must match this fold')
    spec = runner.build_spec('A1B', phase=spec['phase'], fold=fold, gpu=manifest['gpu'])
    runner._gpu_empty(manifest['gpu'])
    runner.validate_manifest(manifest, spec)
    runner.validate_input_identity(manifest, spec)
    require(runner.sha(output/'config.yaml') == manifest['config_sha256'], 'Config changed')
    for path, value in manifest['source_hashes'].items():
        require(runner.sha(ROOT/path) == value, 'Source changed: '+path)
    report = collection_report(phase, fold, runner.sha(__file__))
    if phase == 'bridge':
        proof = audit.read_json(output/'bridge_report.json')
        runner.validate_bridge_report(proof, manifest)
        require(proof['manifest_sha256'] == runner.sha(output/'manifest.json'), 'Bridge manifest changed')
        rows = audit.read_jsonl(output/'qualification_steps.jsonl')
        require(len(rows) == 9 and all(row['skipped'] is False for row in rows), 'Wrong own bridge budget')
        expected = [(b, p, i+1) for b in ('native', 'off', 'active_B', 'active_B_repeat')
                    for i, p in enumerate(('cold', 'warm'))] + [('allowed_train_lab_probe_then_step', 'cold', 1)]
        require([(row['branch'], row['phase'], row['update']) for row in rows] == expected, 'Wrong bridge sequence')
        checker = object.__new__(runner._Engine)
        checker.group, checker.spec = 'A1B', spec
        for row in rows:
            checker.check_step(row, context=row['branch'] != 'native',
                               reduction='0' if row['branch'] in ('native', 'off') else 'B')
        overflow = audit.read_jsonl(output/'overflow.jsonl')
        require(len(overflow) == 1 and overflow[0]['skipped'] is True and overflow[0]['ema_calls'] == overflow[0]['clip_calls'] == 0, 'Wrong overflow')
        report.update(successful_updates=9, artificial_overflows=1, bridge_report_sha256=runner.sha(output/'bridge_report.json'))
    else:
        data = audit.preflight('A1B', fold, manifest['gpu'])
        names = (Path(data['split'])/'train_lab.txt').read_text().split()
        validate_audit(audit.read_json(output/'final_train_label_audit/report.json'), data, names)
        report.update(final_dice=data['final']['final_dice'], final_metrics=data['final'], **data['counts'])
    names = output_files(output, 'bridge' if phase == 'bridge' else 'formal')
    paths = {str(PurePosixPath('run')/name): output/name for name in names}
    paths['execution.log'] = Path(spec['log'])
    paths['sources/'+NAME] = Path(__file__)
    for name in runner.contract.ENTRY_SOURCES:
        paths['sources/'+name] = HERE/name
    identity = manifest['dispatch_identity']
    registered = runner.contract.paths(spec['phase'], fold=fold, gpu=manifest['gpu'])
    for name, source, expected in (
        ('preparation.json', BASE/runner.contract.PREPARATION_NAME, identity['preparation_sha256']),
        ('dispatch.json', Path(registered['dispatch']), identity['dispatch_sha256']),
        ('started.json', Path(registered['started']), identity['started_sha256'])):
        require(digest(read_small(source)) == expected, 'Dispatch chain changed: '+name)
        paths['provenance/'+name] = source
    registry_path = BASE/runner.contract.REGISTRY_NAME
    require(manifest['input_registry']['path'] == str(registry_path)
            and digest(read_small(registry_path)) == manifest['input_registry']['sha256'], 'Public05 input registry changed')
    paths['provenance/input_registry.json'] = registry_path
    files = {name: read_small(path) for name, path in paths.items()}
    require(all(read_small(paths[name]) == raw for name, raw in files.items()), 'Evidence changed before publication')
    require('torch' not in sys.modules, 'Unexpected model import')
    return publish(destination, files, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('bridge', 'completion'), required=True)
    parser.add_argument('--fold', type=int, choices=range(5), required=True)
    parser.add_argument('--gpu', choices=('0', '1'))
    args = parser.parse_args()
    print(json.dumps(collect(args.phase, args.fold, args.gpu), allow_nan=False))


