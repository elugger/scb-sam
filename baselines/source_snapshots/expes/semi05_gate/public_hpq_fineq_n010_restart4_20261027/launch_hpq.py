"""Explicit prepare / one-phase launch / worker. No queue, retry or resume."""
import argparse
import contextlib
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

from hpq_contract import ROOT, BASE, STORE, BRIDGE_STORE, INIT, INIT_SHA, require, sha_file, validate_scope
from hpq_inputs import plain_file
from hpq_resources import backend, RAM_REQUIRED, no_symlink
from hpq_engine import write_json
from hpq_preparation import SRC, READY, prepare, validate_preparation, phase_config, imported_project_sources
from gpu0_readonly import read_gpu0
from selection_protocol import POLICY, policy_sha

PYTHON = '__RECORDED_PYTHON__'
CONTROL = BASE.parent.parent / 'control'


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def paths(phase):
    require(phase in ('bridge', 'formal'), 'Unknown phase')
    return dict(output=Path(BASE) / (phase + '_r1'),
                store=Path(STORE if phase == 'formal' else BRIDGE_STORE),
                **{key: Path(CONTROL) / (phase + '.' + key + suffix) for key, suffix in
                   [('intent', '.json'), ('dispatch', '.json'), ('started', '.json'),
                    ('exited', '.json'), ('boot', '.log'), ('worker_failure', '.json')]})


def check_fresh(phase):
    targets = paths(phase)
    for name, path in targets.items():
        no_symlink(path)
        require(not os.path.lexists(path), 'Existing phase evidence, preserve and investigate: ' + name)
    return targets


def gpu_snapshot(gpu):
    require(type(gpu) is int and gpu == 0, 'Only physical GPU0 is authorized')
    snapshot = read_gpu0()
    require(snapshot['process_counts']['compute'] == 0, 'GPU0 has active compute processes')
    display = snapshot.get('graphics', [])
    require(len(display) == snapshot['process_counts']['graphics'] and
            all(p['name'] == 'Xorg' and 0 <= p['memory_bytes'] <= 32 * 1024 ** 2 for p in display),
            'GPU0 has unknown or oversized graphics work')
    require(snapshot['utilization_percent'] == 0 and snapshot['memory_bytes'] < 512 * 1024 ** 2,
            'GPU0 is active or has unexplained device allocations')
    return dict(gpu=gpu, **snapshot, observed_at=now())


def validate_bridge_evidence(preparation_sha):
    p = paths('bridge')
    report_path = p['output'] / 'bridge_report.json'
    for file in (report_path, p['exited'], p['dispatch'], p['started'], p['output'] / 'manifest.json'):
        plain_file(file)
    require(not p['worker_failure'].exists() and not (p['output'] / 'failure.json').exists(), 'Bridge failure exists')
    report = json.loads(report_path.read_text())
    require(report['status'] == 'OWN_BRIDGE_PASS_DISK_GATE_PENDING' and report['successful_updates'] == 9
            and report['artificial_overflows'] == 1 and report['new_test_forwards'] == 0
            and report['permanent_qualification_weight'] is False and report['initializer_sha256'] == INIT_SHA
            and report['preparation_sha256'] == preparation_sha, 'Bridge did not qualify this preparation')
    require(report['manifest_sha256'] == sha_file(p['output'] / 'manifest.json'), 'Bridge manifest changed')
    for key in ('native_context_off_exact', 'active_replay_exact', 'allowed_probe_then_step_exact',
                'six_groups_positive_finite', 'evaluation_then_step_exact'):
        require(report[key] is True, 'Bridge gate missing: ' + key)
    for key in ('six_root_roundtrip_exact', 'native_student_teacher_output_exact', 'declared_state_restored',
                'rng_unchanged', 'optimizer_scaler_unchanged'):
        require(report['roundtrip'][key] is True, 'Bridge roundtrip gate missing: ' + key)
    exited = json.loads(p['exited'].read_text())
    best = json.loads((p['output'] / 'best_probe.json').read_text())
    require(report['best_probe'] == best and best['status'] == 'BEST_PROBE_PASS'
            and best['strict_cpu_readback_verified'] is True
            and best['path'] == str(Path(STORE) / 'best/final.pth'), 'Missing best-model save proof')
    require(not os.path.lexists(best['path']) and not os.path.lexists(best['path'] + '.partial'),
            'Best-model qualification residual')
    require(exited['status'] == 'SUCCESS' and exited['report_sha256'] == sha_file(report_path), 'Bridge exit mismatch')
    # Completion is evidenced by the bound exit receipt; do not poll ended PIDs.
    require(exited['pid'] == json.loads(p['started'].read_text())['pid'], 'Bridge exit identity differs')
    require(not any(p['store'].iterdir()), 'Unexpected permanent bridge artifact')
    from hpq_inspect import read_rows
    from hpq_engine import Execution, validate_style
    rows = read_rows(p['output'] / 'qualification_steps.jsonl')
    require(len(rows) == 9, 'Bridge ledger must contain nine successful steps')
    helper = object.__new__(Execution)
    helper.group, helper.spec = 'A1B', {'phase': 'bridge'}
    for row in rows:
        active = row['branch'].startswith('active') or row['branch'] == 'allowed_train_lab_probe_then_step'
        helper.check_step(row, row['branch'] != 'native', 'B' if active else '0')
        from hpq_inputs import read_bound_splits
        validate_style(row, read_bound_splits()['train_unlab'])
    overflows = read_rows(p['output'] / 'overflow.jsonl')
    require(len(overflows) == 1 and overflows[0]['skipped'] and overflows[0]['clip_calls'] == overflows[0]['ema_calls'] == 0,
            'Bridge overflow ledger mismatch')
    safety = report['safety_state_probe']
    require(safety['status'] == 'FULL_STATE_SERIALIZATION_VERIFIED'
            and safety['continuation_replay_verified'] is False
            and set(safety['parts']) == {'weights', 'optimizer'}, 'Missing full-state serialization proof')
    require(safety == json.loads((p['output'] / 'safety_state_probe.json').read_text()),
            'Safety serialization receipt differs')
    disk = safety.get('disk_roundtrip', {})
    require(disk.get('status') == 'REGISTERED_OWN_PARTIAL_DISK_ROUNDTRIP_VERIFIED'
            and set(disk.get('parts', {})) == {'weights', 'optimizer'},
            'Missing real registered safety disk roundtrip')
    for name, part in disk['parts'].items():
        expected = Path(STORE) / 'safety_slot0' / name / 'final.pth'
        require(part['path'] == str(expected) and 0 < part['bytes'] <= 2 * 1024 ** 3
                and part['own_verified_probe_removed'] is True
                and part['capacity_before_reload']['owned_target'] == str(expected),
                'Safety disk proof identity/bound differs')
        require(not expected.exists() and not Path(str(expected) + '.partial').exists(),
                'Bridge safety disk residual')
    return dict(path=str(report_path), sha256=sha_file(report_path), manifest_sha256=report['manifest_sha256'])


def launch(phase, gpu, preparation_sha):
    validate_scope(2, 2027, phase, gpu)
    validate_preparation(preparation_sha)
    require(Path(__file__).resolve() == Path(SRC) / 'launch_hpq.py', 'Only registered launcher')
    bridge = validate_bridge_evidence(preparation_sha) if phase == 'formal' else None
    guard = backend()
    with guard.original_lock():
        target = check_fresh(phase)
        capacity, occupancy = guard.capacity_snapshot(RAM_REQUIRED), gpu_snapshot(gpu)
        Path(CONTROL).mkdir(parents=True, exist_ok=True)
        intent = dict(phase=phase, gpu=gpu, preparation_sha256=preparation_sha, nonce=uuid.uuid4().hex,
                      created_at=now(), capacity=capacity, gpu_snapshot=occupancy, bridge=bridge)
        write_json(target['intent'], intent)
        env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                   CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=':4096:8')
        args = [PYTHON, str(SRC / 'launch_hpq.py'), '--action', 'worker', '--phase', phase,
                '--gpu', str(gpu), '--preparation-sha', preparation_sha, '--nonce', intent['nonce']]
        with target['boot'].open('x', encoding='utf-8') as boot:
            process = subprocess.Popen(args, cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL,
                                       stdout=boot, stderr=subprocess.STDOUT, start_new_session=True)
        dispatch = dict(phase=phase, gpu=gpu, pid=process.pid, nonce=intent['nonce'],
                        preparation_sha256=preparation_sha, intent_sha256=sha_file(target['intent']),
                        command=args, dispatched_at=now())
        write_json(target['dispatch'], dispatch)
    return dispatch


def worker(phase, gpu, preparation_sha, nonce):
    target = paths(phase)
    engine = None
    try:
        # Only parent/child handoff waits; no autonomous relaunch/retry.
        deadline = time.monotonic() + 15
        while not target['dispatch'].exists() and time.monotonic() < deadline:
            time.sleep(.1)
        plain_file(target['dispatch'])
        dispatch = json.loads(target['dispatch'].read_text())
        intent = json.loads(target['intent'].read_text())
        require(dispatch['pid'] == os.getpid() and dispatch['nonce'] == intent['nonce'] == nonce
                and dispatch['phase'] == phase and dispatch['gpu'] == gpu
                and dispatch['preparation_sha256'] == preparation_sha
                and dispatch['intent_sha256'] == sha_file(target['intent']), 'Dispatch identity mismatch')
        ready, ancestor = validate_preparation(preparation_sha)
        require(os.environ['CUDA_VISIBLE_DEVICES'] == str(gpu), 'Physical GPU selection differs')
        import torch
        from hpq_initialization import make_trainer_class
        from expes.semi05_gate.fcd_deterministic_losses_r2_20261005 import deterministic_training
        from expes.common import seg_metrics
        trainer_class = make_trainer_class()
        for name, digest in imported_project_sources().items():
            require(ready['source_hashes'].get(name) == digest, 'Runtime imported unbound/changed source: ' + name)
        guard = backend()
        with guard.original_lock():
            capacity, occupancy = guard.capacity_snapshot(RAM_REQUIRED), gpu_snapshot(gpu)
            for key in ('output', 'store', 'started', 'exited', 'worker_failure'):
                no_symlink(target[key])
                require(not os.path.lexists(target[key]), 'Worker found residual: ' + key)
            if phase == 'formal':
                require(validate_bridge_evidence(preparation_sha) == intent['bridge'], 'Bridge identity changed')
            target['output'].mkdir(parents=True, exist_ok=False)
            target['store'].mkdir(parents=True, exist_ok=False)
            write_json(target['started'], dict(pid=os.getpid(), phase=phase, gpu=gpu, nonce=nonce, started_at=now(),
                       preparation_sha256=preparation_sha, dispatch_sha256=sha_file(target['dispatch'])))
        cfg = phase_config(phase, gpu, ancestor)
        manifest = dict(phase=phase, gpu=gpu, fold=2, seed=2027, initializer_sha256=INIT_SHA,
                        screen_policy=POLICY, screen_policy_sha256=policy_sha(),
                        candidate_schema=ready['candidate_schema'],
                        scientific_change=ready['scientific_change'],
                        preparation_sha256=preparation_sha, source_hashes=ready['source_hashes'],
                        dispatch_sha256=sha_file(target['dispatch']), started_sha256=sha_file(target['started']),
                        cfg=cfg, splits=ready['splits'], capacity=capacity, gpu_snapshot=occupancy,
                        bridge=intent['bridge'], clean_confirmation=False, gt_roi_used=True,
                        patient_alias_boundary_verified=False, unique_patient_count=None)
        write_json(target['output'] / 'manifest.json', manifest)
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
        with (target['output'] / 'execution.log').open('x', encoding='utf-8', buffering=1) as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log), deterministic_training():
                trainer = trainer_class(cfg, splits=ready['splits'], ancestor=ancestor, phase=phase)
                trainer.qualification_routing = phase == 'bridge'
                trainer.initialize_matrix(str(INIT))
                write_json(target['output'] / 'initialization.json', trainer.hpq_initialization_receipt)
                from hpq_engine import make_engine
                engine = make_engine(torch, trainer, phase, gpu, cfg, manifest, target['output'])
                report = engine.bridge() if phase == 'bridge' else engine.formal()
        report_path = target['output'] / ('bridge_report.json' if phase == 'bridge' else 'selection_metrics.json')
        write_json(target['exited'], dict(status='SUCCESS', pid=os.getpid(), phase=phase, ended_at=now(),
                                         report_sha256=sha_file(report_path)))
        print(json.dumps(dict(status=report['status'], report=str(report_path))), flush=True)
    except BaseException:
        failure = dict(status='FAIL_PRESERVED_NO_AUTOMATIC_RETRY', phase=phase, pid=os.getpid(), failed_at=now(),
                       successful_updates=0 if engine is None else engine.actual_successes,
                       model_updates=None if engine is None else engine.t.matrix_updates,
                       traceback=traceback.format_exc())
        if target['output'].exists():
            write_json(target['output'] / 'failure.json', failure)
        if not target['worker_failure'].exists():
            write_json(target['worker_failure'], failure)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', required=True, choices=['prepare', 'launch', 'worker'])
    parser.add_argument('--phase', choices=['bridge', 'formal'])
    parser.add_argument('--gpu', type=int, choices=[0])
    parser.add_argument('--preparation-sha')
    parser.add_argument('--bundle-sha')
    parser.add_argument('--nonce')
    args = parser.parse_args()
    if args.action == 'prepare':
        result = prepare(args.bundle_sha)
    elif args.action == 'launch':
        result = launch(args.phase, args.gpu, args.preparation_sha)
    else:
        return worker(args.phase, args.gpu, args.preparation_sha, args.nonce)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
