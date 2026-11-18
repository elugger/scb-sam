#!/usr/bin/env python3
"""Read-only final acceptance of the explicitly authorized non-bitwise sup100 recovery.

No model/torch imports, inference, training, checkpoint deserialization, output
writes, resume, launch, queue or deletion. JSON is printed only to stdout.
Full existing ledgers are checked; this tool never upgrades a partial run to a
completed result. Checkpoint SHA verification reads the final only on completion.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys


ROOT = Path('__RECORDED_PROJECT_ROOT__')
CONTROL_REL = Path('expes/work_dir/semi05_gate/PUBLIC_BASIC_SUP_FOLD2_20261018/control')
BASE_REL = CONTROL_REL.parent
STORE_REL = Path('expes/artifacts/FCD_A_MATRIX_20261006/PUBLIC_BASIC_SUP_FOLD2_20261018')
SOURCE_REL = Path('expes/semi05_gate/public_sup_20261018')
ARMS = {'sup100_recovery_r1': (100, 346)}
EVIDENCE_NAMES = ('config.json', 'started.json', 'initialization_load_audit.json', 'initialization.json', 'training_steps.jsonl',
                  'training_epochs.jsonl', 'progress.json', 'save_receipts.jsonl',
                  'test_evaluation_started.json', 'final_reload_verified.json',
                  'final_test_cases.jsonl', 'final_metrics.json', 'failure.json')
MAX_SMALL_FILE = 64 * 1024 * 1024
METRIC_KEYS = ('dice', 'jaccard', 'hd95', 'asd')


class EvidenceError(RuntimeError):
    pass


def check(value, message):
    if not value:
        raise EvidenceError(message)


def timestamp():
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def parse_json(payload):
    def reject_constant(value):
        raise EvidenceError('Nonfinite JSON constant: '+value)
    return json.loads(payload, parse_constant=reject_constant)


def fingerprint(path):
    try:
        info = Path(path).lstat()
    except FileNotFoundError:
        return None
    check(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'Not an ordinary unshared evidence file: '+str(path))
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def read_json(path, optional=False):
    identity = fingerprint(path)
    if optional and identity is None:
        return None
    check(identity is not None, 'Required evidence missing: '+str(path))
    size = identity[2]
    check(size <= MAX_SMALL_FILE, 'Small-evidence size bound exceeded: '+str(path))
    result = parse_json(Path(path).read_text(encoding='utf-8'))
    check(fingerprint(path) == identity, 'Evidence changed during read: '+str(path))
    return result


def evidence_snapshot(directory):
    paths = [Path(directory)/name for name in EVIDENCE_NAMES]
    for attempt in range(1, 4):
        before = {path.name: fingerprint(path) for path in paths}
        raw = {}
        for path in paths:
            if before[path.name] is not None:
                check(before[path.name][2] <= MAX_SMALL_FILE, 'Evidence file exceeds read bound: '+str(path))
                raw[path.name] = path.read_bytes()
        after = {path.name: fingerprint(path) for path in paths}
        if before == after:
            parsed = {}
            for name, payload in raw.items():
                if name.endswith('.jsonl'):
                    check(not payload or payload.endswith(b'\n'), 'Uncommitted final JSONL line; inspect again: '+name)
                    parsed[name] = [parse_json(line) for line in payload.splitlines() if line.strip()]
                else:
                    parsed[name] = parse_json(payload)
            return parsed, {'attempts': attempt, 'files': {
                name: {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
                for name, payload in raw.items()}}
    raise EvidenceError('Evidence changed during all three bounded snapshot attempts; no stable observation')


def number(value, label, minimum=None):
    check(type(value) in (int, float) and math.isfinite(value), 'Nonfinite/non-numeric '+label)
    if minimum is not None:
        check(value >= minimum, label+' below minimum')
    return value


def close(actual, expected, label):
    number(actual, label)
    check(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), label+' mismatch')


def validate_ledgers(train_ids, steps, epochs):
    count = len(train_ids)
    check(count > 0 and len(set(train_ids)) == count, 'Invalid unique training IDs')
    check(len(steps) <= 60*count and len(epochs) <= 60, 'Frozen training budget exceeded')
    seen_by_epoch = {}
    index_for_case = {case: index for index, case in enumerate(train_ids)}
    for index, row in enumerate(steps):
        epoch, position = divmod(index, count)
        check(row['successful_update'] == index+1, 'Noncontinuous successful update at row '+str(index))
        check(row['epoch'] == epoch and row['epoch_number'] == epoch+1 and row['epoch_position'] == position,
              'Step epoch/position mismatch at update '+str(index+1))
        case = row['case_id']
        check(case in index_for_case and row['requested_index'] == index_for_case[case], 'Actual/requested training identity mismatch')
        seen = seen_by_epoch.setdefault(epoch, [])
        check(case not in seen, 'Duplicate actual training case in epoch '+str(epoch))
        seen.append(case)
        check(row['optimizer_steps'] == 1 and row['skipped'] is False and row['gradient_finite'] is True,
              'Optimizer/skip/finite gate violation at update '+str(index+1))
        number(row['loss'], 'supervised loss', 0)
        number(row['amp_scale'], 'AMP scale', 0)
        check(row['amp_scale'] > 0, 'Zero AMP scale')
        number(row['elapsed_seconds'], 'step duration', 0)
        check(type(row['foreground_voxels']) is int and row['foreground_voxels'] >= 0, 'Invalid foreground count')
    for index, row in enumerate(epochs):
        check(row['epoch'] == index and row['epoch_number'] == index+1, 'Noncontinuous complete-epoch ledger')
        check(row['successful_updates'] == (index+1)*count, 'Complete-epoch update count mismatch')
        actual = seen_by_epoch.get(index, [])
        check(len(actual) == count and set(actual) == set(train_ids), 'Incomplete actual-case epoch coverage')
        check(row['actual_case_ids'] == actual, 'Epoch case sequence differs from step ledger')
        check(row['test_evaluations'] == 0, 'Test evaluation recorded during training')
        close(row['mean_supervised_loss'], sum(step['loss'] for step in steps[index*count:(index+1)*count])/count,
              'Mean epoch supervised loss')
        number(row['elapsed_seconds'], 'epoch duration', 0)
    full_step_epochs = len(steps)//count
    allowed_epoch_rows = {full_step_epochs}
    if steps and len(steps) % count == 0:
        allowed_epoch_rows.add(full_step_epochs-1)  # Between step append and epoch append.
    check(len(epochs) in allowed_epoch_rows, 'Epoch records lag beyond a single append boundary')
    recent = [row['elapsed_seconds'] for row in steps[-20:]]
    return {'successful_updates': len(steps), 'total_updates': 60*count,
            'percent': 100*len(steps)/(60*count), 'complete_epoch_records': len(epochs),
            'full_step_coverage_epochs': full_step_epochs, 'current_epoch_zero_based': len(steps)//count if len(steps) < 60*count else 59,
            'current_epoch_cases': len(steps) % count, 'all_existing_steps_verified': True,
            'all_recorded_epochs_exact_actual_case_coverage': True,
            'last_successful_update_time': steps[-1].get('time') if steps else None,
            'latest_complete_epoch_mean_supervised_loss': epochs[-1]['mean_supervised_loss'] if epochs else None,
            'recent_step_mean_seconds': sum(recent)/len(recent) if recent else None,
            'compute_only_remaining_hours': ((60*count-len(steps))*sum(recent)/len(recent)/3600) if recent else None,
            'eta_excludes_future_saves_test_and_other_overhead': True}


def validate_case_metrics(rows, test_ids, final_sha):
    check(len(rows) <= len(test_ids), 'Too many final-test case records')
    for index, row in enumerate(rows):
        check(row['actual_case_id'] == test_ids[index] and row['requested_case_id'] == test_ids[index],
              'Final test actual/requested order mismatch')
        check(row['case_replaced'] is False and row['final_sha256'] == final_sha, 'Final test source/case replacement mismatch')
        check(set(row['metrics']) == set(METRIC_KEYS), 'Unexpected final-test metric keys')
        for key in METRIC_KEYS:
            number(row['metrics'][key], key, 0)
        check(row['metrics']['dice'] <= 1 and row['metrics']['jaccard'] <= 1, 'Overlap metric outside [0,1]')
    return {key: sum(row['metrics'][key] for row in rows)/len(rows) for key in METRIC_KEYS} if rows else None


def gpu_snapshot():
    result = {'gpus': [], 'compute_processes': [], 'error': None}
    try:
        query = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total',
                                '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20, check=True)
        for row in csv.reader(io.StringIO(query.stdout), skipinitialspace=True):
            if row:
                result['gpus'].append({'index': int(row[0]), 'uuid': row[1].strip(),
                                      'utilization_percent': row[2].strip(), 'memory_used_mib': row[3].strip(),
                                      'memory_total_mib': row[4].strip()})
        query = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                                '--format=csv,noheader'], capture_output=True, text=True, timeout=20, check=True)
        for row in csv.reader(io.StringIO(query.stdout), skipinitialspace=True):
            if row:
                result['compute_processes'].append({'gpu_uuid': row[0].strip(), 'pid': int(row[1]),
                                                    'process_name': row[2].strip()})
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as error:
        result['error'] = type(error).__name__+': '+str(error)
    return result


def process_snapshot(dispatch, started, gpu):
    pid = dispatch.get('pid') if dispatch else None
    result = {'pid': pid, 'alive': False, 'identity_match': False, 'gpu_pid_match': False, 'error': None}
    if pid is None:
        return result
    try:
        process = Path('/proc')/str(int(pid))
        status = (process/'stat').read_text().rsplit(')', 1)[1].split()
        command = [item.decode('utf-8') for item in (process/'cmdline').read_bytes().split(b'\0') if item]
        result.update(alive=status[0] != 'Z', process_state=status[0], command=command, start_ticks=status[19])
        result['identity_match'] = command == dispatch['command'] and (not started or
                                   (started['pid'] == pid and str(started.get('proc_start_ticks')) == status[19]))
        result['gpu_pid_match'] = any(row['pid'] == pid and row['gpu_uuid'] == next(g['uuid'] for g in gpu['gpus'] if g['index'] == dispatch['gpu'])
                                     for row in gpu['compute_processes'])
    except FileNotFoundError:
        pass  # Normal after successful exit; completion is verified independently.
    except (OSError, ValueError, IndexError, UnicodeError) as error:
        result['error'] = type(error).__name__+': '+str(error)
    return result


def resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root/path


def inspect_arm(arm, gpu, root=ROOT):
    percent, count = ARMS[arm]
    out, artifacts, control = root/BASE_REL/arm, root/STORE_REL/arm, root/CONTROL_REL
    result = {'arm': arm, 'label_percent': percent, 'state': 'UNKNOWN', 'errors': [], 'warnings': [],
              'output_dir': str(out), 'artifact_dir': str(artifacts), 'failure': None}
    try:
        dispatch = read_json(control/(arm+'.dispatch.json'), optional=True)
        intent = read_json(control/(arm+'.intent.json'), optional=True)
        dispatch_error = read_json(control/(arm+'.dispatch_error.json'), optional=True)
        result.update(dispatch=dispatch, dispatch_error=dispatch_error)
        check(not any(path.is_symlink() for path in (out, artifacts, *out.parents, *artifacts.parents)),
              'Symlink arm directory or ancestor')
        if not out.exists():
            result['process'] = process_snapshot(dispatch, None, gpu)
            result['state'] = ('DISPATCH_FAILED' if dispatch_error else
                               'DISPATCHED_START_PENDING' if dispatch and result['process']['alive'] else
                               'STOPPED_BEFORE_STARTED' if dispatch else 'DISPATCH_INTENT_WITHOUT_PID' if intent
                               else 'NOT_LAUNCHED' if not artifacts.exists() else 'UNEXPLAINED_ARTIFACT_RESIDUE')
            return result
        files, snapshot = evidence_snapshot(out)
        result['snapshot'] = snapshot
        result['failure'] = files.get('failure.json')
        result['progress'] = files.get('progress.json')
        steps, epochs = files.get('training_steps.jsonl', []), files.get('training_epochs.jsonl', [])
        result.update(observed_step_records=len(steps), observed_epoch_records=len(epochs))
        started, cfg = files.get('started.json'), files.get('config.json')
        result['process'] = process_snapshot(dispatch, started, gpu)
        check(dispatch is not None and intent is not None and not dispatch_error, 'Missing/failed dispatch evidence')
        if cfg is None or started is None:
            check(not steps and not epochs, 'Training exists without configuration/started evidence')
            result['state'] = 'STARTUP_EVIDENCE_PENDING' if result['process']['alive'] else 'STOPPED_BEFORE_STARTED'
            return result
        config_path = root/SOURCE_REL/(arm+'.json')
        canonical_cfg = read_json(config_path)
        config_sha = sha_file(config_path)
        check(cfg == canonical_cfg, 'Saved configuration differs from launched configuration')
        check(config_sha == dispatch['config_sha256'] == intent['config_sha256'] == started['config_sha256'],
              'Configuration SHA binding mismatch')
        check(cfg['fold'] == 2 and cfg['seed'] == 2027 and cfg['epochs'] == 60 and
              cfg['label_percent'] == percent and cfg['train_count'] == count and cfg['test_count'] == 87,
              'Authorized arm/fold/seed/budget mismatch')
        check(resolve(root, cfg['output_dir']) == out and resolve(root, cfg['artifact_dir']) == artifacts,
              'Configured artifact/output path mismatch')
        check(started['pid'] == dispatch['pid'] and started['gpu'] == dispatch['gpu'], 'Started PID/GPU differs from dispatch')
        check(started['architecture'] == 'coarse_unet_mask_prompt_sam_basic_cascade' and
              started['hpq_enabled'] is False and started['unlabeled_data_used'] is False and
              started['case_replacement_enabled'] is False and started['foreground_threshold_rejection_enabled'] is False and
              started['test_evaluations_before_final'] == 0 and started['selection_rule'] == 'fixed_epoch_60_student',
              'Scientific/no-HPQ/no-unlabeled/selection boundary mismatch')
        check(started['budget']['total_updates'] == count*60 and started['budget']['updates_per_epoch'] == count,
              'Started budget mismatch')
        check(started['budget']['epochs'] == 60 and started['budget']['train_count'] == count and
              started['budget']['batch_size'] == 1 and started['budget']['replay_per_epoch'] == 0 and
              started['amp_init_scale'] == 1024 and started['continuation_replay_verified'] is False,
              'Started exposure/AMP/continuation contract mismatch')
        checked_sources = []
        for group in ('source_sha256', 'input_sha256'):
            for name, expected in cfg[group].items():
                path = resolve(root, name).resolve()
                check(valid_sha(expected) and sha_file(path) == expected, 'Current source/input SHA changed: '+str(path))
                binding = started['files'].get(str(path))
                check(binding and binding['kind'] == group and binding['sha256'] == expected, 'Started source/input manifest mismatch')
                checked_sources.append({'path': str(path), 'sha256': expected, 'kind': group})
        result['source_and_input_sha_verified'] = checked_sources
        train_ids = [line.strip() for line in resolve(root, cfg['train_list']).read_text().splitlines() if line.strip()]
        test_ids = [line.strip() for line in resolve(root, cfg['test_list']).read_text().splitlines() if line.strip()]
        check(len(train_ids) == count and len(set(train_ids)) == count and len(test_ids) == 87 and len(set(test_ids)) == 87,
              'Invalid declared case-list cardinality or uniqueness')
        check(not set(train_ids) & set(test_ids) and started['train_ids'] == train_ids and started['test_ids'] == test_ids,
              'Case-set role/started identity mismatch')
        result['ledger'] = validate_ledgers(train_ids, steps, epochs)
        import recovery_support_20261019 as support
        support.validate(cfg)
        recovery = read_json(out/'recovery_restored.json')
        check(recovery['status'] == 'RESTORED_NOT_BITWISE_REPLAY' and
              recovery['checkpoint_sha256'] == support.CKPT_SHA and
              recovery['original_ledger_sha256'] == support.LEDGER_SHA and
              recovery['inherited_updates'] == 14878 and recovery['inherited_epochs'] == 43 and
              recovery['discarded_suffix_updates'] == 52 and recovery['remaining_updates'] == 5882 and
              recovery['continuation_replay_verified'] is False and recovery['user_authorized_non_bitwise'] is True,
              'Recovery provenance missing or inconsistent')
        old_out = root/BASE_REL/'sup100'
        old_steps_path = old_out/'training_steps.jsonl'
        check(sha_file(old_steps_path) == support.LEDGER_SHA, 'Original ledger changed')
        old_steps = [parse_json(x) for x in old_steps_path.read_bytes().splitlines()]
        old_epochs = [parse_json(x) for x in (old_out/'training_epochs.jsonl').read_bytes().splitlines()]
        check(len(old_steps) == 14930 and len(old_epochs) == 43, 'Original retained ledger length changed')
        check(steps[:14878] == old_steps[:14878] and epochs[:43] == old_epochs, 'Inherited prefix is not original')
        check(not (old_out/'test_evaluation_started.json').exists(), 'Original failed branch unexpectedly evaluated')
        check(all(r['amp_scale'] == 1024.0 for r in steps[14878:]), 'Recovery fixed-scale policy changed')
        check(started['recovery'] == cfg['recovery'] and started['continuation_replay_verified'] is False,
              'Started recovery contract mismatch')
        result['recovery'] = recovery
        result['new_updates'] = len(steps)-14878
        result['original_prefix_verified'] = True
        load_audit = files.get('initialization_load_audit.json')
        result['initialization_load_audit'] = load_audit
        if steps or files.get('initialization.json'):
            check(load_audit and load_audit['status'] == 'PASS' and load_audit['all_values_checked'] is True and
                  load_audit['actual_model_key_count'] == load_audit['checkpoint_key_count'] == load_audit['tensor_values_checked'] > 0 and
                  load_audit['checkpoint_sha256'] == cfg['sam_checkpoint_sha256'] and
                  load_audit['model_reloaded_or_repaired'] is False and load_audit['model_forward_calls'] == 0 and
                  load_audit['training_updates'] == 0 and load_audit['test_evaluations'] == 0 and
                  not any(load_audit[key] for key in ('missing_keys', 'unexpected_keys', 'shape_mismatches', 'value_mismatches', 'invalid_entries')),
                  'Generic SAM full-load audit missing or failed')
        initialization = files.get('initialization.json')
        result['initialization'] = initialization
        if initialization is not None:
            check(initialization['seed'] == 2027 and initialization['sam_checkpoint_sha256'] == cfg['sam_checkpoint_sha256'] and
                  initialization['data_batches_read_before_hash'] == 0 and initialization['public_checkpoint_loaded'] is False and
                  initialization['test_evaluations'] == 0 and initialization['coarse_checkpoint'] is None,
                  'Initialization policy mismatch')
            check(valid_sha(initialization['full_student_state_sha256']) and valid_sha(initialization['coarse_state_sha256']),
                  'Malformed initialization state hash')
            result['initialization_verified'] = True
        else:
            check(not steps, 'Training updates exist without initialization identity')
        saves = files.get('save_receipts.jsonl', [])
        latest_saves, final_saves = [], []
        for row in saves:
            check(row['kind'] in ('latest', 'final'), 'Unexpected checkpoint kind')
            receipt = row['receipt']
            check(receipt['path'] == str(artifacts/(row['kind']+'.pth')) and valid_sha(receipt['sha256']) and
                  type(receipt['bytes']) is int and receipt['bytes'] > 0, 'Save receipt identity mismatch')
            if row['kind'] == 'latest':
                check(row['epoch'] == 43 + len(latest_saves) and row['epoch'] < len(epochs), 'Latest-save epoch sequence mismatch')
                latest_saves.append(row)
            else:
                final_saves.append(row)
        check(len(final_saves) <= 1 and len(latest_saves) <= len(epochs), 'Duplicate/inconsistent save receipt')
        result['save_receipts'] = {'latest_count': len(latest_saves), 'final_count': len(final_saves),
                                   'most_recent': saves[-1] if saves else None}
        partials = [str(path) for path in (artifacts/'latest.pth.partial', artifacts/'final.pth.partial') if os.path.lexists(path)]
        result['atomic_partial_files'] = partials
        if partials:
            result['warnings'].append('Atomic partial exists: may be in-flight while live; do not delete or relaunch')
        marker, reload_record = files.get('test_evaluation_started.json'), files.get('final_reload_verified.json')
        case_rows, final = files.get('final_test_cases.jsonl', []), files.get('final_metrics.json')
        result['test_case_records'] = len(case_rows)
        if marker is not None:
            check(len(steps) == 60*count and len(epochs) == 60 and len(latest_saves) == 17 and len(final_saves) == 1,
                  'Test began before complete fixed training/save budget')
            check(reload_record and reload_record['strict_load'] is True and marker['test_count'] == 87 and
                  reload_record['path'] == str(artifacts/'final.pth') and
                  marker['final_sha256'] == reload_record['sha256'] == final_saves[0]['receipt']['sha256'],
                  'Test marker/final reload/checkpoint receipt mismatch')
            aggregate = validate_case_metrics(case_rows, test_ids, marker['final_sha256'])
        else:
            check(not case_rows and final is None, 'Test/final results exist without sole evaluation marker')
            aggregate = None
        progress = result['progress']
        if progress is not None and final is None:
            gap = len(steps)-progress['successful_updates']
            check(gap in (0, 1), 'Progress differs from step ledger beyond one append boundary')
            check(progress.get('test_evaluations', 0) == 0, 'Unexpected training-time test count')
            if gap:
                result['warnings'].append('Observed between step append and atomic progress publication')
        if final is not None:
            check(final['status'] == 'COMPLETE' and final['fold'] == 2 and final['label_percent'] == percent and final['seed'] == 2027 and
                  final['completed_epochs'] == 60 and final['successful_updates'] == 60*count and final['test_evaluations'] == 1 and
                  final['selection_rule'] == 'fixed_epoch_60_student' and final['no_hpq'] is True and final['no_unlabeled'] is True,
                  'Completion contract mismatch')
            check(marker and len(case_rows) == 87 and final['test_actual_case_ids'] == test_ids,
                  'Completed final lacks all 87 unique actual cases')
            check(final['final_checkpoint'] == str(artifacts/'final.pth') and final['final_sha256'] == marker['final_sha256'],
                  'Final result/checkpoint binding mismatch')
            check(set(final['metrics']) == set(METRIC_KEYS), 'Final aggregate metric keys mismatch')
            for key in METRIC_KEYS:
                close(final['metrics'][key], aggregate[key], 'Final aggregate '+key)
            check(progress == final, 'Final metrics and progress objects differ; publication may still be in-flight')
            check(not result['failure'] and not partials, 'Failure/partial residue prevents complete acceptance')
            final_path = artifacts/'final.pth'
            final_info = fingerprint(final_path)
            check(final_info and final_info[2] == final_saves[0]['receipt']['bytes'], 'Final artifact missing or size mismatch')
            disk_sha = sha_file(final_path)
            check(fingerprint(final_path) == final_info and disk_sha == final['final_sha256'], 'Final artifact SHA/identity changed')
            result['final_verified'] = {'sha256': disk_sha, 'bytes': final_info[2], 'metrics': final['metrics'],
                                        'final_metrics_file_sha256': snapshot['files']['final_metrics.json']['sha256'],
                                        'recomputed_case_metric_means': aggregate, 'unique_actual_test_cases': 87,
                                        'test_evaluations': 1, 'no_model_loaded_by_inspector': True}
            check(final['recovery'] == cfg['recovery'] and final['continuation_replay_verified'] is False,
                  'Final lacks non-bitwise continuation disclosure')
            for save in saves:
                gate=save['receipt']['capacity_before']
                check(gate['save_lock']['device'] == 2049 and gate['save_lock']['inode'] == 168299140 and
                      gate['save_lock']['nlink'] == 1, 'Wrong original save lock')
                for fs in gate['filesystems']:
                    check(fs['required_bytes'] == fs['remaining_permanent'] + fs['atomic_peak'] + 1024**3 + 16*1024**2
                          and fs['free_bytes'] >= fs['required_bytes'] and
                          fs['margin_bytes'] == fs['free_bytes']-fs['required_bytes'], 'Save capacity equation failed')
                check(gate['memory']['effective_available'] >= gate['memory']['required_bytes'],
                      'Save effective RAM check failed')
            check(len(saves) == 18 and len(latest_saves) == 17, 'Recovery save count mismatch')
            result['recovery_save_capacities_verified'] = True
            result['state'] = 'COMPLETE_VERIFIED'
        elif result['failure'] is not None:
            result['state'] = 'FAILED_PRESERVED_NO_AUTO_RETRY'
        elif result['process']['alive']:
            check(result['process']['identity_match'], 'Live PID does not match dispatch command/start ticks')
            check(gpu['error'] is None, 'GPU query unavailable: '+str(gpu['error']))
            if steps:
                check(result['process']['gpu_pid_match'], 'Live training PID absent from assigned GPU compute list')
            result['state'] = 'FINAL_TEST_IN_PROGRESS' if marker else 'RUNNING' if steps else 'INITIALIZING'
        else:
            result['state'] = 'STOPPED_INCOMPLETE_NO_AUTO_RETRY'
        if result['process']['error']:
            result['warnings'].append('Process inspection unavailable: '+result['process']['error'])
    except (EvidenceError, KeyError, TypeError, ValueError, OSError) as error:
        result['errors'].append(type(error).__name__+': '+str(error))
        result['state'] = 'INVALID_OR_UNSTABLE_EVIDENCE'
    return result


def resources(root):
    result = {'note': 'Read-only current snapshots, not a locked save-capacity approval'}
    try:
        path = root/STORE_REL
        while not path.exists():
            path = path.parent
        fs = os.statvfs(path)
        result['artifact_filesystem_free_bytes'] = fs.f_bavail*fs.f_frsize
        result['artifact_filesystem_device'] = path.stat().st_dev
        info = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        result['host_mem_available_bytes'] = int(info['MemAvailable'].split()[0])*1024
        result['host_memory_is_not_cgroup_effective_memory'] = True
    except (OSError, KeyError, ValueError, AttributeError) as error:
        result['error'] = type(error).__name__+': '+str(error)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', action='append', choices=tuple(ARMS), help='Omit for all three arms; may repeat')
    args = parser.parse_args()
    chosen = list(dict.fromkeys(args.arm or ARMS))
    gpu = gpu_snapshot()
    inspected = {arm: inspect_arm(arm, gpu) for arm in chosen}
    initialized = {arm: row['initialization'] for arm, row in inspected.items() if row.get('initialization_verified')}
    common = None
    if len(initialized) >= 2:
        identities = {(item['full_student_state_sha256'], item['coarse_state_sha256'], item['sam_checkpoint_sha256'])
                      for item in initialized.values()}
        common = {'arms_compared': list(initialized), 'common_initialization_verified': len(identities) == 1}
    payload = {'observed_at': timestamp(), 'inspector_sha256': sha_file(Path(__file__)),
               'scope': 'read_only_existing_evidence_no_model_no_test_no_launch_no_queue_no_output_write',
               'arms': inspected, 'gpu': gpu, 'resources': resources(ROOT), 'cross_arm_initialization': common,
               'automatic_next_action': None, 'single_seed_does_not_establish_stability': True,
               'historically_seen_test_development_context': True}
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False), flush=True)
    bad = any(row['errors'] or row['state'] in ('FAILED_PRESERVED_NO_AUTO_RETRY', 'DISPATCH_FAILED',
              'UNEXPLAINED_ARTIFACT_RESIDUE', 'STOPPED_INCOMPLETE_NO_AUTO_RETRY', 'STOPPED_BEFORE_STARTED',
              'DISPATCH_INTENT_WITHOUT_PID') for row in inspected.values())
    if bad or (common and not common['common_initialization_verified']):
        sys.exit(2)


if __name__ == '__main__':
    main()
