import os
import sys
from pathlib import Path
ASNET_ROOT = Path(os.environ.get('ASNET_ROOT', '.')).resolve()
from collections import Counter
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import re
ROOT = PurePosixPath(str(ASNET_ROOT))
INIT = ROOT / 'expes/artifacts/FCD_A_MATRIX_20261006/MAIN_TABLE_FIXED_FINAL_20261019/basic_cascade_ncase/Public433/lab010/fold2/r1/final.pth'
INIT_SHA = 'e0b0054027306e50905d1092cd4175bec9d8a37983e686071becbe0e3f1a8db2'
METRICS_SHA = 'dcb1ceb6bfe9e502313d7c314a100ba5619a5eead9e6ab5b3f7c66e3825fedc4'
BASE = ROOT / 'expes/work_dir/semi05_gate/PUBLIC_N010_FINEQ_BEST_RESTART4_20261027/fold2/A1B'
STORE = ROOT / 'expes/artifacts/FCD_A_MATRIX_20261006/MAIN_TABLE_FIXED_FINAL_20261019/hpq_ncase/Public433/lab010/fold2/fine_q_best/r2'
BRIDGE_STORE = STORE.parent / 'bridge_restart4_20261027'
EPOCHS, MICRO_PER_EPOCH, ACCUMULATION = (36, 340, 5)
UPDATES_PER_EPOCH, UPDATES, MICROBATCHES = (68, 2448, 12240)

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()

def is_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None

def validate_scope(fold, seed, phase, gpu):
    require(type(fold) is int and fold == 2, 'Only fold2 is authorized')
    require(type(seed) is int and seed == 2027, 'Only seed2027 is authorized')
    require(phase in ('bridge', 'formal'), 'Unknown phase')
    require(type(gpu) is int and gpu == 0, 'Invalid physical GPU')

def validate_scientific_config(cfg, ancestor):
    import copy
    a, b = (copy.deepcopy(ancestor), copy.deepcopy(cfg))
    require(b['dataset']['fold'] == 2 and b['experiment']['seed'] == 2027, 'Wrong fold/seed')
    require(b['dataset']['split_root'] == 'expes/semi05_gate/public_ncase_20261019/splits' and b['dataset']['fold_cv_dir_template'] == 'data/Public_433_Folds/fold_{fold}', 'Wrong dataset')
    require(b['semi']['matrix_group'] == 'A1B' and b['semi']['matrix_c'] == 'B', 'Not original HPQ')
    require(b['semi']['ucp_epochs'] == 36 and b['semi']['ucp_accumulation'] == 5, 'Wrong training budget')
    require(b['semi']['paired_refresh_labels'] is False and b['semi']['paired_workers'] == 0, 'Frozen pairing changed')
    frozen = {'ucp_encoder_lr': 5e-08, 'ucp_decoder_lr': 2e-07, 'ucp_coarse_lr': 2e-07, 'fclr_head_lr': 0.0001, 'zea_lr': 0.0001, 'xsa_cross_gain': 0.5, 'cross_style_complement': 'two-donor-full-decoder'}
    for k, v in frozen.items():
        require(b['semi'].get(k) == v, 'Frozen field differs: ' + k)
    b['dataset']['fold'] = a['dataset']['fold']
    b['dataset']['split_root'] = a['dataset']['split_root']
    for obj in (a, b):
        for key in ('gpu', 'output_dir'):
            obj['experiment'].pop(key, None)
        obj['semi'].pop('paired_checkpoint_dir', None)
    require(a == b, 'Unreviewed scientific config change')

def validate_initialization(path, digest, acceptance):
    require(str(path) == str(INIT) and digest == INIT_SHA, 'Not the accepted own-ten final')
    require(acceptance.get('status') == 'VERIFIED_COMPLETE', 'Own-ten supervisor not accepted')
    require(acceptance.get('final_sha256') == INIT_SHA, 'Acceptance weight differs')
    require(acceptance.get('final_metrics_sha256') == METRICS_SHA, 'Acceptance metrics differ')

def validate_checkpoint_metadata(payload):
    require(payload.get('format') == 'public_basic_sup_fixed_final_student_v1', 'Not basic supervised final')
    for k, v in (('completed_epochs', 60), ('epoch', 59), ('successful_updates', 600), ('test_evaluations_before_save', 0)):
        require(type(payload.get(k)) is int and payload[k] == v, 'Invalid checkpoint field: ' + k)
    require(payload.get('selection_rule') == 'fixed_epoch_60_student', 'Not fixed final selection')
    require(isinstance(payload.get('student'), dict) and payload['student'], 'Missing full student state')

def validate_splits(splits):
    sizes = {'train_lab': 10, 'train_unlab': 336, 'train_whole': 346, 'test': 87}
    require(set(splits) == set(sizes), 'Unexpected split roles')
    sets = {}
    for role, size in sizes.items():
        ids = splits[role]
        require(isinstance(ids, (tuple, list)) and len(ids) == size, 'Count: ' + role)
        require(all((isinstance(x, str) and re.fullmatch('[A-Za-z0-9_-]+', x) for x in ids)), 'Unsafe case ID')
        require(len(set(ids)) == size, 'Duplicate: ' + role)
        sets[role] = set(ids)
    require(not sets['train_lab'] & sets['train_unlab'], 'Lab/unlab overlap')
    require(sets['train_lab'] | sets['train_unlab'] == sets['train_whole'], 'Training union mismatch')
    require(not sets['train_whole'] & sets['test'], 'Train/test overlap')
    return {'file_id_boundary_verified': True, 'patient_alias_boundary_verified': False, 'unique_patient_count': None}

def epoch_tokens(epoch):
    require(type(epoch) is int and 0 <= epoch < EPOCHS, 'Epoch out of budget')
    return [(epoch, 0, i) for i in range(336)] + [(epoch, 1, (epoch * 4 + j) % 336) for j in range(4)]

def validate_loaded_case(sample, expected_id, expected_index, labeled):
    require(sample.get('case_id') == expected_id, 'Actual returned case differs')
    require(type(sample.get('actual_index')) is int and sample['actual_index'] == expected_index, 'Actual returned index differs')
    keys = {'image', 'case_id', 'actual_index', 'gt_roi_used', 'origin_adjusted', 'direction_adjusted'}
    if labeled:
        keys.add('label')
    require(set(sample) == keys, 'Unexpected/missing sample key, possible hidden label')
    require(sample['gt_roi_used'] is True, 'Inherited ROI limitation must not be hidden')
    require(type(sample['origin_adjusted']) is bool and type(sample['direction_adjusted']) is bool, 'Geometry adjustment provenance is missing')

def validate_micro_trace(row, splits, epoch, step, labeled_order):
    require(type(step) is int and 0 <= step < MICRO_PER_EPOCH, 'Microstep out of range')
    require(len(labeled_order) == 10 and set(labeled_order) == set(range(10)) and all((type(i) is int for i in labeled_order)), 'Wrong frozen labeled permutation')
    token = epoch_tokens(epoch)[step]
    li = labeled_order[step % 10]
    ui = token[2]
    expected = {'epoch': epoch, 'step': step, 'labeled_index': li, 'unlabeled_index': ui, 'labeled_case_id': splits['train_lab'][li], 'unlabeled_case_id': splits['train_unlab'][ui], 'unlabeled_replay': token[1], 'padding': bool(token[1]), 'padding_case_id': splits['train_unlab'][ui] if token[1] else None, 'unlabeled_label_returned': False, 'gt_roi_used': True}
    for k, v in expected.items():
        require(type(row.get(k)) is type(v) and row[k] == v, 'Trace mismatch: ' + k)
    return expected

def validate_epoch_trace(rows, splits, epoch, labeled_order):
    require(len(rows) == MICRO_PER_EPOCH, 'Incomplete epoch microtrace')
    for step, row in enumerate(rows):
        validate_micro_trace(row, splits, epoch, step, labeled_order)
    expected = Counter(splits['train_unlab'])
    for j in range(4):
        expected[splits['train_unlab'][(epoch * 4 + j) % 336]] += 1
    require(Counter((r['unlabeled_case_id'] for r in rows)) == expected, 'Actual unlab coverage differs')
    require(set((r['labeled_case_id'] for r in rows)) == set(splits['train_lab']), 'Lab coverage differs')
    return {'epoch': epoch, 'microbatches': 340, 'updates': 68, 'unique_unlabeled': 336, 'unique_labeled': 10, 'padding_case_ids': [splits['train_unlab'][(epoch * 4 + j) % 336] for j in range(4)]}

def validate_test_boundary(state):
    for k, v in (('completed_epochs', 36), ('updates', 2448), ('microbatches', 12240), ('saved_epoch', 35), ('test_evaluations_started', 35)):
        require(type(state.get(k)) is int and state[k] == v, 'Final test budget violation: ' + k)
    require(state.get('phase') == 'formal', 'Bridge cannot evaluate test')
    require(state.get('strict_reload_verified') is True, 'Final not strictly reloaded')
    require(state.get('final_path') == str(STORE / 'final.pth'), 'Noncanonical final target')
    require(is_sha(state.get('final_sha256')) and state.get('reloaded_sha256') == state['final_sha256'], 'Final/reload identity differs')

def claim_final_test(output, state):
    validate_test_boundary(state)
    output = Path(output)
    require(output.as_posix() == str(BASE / 'formal_r1'), 'Wrong final output')
    final = Path(state['final_path'])
    for path in (output, final):
        require(not any((p.is_symlink() for p in (path, *path.parents))), 'Symlink/ancestor not allowed')
    require(output.is_dir() and final.is_file(), 'Missing final output/weight')
    require(sha_file(final) == state['final_sha256'], 'Final changed before evaluation')
    marker = output / 'test_evaluation_started.json'
    with marker.open('x', encoding='utf-8') as f:
        json.dump({'final_sha256': state['final_sha256'], 'test_cases': 87, 'selection_rule': 'fixed_epoch35_native_student', 'automatic_retry': False}, f)
        f.flush()
        os.fsync(f.fileno())
    return marker
