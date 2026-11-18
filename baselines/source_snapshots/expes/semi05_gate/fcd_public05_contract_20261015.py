"""Public-433 5% original A1B: independent, stdlib-only five-fold contract.

Historical per-fold test-selected SelfTrain+SAM initializations are explicitly
development inputs. No result-dependent fold choice or automatic retries.
"""
import copy
import contextlib
import hashlib
import json
import math
import os
import stat
from pathlib import Path, PurePosixPath

ROOT = PurePosixPath('__RECORDED_PROJECT_ROOT__')
BASE = ROOT/'expes/work_dir/semi05_gate/FCD_PUBLIC05_A1B_20261015'
STORE = ROOT/'expes/artifacts/FCD_A_MATRIX_20261006'
SPLIT_ROOT = 'expes/splits/public/public_semi05_cv5'
DATA_TEMPLATE = 'data/Public_433_Folds/fold_{fold}'
PROTOCOL = 'PUBLIC05_A1B_HISTORICAL_DEVELOPMENT_V1'
TAGS = {'bridge': 'bridge05_s2027_r1', 'formal': 'formal05_s2027_r1'}
PREPARATION_NAME = 'preparation_ready.json'
REGISTRY_NAME = 'input_registry.json'
ENTRY_SOURCES = ('fcd_public05_contract_20261015.py', 'run_fcd_public05_a1b_20261015.py',
                 'audit_fcd_public05_a1b_final_20261015.py', 'collect_fcd_public05_a1b_20261015.py',
                 'launch_fcd_public05_a1b_20261015.py', 'fcd_public05_trainer_adapter_20261015.py')
BOUND = 1624671895
EPOCHS, UPDATES_PER_EPOCH, ACCUMULATION = 36, 66, 5
UPDATES, MICROBATCHES = 2376, 11880
TAIL_POLICY = 'complete330_repeat_epoch_mod_count_replay1_no_drop'
PRIMARY_METRIC = 'native_student_epoch35_final'
FROZEN_HELPERS = {
    'paired_fcd_a1b_20261009.py': '877648a845acb362e9880c921ffaaabf18ea33db539f030906a9955b53f186b5',
    'paired_fcd_matrix_20261006.py': '0154d1af46d05bbb85d42ed82e736b30f56af513a23f5864619eb5f79fee31c0',
    'paired_l_cross_style_complement_20261001.py': '1edd7bc52af0004db52399be0230ee22388a5d27e62a1f393585accb6ef3ea89',
    'fcd_cac_fg_20260928.py': '6bb817436037de726c4f26349c0f8106aefa982441850b3f7876f8f1be4100b6',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_scope(group='A1B', ratio='05', fold=0, seed=2027, gpu='0', phase='formal'):
    require(group == 'A1B' and ratio == '05', 'Only original A1B Public05 authorized')
    require(type(fold) is int and fold in range(5) and type(seed) is int and seed == 2027, 'Wrong fold or seed')
    require(str(gpu) in ('0', '1') and phase in TAGS, 'Wrong GPU or phase')
    return dict(group=group, ratio=ratio, fold=fold, seed=seed, gpu=str(gpu), phase=phase, tag=TAGS[phase])


def budget(fold):
    validate_scope(fold=fold)
    unlab = 329 if fold < 3 else 330
    return dict(epochs=EPOCHS, updates_per_epoch=UPDATES_PER_EPOCH,
                microbatches_per_epoch=330, updates=UPDATES, microbatches=MICROBATCHES,
                lab_count=17, unlab_count=unlab, test_count=87 if fold < 3 else 86,
                accumulation=ACCUMULATION, padding_per_epoch=330-unlab, tail_policy=TAIL_POLICY)


def epoch_tokens(fold, epoch):
    require(type(epoch) is int and 0 <= epoch < EPOCHS, 'Epoch outside frozen budget')
    count = budget(fold)['unlab_count']
    tokens = [(epoch, 0, i) for i in range(count)]
    if count == 329:
        tokens.append((epoch, 1, epoch % count))
    require(len(tokens) == 330, 'Unexpected data schedule')
    return tokens


def paths(phase='formal', fold=0, gpu='0'):
    scope = validate_scope(fold=fold, gpu=gpu, phase=phase)
    output = BASE/('fold'+str(fold))/'A1B'/scope['tag']
    weight = (STORE/'PUBLIC05_A1B_20261015'/('fold'+str(fold))/'A1B'/scope['tag']
              if phase == 'formal' else output/'runtime_checkpoints')
    return dict(output=str(output), store=str(weight), log=str(output)+'.log',
                dispatch=str(output)+'.dispatch.json', boot=str(output)+'.boot.log',
                started=str(output)+'.started.json', save_lock=str(STORE/'save.lock'))


def capacity_targets(root=STORE):
    root = PurePosixPath(str(root).replace('\\', '/'))
    return [dict(path=str(root/'PUBLIC05_A1B_20261015'/('fold'+str(fold))/'A1B'/TAGS['formal']/name),
                 maximum_bytes=BOUND) for fold in range(5) for name in ('best.pth', 'final.pth')]


def capacity_bytes(present):
    require(len(present) == 10, 'Exactly ten Public05 best/final targets required')
    require(all(x is None or type(x) is int and 0 < x <= BOUND for x in present), 'Invalid target sizes')
    permanent = sum(BOUND-(x or 0) for x in present)
    return dict(remaining_permanent_bytes=permanent, atomic_bytes=BOUND,
                safety_bytes=1024**3, evidence_bytes=16*1024**2,
                required_bytes=permanent+BOUND+1024**3+16*1024**2)


def assert_fresh(phase='formal', fold=0, gpu='0'):
    for key, value in paths(phase, fold, gpu).items():
        if key == 'save_lock':
            continue
        path = Path(value)
        require(not os.path.lexists(path), 'Existing execution evidence: '+value)
        require(not any(p.is_symlink() for p in path.parents), 'Symlink ancestor')
    for target in capacity_targets():
        require(not os.path.lexists(target['path']+'.partial'), 'Unresolved atomic partial')
    return True


def validate_save_lock(binding, fd=None):
    """Pin the pre-existing canonical lock inode, never create/repoint it."""
    require(isinstance(binding, dict) and set(binding) == {'path', 'device', 'inode', 'nlink'}, 'Incomplete original lock identity')
    path = Path(STORE/'save.lock')
    require(binding['path'] == str(path), 'Noncanonical save lock')
    require(type(binding['device']) is int and binding['device'] >= 0
            and type(binding['inode']) is int and binding['inode'] > 0 and binding['nlink'] == 1, 'Invalid lock identity')
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'Symlink lock/ancestor')
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'Original lock must be regular and single-link')
    require((info.st_dev, info.st_ino) == (binding['device'], binding['inode']), 'Original lock inode replaced')
    if fd is not None:
        actual = os.fstat(fd)
        require(stat.S_ISREG(actual.st_mode) and actual.st_nlink == 1
                and (actual.st_dev, actual.st_ino) == (info.st_dev, info.st_ino), 'Opened lock descriptor identity changed')
    return True


@contextlib.contextmanager
def locked_save_lock(binding):
    """Read/write existing lock only, verify descriptor again after locking."""
    import fcntl
    validate_save_lock(binding)
    fd = os.open(binding['path'], os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0))
    try:
        validate_save_lock(binding, fd)
        with os.fdopen(fd, 'r+b') as handle:
            fd = None
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                validate_save_lock(binding, handle.fileno())
                yield handle
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        if fd is not None:
            os.close(fd)


def validate_lists(lists, fold):
    b = budget(fold)
    expected = dict(train_lab=17, train_unlab=b['unlab_count'], train_whole=17+b['unlab_count'], test=b['test_count'])
    require(set(lists) == set(expected), 'Unexpected split roles')
    sets = {}
    for role, count in expected.items():
        values = lists[role]
        require(len(values) == count and len(set(values)) == count, 'Wrong count or duplicate: '+role)
        require(all(isinstance(x, str) and x and x == x.strip() and '/' not in x and '\\' not in x for x in values), 'Malformed case ID')
        sets[role] = set(values)
    require(not sets['train_lab'] & sets['train_unlab'], 'Lab/unlab overlap')
    require(sets['train_lab'] | sets['train_unlab'] == sets['train_whole'], 'Train union mismatch')
    require(not sets['train_whole'] & sets['test'], 'Train/test overlap')
    return dict(file_id_boundary_pass=True, patient_alias_boundary_verified=False, **expected)


def input_spec(fold, registry=None):
    validate_scope(fold=fold)
    if registry is None:
        registry = load_registry()
    require(registry.get('schema') == 'FCD_PUBLIC05_INPUT_REGISTRY_V1' and set(registry.get('folds', {})) == {str(f) for f in range(5)}, 'Wrong registry scope')
    row = registry['folds'][str(fold)]
    require(row.get('fold') == fold, 'Input fold mismatch')
    init = row.get('init', {})
    expected = 'expes/work_dir/public/selftrain_sam/public_semi05_cv5/fold'+str(fold)+'/best.pth'
    require(init.get('path') == expected and is_sha(init.get('sha256')) and type(init.get('bytes')) is int and init['bytes'] > 0, 'Wrong same-fold initialization')
    hashes = row.get('split_hashes', {})
    require({'train_lab.txt', 'train_unlab.txt', 'train_whole.txt', 'test.txt'} <= set(hashes)
            and set(hashes) <= {'train_lab.txt', 'train_unlab.txt', 'train_whole.txt', 'test.txt', 'fold_manifest.json'}
            and all(is_sha(x) for x in hashes.values()), 'Incomplete split identity')
    require(row.get('historical_test_selected') is True and row.get('file_id_boundary_pass') is True, 'Input provenance not reviewed')
    require(row.get('patient_alias_boundary_verified') is False and row.get('unique_patient_count') is None, 'No unsupported patient-identity claim')
    return copy.deepcopy(row)


def load_registry(expected_sha=None):
    path = Path(BASE)/REGISTRY_NAME
    require(path.is_file() and not path.is_symlink() and not any(x.is_symlink() for x in path.parents), 'Unsafe or missing registry')
    if expected_sha is not None:
        require(is_sha(expected_sha) and sha(path) == expected_sha, 'Input registry changed')
    value = json.loads(path.read_text(encoding='utf-8'))
    for fold in range(5):
        input_spec(fold, value)
    return value


def validate_floor(value, inputs=None, fold=None):
    require(type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 1, 'Invalid initialization Dice')
    if inputs is None:
        require(fold is not None, 'Own fold input identity required')
        inputs = input_spec(fold)
    evidence = inputs.get('historical_floor') or {}
    if evidence.get('same_native_evaluator_verified') is True:
        expected = evidence.get('dice')
        require(type(expected) in (float, int) and math.isfinite(expected) and abs(value-expected) <= 1e-4, 'Confirmed historical native floor mismatch')
    return True


def validate_config(cfg, old05):
    previous, current = copy.deepcopy(old05), copy.deepcopy(cfg)
    fold = current['dataset']['fold']
    validate_scope(fold=fold, gpu=str(current['experiment']['gpu']))
    require(previous['dataset']['split_root'] == 'expes/splits/by_semi05_cv5' and previous['dataset']['fold'] == 2, 'Wrong scientific ancestor')
    require(current['dataset']['split_root'] == SPLIT_ROOT and current['dataset']['fold_cv_dir_template'] == DATA_TEMPLATE, 'Wrong Public dataset')
    for key in ('split_root', 'fold', 'fold_cv_dir_template'):
        current['dataset'][key] = previous['dataset'][key]
    for value in (previous, current):
        for key in ('gpu', 'output_dir'):
            value['experiment'].pop(key, None)
        value['semi'].pop('paired_checkpoint_dir', None)
    require(previous == current, 'Unapproved scientific config change')
    require(cfg['semi']['ucp_epochs'] == EPOCHS and cfg['semi']['ucp_accumulation'] == ACCUMULATION, 'Budget changed')
    require(cfg['semi']['matrix_group'] == 'A1B' and cfg['semi']['matrix_c'] == 'B'
            and cfg['semi']['cross_style_complement'] == 'two-donor-full-decoder', 'Not original HPQ')
    return True


def validate_final(final):
    validate_scope(final['group'], final['ratio'], final['fold'], final['seed'], final.get('gpu', '0'))
    b = budget(final['fold'])
    require(final['completed_epochs'] == b['epochs'] and final['updates'] == b['updates']
            and final['microbatches'] == b['microbatches'] and final['stop_reason'] == 'max_epochs', 'Incomplete budget')
    require(final['primary_metric'] == PRIMARY_METRIC and final['exact_resume_supported'] is False, 'Wrong final identity')
    require(math.isfinite(final['final_dice']) and 0 <= final['final_dice'] <= 1, 'Invalid final Dice')
    require(final['clean_confirmation'] is False and final['protocol'] == PROTOCOL and final.get('tail_policy') == TAIL_POLICY, 'Missing development/data schedule disclosure')
    return True


def readiness(checks):
    required = ('independent_runner', 'independent_auditor', 'independent_collector', 'cpu_tests',
                'source_provenance_review', 'patient_boundary_review', 'tail_schedule_review',
                'floor_protocol', 'actual_capacity_ram', 'gpu_idle', 'fresh_dispatch')
    missing = [key for key in required if checks.get(key) is not True]
    return dict(status='PREPARATION_PENDING' if missing else 'READY_FOR_OWN_BRIDGE', missing=missing, formal_launch_allowed=False)
