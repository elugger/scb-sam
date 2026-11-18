"""Predeclared five/ten-case experiment. No CUDA, metrics-based selection or I/O writes."""
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = PurePosixPath('__RECORDED_PROJECT_ROOT__')
SRC = ROOT / 'expes/semi05_gate/public_ncase_20261019'
BASE = ROOT / 'expes/work_dir/semi05_gate/PUBLIC_NCASE_FOLD2_20261019'
SHARED_STORE = ROOT / 'expes/artifacts/FCD_A_MATRIX_20261006/MAIN_TABLE_FIXED_FINAL_20261019'
SAM_SHA = '899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c'
RANK_PREFIX = 'Public433/fold2/ncase/2027/'


def artifact_dir(n):
    if type(n) is not int or n not in (5, 10):
        raise ValueError('Only five/ten-case design')
    return SHARED_STORE / 'basic_cascade_ncase/Public433' / ('lab%03d' % n) / 'fold2/r1'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def ids(path):
    result = [x.strip() for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
    if not result or len(set(result)) != len(result):
        raise ValueError('Empty/duplicate case IDs')
    if any(not x.startswith('PUBLIC_NATIVE_PREFIX_') or not x[13:].isdigit() for x in result):
        raise ValueError('Unsafe/unexpected case ID')
    return result


def validate_splits(folder):
    folder = Path(folder)
    whole, test, original = [ids(folder / name) for name in ('train_whole.txt', 'test.txt', 'original_lab17.txt')]
    if (len(whole), len(test), len(original)) != (346, 87, 17):
        raise ValueError('Original fold counts differ')
    if set(whole) & set(test) or not set(original) < set(whole):
        raise ValueError('Original fold split overlap/membership')
    ranked = sorted(original, key=lambda c: hashlib.sha256((RANK_PREFIX+c).encode()).hexdigest())
    result = {}
    for n in (5, 10):
        lab, unlab = [ids(folder / ('n%03d_%s.txt' % (n, role))) for role in ('lab', 'unlab')]
        if lab != ranked[:n] or unlab != [x for x in whole if x not in lab]:
            raise ValueError('Count-specific split differs from prospective hash rule')
        if len(unlab) != 346-n or set(lab) & set(unlab) or set(lab+unlab) != set(whole):
            raise ValueError('Count-specific split accounting')
        result[n] = dict(lab=lab, unlab=unlab, test=test)
    return result


def hpq_budget(n):
    artifact_dir(n)
    unlab = 346-n
    micro_per_epoch = ((unlab+4)//5)*5
    return dict(labeled_cases=n, unlabeled_cases=unlab, epochs=36,
                micro_per_epoch=micro_per_epoch, updates_per_epoch=micro_per_epoch//5,
                total_updates=36*micro_per_epoch//5, total_microbatches=36*micro_per_epoch,
                legal_unlabeled_replays_per_epoch=micro_per_epoch-unlab,
                status='PROTOCOL_ONLY_INITIALIZER_SHA_AND_GPU_BRIDGE_PENDING')


def make_config(folder, n):
    folder = Path(folder)
    validate_splits(folder/'splits')
    artifact_dir(n)
    source_pins = json.loads((folder/'dependency_pins.json').read_text())
    for p in sorted(folder.glob('*.py')):
        source_pins[str((SRC/p.name).relative_to(ROOT))] = sha(p)
    input_pins = {str((SRC/'splits'/p.name).relative_to(ROOT)): sha(p)
                  for p in sorted((folder/'splits').glob('*.txt'))}
    for name in ('selection.json', 'dependency_pins.json'):
        input_pins[str((SRC/name).relative_to(ROOT))] = sha(folder/name)
    # Recheck original remote raw lists at preflight. Local snapshots have different newline bytes.
    original = json.loads((folder/'selection.json').read_text())['source_raw_sha256']
    input_pins.update({p: h for p, h in original.items() if '/public_semi05_cv5/' in p})
    return dict(project_root=str(ROOT), output_dir=str(BASE/('sup_n%03d' % n)),
                artifact_dir=str(artifact_dir(n)), fold_cv_dir=str(ROOT/'data/Public_433_Folds/fold_2'),
                train_list=str((SRC/'splits'/('n%03d_lab.txt' % n)).relative_to(ROOT)),
                test_list=str((SRC/'splits/test.txt').relative_to(ROOT)),
                sam_checkpoint='ckpt/sam_med3d_turbo.pth', sam_checkpoint_sha256=SAM_SHA,
                source_sha256=source_pins, input_sha256=input_pins,
                resource_guard_module='ncase_resources', labeled_cases=n, train_count=n, test_count=87,
                fold=2, seed=2027, epochs=60, batch_size=1, num_workers=0,
                test_policy='fixed_last_epoch_once', mode='sup', use_hpq=False, use_unlabeled=False,
                resume=False, amp_init_scale=1024.0, amp_policy='fixed_1024',
                checkpoint_policy='final_student_only_no_resume',
                equal_budget_definition='sixty_exposures_per_labeled_case_not_equal_total_updates',
                data_source_identity_formally_verified=False, patient_alias_boundary_verified=False,
                gt_roi_used=True, fold_selection='historical_development_fold2',
                split_selection='nested_seed2027_hash_rank_of_original17_no_metrics')
