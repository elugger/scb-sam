"""Fixed BY5 shared-supervision contract; no model imports or launch side effects."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

ROOT = PurePosixPath('__RECORDED_PROJECT_ROOT__')
SRC = ROOT / 'expes/semi05_gate/by10_hpq_20261107'
BASE = ROOT / 'expes/work_dir/semi05_gate/BY10_HPQ_20261107'
STORE = ROOT / 'expes/artifacts/FCD_A_MATRIX_20261006/MAIN_TABLE_FIXED_FINAL_20261019/by10_hpq_20261107'
INFRA = ROOT / 'expes/semi05_gate/public_hpq_fineq_n010_restart4_20261027'
CONTROL = BASE / 'control'
GPU_UUID = '__RECORDED_GPU_UUID__'
SAM_SHA = '899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c'
SPLIT = ROOT / 'expes/splits/by_semi10_cv5/fold2'
SPLIT_SHA = {
    'train_lab.txt': '5a63114a06896d1721a8626c660644de63f420ddde55717cebd49d1ddc0c63cb',
    'train_unlab.txt': '744d833bd4aa31c12ea5bf858ecc8cfbe13bc1ffe7db2d0cf09cd3006e34fde0',
    'test.txt': '60ed81fb7520f795f52d62c9e3c83235969671453c810b8f9b22defb02550eda'}
GROUPS = ('HPQ',)
GIB = 1024**3


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def targets():
    rows = [(str(STORE / 'supervised/final.pth'), GIB)]
    rows += [(str(STORE / group / 'final.pth'), 2 * GIB) for group in GROUPS]
    rows += [(str(STORE / name / 'evidence_capacity/final.pth'), 64 * 1024**2)
             for name in ('supervised',) + GROUPS]
    return rows


def base_config():
    return dict(project_root=str(ROOT), output_dir=str(BASE / 'supervised'),
        artifact_dir=str(STORE / 'supervised'), fold_cv_dir=str(ROOT / 'data/BY_125_Folds/fold_2'),
        train_list=str(SPLIT / 'train_lab.txt'), test_list=str(SPLIT / 'test.txt'),
        sam_checkpoint=str(ROOT / 'ckpt/sam_med3d_turbo.pth'), sam_checkpoint_sha256=SAM_SHA,
        resource_guard_module='by_resources', dataset='BY', labeled_cases=10, train_count=10,
        test_count=25, fold=2, seed=2027, epochs=60, batch_size=1, num_workers=0,
        test_policy='fixed_last_epoch_once', mode='sup', use_hpq=False, use_unlabeled=False,
        resume=False, amp_init_scale=1024.0, amp_policy='fixed_1024',
        checkpoint_policy='final_student_only_no_resume',
        equal_budget_definition='sixty_exposures_per_labeled_case',
        gt_roi_used=True, fold_selection='historical_development_fold2',
        split_selection='unchanged_original_BY10_lists',
        patient_alias_boundary_verified=False,
        source_sha256={}, input_sha256={str(SPLIT / n): h for n, h in SPLIT_SHA.items()})


def validate_supervised_config(cfg):
    fixed = base_config()
    for key, value in fixed.items():
        if key not in ('source_sha256', 'input_sha256'):
            require(key in cfg and type(cfg[key]) is type(value) and cfg[key] == value,
                    'Frozen BY5 setting differs: ' + key)
    for key in ('init_from', 'coarse_checkpoint', 'teacher_checkpoint', 'best_checkpoint', 'resume_from'):
        require(cfg.get(key) is None, 'No cohort-selected initializer or resume: ' + key)
    require('label_percent' not in cfg, 'Use exact ten-case identity')
    optional = dict(img_size=128, model_type='vit_b_ori', num_classes=2, deep_supervision=False,
        lr=1e-4, lr_encoder_scale=1.0, lr_decoder_scale=0.1, lr_coarse_scale=1.25,
        weight_decay=0.1, aux_weight=0.3, beta1=0.9, beta2=0.999)
    for key, expected in optional.items():
        require(key not in cfg or (not isinstance(cfg[key], bool) if isinstance(expected, float) else True)
                and cfg[key] == expected, 'Frozen optimizer/model setting differs: ' + key)
    for group in ('source_sha256', 'input_sha256'):
        require(isinstance(cfg.get(group), dict) and cfg[group], 'Missing source/input manifest')
        for digest in cfg[group].values():
            require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest), 'Invalid SHA')
    require(all(cfg['input_sha256'].get(p) == h for p, h in fixed['input_sha256'].items()),
            'Original BY5 split hashes changed')
    return dict(epochs=60, train_count=10, updates_per_epoch=10, total_updates=600,
                batch_size=1, replay_per_epoch=0)


def verify_bundle(expected_sha, folder=None):
    folder = Path(folder or SRC)
    require(sha(folder / 'source_manifest.json') == expected_sha, 'Bundle manifest SHA changed')
    manifest = json.loads((folder / 'source_manifest.json').read_text())
    require(set(manifest) == {'files'}, 'Unexpected source manifest schema')
    require(isinstance(manifest['files'], dict) and bool(manifest['files']), 'Empty source manifest')
    actual = {}
    for name, digest in manifest['files'].items():
        require(Path(name).name == name and name != 'source_manifest.json', 'Unsafe source path')
        path = folder / name
        require(not path.is_symlink() and path.is_file() and sha(path) == digest, 'Source changed: ' + name)
        actual[str(SRC / name)] = digest
    return actual
