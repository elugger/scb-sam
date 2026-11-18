import os
import sys
from pathlib import Path
ASNET_ROOT = Path(os.environ.get('ASNET_ROOT', '.')).resolve()
import copy, hashlib, json, os, sys
from pathlib import Path
ROOT = Path(str(ASNET_ROOT))
PREFIX = ROOT / 'expes/semi05_gate/by5_four_arm_rerun_20261030'
CHECK = ROOT / 'expes/semi05_gate/by5_four_arm_initialization_check_20261030'
SRC = ROOT / 'expes/semi05_gate/by5_four_arm_training_20261030'
BASE = ROOT / 'expes/work_dir/semi05_gate/BY5_FOUR_ARM_RERUN_20261030'
STORE = ROOT / 'expes/artifacts/FCD_A_MATRIX_20261006/MAIN_TABLE_FIXED_FINAL_20261019/by5_four_arm_rerun_20261030'
CONTROL = BASE / 'four_arm_control'
GROUPS = {'H': ('A00', '0'), 'HP': ('A10', '0'), 'HQ': ('A0B', 'B'), 'HPQ': ('A1B', 'B')}
INIT_SHA = 'b113d1731a8e535ff482bebe7c2ca0d821760d44e5ea02e59b33cae58557c4be'
CHECK_SHA = '31b81db4d5fd7ebef95944b3726754a1cdb35c14e2b838b4daaa250eef379672'
GPU = os.environ['ASNET_GPU_UUID']
sys.path.insert(0, str(PREFIX))
sys.path.insert(0, str(CHECK))
from by_protocol import require, sha
from prepare_by5 import dump_new, stamp
from by_resources import original_lock, capacity_snapshot, infra

def verify_sources(expected):
    require(__debug__ and Path(__file__).resolve() == SRC / 'four_contract.py', 'Noncanonical contract')
    require(sha(SRC / 'manifest.json') == expected, 'Bundle manifest changed')
    manifest = json.loads((SRC / 'manifest.json').read_text())
    require(set(manifest['files']) == {'four_contract.py', 'four_training.py', 'four_control.py', 'test_four.py'}, 'Bundle set changed')
    for n, h in manifest['files'].items():
        require(Path(n).name == n and (not (SRC / n).is_symlink()) and (sha(SRC / n) == h), 'Source changed: ' + n)
    import qualify_four as q
    q.EXPECTED_MANIFEST_SHA = CHECK_SHA
    cfg, prep, ancestry, accepted = q.verify()
    result_path = BASE / 'four_arm_initialization_check/result.json'
    exited = json.loads((result_path.parent / 'exited.json').read_text())
    result = json.loads(result_path.read_text())
    require(exited['status'] == result['status'] == 'PASS_FOUR_ARM_COMMON_INITIALIZATION', 'Initialization check incomplete')
    require(result['temporary_optimizer_updates'] == result['development_evaluations'] == 0, 'Wrong initialization-check budget')
    require(set(result['results']) == set(GROUPS), 'Missing initialization arm')
    for g, row in result['results'].items():
        require(row == json.loads((result_path.parent / g / 'result.json').read_text()), 'Initialization child receipt changed')
    return (cfg, ancestry, accepted, result)

def verify_prepared(bundle_sha, prepare_sha):
    cfg, ancestor, accepted, initial = verify_sources(bundle_sha)
    require(sha(CONTROL / 'preparation.json') == prepare_sha, 'Preparation changed')
    prep = json.loads((CONTROL / 'preparation.json').read_text())
    require(prep['status'] == 'PREPARED_NO_TRAINING' and prep['bundle_sha256'] == bundle_sha, 'Wrong preparation')
    require(sha(BASE / 'four_arm_initialization_check/result.json') == prep['initialization_check_sha256'], 'Initialization reference changed')
    for n, h in prep['source_hashes'].items():
        require(sha(n) == h, 'Dependency changed: ' + n)
    from by_inputs import identity
    for p, row in prep['inputs'].items():
        require(identity(p, ROOT / 'data') == row['identity'] and sha(p) == row['sha256'], 'Input changed: ' + p)
    return (cfg, ancestor, accepted, initial, prep)

def group_config(ancestor, group, phase):
    require(group in GROUPS and phase in ('bridge', 'formal'), 'Invalid arm/phase')
    cfg = copy.deepcopy(ancestor)
    cfg['experiment'].update(output_dir=str(BASE / phase / group), gpu=GPU, seed=2027)
    cfg['semi'].update(paired_checkpoint_dir=str(STORE / group if phase == 'formal' else BASE / phase / group / 'no_weights'), paired_workers=0, matrix_group=GROUPS[group][0], matrix_c=GROUPS[group][1])
    require(cfg['semi']['ucp_epochs'] == 36 and cfg['semi']['ucp_accumulation'] == 5, 'Wrong fixed budget')
    return cfg

def check_step(row, group, overflow=False):
    require(row['skipped'] is overflow and row['teacher_no_grad'], 'Unexpected overflow or teacher gradient')
    require(row['clip_calls'] == row['ema_calls'] == (0 if overflow else 1), 'Clip/EMA count differs')
    require(row['context_computed'] and len(row['microbatches']) == 5, 'Wrong accumulation')
    if not overflow:
        require(len(row['gradient_groups']) == 6 and all((v is not None and __import__('math').isfinite(v) and (v > 0) for v in row['gradient_groups'].values())), 'Nonfinite/zero component gradient')
    for m in row['microbatches']:
        c = m['context']
        require(c['reduction'] == GROUPS[group][1] and c['coefficient'] == 1 and (c['applied'] == (group in ('HQ', 'HPQ'))), 'CCBS switch differs')
        require(c['no_gt_argument'] and c['declared_state_restored'] and (not c['relation_computed']), 'CCBS boundary differs')
        require(c['foreground_voxels'] + c['background_voxels'] == 64 * c['reliable_blocks'], 'CCBS counts differ')
        require(all((__import__('math').isfinite(float(m[k])) for k in ('supervised', 'pseudo', 'wsv', 'baseline_total'))), 'Nonfinite loss')

def validate_bridge(bundle_sha, prepare_sha):
    path = BASE / 'bridge'
    exit_row = json.loads((path / 'exited.json').read_text())
    result = json.loads((path / 'result.json').read_text())
    require(exit_row['status'] == result['status'] == 'PASS_FOUR_ARM_FINITE_BRIDGE', 'Bridge incomplete/failed')
    require(result['bundle_sha256'] == bundle_sha and result['preparation_sha256'] == prepare_sha, 'Bridge source/preparation differs')
    initial = json.loads((BASE / 'four_arm_initialization_check/result.json').read_text())
    data = []
    for group in GROUPS:
        row = json.loads((path / group / 'result.json').read_text())
        require(sha(path / group / 'result.json') == result['child_result_sha256'][group], 'Bridge child changed')
        require(row['status'] == 'PASS_FINITE_BRIDGE' and row['successful_updates'] == 2 and (row['optimizer_successful_calls'] == 2) and (row['attempted_updates'] == 3) and (row['artificial_overflow_attempts'] == 1) and (row['development_evaluations'] == 0), 'Bridge budget differs')
        require(row['initializer_sha256'] == INIT_SHA and row['initial_roots'] == initial['results'][group]['root_sha256'], 'Bridge initialization differs')
        require(row['overflow_model_optimizer_unchanged'] and row['strict_memory_roundtrip'], 'Bridge invariant failed')
        steps = [json.loads(s) for s in (path / group / 'steps.jsonl').read_text().splitlines()]
        require(len(steps) == 3, 'Missing bridge step evidence')
        for i, s in enumerate(steps):
            check_step(s, group, overflow=i == 2)
        pairs = [json.loads(s) for s in (path / group / 'pairs.jsonl').read_text().splitlines()]
        require(len(pairs) == 15, 'Missing bridge actual pair evidence')
        data.append([{k: p[k] for k in ('epoch', 'step', 'labeled_case', 'unlabeled_case', 'label_seed', 'unlab_seed', 'label_image_sha', 'unlab_image_sha')} for p in pairs])
        for n, h in row['evidence_sha256'].items():
            require(sha(path / group / n) == h, 'Bridge evidence changed: ' + n)
    require(all((d == data[0] for d in data)), 'Four-arm main input path differs')
    return dict(path=str(path / 'result.json'), sha256=sha(path / 'result.json'), validated_at=stamp(), status='PASS_INDEPENDENT_BRIDGE_VALIDATION')
