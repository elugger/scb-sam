"""One-shot Public05/fold0-4/2027 final audit. No test forward or optimizer update.

Pure validators and --help import only the standard library. Execution reuses
the frozen helpers and the new bridge/disk-gate contract, never an old PASS as
a new-dataset/fold qualification. Existing audit directories, including failures, are
not retryable. This is development evidence, not clean confirmation.
"""
import argparse
import copy
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import time
import traceback

GROUPS = ('A1B',)
GRADIENT_GROUPS = {'encoder', 'decoder', 'prompt', 'coarse', 'head', 'adapter'}
ROOT_NAMES = {'student', 'head', 'adapter', 'teacher', 'teacher_head', 'teacher_adapter'}
if __package__:
    contract = importlib.import_module('.fcd_public05_contract_20261015', __package__)
else:
    contract = importlib.import_module('fcd_public05_contract_20261015')
PROTOCOL = 'PUBLIC05_A1B_HISTORICAL_DEVELOPMENT_V1'
PURPOSE = 'historical_test_selected_Public05_original_A1B_development'
TAG = 'formal05_s2027_r1'


def _budget(fold):
    require(type(fold) is int and fold in range(5), 'An explicit Public05 fold0-4 is required')
    result = contract.budget(fold)
    require(result['epochs'] == 36 and result['updates_per_epoch'] == 66
            and result['microbatches_per_epoch'] == 330 and result['updates'] == 2376
            and result['microbatches'] == 11880 and result['accumulation'] == 5
            and result['lab_count'] == 17 and result['unlab_count'] == (329 if fold < 3 else 330),
            'Frozen Public05 per-fold exposure/budget changed')
    return result


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def _reject_constant(value):
    raise ValueError('Non-finite JSON: ' + value)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=_reject_constant)


def read_jsonl(path):
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    require(lines and all(line.strip() for line in lines), 'Missing/blank ledger row: ' + str(path))
    return [json.loads(line, parse_constant=_reject_constant) for line in lines]


def finite(value, minimum=0., strict=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > minimum if strict else value >= minimum)


def _same(left, right, tolerance=1e-12):
    return finite(left) and finite(right) and math.isclose(left, right, rel_tol=0., abs_tol=tolerance)


def _gradient_map(value, positive):
    require(isinstance(value, dict) and set(value) == GRADIENT_GROUPS, 'Wrong six gradient groups')
    require(all(finite(item, strict=positive) for item in value.values()), 'Non-finite/nonpositive gradient group')


def stream_seed(base, stream, *keys):
    """Read-only replay of the frozen paired_rng_data stream-seed formula."""
    payload = repr((int(base), stream, keys)).encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], 'little')


def validate_identity(group, manifest, final, cfg):
    """Pure identity/budget validator; actual file hashes are checked separately."""
    fold = manifest.get('fold')
    budget = _budget(fold)
    require(group in GROUPS, 'Only original A1B is authorized')
    for record in (manifest, final):
        require(record.get('group') == group and record.get('ratio') == '05'
                and type(record.get('fold')) is int and record['fold'] == fold
                and type(record.get('seed')) is int and record['seed'] == 2027, 'Scope mismatch')
        require(record.get('protocol') == PROTOCOL
                and record.get('primary_metric') == 'native_student_epoch35_final', 'Protocol mismatch')
        require(record.get('purpose') == PURPOSE and record.get('clean_confirmation') is False,
                'Historical initialization must remain development-only')
    require(manifest.get('phase') == 'formal' and manifest.get('tag') == TAG, 'Not the registered formal run')
    require(manifest.get('budget') == budget and manifest.get('tail_policy') == final.get('tail_policy') == contract.TAIL_POLICY,
            'Frozen per-fold budget/tail-policy identity changed')
    require(final.get('completed_epochs') == 36 and final.get('updates') == 2376
            and final.get('microbatches') == 11880 and final.get('stop_reason') == 'max_epochs', 'Not a normal complete run')
    require(final.get('exact_resume_supported') is False, 'Research archive cannot claim exact resume')
    require(finite(final.get('wall_seconds'), strict=True) and type(final.get('peak_bytes')) is int
            and final['peak_bytes'] > 0, 'Missing actual resource measurements')
    require(set(final.get('weights', {})) == {'best.pth', 'final.pth'}, 'Both research archives required')
    require(cfg['experiment']['seed'] == 2027 and cfg['dataset']['fold'] == fold
            and cfg['dataset']['split_root'] == 'expes/splits/public/public_semi05_cv5', 'Config scope mismatch')
    semi = cfg['semi']
    require(semi['matrix_group'] == group and semi['matrix_c'] == ('0' if group == 'A00' else 'B'), 'Wrong frozen recipe')
    require(semi['ucp_epochs'] == 36 and semi['ucp_accumulation'] == 5
            and semi['paired_refresh_labels'] is False, 'Frozen budget/label sampling changed')
    if group == 'A1B':
        require(semi['cross_style_complement'] == 'two-donor-full-decoder', 'Missing frozen S path')
    return True


def validate_ledgers(group, final, epochs, steps, metrics, traces, train_unlab_names, stream_seed_fn=None, train_lab_names=None):
    """Pure full-ledger validation; no model or new performance measurement.

    Schema is the frozen matrix ledger plus measured ema_calls on all groups.
    Only recorded branch signals are checked; no unrecorded embedding-gradient
    or teacher-energy-variance measurement is implied.
    """
    require(group in GROUPS, 'Unsupported group')
    budget = _budget(final.get('fold'))
    require(len(epochs) == 36 and len(steps) == 2376 and len(traces) == 11880 and len(metrics) == 37,
            'Incomplete full ledger')
    require([r['epoch'] for r in epochs] == list(range(36))
            and [r['epoch'] for r in metrics] == list(range(-1, 36)), 'Non-contiguous/duplicate epoch metrics')
    require(final['updates'] == 2376 and final['microbatches'] == 11880, 'Wrong final totals')
    require(all(finite(r['dice']) and r['dice'] <= 1. for r in metrics), 'Invalid Dice scale')
    require(_same(final['floor_dice'], metrics[0]['dice'], 1e-8), 'Floor mismatch')
    best, best_epoch = final['floor_dice'], -1
    for index, row in enumerate(epochs):
        require(row['updates'] == (index + 1) * 66 and _same(row['dice'], metrics[index + 1]['dice']), 'Epoch update/metric mismatch')
        require(finite(row['seconds'], strict=True), 'Invalid epoch wall time')
        if row['dice'] > best:
            best, best_epoch = row['dice'], index
        require(_same(row['best'], best), 'Best history changed')
    require(_same(final['final_dice'], epochs[-1]['dice']) and _same(final['best_dice'], best)
            and final['best_epoch'] == best_epoch, 'Final/best selection mismatch')
    legal = set(train_unlab_names)
    require(len(legal) == len(train_unlab_names) == budget['unlab_count'], 'Missing/duplicate/wrong-count legal donor names')
    require(train_lab_names is not None and len(set(train_lab_names)) == len(train_lab_names) == 17
            and not legal.intersection(train_lab_names), 'Missing/duplicate/overlapping labeled split')
    seed_fn = stream_seed_fn or stream_seed
    for index, row in enumerate(steps):
        epoch, step_start = index // 66, (index % 66) * 5
        require(row['epoch'] == epoch and row['update'] == index + 1 and row['group'] == group, 'Update discontinuity')
        require(row['skipped'] is False and row['teacher_no_grad'] is True and row['clip_calls'] == 1
                and row['ema_calls'] == 1 and row['context_computed'] is True, 'Update/teacher/clip/EMA gate failed')
        require(finite(row['scale_before'], strict=True) and finite(row['scale_after'], strict=True)
                and row['scale_after'] >= row['scale_before'], 'Invalid scaler/no-skip accounting')
        require(finite(row['seconds'], strict=True) and len(row['microbatches']) == 5, 'Invalid accumulated update')
        _gradient_map(row['gradient_groups'], True)
        if group == 'A1B':
            require(row['branch_routing_measured'] is False, 'Qualification branch probe executed in formal')
        for offset, micro in enumerate(row['microbatches']):
            step = step_start + offset
            require(micro['step'] == step and 'branch_routes' not in micro, 'Microbatch discontinuity/retained branch probe')
            require(all(finite(micro[key]) for key in ('supervised', 'pseudo', 'wsv', 'baseline_total')), 'Invalid native losses')
            c = micro['context']
            require(c['axis'] == (epoch + step) % 3 and c['reduction'] == ('0' if group == 'A00' else 'B')
                    and c['coefficient'] == 1. and c['applied'] is (group != 'A00'), 'Context routing/application changed')
            require(c['declared_state_restored'] is True and c['no_gt_argument'] is True
                    and c['relation_computed'] is False, 'Context state/data boundary failed')
            bg, fg, blocks = c['background_voxels'], c['foreground_voxels'], c['reliable_blocks']
            require(all(type(v) is int and v >= 0 for v in (bg, fg, blocks)) and bg + fg == 64 * blocks, 'Invalid 64-voxel counts')
            counts = [v for v in (bg, fg) if v]
            terms = c['class_losses']
            require(len(terms) == len(counts) and all(finite(v) for v in terms), 'Invalid present-class losses')
            balanced = sum(terms) / len(terms) if terms else 0.
            voxel = sum(n * loss for n, loss in zip(counts, terms)) / sum(counts) if counts else 0.
            require(_same(c['balanced'], balanced, 2e-6) and _same(c['voxel'], voxel, 2e-6), 'Class reduction formula mismatch')
            multiplier = (bg + fg) / len(counts) / fg if fg else 0.
            require(_same(c['foreground_multiplier'], multiplier, 1e-9), 'Foreground multiplier mismatch')
            _gradient_map(c['raw_gradient_norms'], blocks > 0)
            if group == 'A1B':
                style = micro['style']
                require(style['donor1'] in legal and style['donor2'] in legal and style['donor1'] != style['donor2'], 'Illegal/same donor')
                first = seed_fn(2027, 'B-style-donor-1', epoch, step) % len(train_unlab_names)
                second = seed_fn(2027, 'B-style-donor-2', epoch, step) % len(train_unlab_names)
                if second == first:
                    second = (second + 1) % len(train_unlab_names)
                require(style['donor1'] == train_unlab_names[first] and style['donor2'] == train_unlab_names[second],
                        'Style donor stream is not the frozen seed2027 stream')
                require(style['donor_labels_used'] is False and style['native_labeled_supervision'] is True
                        and style['native_teacher'] is True and style['old_fvc_active'] is False
                        and style['complementary_sum'] == 2., 'S route is not a replacement with complementary channels')
                require(all(finite(style[k], strict=True) for k in ('pseudo1', 'pseudo2', 'extra1', 'extra2'))
                        and all(finite(style[k], minimum=1e-3, strict=True) for k in ('style_delta1', 'style_delta2')), 'Missing recorded two-branch action')
                require(math.isclose(micro['pseudo'], .5 * (style['pseudo1'] + style['pseudo2']), rel_tol=.002, abs_tol=1e-5)
                        and math.isclose(micro['wsv'], .5 * (style['extra1'] + style['extra2']), rel_tol=.002, abs_tol=1e-5), 'Two-branch average mismatch')
    for index, row in enumerate(traces):
        epoch, step = divmod(index, 330)
        require(row['epoch'] == epoch and row['step'] == step, 'Data trace discontinuity')
        require(row['student_seed'] == seed_fn(2027, 'student-step', epoch, step), 'Student stream is not seed2027')
        require(all(type(row[k]) is int and row[k] >= 0 for k in ('labeled_index', 'label_seed', 'unlabeled_index', 'unlabeled_seed')),
                'Invalid data stream trace')
        expected_index = step if step < budget['unlab_count'] else epoch % budget['unlab_count']
        expected_replay = 0 if step < budget['unlab_count'] else 1
        require(row['unlabeled_index'] == expected_index, 'Unlabeled order/padding index changed')
        require(row.get('unlabeled_replay') == expected_replay
                and row.get('padding') is (expected_replay == 1), 'Explicit tail exposure trace changed')
        require(row.get('padding_case_id') == (train_unlab_names[expected_index] if expected_replay else None),
                'Padding case must be the declared legal train_unlab case')
        require(row['labeled_index'] < len(train_lab_names), 'Labeled index outside legal split')
        require(row['label_seed'] == seed_fn(2027, 'lab', epoch, 0, row['labeled_index'])
                and row['unlabeled_seed'] == seed_fn(2027, 'unlab', epoch, expected_replay, row['unlabeled_index']),
                'Label/unlabeled stream is not frozen seed2027 with declared replay0/replay1')
        first_cycle = [item['labeled_index'] for item in traces[epoch * 330:epoch * 330 + len(train_lab_names)]]
        require(set(first_cycle) == set(range(len(train_lab_names)))
                and row['labeled_index'] == first_cycle[step % len(train_lab_names)], 'Frozen label cycle changed')
        if epoch == 0:
            require(all(isinstance(row.get(k), str) and len(row[k]) == 64 for k in ('label_image_sha', 'unlab_image_sha')), 'Missing first-epoch image fingerprints')
    return dict(completed_epochs=36, total_updates=2376, context_microbatches=11880,
                natural_unlab_count=budget['unlab_count'], tail_padding_microbatches=36*(330-budget['unlab_count']),
                context_applied_microbatches=0 if group == 'A00' else 11880,
                data_trace_rows=11880, complete_epoch_checks=[0, 11, 29, 30, 35],
                formal_branch_embedding_gradients_measured=False, teacher_energy_variance_measured=False)


def _runner():
    if __package__:
        return importlib.import_module('.run_fcd_public05_a1b_20261015', __package__)
    return importlib.import_module('run_fcd_public05_a1b_20261015')


def validate_cross_styles(rows, steps):
    require(len(rows) == 11880 and len(steps) == 2376, 'Incomplete cross-style/step ledger')
    expected = [micro['style'] for step in steps for micro in step['microbatches']]
    require(rows == expected, 'Cross-style records differ from actual step records')
    for index, row in enumerate(rows):
        epoch, step = divmod(index, 330)
        require(row['epoch'] == epoch and row['step'] == step, 'Cross-style order changed')
    return True


def _file(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size > 0, 'Missing/nonregular evidence: ' + str(path))
    return path


def exact_dataset_class(base):
    """Retain native transforms/thresholds; refuse recursive case substitution."""
    class ExactCaseDataset(base):
        def __getitem__(self, index):
            require(not getattr(self, '_audit_loading', False), 'Native dataset attempted case resampling during per-case audit')
            self._audit_loading = True
            try:
                return super().__getitem__(index)
            finally:
                self._audit_loading = False
    return ExactCaseDataset


def audit_state(reader, trainer):
    """CPU fingerprints using the own runner; no extra model forward."""
    roots = trainer.roots()
    return dict(roots_optimizer_scaler=reader.state(), rng=reader.rng(),
                modes={key: [m.training for m in root.modules()] for key, root in roots.items()},
                gradients=reader.signature({key: [p.grad for p in root.parameters()] for key, root in roots.items()}),
                updates=trainer.matrix_updates)


def preflight(group, fold, gpu=None):
    """Read-only complete filesystem preflight. No torch/model construction."""
    runner = _runner()
    spec = runner.build_spec(group, gpu='0' if gpu is None else str(gpu), fold=fold)
    output = Path(spec['output'])
    manifest = read_json(_file(output / 'manifest.json'))
    spec = runner.build_spec(group, gpu=manifest['gpu'], fold=fold)
    require(gpu is None or str(gpu) == manifest['gpu'], 'Requested audit GPU differs from this fold')
    runner.validate_manifest(manifest, spec)
    runner.validate_input_identity(manifest, spec)
    final = read_json(_file(output / 'final_metrics.json'))
    require(not (output / 'failure.json').exists(), 'Formal failure evidence exists')
    require(sha(_file(output / 'config.yaml')) == manifest['config_sha256'], 'Saved config changed')
    import yaml
    cfg = yaml.safe_load((output / 'config.yaml').read_text(encoding='utf-8'))
    validate_identity(group, manifest, final, cfg)
    require(cfg['experiment']['output_dir'] == spec['output'] and str(cfg['experiment']['gpu']) == manifest['gpu']
            and cfg['semi']['paired_checkpoint_dir'] == spec['store'], 'Saved config path/device identity changed')
    require(runner.json_sha(runner.semantic_config(cfg)) == manifest['semantic_config_sha256'], 'Semantic config changed')
    root = Path(str(runner.ROOT))
    def rooted(path):
        path = Path(path)
        return path if path.is_absolute() else root / path
    for path, digest in manifest['source_hashes'].items():
        require(sha(_file(rooted(path))) == digest, 'Frozen/new source changed: ' + path)
    require(str(root / 'expes/semi05_gate/audit_fcd_public05_a1b_final_20261015.py') in manifest['source_hashes'], 'Auditor not bound into source chain')
    require(sha(_file(rooted(manifest['init_from']))) == manifest['init_sha256'], 'Actual initialization changed')
    split = rooted(cfg['dataset']['split_root']) / ('fold'+str(fold))
    require({name: sha(_file(split / name)) for name in manifest['split_hashes']} == manifest['split_hashes'], 'Actual split changed')
    ancestor = manifest['ancestor']
    require(sha(_file(ancestor['proof_path'])) == ancestor['proof_sha256'], 'Engineering ancestor proof changed')
    ancestor_config = _file(ancestor['config_path'])
    require(sha(ancestor_config) == ancestor['config_sha256'], 'Engineering ancestor scientific config changed')
    runner.validate_config_bridge(cfg, yaml.safe_load(ancestor_config.read_text(encoding='utf-8')))
    old_weight = _file(ancestor['checkpoint']['path'])
    require(old_weight.stat().st_size == ancestor['checkpoint']['bytes'] and sha(old_weight) == ancestor['checkpoint']['sha256'],
            'Engineering ancestor weight changed')
    bridge_path = _file(spec['bridge_report'])
    require(manifest['bridge']['path'] == str(bridge_path) and sha(bridge_path) == manifest['bridge']['sha256'], 'Wrong new bridge receipt')
    bridge, bridge_manifest = read_json(bridge_path), read_json(_file(bridge_path.parent / 'manifest.json'))
    runner.validate_manifest(bridge_manifest, runner.build_spec(group, gpu=bridge_manifest['gpu'], phase='bridge', fold=fold))
    runner.validate_bridge_report(bridge, manifest)
    require(bridge['manifest_sha256'] == manifest['bridge']['manifest_sha256'] == sha(bridge_path.parent / 'manifest.json'), 'Bridge manifest changed')
    require(not (bridge_path.parent / 'failure.json').exists(), 'New bridge failed')
    gate_path = _file(spec['startup_gate'])
    gate = read_json(gate_path)
    runner.validate_startup_gate(gate, manifest)
    require(gate['manifest_sha256'] == sha(output / 'manifest.json') and gate['bridge_sha256'] == sha(bridge_path)
            and final['startup_gate_sha256'] == sha(gate_path), 'Update-zero disk gate is not bound to this run')
    require(gate['checkpoint']['path'] == str(Path(spec['store']) / 'best.pth'), 'Disk gate points outside this run')
    # Initial best may be replaced later. Its receipt is an event, not a claim
    # that the current best file still has the update-zero SHA.
    for name, metadata in final['weights'].items():
        expected = Path(spec['store']) / name
        require(metadata['path'] == str(expected), 'Weight points outside registered run')
        path = _file(expected)
        require(path.stat().st_size == metadata['bytes'] and 0 < metadata['bytes'] <= runner.CHECKPOINT_BOUND
                and sha(path) == metadata['sha256'], 'Actual research archive changed')
        require(not Path(str(path) + '.partial').exists(), 'Unresolved save partial')
    names = [line.strip() for line in (split / 'train_unlab.txt').read_text().splitlines() if line.strip()]
    lab_names = [line.strip() for line in (split / 'train_lab.txt').read_text().splitlines() if line.strip()]
    epochs, steps = read_jsonl(output / 'train_epochs.jsonl'), read_jsonl(output / 'matrix_steps.jsonl')
    metrics, traces = read_jsonl(output / 'metrics.jsonl'), read_jsonl(output / 'pair_trace.jsonl')
    counts = validate_ledgers(group, final, epochs, steps, metrics, traces, names, train_lab_names=lab_names)
    require(_same(gate['floor_dice'], final['floor_dice'], 1e-12), 'Floor precision/identity mismatch')
    runner.contract.validate_floor(final['floor_dice'], fold=fold)
    style_rows = read_jsonl(output / 'cross_style_steps.jsonl')
    validate_cross_styles(style_rows, steps)
    counts['cross_style_rows'] = len(style_rows)
    # The original loader uses NumPy's legacy RandomState for its requested
    # label ordering. Recompute only this CPU token sequence, not any image
    # transform or model forward. Requested indices do not claim that legacy
    # dataset low-volume resampling was absent.
    import numpy as np
    for epoch in range(36):
        order = np.random.RandomState(stream_seed(2027, 'lab-order', epoch)).permutation(len(lab_names)).tolist()
        require([row['labeled_index'] for row in traces[epoch * 330:(epoch + 1) * 330]]
                == [order[step % len(order)] for step in range(330)], 'Seed2027 requested label permutation changed')
    counts['label_order_seed2027_verified'] = True
    counts['trace_scope'] = 'Requested indices and recorded augmentation/student/donor streams; legacy dataset resampling is unchanged, not independently excluded.'
    evidence = {name: sha(_file(output / name)) for name in ('manifest.json', 'config.yaml', 'final_metrics.json',
                'train_epochs.jsonl', 'matrix_steps.jsonl', 'metrics.jsonl', 'pair_trace.jsonl', 'cross_style_steps.jsonl', 'startup_disk_gate.json')}
    return dict(spec=spec, manifest=manifest, final=final, cfg=cfg, epochs=epochs, counts=counts, evidence_sha256=evidence,
                bridge_sha256=sha(bridge_path), split=str(split))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=GROUPS, required=True)
    parser.add_argument('--gpu', choices=('0', '1'), required=True)
    parser.add_argument('--fold', type=int, choices=range(5), required=True)
    parser.add_argument('--cpu-preflight', action='store_true', help='Read-only file/ledger checks; no model, audit directory or CUDA')
    args = parser.parse_args()
    require(__debug__, 'Do not use -O; frozen helper assertions are required')
    runner = _runner()
    require(Path(__file__).resolve() == Path(str(runner.ROOT)) / 'expes/semi05_gate/audit_fcd_public05_a1b_final_20261015.py',
            'Execute the registered deployed auditor only')
    os.chdir(str(runner.ROOT))
    data = preflight(args.group, args.fold, args.gpu)
    if args.cpu_preflight:
        print(json.dumps(dict(status='CPU_PREFLIGHT_PASS_NOT_GPU_AUDIT', group=args.group,
                              **data['counts'], evidence_sha256=data['evidence_sha256']), allow_nan=False))
        return
    destination = Path(data['spec']['output']) / 'final_train_label_audit'
    require(not os.path.lexists(destination), 'Existing audit, including empty/failed output; do not retry')
    require(not list(Path(data['spec']['output']).glob('*audit*')), 'Any prior audit residue must be investigated')
    require(not list(Path(data['spec']['output']).parent.glob(Path(data['spec']['output']).name+'*audit*')), 'Sibling audit log residue')
    runner._memory_gate()
    runner._gpu_empty(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from expes.semi05_gate.paired_hard_relations_20261005 import tensor_state_digest
    from expes.semi05_gate.fcd_deterministic_losses_r2_20261005 import deterministic_training
    from expes.common.semi_data import CaseListDataset, build_eval_transform, read_case_list
    module, class_name = runner.HELPERS[args.group]
    frozen = getattr(importlib.import_module('expes.semi05_gate.' + module), class_name)
    for base in frozen.__mro__:
        if base.__module__.startswith('expes.'):
            source = str(Path(inspect.getfile(base)).resolve())
            require(data['manifest']['source_hashes'].get(source) == sha(_file(source)), 'Audit MRO source changed: '+source)
    class AuditOnly(frozen):
        def _evaluate(self, *unused, **ignored):
            raise AssertionError('Test evaluation is forbidden in final train-label audit')
        def step_matrix(self, *unused, **ignored):
            raise AssertionError('Optimizer update is forbidden in final train-label audit')
    destination.mkdir(exist_ok=False)
    report = dict(schema='FCD_PUBLIC05_A1B_FINAL_AUDIT_V1', status='RUNNING', group=args.group, ratio='05', fold=args.fold, seed=2027,
                  purpose=PURPOSE, clean_confirmation=False, test_forwards=0, new_test_forwards=0, optimizer_updates=0,
                  protocol=PROTOCOL, primary_metric='native_student_epoch35_final', final_dice=data['final']['final_dice'],
                  cases={}, weights={}, source_hashes=data['manifest']['source_hashes'], init_from=data['manifest']['init_from'],
                  init_sha256=data['manifest']['init_sha256'], split_hashes=data['manifest']['split_hashes'],
                   input_registry=data['manifest']['input_registry'], budget=data['manifest']['budget'],
                   tail_policy=data['manifest']['tail_policy'],
                  evidence_sha256=data['evidence_sha256'], bridge_sha256=data['bridge_sha256'], **data['counts'])
    started = time.monotonic()
    def write():
        (destination / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    write()
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        cfg = copy.deepcopy(data['cfg'])
        cfg['experiment'].update(gpu=args.gpu, output_dir=str(destination / 'runtime'))
        with deterministic_training():
            trainer = AuditOnly(cfg)
            trainer.matrix_updates = 0  # Audit-local counter, not a rewrite of checkpoint metadata.
            if args.group == 'A0B': trainer.qualification_component_checks = False
            if args.group == 'A1B': trainer.qualification_routing = False
            require(set(trainer.roots()) == ROOT_NAMES and trainer.pair_seed == 2027, 'Wrong constructed roots/seed')
            reader = object.__new__(runner._Engine)
            reader.torch, reader.t = torch, trainer
            names = read_case_list(Path(data['split']) / 'train_lab.txt')
            require(len(names) == len(set(names)) == 17, 'Missing/duplicate train_lab cases')
            dataset = exact_dataset_class(CaseListDataset)(trainer.lab_set.fold_cv_dir, names, build_eval_transform(128), trainer.lab_set.threshold, split='train')
            for name in ('best.pth', 'final.pth'):
                metadata = data['final']['weights'][name]
                require(sha(metadata['path']) == metadata['sha256'], 'Archive changed after preflight')
                import fcntl
                with Path(data['spec']['save_lock']).open('a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    runner._memory_gate()
                    ck = torch.load(metadata['path'], map_location='cpu')
                require(ck['schema'] == 'FCD_PUBLIC05_A1B_WEIGHT_V1' and ck['group'] == args.group and ck['ratio'] == '05'
                        and ck['fold'] == args.fold and ck['seed'] == 2027 and ck['exact_resume_supported'] is False,
                        'Archive metadata mismatch')
                require(ck['cfg'] == data['cfg'] and set(ck['roots']) == ROOT_NAMES, 'Archive config/root mismatch')
                expected_epoch = 35 if name == 'final.pth' else data['final']['best_epoch']
                require(ck['saved_epoch'] == expected_epoch and ck['updates'] == (expected_epoch + 1) * 66, 'Wrong best/final epoch or updates')
                for key, model in trainer.roots().items():
                    model.load_state_dict(ck['roots'][key], strict=True)
                    loaded = model.state_dict()
                    require(set(loaded) == set(ck['roots'][key]) and all(loaded[k].dtype == value.dtype
                            and torch.equal(loaded[k].detach().cpu(), value)
                            for k, value in ck['roots'][key].items()), 'Six-root loaded values differ: ' + key)
                    model.eval()
                saved_epoch, updates = ck['saved_epoch'], ck['updates']
                del ck
                trainer.optimizer.zero_grad(set_to_none=True)
                before = audit_state(reader, trainer)
                rows = []
                with trainer.isolated_context(-1, -1), torch.no_grad(), torch.cuda.amp.autocast():
                    for index in range(len(dataset)):
                        item = dataset[index]
                        image, label = trainer._prepare_batch({k: v.unsqueeze(0) for k, v in item.items() if torch.is_tensor(v)})
                        logits = trainer._forward_main_logits(trainer.student, image)
                        truth, prediction = label[:, 0] > 0, logits.argmax(1).bool()
                        tp, fp, fn = int((prediction & truth).sum()), int((prediction & ~truth).sum()), int((~prediction & truth).sum())
                        rows.append(dict(case=names[index], tp=tp, fp=fp, fn=fn, dice=2 * tp / max(1, 2 * tp + fp + fn)))
                        del image, label, logits
                        trainer.clear_relation_graph()
                after = audit_state(reader, trainer)
                require(after == before, 'Allowed audit changed roots/optimizer/scaler/RNG/modes/gradients/updates')
                require(all(parameter.grad is None for model in trainer.roots().values() for parameter in model.parameters()), 'Audit created gradients')
                report['cases'][name] = rows
                report['weights'][name] = dict(**metadata, saved_epoch=saved_epoch, updates=updates, all_six_roots_loaded=True,
                                               all_six_root_values_exact=True, zero_update_state_unchanged=True,
                                               requested_case_identity_preserved=True, declared_forward_transaction_restored=True,
                                               full_state_before=before, full_state_after=after,
                                               totals={k: sum(row[k] for row in rows) for k in ('tp', 'fp', 'fn')})
                write()
        report.update(status='COMPLETE', epoch35=data['epochs'][-1], training_wall_seconds=data['final']['wall_seconds'],
                      training_peak_bytes=data['final']['peak_bytes'], audit_seconds=time.monotonic() - started,
                      limitation='Train-label internal FP/FN, not held-out causal evidence. Old test-selected Semi, GT ROI and seen test curves remain; not clean confirmation or exact resume.')
        write()
        print('PUBLIC05_A1B_AUDIT_COMPLETE ' + json.dumps({k: v for k, v in report.items() if k != 'cases'}, allow_nan=False), flush=True)
    except Exception:
        report.update(status='FAIL_ENGINEERING', error=traceback.format_exc(), audit_seconds=time.monotonic() - started)
        write()
        raise


if __name__ == '__main__':
    main()


