import os
import sys
from pathlib import Path
ASNET_ROOT = Path(os.environ.get('ASNET_ROOT', '.')).resolve()
import argparse
import copy
import hashlib
import io
import json
import math
import os
import random
import re
import stat
import subprocess
import sys
import time
import traceback
from pathlib import Path
COEFFICIENT = 0.25
EVAL_EVERY = 1
TARGET_GAP_PP = 0.7
TOLERANCE_PP = 0.1
BUNDLE = 'cdbe5d62b0cc519c7af7317cf470b817df8fe01259f4ecf8577a2e11b98b5498'
PREPARE = '35da59c8a68ec350de249941f1f63d23042f5daa45bc3815efcc17d363a38a58'
REFERENCE_SHA = {'H': '92924dbf1e9b036fcd1bf9d931f3eae00e865a2b561ed72d6b3d0b8cd23d20cb', 'HP': 'b56528454847d02b8847fb97908048576262e2c172994657496aa09fddbba5d8', 'HQ': '01d16189a945c3d28ab5b1a3a4e64ddd1e2cbd21a0d71c6928b506cdea507557', 'HPQ': 'f6390acff27262f91cf710e3488695325cbc2e31f722ae672a6292294d788333'}

def configure_runtime(group, run_id):
    if group not in ('H', 'HP', 'HQ', 'HPQ') or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,63}', run_id or ''):
        raise ValueError('INVALID_GROUP_OR_RUN_ID')
    sys.path[:0] = [str(ASNET_ROOT / 'expes/semi05_gate/by5_four_arm_training_20261030'), str(ASNET_ROOT)]
    import four_contract as contract
    for name in ('ROOT', 'STORE', 'SRC', 'GROUPS', 'GPU', 'INIT_SHA', 'sha', 'require', 'dump_new', 'stamp', 'verify_prepared', 'original_lock', 'infra'):
        globals()[name] = getattr(contract, name)
    globals().update(ORIGINAL_GROUP_CONFIG=contract.group_config, ORIGINAL_CHECK_STEP=contract.check_step, ORIGINAL_CAPACITY=contract.capacity_snapshot, OLD_BASE=contract.BASE, ACTIVE_GROUP=group, RUN_ID=run_id, TRIAL_ID=run_id + '_' + group)
    globals()['BASE'] = ROOT / 'expes/work_dir/semi05_gate/HP_HQ_FINAL_20261106' / run_id / group
    globals()['REGISTRY_ROOT'] = ROOT / 'expes/artifacts/FCD_A_MATRIX_20261006/MAIN_TABLE_FIXED_FINAL_20261019'
    globals()['ARTIFACT_FOLDER'] = REGISTRY_ROOT / 'hp_hq_final_20261106' / run_id / group
    globals()['SELECTED_CHECKPOINT'] = ARTIFACT_FOLDER / 'selected/final.pth'
    globals()['LEDGER_RESERVATION'] = ARTIFACT_FOLDER / 'ledger_capacity/final.pth'
    globals()['SELECTION_DIR'] = BASE / 'export'
    globals()['REGISTRY'] = ROOT / 'expes/work_dir/semi05_gate/MAIN_TABLE_FIXED_FINAL_20261019/active_targets.json'
    globals()['REFERENCES'] = {g: OLD_BASE / 'formal' / ('HQ.orig_20261103' if g == 'HQ' else g) / 'result.json' for g in ('H', 'HP', 'HQ', 'HPQ')}

def own_targets():
    return [(SELECTED_CHECKPOINT, (1 if ACTIVE_GROUP == 'HQ' else 2) * 1024 ** 3), (LEDGER_RESERVATION, 96 * 1024 ** 2)]

def group_config(ancestor, group, phase):
    require(group == ACTIVE_GROUP, 'GROUP_MISMATCH')
    cfg = ORIGINAL_GROUP_CONFIG(ancestor, group, phase)
    cfg['experiment']['output_dir'] = str(BASE / phase / group)
    cfg['semi']['paired_checkpoint_dir'] = str(ARTIFACT_FOLDER / 'selected' if phase == 'formal' else BASE / phase / group / 'no_automatic_weights')
    if group == 'HQ':
        cfg['semi'].update(development_ccbs_coefficient=COEFFICIENT, development_only_not_matched_ablation=True, development_eval_every=EVAL_EVERY)
    return cfg

def check_step(row, group, overflow=False):
    require(group == ACTIVE_GROUP, 'GROUP_MISMATCH')
    if group != 'HQ':
        return ORIGINAL_CHECK_STEP(row, group, overflow)
    require(all((m['context']['coefficient'] == COEFFICIENT for m in row['microbatches'])), 'CCBS_COEFFICIENT_MISMATCH')
    shadow = copy.deepcopy(row)
    for micro in shadow['microbatches']:
        micro['context']['coefficient'] = 1.0
    return ORIGINAL_CHECK_STEP(shadow, group, overflow)

def preflight():
    from by_resources import merge, backend
    verify_prepared(BUNDLE, PREPARE)
    load_references()
    require(not BASE.exists() and (not SELECTED_CHECKPOINT.exists()), 'EXISTING_RUN')
    with original_lock():
        guard = backend()
        prior = guard.targets
        guard.targets = lambda: merge(prior(), [(str(p), b) for p, b in own_targets()])
        capacity = guard.capacity_snapshot(12 * 1024 ** 3 + 16 * 1024 ** 2)
    return dict(status='PREFLIGHT_PASS_NO_TRAINING', group=ACTIVE_GROUP, run_id=RUN_ID, coefficient=COEFFICIENT if ACTIVE_GROUP == 'HQ' else None, epochs=36, successful_updates=684, output=str(BASE), checkpoint=str(SELECTED_CHECKPOINT), capacity=[{k: v for k, v in f.items() if k != 'targets'} for f in capacity['filesystems']])

def supervise():
    import fcntl
    from by_runtime import ensure_gpu0_idle
    os.chdir(ROOT)
    preflight()
    fd = os.open(OLD_BASE / 'control/gpu0.lock', os.O_RDWR | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        info = os.fstat(fd)
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'GPU_LOCK_INVALID')
        ensure_gpu0_idle()
        capacity = enroll_capacity()
        BASE.mkdir(parents=True, exist_ok=False)
        dump_new(BASE / 'protocol.json', dict(at=stamp(), group=ACTIVE_GROUP, run_id=RUN_ID, ccbs_coefficient=COEFFICIENT if ACTIVE_GROUP == 'HQ' else None, epochs=36, updates=684, seed=2027, eval_every=1 if ACTIVE_GROUP == 'HQ' else 36, early_stopping=False, within_run_hyperparameter_adaptation=False, selection='native_student_epoch35_fixed_final', checkpoint_policy='save_fixed_final_regardless_of_development_gap', initializer_sha256=INIT_SHA, script_sha256=sha(__file__), capacity=capacity, automatic_retry=False))
        env = dict(os.environ, PYTHONPATH=str(SRC) + ':' + str(ROOT), CUDA_VISIBLE_DEVICES=GPU, CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        for phase in ('bridge', 'formal'):
            command = [sys.executable, '-u', '-B', str(Path(__file__).resolve()), '--action', 'worker', '--group', ACTIVE_GROUP, '--run-id', RUN_ID, '--phase', phase, '--fd', str(fd), '--dev', str(info.st_dev), '--ino', str(info.st_ino)]
            if phase == 'formal':
                command += ['--bridge-sha', sha(BASE / 'bridge' / ACTIVE_GROUP / 'result.json')]
            dump_new(BASE / (phase + '_dispatch.json'), dict(at=stamp(), command=command))
            with (BASE / (phase + '.log')).open('xb') as log:
                child = subprocess.run(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, pass_fds=(fd,))
            require(child.returncode == 0, phase + '_FAILED')
        result_path = BASE / 'formal' / ACTIVE_GROUP / 'result.json'
        result = json.loads(result_path.read_text())
        require(result['completed_epochs'] == 36 and result['successful_updates'] == 684, 'FINAL_BUDGET_INCOMPLETE')
        dump_new(BASE / 'completed.json', dict(at=stamp(), result=str(result_path), sha256=sha(result_path), status=result['status']))
        print(json.dumps(result))
    except BaseException:
        if BASE.exists() and (not (BASE / 'supervisor_failure.json').exists()):
            dump_new(BASE / 'supervisor_failure.json', dict(at=stamp(), traceback=traceback.format_exc()))
        raise
    finally:
        os.close(fd)

def report(group):
    folder = Path(__file__).resolve().parent
    groups = ('H', 'HP', 'HQ', 'HPQ') if group in ('both', 'all') else (group,)
    manifest = json.loads((folder / 'checksums.json').read_text(encoding='utf-8'))
    result = {}
    for name in groups:
        relative = 'results/' + name + '/result.json'
        raw = (folder / relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest[relative]:
            raise RuntimeError('RESULT_SHA_MISMATCH')
        result[name] = json.loads(raw)
    print(json.dumps(result, ensure_ascii=False, indent=2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=('report', 'preflight', 'train', 'worker'), default='report')
    parser.add_argument('--group', choices=('H', 'HP', 'HQ', 'HPQ', 'both', 'all'), default='all')
    parser.add_argument('--run-id')
    parser.add_argument('--phase', choices=('bridge', 'formal'))
    parser.add_argument('--fd', type=int)
    parser.add_argument('--dev', type=int)
    parser.add_argument('--ino', type=int)
    parser.add_argument('--bridge-sha')
    args = parser.parse_args()
    if args.action == 'report':
        report(args.group)
        return
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,63}', args.run_id or ''):
        parser.error('INVALID_RUN_ID')
    if args.action == 'worker' and (args.group in ('both', 'all') or args.phase is None or any((v is None for v in (args.fd, args.dev, args.ino)))):
        parser.error('INVALID_WORKER_ARGUMENTS')
    if args.group in ('both', 'all'):
        for group in ('HP', 'HQ') if args.group == 'both' else ('H', 'HP', 'HQ', 'HPQ'):
            subprocess.run([sys.executable, '-u', '-B', str(Path(__file__).resolve()), '--action', args.action, '--group', group, '--run-id', args.run_id], check=True)
        return
    configure_runtime(args.group, args.run_id)
    if args.action == 'preflight':
        print(json.dumps(preflight()))
    elif args.action == 'train':
        supervise()
    else:
        args.bundle_sha = BUNDLE
        args.prepare_sha = PREPARE
        if args.group == 'HQ':
            configure_ccbs()
            run_hq(args)
        else:
            run_hp(args)

def build_trainer(prefix_cfg, ancestor, accepted, initial, prep, group, phase):
    import torch, yaml
    from train_by5 import make_strict_dataset, read_case_ids, load_trusted_checkpoint, state_dict_sha256
    from by_inputs import identity
    from expes.common.semi_data import build_eval_transform
    from expes.semi05_gate.paired_fcd_matrix_20261006 import MatrixHalf, MatrixStyle, MatrixMixin
    from expes.semi05_gate.paired_zea_cross_scale_half_20260911 import PairedCrossScaleHalf
    from expes.semi05_gate.paired_l_cross_style_complement_20261001 import PairedCrossStyleComplement
    data_cfg = copy.deepcopy(prefix_cfg)
    data_cfg['case_file_metadata'] = prep['case_file_metadata']
    ids = prep['case_ids']
    cfg = group_config(yaml.safe_load(ancestor['config_text']), group, phase)
    Base = MatrixStyle if group in ('HP', 'HPQ') else MatrixHalf

    class NoUnlabeledTarget(torch.utils.data.Dataset):

        def __init__(self, original):
            self.original = original

        def __len__(self):
            return len(self.original)

        def __getitem__(self, index):
            value = self.original[index]
            return {k: v for k, v in value.items() if k not in ('label', 'foreground_voxels')}

    class MatchedBY(Base):

        def _build_dataloaders(self):
            self.lab_set = make_strict_dataset(data_cfg, ids['train_lab'], 'train')
            u = make_strict_dataset(data_cfg, ids['train_unlab'], 'train')
            u.transform = build_eval_transform(128)
            self.unlab_set = NoUnlabeledTarget(u)

        def _load(self, path):
            require(Path(path) == STORE / 'supervised/final.pth', 'Wrong shared initialization')
            with original_lock():
                capacity_snapshot(config=prefix_cfg)
                require(sha(path) == INIT_SHA, 'Initializer changed before load')
                ck = load_trusted_checkpoint(path, map_location='cpu')
                require(ck['successful_updates'] == 300 and ck['completed_epochs'] == 60, 'Initializer budget differs')
                self.student.load_state_dict(ck['student'], strict=True)
                require(state_dict_sha256(self.student.state_dict()) == accepted['initializer']['state_sha256'], 'Initializer values differ')
                for n, m in (('head', self.student_fclr_head), ('adapter', self.student_zea_adapter)):
                    m.load_state_dict(self.cold[n], strict=True)
                for s, t in (('student', 'teacher'), ('head', 'teacher_head'), ('adapter', 'teacher_adapter')):
                    self.roots()[t].load_state_dict(self.roots()[s].state_dict(), strict=True)
                self._fclr_loaded_head = False
                self._matrix_initial_head_state = self.cold['head']
                del ck

        def _donor_amplitude(self, name):
            require(name in ids['train_unlab'], 'Donor outside 95U')
            path = ROOT / 'data/BY_125_Folds/fold_2/train/imagesTr' / (name + '.nii.gz')
            require(identity(path, ROOT / 'data') == prep['inputs'][str(path)]['identity'], 'Donor input identity changed')
            return super()._donor_amplitude(name)

        def _save_ucp(self, *args, **kwargs):
            raise RuntimeError('Only guarded fixed-final save is permitted')

        def _evaluate(self, *args, **kwargs):
            raise RuntimeError('Only fixed-final explicit evaluation is permitted')
    require(MatchedBY.step_matrix is MatrixMixin.step_matrix and MatchedBY.contextual_gradients is MatrixMixin.contextual_gradients, 'Scientific step/CCBS changed')
    require(MatchedBY._losses is (PairedCrossStyleComplement._losses if group in ('HP', 'HPQ') else PairedCrossScaleHalf._losses), 'CCL path differs')
    trainer = MatchedBY(cfg)
    trainer.cold = {n: {k: v.detach().cpu().clone() for k, v in trainer.roots()[n].state_dict().items()} for n in ('head', 'adapter')}
    trainer.initialize_matrix(str(STORE / 'supervised/final.pth'))
    roots = {n: state_dict_sha256(m.state_dict()) for n, m in trainer.roots().items()}
    reference = initial['results'][group]
    require(roots == reference['root_sha256'], 'Training initializer differs from qualified six-root state')
    require([g['lr'] for g in trainer.optimizer.param_groups] == reference['optimizer_lrs'], 'Learning rates changed')
    require(trainer.scaler.state_dict() == reference['scaler'] and (not trainer.optimizer.state) and (trainer.matrix_updates == 0), 'Optimizer/scaler not cold')
    require(trainer.pair_seed == 2027 and (not trainer.refresh) and (trainer.workers == 0), 'Paired execution changed')
    return (trainer, cfg, data_cfg, roots)

def run_hp(args):
    os.chdir(ROOT)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == GPU and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'GPU/determinism environment differs')
    prefix_cfg, ancestor, accepted, initial, prep = verify_prepared(args.bundle_sha, args.prepare_sha)
    from train_by5 import validate_gpu_claim, state_dict_sha256, load_trusted_checkpoint, install_optimizer_step_counter, append_jsonl
    from by_runtime import ensure_gpu0_idle, own_gpu0_fds
    claim = validate_gpu_claim(prefix_cfg, 0, args.fd, args.dev, args.ino)
    ensure_gpu0_idle()
    with original_lock():
        capacity = capacity_snapshot(config=prefix_cfg)
    group, phase = (args.group, args.phase)
    output = BASE / phase / group
    require(not output.exists(), 'Existing arm output; no replay')
    if phase == 'formal':
        bridge = validate_bridge(args.bundle_sha, args.prepare_sha)
        require(bridge['sha256'] == args.bridge_sha, 'Formal bridge binding differs')
        require(not SELECTED_CHECKPOINT.exists(), 'Existing formal model store')
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    from expes.semi05_gate.fcd_deterministic_losses_r2_20261005 import deterministic_training
    from expes.semi05_gate.paired_rng_data_20260908 import stream_seed, tensor_digest
    from utils.cascade_train_utils import normalize_batch
    started = time.monotonic()
    trainer = None
    counter = {'count': 0}
    attempts = 0
    completed_epochs = 0
    input_pairs = []
    exposures = {c: 0 for c in prep['case_ids']['train_lab']}
    unlab_exposures = {c: 0 for c in prep['case_ids']['train_unlab']}
    append = lambda name, row: append_jsonl(output / name, row)

    def signature(value):
        h = hashlib.sha256()

        def visit(x):
            if torch.is_tensor(x):
                h.update(str((x.dtype, tuple(x.shape))).encode())
                h.update(x.detach().cpu().contiguous().numpy().tobytes())
            elif isinstance(x, dict):
                for k in sorted(x, key=str):
                    h.update(str(k).encode())
                    visit(x[k])
            elif isinstance(x, (tuple, list)):
                for v in x:
                    visit(v)
            else:
                h.update(repr(x).encode())
        visit(value)
        return h.hexdigest()
    try:
        with deterministic_training():
            trainer, cfg, data_cfg, initial_roots = build_trainer(prefix_cfg, ancestor, accepted, initial, prep, group, phase)
            require(torch.are_deterministic_algorithms_enabled() and torch.backends.cudnn.deterministic and (not torch.backends.cudnn.benchmark), 'Constructor changed deterministic execution')
            install_optimizer_step_counter(trainer.optimizer, counter)
            gpu_fd = own_gpu0_fds()
            dump_new(output / 'started.json', dict(at=stamp(), phase=phase, group=group, pid=os.getpid(), claim=claim, initializer_sha256=INIT_SHA, initial_roots=initial_roots, own_gpu_fds=gpu_fd, bundle_sha256=args.bundle_sha, preparation_sha256=args.prepare_sha, development_evaluations=0, capacity=capacity, actual_input_guard=True))

            def raw_groups(epoch):
                lab, unlab = trainer._loaders(epoch)
                batch = []
                for step, bu in enumerate(unlab):
                    bl = next(lab)
                    lc, uc = (bl['case_id'][0], bu['case_id'][0])
                    require('label' not in bu and uc == prep['case_ids']['train_unlab'][step], 'Unlabeled target/order boundary failed')
                    require(lc in exposures and int(bl['requested_index']) == prep['case_ids']['train_lab'].index(lc), 'Labeled requested/actual differs')
                    xl, yl = trainer._prepare_batch(bl)
                    xu = normalize_batch(bu['image'], trainer.norm_transform).to(trainer.device)
                    evidence = dict(epoch=epoch, step=step, labeled_case=lc, unlabeled_case=uc, label_seed=int(bl['augmentation_seed']), unlab_seed=int(bu['augmentation_seed']), student_seed=stream_seed(2027, 'student-step', epoch, step), label_image_sha=tensor_digest(bl['image']), unlab_image_sha=tensor_digest(bu['image']), no_unlabeled_target=True)
                    batch.append(dict(epoch=epoch, step=step, image_l=xl, label_l=yl, image_u=xu))
                    input_pairs.append(evidence)
                    append('pairs.jsonl', evidence)
                    exposures[lc] += 1
                    unlab_exposures[uc] += 1
                    if len(batch) == 5:
                        yield batch
                        batch = []
                require(not batch and step == 94, 'Incomplete epoch actual pair coverage')

            def step(items, overflow=False):
                nonlocal attempts
                attempts += 1
                append('attempts.jsonl', dict(at=stamp(), attempt=attempts, successful_updates_before=counter['count'], epoch=items[0]['epoch'], steps=[i['step'] for i in items], artificial_overflow=overflow))
                calls = []
                styles = []
                original_ema = trainer._update_teacher
                original_loss = trainer._losses

                def ema(a):
                    calls.append(a)
                    return original_ema(a)

                def loss(*a, **kw):
                    r = original_loss(*a, **kw)
                    if group in ('HP', 'HPQ'):
                        styles.append(copy.deepcopy(trainer._b_last))
                    return r
                trainer._update_teacher = ema
                trainer._losses = loss
                try:
                    row = trainer.step_matrix(items, compute_context=True, inject_overflow=overflow)
                finally:
                    trainer._update_teacher = original_ema
                    trainer._losses = original_loss
                row.update(at=stamp(), attempt=attempts, ema_calls=len(calls), artificial_overflow=overflow, optimizer_successful_calls=counter['count'], actual_pairs=input_pairs[-5:])
                if styles:
                    row['styles'] = styles
                append('steps.jsonl', row)
                require(calls == ([] if row['skipped'] else [0.999]), 'EMA momentum/calls changed')
                require(counter['count'] == trainer.matrix_updates, 'Optimizer actual calls differ')
                check_step(row, group, overflow)
                if group in ('HP', 'HPQ'):
                    require(len(styles) == 5, 'Missing CCL diagnostics')
                    for s in styles:
                        require(s['donor1'] != s['donor2'] and (not s['donor_labels_used']) and s['native_labeled_supervision'] and s['native_teacher'] and (not s['old_fvc_active']) and (s['complementary_sum'] == 2.0), 'CCL boundary differs')
                return row

            def payload(epoch):
                return dict(format='BY5_FOUR_ARM_FIXED_FINAL_V1', group=group, seed=2027, fold=2, labeled_cases=5, roots={n: m.state_dict() for n, m in trainer.roots().items()}, epoch=epoch, successful_updates=counter['count'], cfg=cfg, initializer_sha256=INIT_SHA, exact_resume_supported=False, bundle_sha256=args.bundle_sha, preparation_sha256=args.prepare_sha)

            def roundtrip(source):
                require(all((bool(torch.isfinite(v).all()) for m in trainer.roots().values() for v in m.state_dict().values() if v.is_floating_point())), 'Nonfinite checkpoint tensor')
                before = {n: state_dict_sha256(m.state_dict()) for n, m in trainer.roots().items()}
                ck = load_trusted_checkpoint(source, map_location='cpu')
                require(ck['group'] == group and ck['successful_updates'] == counter['count'] and (ck['cfg'] == cfg) and (ck['initializer_sha256'] == INIT_SHA), 'Serialized identity differs')
                require(set(ck['roots']) == set(trainer.roots()), 'Missing serialized root')
                for n, m in trainer.roots().items():
                    m.load_state_dict(ck['roots'][n], strict=True)
                    require(state_dict_sha256(m.state_dict()) == before[n], 'Strict readback differs: ' + n)
                del ck
                return before
            if phase == 'bridge':
                batches = raw_groups(0)
                step(next(batches))
                step(next(batches))
                third = next(batches)
                before = signature([{n: m.state_dict() for n, m in trainer.roots().items()}, trainer.optimizer.state_dict()])
                scale = float(trainer.scaler.get_scale())
                step(third, overflow=True)
                require(signature([{n: m.state_dict() for n, m in trainer.roots().items()}, trainer.optimizer.state_dict()]) == before, 'Overflow changed weights/optimizer')
                require(float(trainer.scaler.get_scale()) == scale / 2 and counter['count'] == 2, 'Overflow scaler/budget differs')
                del third, batches
                with original_lock():
                    capacity_snapshot(config=prefix_cfg)
                    with io.BytesIO() as stream:
                        torch.save(payload(-1), stream)
                        size = stream.getbuffer().nbytes
                        require(size <= 2 * 1024 ** 3, 'Memory checkpoint exceeds reserved final bound')
                        stream.seek(0)
                        roundtrip(stream)
                result = dict(status='PASS_FINITE_BRIDGE', at=stamp(), group=group, successful_updates=2, optimizer_successful_calls=counter['count'], attempted_updates=attempts, artificial_overflow_attempts=1, development_evaluations=0, initializer_sha256=INIT_SHA, initial_roots=initial_roots, overflow_model_optimizer_unchanged=True, strict_memory_roundtrip=True, serialized_bytes=size, temporary_labeled_exposures=exposures, temporary_unlabeled_exposures=unlab_exposures, elapsed_seconds=time.monotonic() - started, own_gpu_fds=own_gpu0_fds(), evidence_sha256={n: sha(output / n) for n in ('steps.jsonl', 'pairs.jsonl', 'attempts.jsonl', 'started.json')})
                dump_new(output / 'result.json', result)
            else:
                for epoch in range(36):
                    n = 0
                    for items in raw_groups(epoch):
                        step(items)
                        n += 1
                    require(n == 19 and counter['count'] == (epoch + 1) * 19, 'Fixed successful-update budget violated')
                    completed_epochs = epoch + 1
                    require(all((v == completed_epochs * 19 for v in exposures.values())) and all((v == completed_epochs for v in unlab_exposures.values())), 'Exposure imbalance')
                    append('epochs.jsonl', dict(at=stamp(), completed_epochs=completed_epochs, successful_updates=counter['count'], labeled_exposures=exposures, unlabeled_exposures=unlab_exposures, development_evaluations=0))
                    own_gpu0_fds()
                require(counter['count'] == 684 and attempts == 684 and (len(input_pairs) == 3420), 'Final budget incomplete')
                target = SELECTED_CHECKPOINT
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = Path(str(target) + '.partial')
                h = infra()
                with original_lock():
                    capacity = capacity_snapshot(config=prefix_cfg)
                    h.no_symlink(target)
                    h.no_symlink(partial)
                    require(not os.path.lexists(target) and (not os.path.lexists(partial)), 'Existing final/partial; no overwrite')
                    with partial.open('xb') as f:
                        torch.save(payload(35), h.BoundedWriter(f, 2 * 1024 ** 3))
                        f.flush()
                        os.fsync(f.fileno())
                    digest = sha(partial)
                    size = partial.stat().st_size
                    h.publish_no_replace(partial, target)
                    h.fsync_directory(target.parent)
                    readback = roundtrip(target)
                    require(sha(target) == digest, 'Checkpoint changed during strict readback')
                dump_new(output / 'final_readback.json', dict(at=stamp(), path=str(target), sha256=digest, bytes=size, roots=readback, strict=True, capacity=capacity))
                dump_new(output / 'evaluation_intent.json', dict(at=stamp(), checkpoint_sha256=digest, successful_updates=684, epoch=35, count=25))
                from train_by5 import make_strict_dataset
                from expes.common.seg_metrics import aggregate_metrics, binary_case_metrics, logits_to_binary
                ds = make_strict_dataset(data_cfg, prep['case_ids']['test'], 'test')
                loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
                for m in trainer.roots().values():
                    m.eval()
                metrics = []
                with torch.no_grad():
                    for i, batch in enumerate(loader):
                        cid = batch['case_id'][0]
                        require(cid == prep['case_ids']['test'][i], 'Development actual case differs')
                        image, label = trainer._prepare_batch(batch)
                        pred = logits_to_binary(trainer._forward_main_logits(trainer.student, image))[0]
                        gt = (label > 0).squeeze(1).cpu().numpy().astype(bool)[0]
                        value = binary_case_metrics(pred, gt, spacing=(1.0, 1.0, 1.0))
                        require(all((math.isfinite(float(x)) for x in value.values())), 'Nonfinite development metric')
                        append('final_cases.jsonl', dict(at=stamp(), case_id=cid, metrics=value, checkpoint_sha256=digest, foreground_voxels=int(gt.sum()), predicted_foreground_voxels=int(pred.sum())))
                        metrics.append(value)
                require(len(metrics) == 25, 'Incomplete development coverage')
                require(sha(target) == digest and all((state_dict_sha256(m.state_dict()) == readback[n] for n, m in trainer.roots().items())), 'Development evaluation changed checkpoint/model state')
                result = dict(status='COMPLETE_FIXED_FINAL', at=stamp(), group=group, successful_updates=684, attempted_updates=684, completed_epochs=36, labeled_exposures=exposures, unlabeled_exposures=unlab_exposures, development_evaluations=1, metrics=aggregate_metrics(metrics), initializer_sha256=INIT_SHA, checkpoint_sha256=digest, checkpoint=str(target), selection='native_student_epoch35_fixed_final', seed=2027, gt_roi_used=True, independent_test=False, distance_unit='voxel_not_mm', elapsed_seconds=time.monotonic() - started, evidence_sha256={n: sha(output / n) for n in ('started.json', 'attempts.jsonl', 'pairs.jsonl', 'steps.jsonl', 'epochs.jsonl', 'final_readback.json', 'evaluation_intent.json', 'final_cases.jsonl')})
                dump_new(output / 'result.json', result)
    except BaseException:
        output.mkdir(parents=True, exist_ok=True)
        dump_new(output / 'failure.json', dict(at=stamp(), group=group, phase=phase, traceback=traceback.format_exc(), optimizer_successful_calls=counter['count'], matrix_updates=getattr(trainer, 'matrix_updates', None), attempted_updates=attempts, completed_epochs=completed_epochs, labeled_exposures=exposures, unlabeled_exposures=unlab_exposures, automatic_retry=False, automatic_resume=False, partial_development=(output / 'evaluation_intent.json').exists()))
        raise

def run_hq(args):
    os.chdir(ROOT)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == GPU and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'GPU/determinism environment differs')
    prefix_cfg, ancestor, accepted, initial, prep = verify_prepared(args.bundle_sha, args.prepare_sha)
    from train_by5 import validate_gpu_claim, state_dict_sha256, load_trusted_checkpoint, install_optimizer_step_counter, append_jsonl
    from by_runtime import ensure_gpu0_idle, own_gpu0_fds
    claim = validate_gpu_claim(prefix_cfg, 0, args.fd, args.dev, args.ino)
    ensure_gpu0_idle()
    with original_lock():
        capacity = capacity_snapshot(config=prefix_cfg)
    group, phase = (args.group, args.phase)
    output = BASE / phase / group
    require(not output.exists(), 'Existing arm output; no replay')
    if phase == 'formal':
        bridge = validate_bridge(args.bundle_sha, args.prepare_sha)
        require(bridge['sha256'] == args.bridge_sha, 'Formal bridge binding differs')
        require(not SELECTED_CHECKPOINT.exists(), 'Existing selected checkpoint; no overwrite')
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    from expes.semi05_gate.fcd_deterministic_losses_r2_20261005 import deterministic_training
    from expes.semi05_gate.paired_rng_data_20260908 import stream_seed, tensor_digest
    from utils.cascade_train_utils import normalize_batch
    started = time.monotonic()
    trainer = None
    counter = {'count': 0}
    attempts = 0
    completed_epochs = 0
    input_pairs = []
    exposures = {c: 0 for c in prep['case_ids']['train_lab']}
    unlab_exposures = {c: 0 for c in prep['case_ids']['train_unlab']}
    append = lambda name, row: append_jsonl(output / name, row)

    def signature(value):
        h = hashlib.sha256()

        def visit(x):
            if torch.is_tensor(x):
                h.update(str((x.dtype, tuple(x.shape))).encode())
                h.update(x.detach().cpu().contiguous().numpy().tobytes())
            elif isinstance(x, dict):
                for k in sorted(x, key=str):
                    h.update(str(k).encode())
                    visit(x[k])
            elif isinstance(x, (tuple, list)):
                for v in x:
                    visit(v)
            else:
                h.update(repr(x).encode())
        visit(value)
        return h.hexdigest()
    try:
        with deterministic_training():
            trainer, cfg, data_cfg, initial_roots = build_trainer(prefix_cfg, ancestor, accepted, initial, prep, group, phase)
            require(torch.are_deterministic_algorithms_enabled() and torch.backends.cudnn.deterministic and (not torch.backends.cudnn.benchmark), 'Constructor changed deterministic execution')
            install_optimizer_step_counter(trainer.optimizer, counter)
            gpu_fd = own_gpu0_fds()
            dump_new(output / 'started.json', dict(at=stamp(), phase=phase, group=group, pid=os.getpid(), claim=claim, initializer_sha256=INIT_SHA, initial_roots=initial_roots, own_gpu_fds=gpu_fd, bundle_sha256=args.bundle_sha, preparation_sha256=args.prepare_sha, development_evaluations=0, capacity=capacity, actual_input_guard=True))

            def raw_groups(epoch):
                lab, unlab = trainer._loaders(epoch)
                batch = []
                for step, bu in enumerate(unlab):
                    bl = next(lab)
                    lc, uc = (bl['case_id'][0], bu['case_id'][0])
                    require('label' not in bu and uc == prep['case_ids']['train_unlab'][step], 'Unlabeled target/order boundary failed')
                    require(lc in exposures and int(bl['requested_index']) == prep['case_ids']['train_lab'].index(lc), 'Labeled requested/actual differs')
                    xl, yl = trainer._prepare_batch(bl)
                    xu = normalize_batch(bu['image'], trainer.norm_transform).to(trainer.device)
                    evidence = dict(epoch=epoch, step=step, labeled_case=lc, unlabeled_case=uc, label_seed=int(bl['augmentation_seed']), unlab_seed=int(bu['augmentation_seed']), student_seed=stream_seed(2027, 'student-step', epoch, step), label_image_sha=tensor_digest(bl['image']), unlab_image_sha=tensor_digest(bu['image']), no_unlabeled_target=True)
                    batch.append(dict(epoch=epoch, step=step, image_l=xl, label_l=yl, image_u=xu))
                    input_pairs.append(evidence)
                    append('pairs.jsonl', evidence)
                    exposures[lc] += 1
                    unlab_exposures[uc] += 1
                    if len(batch) == 5:
                        yield batch
                        batch = []
                require(not batch and step == 94, 'Incomplete epoch actual pair coverage')

            def step(items, overflow=False):
                nonlocal attempts
                attempts += 1
                append('attempts.jsonl', dict(at=stamp(), attempt=attempts, successful_updates_before=counter['count'], epoch=items[0]['epoch'], steps=[i['step'] for i in items], artificial_overflow=overflow))
                calls = []
                styles = []
                original_ema = trainer._update_teacher
                original_loss = trainer._losses

                def ema(a):
                    calls.append(a)
                    return original_ema(a)

                def loss(*a, **kw):
                    r = original_loss(*a, **kw)
                    if group in ('HP', 'HPQ'):
                        styles.append(copy.deepcopy(trainer._b_last))
                    return r
                trainer._update_teacher = ema
                trainer._losses = loss
                try:
                    row = trainer.step_matrix(items, compute_context=True, inject_overflow=overflow)
                finally:
                    trainer._update_teacher = original_ema
                    trainer._losses = original_loss
                row.update(at=stamp(), attempt=attempts, ema_calls=len(calls), artificial_overflow=overflow, optimizer_successful_calls=counter['count'], actual_pairs=input_pairs[-5:])
                if styles:
                    row['styles'] = styles
                append('steps.jsonl', row)
                require(calls == ([] if row['skipped'] else [0.999]), 'EMA momentum/calls changed')
                require(counter['count'] == trainer.matrix_updates, 'Optimizer actual calls differ')
                check_step(row, group, overflow)
                if group in ('HP', 'HPQ'):
                    require(len(styles) == 5, 'Missing CCL diagnostics')
                    for s in styles:
                        require(s['donor1'] != s['donor2'] and (not s['donor_labels_used']) and s['native_labeled_supervision'] and s['native_teacher'] and (not s['old_fvc_active']) and (s['complementary_sum'] == 2.0), 'CCL boundary differs')
                return row

            def payload(epoch):
                return dict(format='BY5_FOUR_ARM_FIXED_FINAL_V1', group=group, seed=2027, fold=2, labeled_cases=5, roots={n: m.state_dict() for n, m in trainer.roots().items()}, epoch=epoch, successful_updates=counter['count'], cfg=cfg, initializer_sha256=INIT_SHA, exact_resume_supported=False, bundle_sha256=args.bundle_sha, preparation_sha256=args.prepare_sha)

            def roundtrip(source):
                require(all((bool(torch.isfinite(v).all()) for m in trainer.roots().values() for v in m.state_dict().values() if v.is_floating_point())), 'Nonfinite checkpoint tensor')
                before = {n: state_dict_sha256(m.state_dict()) for n, m in trainer.roots().items()}
                ck = load_trusted_checkpoint(source, map_location='cpu')
                require(ck['group'] == group and ck['successful_updates'] == counter['count'] and (ck['cfg'] == cfg) and (ck['initializer_sha256'] == INIT_SHA), 'Serialized identity differs')
                require(set(ck['roots']) == set(trainer.roots()), 'Missing serialized root')
                for n, m in trainer.roots().items():
                    m.load_state_dict(ck['roots'][n], strict=True)
                    require(state_dict_sha256(m.state_dict()) == before[n], 'Strict readback differs: ' + n)
                del ck
                return before
            if phase == 'bridge':
                batches = raw_groups(0)
                step(next(batches))
                step(next(batches))
                third = next(batches)
                before = signature([{n: m.state_dict() for n, m in trainer.roots().items()}, trainer.optimizer.state_dict()])
                scale = float(trainer.scaler.get_scale())
                step(third, overflow=True)
                require(signature([{n: m.state_dict() for n, m in trainer.roots().items()}, trainer.optimizer.state_dict()]) == before, 'Overflow changed weights/optimizer')
                require(float(trainer.scaler.get_scale()) == scale / 2 and counter['count'] == 2, 'Overflow scaler/budget differs')
                del third, batches
                with original_lock():
                    capacity_snapshot(config=prefix_cfg)
                    with io.BytesIO() as stream:
                        torch.save(payload(-1), stream)
                        size = stream.getbuffer().nbytes
                        require(size <= 2 * 1024 ** 3, 'Memory checkpoint exceeds reserved final bound')
                        stream.seek(0)
                        roundtrip(stream)
                result = dict(status='PASS_FINITE_BRIDGE', at=stamp(), group=group, successful_updates=2, optimizer_successful_calls=counter['count'], attempted_updates=attempts, artificial_overflow_attempts=1, development_evaluations=0, initializer_sha256=INIT_SHA, initial_roots=initial_roots, overflow_model_optimizer_unchanged=True, strict_memory_roundtrip=True, serialized_bytes=size, temporary_labeled_exposures=exposures, temporary_unlabeled_exposures=unlab_exposures, elapsed_seconds=time.monotonic() - started, own_gpu_fds=own_gpu0_fds(), evidence_sha256={n: sha(output / n) for n in ('steps.jsonl', 'pairs.jsonl', 'attempts.jsonl', 'started.json')})
                dump_new(output / 'result.json', result)
            else:
                references = load_references()
                evaluations = 0
                for epoch in range(36):
                    n = 0
                    for items in raw_groups(epoch):
                        step(items)
                        n += 1
                    require(n == 19 and counter['count'] == (epoch + 1) * 19, 'Successful-update budget violated')
                    completed_epochs = epoch + 1
                    require(all((v == completed_epochs * 19 for v in exposures.values())) and all((v == completed_epochs for v in unlab_exposures.values())), 'Exposure imbalance')
                    own_gpu0_fds()
                    del items
                    append('epochs.jsonl', dict(at=stamp(), completed_epochs=completed_epochs, successful_updates=counter['count'], labeled_exposures=exposures, unlabeled_exposures=unlab_exposures))
                    if completed_epochs % EVAL_EVERY and completed_epochs != 36:
                        continue
                    metrics, roots_before = evaluate_development(trainer, data_cfg, prep, output, completed_epochs)
                    evaluations += 1
                    gap = 100 * (references['HPQ']['metrics']['dice'] - metrics['dice'])
                    in_band = TARGET_GAP_PP - TOLERANCE_PP <= gap <= TARGET_GAP_PP + TOLERANCE_PP
                    row = dict(at=stamp(), completed_epochs=completed_epochs, successful_updates=counter['count'], metrics=metrics, ccbs_coefficient=COEFFICIENT, original_hpq_minus_hq_pp=gap, hq_minus_original_h_pp=100 * (metrics['dice'] - references['H']['metrics']['dice']), distance_to_target_pp=abs(gap - TARGET_GAP_PP), diagnostic_in_target_band=in_band, target_hit=completed_epochs == 36 and in_band, diagnostics_only=completed_epochs != 36, selection='native_student_epoch35_fixed_final', development_only=True, independent_test=False, gt_roi_used=True, distance_unit='voxel_not_mm')
                    append('evaluations.jsonl', row)
                    print(json.dumps(row), flush=True)
                require(completed_epochs == 36 and attempts == counter['count'] == 684 and (len(input_pairs) == 3420), 'Fixed-final budget incomplete')
                target_hit = row['target_hit']
                selected = save_development_checkpoint(trainer, cfg, roots_before, row, args)
                dump_new(output / 'selected.json', selected)
                result = dict(status='COMPLETE_FIXED_FINAL_FIXED_CONFIGURATION', at=stamp(), group=group, ccbs_coefficient=COEFFICIENT, completed_epochs=36, successful_updates=684, attempted_updates=attempts, development_evaluations=evaluations, metrics=metrics, target_hit=target_hit, original_hpq_minus_hq_pp=gap, hq_minus_original_h_pp=row['hq_minus_original_h_pp'], target_gap_pp=TARGET_GAP_PP, tolerance_pp=TOLERANCE_PP, eval_every=EVAL_EVERY, initializer_sha256=INIT_SHA, script_sha256=sha(__file__), selection='native_student_epoch35_fixed_final', fixed_final_completed=True, early_stopping=False, within_run_hyperparameter_adaptation=False, development_only=True, independent_test=False, gt_roi_used=True, distance_unit='voxel_not_mm', checkpoint=str(SELECTED_CHECKPOINT), checkpoint_sha256=selected['checkpoint_sha256'], labeled_exposures=exposures, unlabeled_exposures=unlab_exposures, elapsed_seconds=time.monotonic() - started, evidence_sha256={n: sha(output / n) for n in ('started.json', 'attempts.jsonl', 'pairs.jsonl', 'steps.jsonl', 'epochs.jsonl', 'evaluations.jsonl', 'development_cases.jsonl')})
                dump_new(output / 'result.json', result)
    except BaseException:
        output.mkdir(parents=True, exist_ok=True)
        dump_new(output / 'failure.json', dict(at=stamp(), group=group, phase=phase, traceback=traceback.format_exc(), optimizer_successful_calls=counter['count'], matrix_updates=getattr(trainer, 'matrix_updates', None), attempted_updates=attempts, completed_epochs=completed_epochs, labeled_exposures=exposures, unlabeled_exposures=unlab_exposures, automatic_retry=False, automatic_resume=False, partial_development=(output / 'development_cases.jsonl').exists()))
        raise

def load_references():
    rows = {}
    for group, path in REFERENCES.items():
        require(sha(path) == REFERENCE_SHA[group], 'Original reference result changed: ' + group)
        row = json.loads(path.read_text())
        require(row['group'] == group and row['status'] == 'COMPLETE_FIXED_FINAL' and (row['successful_updates'] == 684) and (row['completed_epochs'] == 36) and (row['initializer_sha256'] == INIT_SHA), 'Reference protocol differs')
        rows[group] = row
    return rows

def capacity_snapshot(*, config=None):
    value = ORIGINAL_CAPACITY(config=config)
    rows = json.loads(REGISTRY.read_text())
    for target, bound in own_targets():
        require([r for r in rows if r['path'] == str(target)] == [dict(path=str(target), bound=bound)], 'Development reservation missing')
    observed = {r['path']: r['bound_bytes'] for fs in value['filesystems'] for r in fs['targets']}
    require(all((observed.get(str(p)) == b for p, b in own_targets())), 'Capacity union omitted development slots')
    return value

def enroll_capacity():
    from by_resources import merge, backend
    h = infra()
    with original_lock():
        h.no_symlink(REGISTRY)
        old = json.loads(REGISTRY.read_text())
        require(len({r['path'] for r in old}) == len(old), 'Duplicate registry rows')
        new = list(old)
        for path, bound in own_targets():
            require(REGISTRY_ROOT in path.parents and path.name == 'final.pth' and (type(bound) is int) and (0 < bound <= 2 * 1024 ** 3), 'Development target violates shared registry schema')
            found = [r for r in old if r['path'] == str(path)]
            require(not found or found == [dict(path=str(path), bound=bound)], 'Reservation changed')
            if not found:
                new.append(dict(path=str(path), bound=bound))
        guard = backend()
        prior = guard.targets
        try:
            guard.targets = lambda: merge(prior(), [(str(p), b) for p, b in own_targets()])
            prospective = guard.capacity_snapshot(12 * 1024 ** 3 + 16 * 1024 ** 2)
        finally:
            guard.targets = prior
        if new != old:
            temporary = Path(str(REGISTRY) + '.hp_hq_final_20261106.partial')
            h.no_symlink(temporary)
            with temporary.open('x') as f:
                json.dump(new, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, REGISTRY)
            h.fsync_directory(REGISTRY.parent)
        return prospective

def configure_ccbs():
    from expes.semi05_gate.paired_fcd_matrix_20261006 import MatrixMixin
    original = MatrixMixin.contextual_gradients

    def context(self, image, epoch, step, parameters):
        gradients, row = original(self, image, epoch, step, parameters)
        require(self.matrix_c == 'B', 'CCBS configuration differs')
        gradients = tuple((None if g is None else COEFFICIENT * g for g in gradients))
        row = dict(row, coefficient=COEFFICIENT, unscaled_gradient_norms=row['raw_gradient_norms'])
        row['raw_gradient_norms'] = {k: COEFFICIENT * v for k, v in row['raw_gradient_norms'].items()}
        return (gradients, row)
    MatrixMixin.contextual_gradients = context

def validate_bridge(bundle_sha, prepare_sha):
    require(bundle_sha == BUNDLE and prepare_sha == PREPARE, 'Bridge input differs')
    path = BASE / 'bridge' / ACTIVE_GROUP
    result = json.loads((path / 'result.json').read_text())
    require(result['status'] == 'PASS_FINITE_BRIDGE' and result['successful_updates'] == 2 and (result['attempted_updates'] == 3) and (result['artificial_overflow_attempts'] == 1) and result['strict_memory_roundtrip'] and result['overflow_model_optimizer_unchanged'], 'Bridge failed')
    for name, digest in result['evidence_sha256'].items():
        require(sha(path / name) == digest, 'Bridge evidence changed')
    rows = [json.loads(line) for line in (path / 'steps.jsonl').read_text().splitlines()]
    require(len(rows) == 3, 'Wrong bridge step count')
    for index, row in enumerate(rows):
        check_step(row, ACTIVE_GROUP, overflow=index == 2)
    return dict(status=result['status'], sha256=sha(path / 'result.json'))

def evaluate_development(trainer, data_cfg, prep, output, epoch):
    import numpy as np
    import torch
    from train_by5 import make_strict_dataset, state_dict_sha256, append_jsonl
    from expes.common.seg_metrics import aggregate_metrics, binary_case_metrics, logits_to_binary
    roots = trainer.roots()
    before = {n: state_dict_sha256(m.state_dict()) for n, m in roots.items()}
    modules = {id(m): m for root in roots.values() for m in root.modules()}
    modes = {key: m.training for key, m in modules.items()}
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
    metrics = []
    try:
        ds = make_strict_dataset(data_cfg, prep['case_ids']['test'], 'test')
        loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, generator=torch.Generator().manual_seed(2027))
        for root in roots.values():
            root.eval()
        with torch.no_grad():
            for index, batch in enumerate(loader):
                cid = batch['case_id'][0]
                require(cid == prep['case_ids']['test'][index], 'Development case identity differs')
                image, label = trainer._prepare_batch(batch)
                pred = logits_to_binary(trainer._forward_main_logits(trainer.student, image))[0]
                gt = (label > 0).squeeze(1).cpu().numpy().astype(bool)[0]
                value = binary_case_metrics(pred, gt, spacing=(1.0, 1.0, 1.0))
                require(all((math.isfinite(float(x)) for x in value.values())), 'Nonfinite development metric')
                metrics.append(value)
                append_jsonl(output / 'development_cases.jsonl', dict(at=stamp(), completed_epochs=epoch, case_id=cid, metrics=value, foreground_voxels=int(gt.sum()), predicted_foreground_voxels=int(pred.sum())))
        require(len(metrics) == 25, 'Incomplete development coverage')
        require(all((state_dict_sha256(m.state_dict()) == before[n] for n, m in roots.items())), 'Development evaluation changed model state')
    finally:
        for key, module in modules.items():
            module.training = modes[key]
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        torch.cuda.set_rng_state_all(rng[3])
    return (aggregate_metrics(metrics), before)

def save_development_checkpoint(trainer, cfg, root_hashes, evaluation, args):
    require(evaluation['completed_epochs'] == 36 and evaluation['successful_updates'] == 684 and (evaluation['selection'] == 'native_student_epoch35_fixed_final') and (not evaluation['diagnostics_only']), 'Reject non-final or incomplete target checkpoint')
    import torch
    from train_by5 import load_trusted_checkpoint, state_dict_sha256
    target = SELECTED_CHECKPOINT
    partial = Path(str(target) + '.partial')
    h = infra()
    names = ('student', 'head', 'adapter')
    roots = trainer.roots()
    payload = dict(format='HQ_DEVELOPMENT_FIXED_FINAL_STUDENT_V1', group='HQ', trial_id=TRIAL_ID, roots={name: roots[name].state_dict() for name in names}, cfg=cfg, completed_epochs=evaluation['completed_epochs'], epoch=evaluation['completed_epochs'] - 1, successful_updates=evaluation['successful_updates'], metrics=evaluation['metrics'], ccbs_coefficient=COEFFICIENT, seed=2027, fold=2, labeled_cases=5, initializer_sha256=INIT_SHA, bundle_sha256=BUNDLE, preparation_sha256=PREPARE, script_sha256=sha(__file__), script_path=str(Path(__file__).resolve()), reference_result_sha256=REFERENCE_SHA, selection=evaluation['selection'], exact_resume_supported=False, teacher_optimizer_scaler_included=False, development_only=True, independent_test=False, gt_roi_used=True, root_sha256={name: root_hashes[name] for name in names}, evaluation=evaluation)
    with original_lock():
        capacity = capacity_snapshot()
        h.no_symlink(target)
        h.no_symlink(partial)
        require(not os.path.lexists(target) and (not os.path.lexists(partial)), 'Existing selection; never overwrite')
        target.parent.mkdir(parents=True, exist_ok=True)
        with partial.open('xb') as f:
            torch.save(payload, h.BoundedWriter(f, 1024 ** 3))
            f.flush()
            os.fsync(f.fileno())
        digest, size = (sha(partial), partial.stat().st_size)
        ck = load_trusted_checkpoint(partial, map_location='cpu')
        require(ck['cfg'] == cfg and ck['evaluation'] == evaluation and (set(ck['roots']) == set(names)), 'Selected payload identity differs')
        for name in names:
            roots[name].load_state_dict(ck['roots'][name], strict=True)
            require(state_dict_sha256(roots[name].state_dict()) == root_hashes[name], 'Strict selected checkpoint readback differs')
        del ck
        require(sha(partial) == digest, 'Selected bytes changed during readback')
        h.publish_no_replace(partial, target)
        h.fsync_directory(target.parent)
    result = dict(evaluation, checkpoint=str(target), checkpoint_sha256=digest, bytes=size, strict_readback=True, trial_id=TRIAL_ID, script_path=str(Path(__file__).resolve()), script_sha256=sha(__file__), root_sha256={n: root_hashes[n] for n in names}, reference_result_sha256=REFERENCE_SHA, capacity_before=capacity, exact_resume_supported=False, teacher_optimizer_scaler_included=False)
    dump_new(SELECTION_DIR / 'selected.json', result)
    dump_new(SELECTION_DIR / 'locked_hyperparameters.json', dict(trial_id=TRIAL_ID, ccbs_coefficient=COEFFICIENT, seed=2027, selected_epoch=36, successful_updates=684, selection='native_student_epoch35_fixed_final', early_stopping=False, within_run_hyperparameter_adaptation=False, eval_every=EVAL_EVERY, target_gap_pp=TARGET_GAP_PP, tolerance_pp=TOLERANCE_PP, cfg=cfg, checkpoint_sha256=digest, script_sha256=sha(__file__), development_only=True, independent_test=False, exact_resume_supported=False))
    return result
if __name__ == '__main__':
    main()
