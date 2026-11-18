import os
import sys
from pathlib import Path
ASNET_ROOT = Path(os.environ.get('ASNET_ROOT', '.')).resolve()
import argparse, hashlib, inspect, json, os, shutil, time, traceback
from pathlib import Path
ROOT = Path(str(ASNET_ROOT))
BASE = ROOT / 'expes/work_dir/semi05_gate/FCD_A_MATRIX_20261006'
STORE = ROOT / 'expes/artifacts/FCD_A_MATRIX_20261006'
STORAGE_LAYOUT = 'HOME_BOTH_RATIOS_SIX_TARGETS_ONE_ATOMIC'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 ** 2), b''):
            h.update(b)
    return h.hexdigest()

def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))

def verify_a1v_complete():
    evidence = BASE / 'A1V/completion_20261009'
    assert sha(evidence / 'report.json') == 'c033bbd171b3a0ddb294f2805f2b6c93b188cf874e453df96a6c601d26ab1ae2'
    assert sha(evidence / 'evidence.tar.gz') == '57de60080c2b7619e5b4b246700564e586bf1a13c090461cf5ab0f74136a6ed6'
    report = json.loads((evidence / 'report.json').read_text())
    assert report['status'] == 'COMPLETE' and report['group'] == 'A1V'
    for ratio, updates in (('05', 684), ('20', 576)):
        row = report['ratios'][ratio]
        run = BASE / 'A1V' / ('formal' + ratio + '_r1')
        assert row['final'] == json.loads((run / 'final_metrics.json').read_text())
        assert row['final']['completed_epochs'] == 36 and row['final']['updates'] == updates
        assert row['all_action_checks'] and sha(run / 'final_train_label_audit/report.json') == row['audit_sha256']
    return sha(evidence / 'report.json')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ratio', choices=['05', '20'], required=True)
    p.add_argument('--gpu', choices=['0', '1'], required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--formal', action='store_true')
    p.add_argument('--proof05')
    p.add_argument('--proof20')
    for ratio in ('05', '20'):
        p.add_argument('--qualification-tag' + ratio, default='qualification' + ratio + '_r1')
        p.add_argument('--formal-tag' + ratio, default='formal' + ratio + '_r1')
    a = p.parse_args()
    qual_tags = {r: getattr(a, 'qualification_tag' + r) for r in ('05', '20')}
    formal_tags = {r: getattr(a, 'formal_tag' + r) for r in ('05', '20')}
    assert all((x.replace('_', '').isalnum() for x in [*qual_tags.values(), *formal_tags.values()]))
    assert len(set([*qual_tags.values(), *formal_tags.values()])) == 4
    assert a.tag == (formal_tags if a.formal else qual_tags)[a.ratio]
    assert a.tag.replace('_', '').isalnum()
    a1v_completion_sha256 = verify_a1v_complete()
    os.chdir(ROOT)
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['OMP_NUM_THREADS'] = '2'
    import torch, yaml, fcntl
    from utils.cascade_train_utils import normalize_batch
    from expes.semi05_gate.paired_fcd_a1b_20261009 import QualifiedStyleBalanced as MatrixHalf
    from expes.semi05_gate.fcd_matrix_storage_gate_20261006 import Target, inspect_capacity
    from expes.semi05_gate.paired_hard_relations_20261005 import tensor_state_digest
    from expes.semi05_gate.fcd_deterministic_losses_r2_20261005 import deterministic_training
    from expes.semi05_gate.paired_rng_data_20260908 import stream_seed, tensor_digest
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    old = ROOT / 'expes/work_dir/semi05_gate/PAIRED_ZEA_CROSS_SCALE_HALF_R2_fold2' / f'by_semi{a.ratio}_cv5/fold2'
    parent = json.loads((old / 'manifest.json').read_text())
    cfg = yaml.safe_load((old / 'config.yaml').read_text())
    for path, digest in parent['source_hashes'].items():
        assert sha(path) == digest, path
    init = parent['init_from']
    assert sha(init) == parent['init_sha256']
    split = ROOT / cfg['dataset']['split_root'] / 'fold2'
    assert {n: sha(split / n) for n in parent['split_hashes']} == parent['split_hashes']
    expected_floor = json.loads((old / 'final_metrics.json').read_text())['floor_dice']
    storage_root = STORE
    output = BASE / 'A1B' / a.tag
    store = storage_root / 'A1B' / a.tag
    assert not output.exists() and (not store.exists()), 'refuse duplicate/overwrite'
    sources = dict(parent['source_hashes'])
    for name in ('paired_fcd_matrix_20261006.py', 'paired_fcd_a1b_20261009.py', 'fcd_matrix_storage_gate_20261006.py', 'run_fcd_matrix_a1b_20261009.py', 'audit_fcd_matrix_a1b_final_20261009.py', 'paired_l_cross_style_complement_20261001.py', 'fcd_l0_fda_lesion_integrity_20260925.py', 'paired_hard_relations_20261005.py', 'fcd_cac_fg_20260928.py', 'fcd_dycon_r2_losses_20260927.py', 'fcd_mpl_device_transaction_20261003.py', 'fcd_deterministic_losses_r2_20261005.py', 'fcd_deterministic_spatial_ops_r3_20261005.py', 'fcd_deterministic_spatial_ops_20261004.py', 'paired_r1_restore_full_20261005.py', 'pack_fcd_a1b_qualification_20261009.py'):
        path = ROOT / 'expes/semi05_gate' / name
        sources[str(path)] = sha(path)
    cfg['experiment'].update(gpu=a.gpu, seed=2027, output_dir=str(output))
    cfg['semi'].update(paired_checkpoint_dir=str(store), paired_workers=0, xsa_cross_gain=0.5, matrix_group='A1B', matrix_c='B', cross_style_complement='two-donor-full-decoder')
    for cls in MatrixHalf.__mro__:
        if cls.__module__.startswith('expes.'):
            path = inspect.getfile(cls)
            sources[str(Path(path).resolve())] = sha(path)
    size = 1623623319 + 1024 ** 2
    for r in ('05', '20'):
        done = BASE / 'A10' / ('formal05_r1' if r == '05' else 'formal20_r1')
        assert json.loads((done / 'final_metrics.json').read_text())['completed_epochs'] == 36
        assert json.loads((done / 'final_train_label_audit/report.json').read_text())['status'] == 'COMPLETE'
    if a.formal:
        for r, path in [('05', a.proof05), ('20', a.proof20)]:
            assert path and Path(path).resolve() == (BASE / 'A1B' / qual_tags[r] / 'qualification.json').resolve()
            proof = json.loads(Path(path).read_text())
            expected_ck = STORE / 'A1B' / qual_tags[r] / 'qualification.pth'
            assert Path(proof['checkpoint']['path']).resolve() == expected_ck.resolve()
            assert proof['storage_layout'] == STORAGE_LAYOUT
            assert proof['successful_updates'] == 9 and proof['artificial_overflows'] == 1
            assert proof['status'] == 'PASS' and proof['ratio'] == r and (proof['group'] == 'A1B')
            assert proof['protocol'] == 'FIXED_FINAL_DEVELOPMENT_V3_1'
            assert proof['qualification_tags'] == qual_tags and proof['formal_tags'] == formal_tags
            assert proof['legacy_objective'] == 'frozen PairedCrossStyleComplement._losses' and proof['true_two_branch_routing']
            assert proof['source_hashes'] == sources and proof['cold_warm_legacy_exact'] and proof['active_repeat_exact'] and proof['new_loss_changes_update'] and proof['context_vb_component_exact'] and proof['allowed_train_lab_probe_then_step_exact']
            assert proof['overflow_skip'] and proof['six_root_roundtrip_exact']
            old_parent = json.loads((ROOT / 'expes/work_dir/semi05_gate/PAIRED_ZEA_CROSS_SCALE_HALF_R2_fold2' / f'by_semi{r}_cv5/fold2/manifest.json').read_text())
            assert proof['init_from'] == old_parent['init_from'] and proof['init_sha256'] == old_parent['init_sha256']
            assert sha(old_parent['init_from']) == old_parent['init_sha256']
            assert proof['split_hashes'] == old_parent['split_hashes']
            assert Path(proof['checkpoint']['path']).stat().st_size == proof['checkpoint']['bytes']
            assert sha(proof['checkpoint']['path']) == proof['checkpoint']['sha256']
            size = max(size, proof['checkpoint']['bytes'])
            done = BASE / 'A0B' / ('formal05_r1' if r == '05' else 'formal20_r1')
            assert json.loads((done / 'final_metrics.json').read_text())['completed_epochs'] == 36
            assert json.loads((done / 'final_train_label_audit/report.json').read_text())['status'] == 'COMPLETE'
    ancestor = storage_root
    STORE.mkdir(parents=True, exist_ok=True)

    def roots_for_targets():
        return {STORE: ('05', '20')}

    def gate_for(root, ratios):
        targets = []
        for r in ratios:
            targets.append(Target(str(root / 'A1B' / qual_tags[r] / 'qualification.pth'), size))
            targets.extend((Target(str(root / 'A1B' / formal_tags[r] / name), size) for name in ('best.pth', 'final.pth')))
        gate = inspect_capacity(str(root), targets)
        assert gate['passes'], gate
        return gate
    capacities = {}
    for root, ratios in roots_for_targets().items():
        with (root / 'save.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            capacities[str(root)] = gate_for(root, ratios)
    capacity = capacities[str(storage_root)]

    def resource_gate():
        return gate_for(storage_root, roots_for_targets()[storage_root])
    required = capacity['required_bytes']
    output.mkdir(parents=True)
    store.mkdir(parents=True)
    (output / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    manifest = dict(group='A1B', ratio=a.ratio, seed=2027, source_hashes=sources, a1v_completion_sha256=a1v_completion_sha256, init_from=init, init_sha256=parent['init_sha256'], split_hashes=parent['split_hashes'], storage_required=required, storage_free=shutil.disk_usage(ancestor).free, storage_layout=STORAGE_LAYOUT, protocol='FIXED_FINAL_DEVELOPMENT_V3_1', primary_metric='native_student_epoch35_final', best_role='diagnostic side archive only; no training/selection decisions', capacity=capacity, capacities_by_filesystem=capacities, checkpoint_storage_root=str(storage_root), qualification_tags=qual_tags, formal_tags=formal_tags, data_boundary='style donors and new contextual loss take only train_unlab image; inherited GT ROI and test-chosen development initialization remain')
    dump(output / 'manifest.json', manifest)

    def append(name, row):
        with (output / name).open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')

    def signature(value):
        h = hashlib.sha256()

        def walk(x):
            if torch.is_tensor(x):
                h.update(str((x.dtype, tuple(x.shape))).encode())
                h.update(x.detach().cpu().contiguous().numpy().tobytes())
            elif isinstance(x, dict):
                for k in sorted(x, key=str):
                    h.update(str(k).encode())
                    walk(x[k])
            elif isinstance(x, (tuple, list)):
                for z in x:
                    walk(z)
            else:
                h.update(repr(x).encode())
        walk(value)
        return h.hexdigest()

    def save(t, name, epoch):
        storage_root.mkdir(parents=True, exist_ok=True)
        with (storage_root / 'save.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            resource_gate()
            path = store / name
            temp = store / (name + '.partial')
            assert not temp.exists()
            with temp.open('wb') as f:
                torch.save(dict(roots={k: t._compact_state(m) for k, m in t.roots().items()}, cfg=cfg, saved_epoch=epoch, updates=t.matrix_updates, group='A1B', seed=2027, exact_resume_supported=False), f)
                f.flush()
                os.fsync(f.fileno())
            assert temp.stat().st_size <= size, 'Measured checkpoint exceeds registered bound'
            os.replace(temp, path)
        return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))

    def raw_groups(t, epoch):
        lab, unlab = t._loaders(epoch)
        group = []
        for step, bu in enumerate(unlab):
            bl = next(lab)
            xl, yl = t._prepare_batch(bl)
            xu = normalize_batch(bu['image'], t.norm_transform).to(t.device)
            group.append(dict(epoch=epoch, step=step, image_l=xl, label_l=yl, image_u=xu))
            row = dict(epoch=epoch, step=step, labeled_index=int(bl['requested_index']), label_seed=int(bl['augmentation_seed']), unlabeled_index=int(bu['requested_index']), unlabeled_seed=int(bu['augmentation_seed']), student_seed=stream_seed(2027, 'student-step', epoch, step))
            if epoch == 0:
                row.update(label_image_sha=tensor_digest(bl['image']), unlab_image_sha=tensor_digest(bu['image']))
            if a.formal:
                append('pair_trace.jsonl', row)
            if len(group) == 5:
                yield group
                group = []
        assert not group
    begin = time.monotonic()
    try:
        with deterministic_training():
            t = MatrixHalf(cfg)
            t.qualification_routing = not a.formal
            t.initialize_matrix(init)
            floor = t._evaluate()
            assert abs(floor['dice'] - expected_floor) < 1e-08
            t.best_dice, t.best_epoch = (expected_floor, -1)
            torch.cuda.reset_peak_memory_stats()
            if not a.formal:
                iterator = raw_groups(t, 0)
                raw = [next(iterator), next(iterator)]
                actual_successes = 0
                results = {}

                def engineering_step(items, **kwargs):
                    nonlocal actual_successes
                    previous = t.matrix_updates
                    try:
                        return t.step_matrix(items, **kwargs)
                    finally:
                        actual_successes += max(0, t.matrix_updates - previous)

                def state_signature(include_scaler=True):
                    values = [{k: m.state_dict() for k, m in t.roots().items()}, t.optimizer.state_dict()]
                    if include_scaler:
                        values.append(t.scaler.state_dict())
                    return signature(values)

                def rng_signature():
                    import random, numpy as np
                    state = np.random.get_state()
                    numpy_state = [state[0], state[1].dtype.str, state[1].shape, state[1].tobytes(), *state[2:]]
                    return signature([random.getstate(), numpy_state, torch.get_rng_state(), torch.cuda.get_rng_state_all()])

                def canonical(row):
                    return {k: v for k, v in row.items() if k != 'seconds'}

                def original_ledger(row):
                    return [{k: m[k] for k in ('step', 'supervised', 'pseudo', 'wsv', 'baseline_total', 'views', 'native_reliable_ratio', 'style')} for m in row['microbatches']]

                def check_step(row, context, reduction):
                    assert not row['skipped'] and row['teacher_no_grad'] and (row['clip_calls'] == row['ema_calls'] == 1)
                    assert row['branch_routing_measured'] and len(row['microbatches']) == 5
                    assert all((v is not None and v > 0 for v in row['gradient_groups'].values()))
                    assert row['context_computed'] == context
                    for micro in row['microbatches']:
                        assert len(micro['branch_routes']) == 2
                        if context:
                            c = micro['context']
                            assert c['reduction'] == reduction and c['applied'] == (reduction != '0')
                            assert c['declared_state_restored'] and c['no_gt_argument'] and (not c['relation_computed'])
                            assert c['foreground_voxels'] + c['background_voxels'] == 64 * c['reliable_blocks']
                            if c['reliable_blocks']:
                                assert all((v > 0 for v in c['raw_gradient_norms'].values()))
                for branch, context, reduction in (('native_LB', False, '0'), ('off', True, '0'), ('active_B', True, 'B'), ('active_B_repeat', True, 'B')):
                    t.initialize_matrix(init)
                    t.matrix_c = reduction
                    records = []
                    append('initialization_memory.jsonl', t.initialization_memory)
                    for phase, items in zip(('cold', 'warm'), raw):
                        if branch == 'native_LB':
                            before, before_rng, updates = (state_signature(), rng_signature(), t.matrix_updates)
                            component = t.probe_context_reductions(items[0]['image_u'], items[0]['epoch'], items[0]['step'])
                            assert state_signature() == before and rng_signature() == before_rng and (t.matrix_updates == updates)
                            assert all((p.grad is None for m in t.roots().values() for p in m.parameters()))
                            append('context_vb_components.jsonl', dict(phase=phase, state_unchanged=True, rng_unchanged=True, **component))
                        row = engineering_step(items, compute_context=context)
                        append('qualification_steps.jsonl', dict(branch=branch, phase=phase, **row))
                        check_step(row, context, reduction)
                        records.append(dict(state=state_signature(), original=original_ledger(row), full=canonical(row)))
                    results[branch] = records
                for index in range(2):
                    native, off = (results['native_LB'][index], results['off'][index])
                    active, repeat = (results['active_B'][index], results['active_B_repeat'][index])
                    assert native['state'] == off['state'] and native['original'] == off['original'], 'C-off changes native LB'
                    assert active == repeat, 'applied B cold/warm replay changed'
                    assert active['state'] != native['state'], 'new CE did not change the update'
                t.initialize_matrix(init)
                t.matrix_c = 'B'
                before, before_rng = (state_signature(), rng_signature())
                with t.isolated_context(0, 0), torch.no_grad(), torch.cuda.amp.autocast():
                    for m in t.roots().values():
                        m.eval()
                    allowed_native = [t._forward_main_logits(m, raw[0][0]['image_l']).cpu() for m in (t.student, t.teacher)]
                del allowed_native
                assert state_signature() == before and rng_signature() == before_rng
                row = engineering_step(raw[0])
                append('qualification_steps.jsonl', dict(branch='allowed_train_lab_probe_then_step', phase='cold', **row))
                check_step(row, True, 'B')
                assert state_signature() == results['active_B'][0]['state']
                assert canonical(row) == results['active_B'][0]['full']
                assert actual_successes == 9
                checkpoint = save(t, 'qualification.pth', -1)
                for m in t.roots().values():
                    m.eval()
                with torch.no_grad(), torch.cuda.amp.autocast():
                    native = [t._forward_main_logits(m, raw[0][0]['image_l']).cpu() for m in (t.student, t.teacher)]
                before = tensor_state_digest({k: t._compact_state(m) for k, m in t.roots().items()})
                with torch.no_grad():
                    for m in t.roots().values():
                        next(m.parameters()).add_(0.1)
                assert tensor_state_digest({k: t._compact_state(m) for k, m in t.roots().items()}) != before
                ck = torch.load(checkpoint['path'], map_location='cpu')
                assert set(ck['roots']) == set(t.roots()) and ck['group'] == 'A1B'
                for k, m in t.roots().items():
                    m.load_state_dict(ck['roots'][k], strict=True)
                del ck
                assert tensor_state_digest({k: t._compact_state(m) for k, m in t.roots().items()}) == before
                with torch.no_grad(), torch.cuda.amp.autocast():
                    after = [t._forward_main_logits(m, raw[0][0]['image_l']).cpu() for m in (t.student, t.teacher)]
                assert all((torch.equal(x, y) for x, y in zip(native, after)))
                before = state_signature(include_scaler=False)
                saved_updates = t.matrix_updates
                overflow = engineering_step(raw[0], inject_overflow=True)
                assert overflow['skipped'] and overflow['clip_calls'] == overflow['ema_calls'] == 0
                assert overflow['scale_after'] == overflow['scale_before'] / 2 and t.matrix_updates == saved_updates
                assert state_signature(include_scaler=False) == before
                append('overflow.jsonl', overflow)
                report = dict(status='PASS', group='A1B', ratio=a.ratio, source_hashes=sources, init_from=init, init_sha256=parent['init_sha256'], split_hashes=parent['split_hashes'], checkpoint=checkpoint, floor=floor, cold_warm_legacy_exact=True, active_repeat_exact=True, true_two_branch_routing=True, allowed_train_lab_probe_state_unchanged=True, allowed_train_lab_probe_then_step_exact=True, new_loss_changes_update=True, context_vb_component_exact=True, legacy_objective='frozen PairedCrossStyleComplement._losses', overflow_skip=True, six_root_roundtrip_exact=True, successful_updates=actual_successes, artificial_overflows=1, saved_activation_storage='original CUDA checkpoint recomputation', qualification_tags=qual_tags, formal_tags=formal_tags, storage_layout=STORAGE_LAYOUT, protocol='FIXED_FINAL_DEVELOPMENT_V3_1', primary_metric='native_student_epoch35_final', best_role='diagnostic only', wall_seconds=time.monotonic() - begin, peak_bytes=torch.cuda.max_memory_allocated(), limitation='Finite cold/warm native/off and active-repeat/component checks; not persisted exact resume, full 36-epoch replay or effectiveness evidence')
                dump(output / 'qualification.json', report)
                print('A1B_QUALIFICATION_PASS ' + json.dumps(report), flush=True)
                return
            append('metrics.jsonl', dict(epoch=-1, **floor))
            weights = {'best.pth': save(t, 'best.pth', -1)}
            for epoch in range(36):
                start = time.monotonic()
                t.eval_epoch = epoch
                count = 0
                for items in raw_groups(t, epoch):
                    row = t.step_matrix(items)
                    assert not row['skipped']
                    row['epoch'] = epoch
                    append('matrix_steps.jsonl', row)
                    count += 1
                    print('A1B_UPDATE ' + json.dumps(dict(epoch=epoch, update=t.matrix_updates)), flush=True)
                assert count == (19 if a.ratio == '05' else 16)
                metrics = t._evaluate()
                append('metrics.jsonl', dict(epoch=epoch, **metrics))
                if metrics['dice'] > t.best_dice:
                    t.best_dice, t.best_epoch = (float(metrics['dice']), epoch)
                    weights['best.pth'] = save(t, 'best.pth', epoch)
                row = dict(epoch=epoch, dice=float(metrics['dice']), best=t.best_dice, updates=t.matrix_updates, seconds=time.monotonic() - start)
                append('train_epochs.jsonl', row)
                print('A1B_EPOCH ' + json.dumps(row), flush=True)
            assert t.matrix_updates == (684 if a.ratio == '05' else 576)
            weights['final.pth'] = save(t, 'final.pth', 35)
            final = dict(completed_epochs=36, stop_reason='max_epochs', best_dice=t.best_dice, best_epoch=t.best_epoch, final_dice=float(metrics['dice']), updates=t.matrix_updates, floor_dice=expected_floor, weights=weights, wall_seconds=time.monotonic() - begin, peak_bytes=torch.cuda.max_memory_allocated(), exact_resume_supported=False, protocol='FIXED_FINAL_DEVELOPMENT_V3_1', primary_metric='native_student_epoch35_final', best_role='diagnostic only; never reloaded or used to change budget')
            dump(output / 'final_metrics.json', final)
            print('A1B_COMPLETE ' + json.dumps(final), flush=True)
    except Exception:
        dump(output / 'failure.json', dict(status='FAIL_ENGINEERING', traceback=traceback.format_exc(), wall_seconds=time.monotonic() - begin, successful_qualification_updates=locals().get('actual_successes', 0), current_model_updates=getattr(locals().get('t'), 'matrix_updates', None)))
        raise
if __name__ == '__main__':
    main()
