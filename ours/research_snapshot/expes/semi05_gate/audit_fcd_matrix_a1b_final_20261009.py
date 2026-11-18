import argparse, json, os, time, traceback
from pathlib import Path
from expes.semi05_gate.run_fcd_matrix_a1b_20261009 import sha

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ratio', choices=('05', '20'), required=True)
    p.add_argument('--gpu', choices=('0', '1'), required=True)
    p.add_argument('--tag', required=True)
    a = p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch, yaml
    from expes.semi05_gate.paired_fcd_a1b_20261009 import QualifiedStyleBalanced as MatrixHalf
    from expes.semi05_gate.paired_hard_relations_20261005 import tensor_state_digest
    from expes.semi05_gate.fcd_deterministic_losses_r2_20261005 import deterministic_training
    from expes.common.semi_data import CaseListDataset, build_eval_transform, read_case_list
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    output = Path('expes/work_dir/semi05_gate/FCD_A_MATRIX_20261006/A1B') / a.tag
    final = json.loads((output / 'final_metrics.json').read_text())
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['ratio'] == a.ratio and manifest['group'] == 'A1B' and (manifest['seed'] == 2027)
    assert manifest['protocol'] == final['protocol'] == 'FIXED_FINAL_DEVELOPMENT_V3_1'
    assert sha(manifest['init_from']) == manifest['init_sha256']
    assert final['primary_metric'] == 'native_student_epoch35_final'
    assert final['completed_epochs'] == 36 and final['stop_reason'] == 'max_epochs'
    for path, digest in manifest['source_hashes'].items():
        assert sha(path) == digest, path
    epochs = [json.loads(x) for x in (output / 'train_epochs.jsonl').read_text().splitlines()]
    steps = [json.loads(x) for x in (output / 'matrix_steps.jsonl').read_text().splitlines()]
    expected = 684 if a.ratio == '05' else 576
    per_epoch = expected // 36
    assert [r['epoch'] for r in epochs] == list(range(36)) and len(steps) == expected and (final['updates'] == expected)
    for i, r in enumerate(steps):
        assert r['epoch'] == i // per_epoch and r['update'] == i + 1 and (r['group'] == 'A1B')
        assert not r['skipped'] and r['teacher_no_grad'] and (r['clip_calls'] == 1) and (len(r['microbatches']) == 5) and r['context_computed'] and (r['ema_calls'] == 1) and (not r['branch_routing_measured'])
        assert all((v is not None and v > 0 for v in r['gradient_groups'].values()))
        for m in r['microbatches']:
            style = m['style']
            assert style['donor1'] != style['donor2'] and (not style['donor_labels_used'])
            assert style['native_labeled_supervision'] and style['native_teacher'] and (not style['old_fvc_active'])
            assert style['complementary_sum'] == 2 and style['style_delta1'] > 0.001 and (style['style_delta2'] > 0.001)
            c = m['context']
            assert c['applied'] and (not c['relation_computed']) and c['no_gt_argument'] and c['declared_state_restored']
            assert c['reduction'] == 'B' and c['background_voxels'] + c['foreground_voxels'] == 64 * c['reliable_blocks']
            if c['reliable_blocks']:
                assert all((v > 0 for v in c['raw_gradient_norms'].values()))
    destination = output / 'final_train_label_audit'
    assert not destination.exists()
    destination.mkdir()
    cfg = yaml.safe_load((output / 'config.yaml').read_text())
    cfg['experiment'].update(gpu=a.gpu, output_dir=str(destination / 'runtime'))
    report = dict(status='RUNNING', group='A1B', ratio=a.ratio, test_forwards=0, cases={}, weights={})
    start = time.monotonic()

    def write():
        (destination / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    try:
        with deterministic_training():
            t = MatrixHalf(cfg)
            split = Path(cfg['dataset']['split_root']) / 'fold2'
            assert {n: sha(split / n) for n in manifest['split_hashes']} == manifest['split_hashes']
            names = read_case_list(split / 'train_lab.txt')
            dataset = CaseListDataset(t.lab_set.fold_cv_dir, names, build_eval_transform(128), t.lab_set.threshold, split='train')
            for name in ('best.pth', 'final.pth'):
                metadata = final['weights'][name]
                path = Path(metadata['path'])
                assert sha(path) == metadata['sha256'] and path.stat().st_size == metadata['bytes']
                ck = torch.load(path, map_location='cpu')
                assert set(ck['roots']) == set(t.roots()) and ck['group'] == 'A1B' and (ck['seed'] == 2027)
                if name == 'final.pth':
                    assert ck['saved_epoch'] == 35 and ck['updates'] == expected
                for k, m in t.roots().items():
                    m.load_state_dict(ck['roots'][k], strict=True)
                    m.eval()
                saved_epoch, updates = (ck['saved_epoch'], ck['updates'])
                del ck
                before = tensor_state_digest({k: t._compact_state(m) for k, m in t.roots().items()})
                rows = []
                with torch.no_grad(), torch.cuda.amp.autocast():
                    for i in range(len(dataset)):
                        item = dataset[i]
                        image, label = t._prepare_batch({k: v.unsqueeze(0) for k, v in item.items() if torch.is_tensor(v)})
                        logits = t._forward_main_logits(t.student, image)
                        truth = label[:, 0] > 0
                        pred = logits.argmax(1).bool()
                        tp = int((pred & truth).sum())
                        fp = int((pred & ~truth).sum())
                        fn = int((~pred & truth).sum())
                        rows.append(dict(case=names[i], tp=tp, fp=fp, fn=fn, dice=2 * tp / max(1, 2 * tp + fp + fn)))
                        del image, label, logits
                        t.clear_relation_graph()
                assert tensor_state_digest({k: t._compact_state(m) for k, m in t.roots().items()}) == before
                report['cases'][name] = rows
                report['weights'][name] = dict(**metadata, saved_epoch=saved_epoch, updates=updates, all_six_roots_loaded=True, zero_update_state_unchanged=True)
                write()
            report.update(status='COMPLETE', total_updates=expected, completed_epochs=36, epoch35=epochs[-1], training_wall_seconds=final['wall_seconds'], training_peak_bytes=final['peak_bytes'], context_microbatches=expected * 5, context_applied_microbatches=expected * 5, audit_seconds=time.monotonic() - start, limitation='Training-internal FP/FN, not held-out causal or independent generalization evidence; historical test-selected init and ROI limitations remain.')
            write()
            print('A1B_AUDIT_COMPLETE ' + json.dumps({k: v for k, v in report.items() if k != 'cases'}), flush=True)
    except Exception:
        report.update(status='FAIL_ENGINEERING', error=traceback.format_exc())
        write()
        raise
if __name__ == '__main__':
    main()
