import argparse
import json
import os
from pathlib import Path
import shutil
import yaml
from expes.semi05_gate.train_paired_wsv_control_20260908 import sha256

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', choices=('05', '20'), required=True)
    parser.add_argument('--gpu', choices=('0', '1'), required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    import torch
    from expes.semi05_gate.paired_zea_cross_scale_20260911 import PairedCrossScale
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(2)
    base = Path('expes/work_dir/semi05_gate')
    root = base / 'PAIRED_ZEA_CROSS_SCALE_R1_fold2'
    suffix = f'preflight{args.ratio}' if args.preflight_only else f'by_semi{args.ratio}_cv5/fold2'
    output = root / suffix
    source = base / f'PAIRED_ZEA_SPATIAL_R1_fold2/by_semi{args.ratio}_cv5/fold2'
    parent = json.loads((source / 'manifest.json').read_text())
    cfg = yaml.safe_load((source / 'config.yaml').read_text())
    floor = json.loads((source / 'final_metrics.json').read_text())['floor_dice']
    init = Path(f'expes/artifacts/selftrain-insam/ckpt/Semi-{args.ratio}/fold2.pth')
    store_root = Path('/var/tmp/zc_asnet_zea_spatial_20260910/CROSS_SCALE_R1')
    store = store_root / suffix
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Nonempty output')
    if store.exists() and any(store.iterdir()):
        raise RuntimeError('Nonempty checkpoint store')
    if args.preflight_only and (not (root / 'preflight05/preflight.json').exists()):
        if shutil.disk_usage('/var/tmp').free < 4.95 * 1024 ** 3:
            raise RuntimeError('Persistent storage budget insufficient before preflight')
    if shutil.disk_usage(base).free < 0.5 * 1024 ** 3:
        raise RuntimeError('Log space low')
    extra = ['expes/semi05_gate/' + name for name in ('paired_zea_spatial_20260910.py', 'train_paired_zea_spatial_20260910.py', 'zea_teacher_quality_audit_20260910.py', 'train_paired_wsv_control_20260908.py', 'paired_zea_cross_scale_20260911.py', 'train_paired_zea_cross_scale_20260911.py')]
    hashes = {p: sha256(p) for p in sorted(set(parent['source_hashes']) | set(extra))}
    assert all((hashes[p] == h for p, h in parent['source_hashes'].items()))
    split = Path(cfg['dataset']['split_root']) / 'fold2'
    split_hashes = {name: sha256(split / name) for name in ('train_lab.txt', 'train_unlab.txt', 'test.txt')}
    assert split_hashes == parent['split_hashes']
    assert int(cfg['semi']['ucp_epochs']) == 36
    if not args.preflight_only:
        for ratio in ('05', '20'):
            report = json.loads((root / f'preflight{ratio}/preflight.json').read_text())
            manifest = json.loads((root / f'preflight{ratio}/manifest.json').read_text())
            assert report['status'] == report['zea_status'] == 'PASS'
            assert report['control'] == 'ZEA-CROSS-SCALE-R1' and report['adapter_parameters'] == 14176
            assert report['initial_logits_bitwise_equal'] and all((v > 0 for v in report['adapter_gradient_norms'].values()))
            assert manifest['source_hashes'] == hashes
            assert manifest['init_sha256'] == sha256(f'expes/artifacts/selftrain-insam/ckpt/Semi-{ratio}/fold2.pth')
            if ratio == '05':
                proof = report['persistent_disk_roundtrip']
                assert proof['logits_bitwise_equal'] and Path(proof['path']).stat().st_size == proof['bytes']
        if shutil.disk_usage('/var/tmp').free < 4.1 * 1024 ** 3:
            raise RuntimeError('Insufficient remaining dual-ratio checkpoint budget')
    cfg['experiment'].update(gpu=args.gpu, output_dir=str(output), seed=2026)
    cfg['semi'].update(paired_checkpoint_dir=str(store), zea_bottleneck=16, zea_lr=0.0001, zea_spatial_kernel=3, zea_spatial_identity_init=True, xsa_coarse_feature_conditioning=True)
    store.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(control='ZEA-CROSS-SCALE-R1', paired_base='ZEA-SPATIAL-R1', ratio=args.ratio, preflight_only=args.preflight_only, source_hashes=hashes, init_from=str(init), init_sha256=sha256(init), parent_config_sha256=sha256(source / 'config.yaml'), split_hashes=split_hashes, visible_physical_gpu=args.gpu, device=torch.cuda.get_device_name(0), checkpoint_working_store=str(store), checkpoint_policy='persistent alias-preserving best+final; fsync+SHA+readback; no quantization', architecture='spatial adapter + zero-init 64->16 projection of detached pooled coarse features', only_algorithm_change='add same-forward detached coarse feature conditioning, 1040 params', independent_run_only=True, unlabeled_gt_extra=False, protocol='inherited label-localized ROI; SAM trainable; single native view no TTA; paired 36 epochs', selection='fixed original highest test Dice per epoch', direct_ablation_parent=str(source))
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    trainer = PairedCrossScale(cfg)
    if args.preflight_only:
        trainer.preflight(init, floor)
    else:
        trainer.run_paired(init, floor)
if __name__ == '__main__':
    main()
