import argparse
import json
import os
from pathlib import Path
import shutil
import yaml
from expes.semi05_gate.train_paired_wsv_control_20260908 import sha256

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', choices=['05', '20'], required=True)
    parser.add_argument('--gpu', choices=['0', '1'], required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    import torch
    from expes.semi05_gate.paired_zea_cross_scale_half_20260911 import PairedCrossScaleHalf
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(2)
    base = Path('expes/work_dir/semi05_gate')
    root = base / 'PAIRED_ZEA_CROSS_SCALE_HALF_R2_fold2'
    suffix = f'preflight{args.ratio}' if args.preflight_only else f'by_semi{args.ratio}_cv5/fold2'
    output = root / suffix
    source = base / f'PAIRED_ZEA_CROSS_SCALE_R1_fold2/by_semi{args.ratio}_cv5/fold2'
    parent = json.loads((source / 'manifest.json').read_text())
    cfg = yaml.safe_load((source / 'config.yaml').read_text())
    floor = json.loads((source / 'final_metrics.json').read_text())['floor_dice']
    init = Path(f'expes/artifacts/selftrain-insam/ckpt/Semi-{args.ratio}/fold2.pth')
    store = Path('/var/tmp/zc_asnet_zea_spatial_20260910/CROSS_SCALE_HALF_R2') / suffix
    assert not (output.exists() and any(output.iterdir())), 'Preserve existing output'
    assert not (store.exists() and any(store.iterdir())), 'Preserve existing checkpoint store'
    required = 4.95 if args.preflight_only else 4.1
    assert shutil.disk_usage('/var/tmp').free > required * 1024 ** 3, 'Insufficient persistent save budget'
    assert shutil.disk_usage(base).free > 0.5 * 1024 ** 3, 'Insufficient log space'
    extras = ['expes/semi05_gate/' + n for n in ('paired_zea_cross_scale_half_20260911.py', 'train_paired_zea_cross_scale_half_20260911.py', 'xsa_branch_strength_audit_20260911.py')]
    hashes = {p: sha256(p) for p in sorted(set(parent['source_hashes']) | set(extras))}
    assert all((hashes[p] == h for p, h in parent['source_hashes'].items()))
    split = Path(cfg['dataset']['split_root']) / 'fold2'
    split_hashes = {n: sha256(split / n) for n in ('train_lab.txt', 'train_unlab.txt', 'test.txt')}
    assert split_hashes == parent['split_hashes'] and int(cfg['semi']['ucp_epochs']) == 36
    if not args.preflight_only:
        for ratio in ('05', '20'):
            report = json.loads((root / f'preflight{ratio}/preflight.json').read_text())
            evidence = json.loads((root / f'preflight{ratio}/manifest.json').read_text())
            assert report['status'] == report['zea_status'] == 'PASS'
            assert report['control'] == 'ZEA-CROSS-SCALE-HALF-R2' and report['xsa_cross_gain'] == 0.5
            assert report['initial_logits_bitwise_equal'] and all(report['gain_applies_to_student_and_teacher'].values())
            assert all((v > 0 for v in report['adapter_gradient_norms'].values()))
            assert evidence['source_hashes'] == hashes
            assert evidence['init_sha256'] == sha256(f'expes/artifacts/selftrain-insam/ckpt/Semi-{ratio}/fold2.pth')
            if ratio == '05':
                proof = report['persistent_disk_roundtrip']
                assert proof['logits_bitwise_equal'] and Path(proof['path']).stat().st_size == proof['bytes']
    cfg['experiment'].update(gpu=args.gpu, output_dir=str(output), seed=2026)
    cfg['semi'].update(paired_checkpoint_dir=str(store), xsa_cross_gain=0.5)
    store.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(control='ZEA-CROSS-SCALE-HALF-R2', paired_base='ZEA-CROSS-SCALE-R1', ratio=args.ratio, preflight_only=args.preflight_only, source_hashes=hashes, init_from=str(init), init_sha256=sha256(init), parent_config_sha256=sha256(source / 'config.yaml'), split_hashes=split_hashes, visible_physical_gpu=args.gpu, device=torch.cuda.get_device_name(0), checkpoint_working_store=str(store), direct_ablation_parent=str(source), only_algorithm_change='Cross projection output multiplied by 0.5, student and EMA, train and inference', architecture='XSA R1 same 14176 trainable adapter parameters, fixed cross gain 0.5', checkpoint_reconstruction_class='PairedCrossScaleHalf; saved cfg xsa_cross_gain=0.5 is required', checkpoint_policy='Persistent alias-preserving best+final; inherited XSA tensor schema, cfg defines gain', independent_run_only=True, unlabeled_gt_extra=False, protocol='Inherited ROI/SAM trainable; fixed single view no TTA; original 36 epochs/every-epoch highest test Dice', rationale='Training-labeled component audit half gain modestly improved mean Dice in both ratios; exploratory not generalization proof')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    trainer = PairedCrossScaleHalf(cfg)
    if args.preflight_only:
        trainer.preflight(init, floor)
    else:
        trainer.run_paired(init, floor)
if __name__ == '__main__':
    main()
