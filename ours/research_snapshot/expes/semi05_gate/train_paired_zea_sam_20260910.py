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
    from expes.semi05_gate.paired_zea_sam_20260910 import PairedZEASAM
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(2)
    base = Path('expes/work_dir/semi05_gate')
    for ratio in ('05', '20'):
        ghr = json.loads((base / f'PAIRED_GHR_FVC_R1_fold2/by_semi{ratio}_cv5/fold2/final_metrics.json').read_text())
        assert ghr['completed_epochs'] == 36 and ghr['stop_reason'] == 'max_epochs'
    root = base / 'PAIRED_ZEA_SAM_R1_fold2'
    suffix = f'preflight{args.ratio}' if args.preflight_only else f'by_semi{args.ratio}_cv5/fold2'
    output = root / suffix
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Refusing nonempty ZEA output')
    source = base / f'PAIRED_GHR_FVC_R1_fold2/by_semi{args.ratio}_cv5/fold2'
    cfg = yaml.safe_load((source / 'config.yaml').read_text())
    floor = json.loads((source / 'final_metrics.json').read_text())['floor_dice']
    init = Path(f'expes/artifacts/selftrain-insam/ckpt/Semi-{args.ratio}/fold2.pth')
    store = Path('/var/tmp/zc_asnet_paired_20260908/ZEA_SAM_R1') / suffix
    store.mkdir(parents=True, exist_ok=True)
    if any(store.iterdir()) or shutil.disk_usage(store).free < 8 * 1024 ** 3:
        raise RuntimeError('ZEA checkpoint store nonempty or insufficient space')
    if shutil.disk_usage(base).free < 1024 ** 3:
        raise RuntimeError('Insufficient log space')
    sources = ['expes/semi05_gate/' + name for name in ('paired_zea_sam_20260910.py', 'train_paired_zea_sam_20260910.py', 'paired_ghr_fvc_20260909.py', 'paired_hr_fvc_20260909.py', 'paired_hr_fclr_20260909.py', 'paired_fvc_control_20260909.py', 'paired_fclr_control_20260909.py', 'paired_wsv_control_20260908.py', 'paired_rng_data_20260908.py', 'final_logit_contextual_residual_wsv_ucp.py', 'weak_strong_view_ucp_trainer.py', 'uncertainty_curriculum_trainer.py')]
    hashes = {path: sha256(path) for path in sources}
    split = Path(cfg['dataset']['split_root']) / 'fold2'
    split_hashes = {name: sha256(split / name) for name in ('train_lab.txt', 'train_unlab.txt', 'test.txt')}
    sets = {name: set((split / name).read_text().splitlines()) for name in split_hashes}
    assert len(sets['train_lab.txt']) == int(args.ratio)
    assert len(sets['train_unlab.txt']) == 100 - int(args.ratio) and len(sets['test.txt']) == 25
    assert not sets['train_lab.txt'] & sets['train_unlab.txt']
    assert not (sets['train_lab.txt'] | sets['train_unlab.txt']) & sets['test.txt']
    assert split_hashes == json.loads((source / 'manifest.json').read_text())['split_hashes']
    if not args.preflight_only:
        for ratio in ('05', '20'):
            report = json.loads((root / f'preflight{ratio}/preflight.json').read_text())
            manifest = json.loads((root / f'preflight{ratio}/manifest.json').read_text())
            assert report['status'] == report['zea_status'] == 'PASS'
            assert report['control'] == 'ZEA-SAM-R1' and report['initial_logits_bitwise_equal']
            assert report['adapter_parameters'] == 12688 and report['adapter_lr'] == 0.0001
            assert all((v > 0 for v in report['adapter_gradient_norms'].values()))
            assert manifest['source_hashes'] == hashes
            assert manifest['init_sha256'] == sha256(f'expes/artifacts/selftrain-insam/ckpt/Semi-{ratio}/fold2.pth')
    cfg['experiment'].update(gpu=args.gpu, output_dir=str(output), seed=2026)
    cfg['semi'].update(paired_checkpoint_dir=str(store.resolve()), ghr_gate=True, zea_bottleneck=16, zea_lr=0.0001)
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(control='ZEA-SAM-R1', paired_base='GHR-FVC-R1', ratio=args.ratio, preflight_only=args.preflight_only, source_hashes=hashes, init_from=str(init), init_sha256=sha256(init), parent_config_sha256=sha256(source / 'config.yaml'), split_hashes=split_hashes, device=torch.cuda.get_device_name(0), visible_physical_gpu=args.gpu, checkpoint_working_store=str(store.resolve()), checkpoint_archive_pending=True, protocol='inherited label-localized ROI; ZEA adapter in SAM test path; training-only FVC; GHR single-view output; no TTA', architecture='zero-init residual adapter 384->16->384 after ImageEncoderViT3D neck', independent_run_only=True, unlabeled_gt_extra=False, selection='fixed original highest test Dice per epoch')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    trainer = PairedZEASAM(cfg)
    if args.preflight_only:
        trainer.preflight(init, floor)
    else:
        trainer.run_paired(init, floor)
if __name__ == '__main__':
    main()
