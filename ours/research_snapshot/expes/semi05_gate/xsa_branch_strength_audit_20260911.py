import argparse
import hashlib
import json
import os
from pathlib import Path
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', choices=['05', '20'], required=True)
    parser.add_argument('--gpu', choices=['0', '1'], required=True)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    import torch
    from expes.common.semi_data import CaseListDataset, build_eval_transform
    from expes.semi05_gate.paired_zea_cross_scale_20260911 import PairedCrossScale
    from expes.semi05_gate.paired_rng_data_20260908 import scoped_rng, stream_seed
    from expes.semi05_gate.train_paired_wsv_control_20260908 import sha256
    torch.set_num_threads(2)
    base = Path('expes/work_dir/semi05_gate')
    source = base / f'PAIRED_ZEA_CROSS_SCALE_R1_fold2/by_semi{args.ratio}_cv5/fold2'
    output = base / f'XSA_BRANCH_STRENGTH_AUDIT_20260911/{args.ratio}'
    assert not output.exists(), 'Preserve previous audit'
    cfg = yaml.safe_load((source / 'config.yaml').read_text())
    manifest = json.loads((source / 'manifest.json').read_text())
    assert all((sha256(p) == h for p, h in manifest['source_hashes'].items()))
    cfg['experiment'].update(gpu=args.gpu, output_dir=str(output), seed=2026)
    cfg['semi'].update(paired_checkpoint_dir=str(output / 'unused'))
    output.mkdir(parents=True)
    checkpoint_hash = sha256(source / 'best.pth')
    trainer = PairedCrossScale(cfg)
    trainer._load(source / 'best.pth')
    for module in (trainer.student, trainer.student_fclr_head, trainer.student_zea_adapter):
        module.eval().requires_grad_(False)
    adapter = trainer.student_zea_adapter
    before = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
    names = [Path(p).name.removesuffix('.nii.gz') for p in trainer.lab_set.image_paths]
    test_names = {Path(p).name.removesuffix('.nii.gz') for p in trainer.test_set.image_paths}
    assert not set(names) & test_names and len(names) == int(args.ratio)
    dataset = CaseListDataset(trainer.lab_set.fold_cv_dir, names, build_eval_transform(128), trainer.lab_set.threshold, split='train')
    runtime = {'scale': 1.0}

    def down_hook(_module, _inputs, value):
        runtime['down_norm'] = float(value.float().norm())

    def cross_hook(_module, _inputs, value):
        runtime['cross_norm'] = float(value.float().norm())
        return value * runtime['scale']
    handles = [adapter.down.register_forward_hook(down_hook), adapter.cross_projection.register_forward_hook(cross_hook)]
    records = []
    try:
        with torch.no_grad():
            for index in range(len(dataset)):
                with scoped_rng(stream_seed(trainer.pair_seed, 'xsa-branch-audit', index)):
                    item = dataset[index]
                image, label = trainer._prepare_batch({k: v.unsqueeze(0) for k, v in item.items() if torch.is_tensor(v)})
                gt = label[:, 0] > 0
                for scale in (1.0, 0.5, 0.0):
                    runtime['scale'] = scale
                    with torch.cuda.amp.autocast():
                        logits = trainer._forward_main_logits(trainer.student, image)
                    assert not trainer._xsa_context, 'Fresh context must be consumed'
                    mask = logits.argmax(1) > 0
                    probability = logits.float().softmax(1)[:, 1]
                    core = probability >= 0.92
                    tp, fp, fn = [int(v.sum()) for v in (mask & gt, mask & ~gt, ~mask & gt)]
                    ctp, cfp = (int((core & gt).sum()), int((core & ~gt).sum()))
                    records.append(dict(case=names[index], scale=scale, dice=2 * tp / max(1, 2 * tp + fp + fn), tp=tp, fp=fp, fn=fn, core_tp=ctp, core_fp=cfp, core_recall=ctp / max(1, int(gt.sum())), cross_to_down_norm=runtime['cross_norm'] / max(1e-12, runtime['down_norm'])))
                print('case_complete', index + 1, flush=True)
    finally:
        for handle in handles:
            handle.remove()
    assert all((torch.equal(v, adapter.state_dict()[k]) for k, v in before.items()))
    assert sha256(source / 'best.pth') == checkpoint_hash
    aggregates = {}
    for scale in (1.0, 0.5, 0.0):
        rr = [r for r in records if r['scale'] == scale]
        ctp, cfp = (sum((r['core_tp'] for r in rr)), sum((r['core_fp'] for r in rr)))
        aggregates[str(scale)] = dict(mean_dice=sum((r['dice'] for r in rr)) / len(rr), tp=sum((r['tp'] for r in rr)), fp=sum((r['fp'] for r in rr)), fn=sum((r['fn'] for r in rr)), core_precision=ctp / max(1, ctp + cfp), mean_core_recall=sum((r['core_recall'] for r in rr)) / len(rr), mean_cross_to_down_norm=sum((r['cross_to_down_norm'] for r in rr)) / len(rr))
    result = dict(status='COMPLETE', ratio=args.ratio, source=str(source / 'best.pth'), source_sha256=checkpoint_hash, audit_code_sha256=sha256(__file__), parent_source_hashes=manifest['source_hashes'], split_hashes=manifest['split_hashes'], scope='Fixed saved XSA student; all labeled training cases; cross projection output scaled only', optimizer_updates=0, test_iterations=0, unlabeled_iterations=0, parameters_unchanged=True, checkpoint_unchanged=True, limitation='Component intervention, not retrained ablation or unbiased generalization estimate', projection_weight_norm=float(adapter.cross_projection.weight.float().norm()), projection_bias_norm=float(adapter.cross_projection.bias.float().norm()), aggregates=aggregates, records=records)
    (output / 'audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'parent_source_hashes')}, indent=2))
if __name__ == '__main__':
    main()
