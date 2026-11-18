import argparse
import copy
import io
import json
import os
from pathlib import Path
import yaml

def alias_preserving_cpu_state(module):
    memo, result = ({}, {})
    for name, tensor in module.state_dict().items():
        identity = (str(tensor.device), tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype))
        if identity not in memo:
            memo[identity] = tensor.detach().cpu().clone()
        result[name] = memo[identity]
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', choices=['05', '20'], required=True)
    parser.add_argument('--gpu', choices=['0', '1'], required=True)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    import torch
    import torch.nn.functional as F
    from expes.common.semi_data import CaseListDataset, build_eval_transform
    from expes.semi05_gate.paired_zea_sam_20260910 import PairedZEASAM
    from expes.semi05_gate.paired_rng_data_20260908 import scoped_rng, stream_seed
    torch.set_num_threads(2)
    base = Path('expes/work_dir/semi05_gate')
    source = base / f'PAIRED_ZEA_SAM_R1_fold2/by_semi{args.ratio}_cv5/fold2'
    output = base / f'ZEA_TEACHER_QUALITY_AUDIT_20260910/{args.ratio}'
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Nonempty audit output')
    cfg = yaml.safe_load((source / 'config.yaml').read_text())
    cfg['experiment'].update(gpu=args.gpu, output_dir=str(output), seed=2026)
    cfg['semi'].update(paired_checkpoint_dir=str(output / 'unused'), zea_lr=0.0001)
    output.mkdir(parents=True, exist_ok=True)
    trainer = PairedZEASAM(cfg)
    trainer._load(source / 'best.pth')
    trainer.student.eval().requires_grad_(False)
    trainer.student_fclr_head.eval().requires_grad_(False)
    trainer.student_zea_adapter.eval().requires_grad_(False)
    student = copy.deepcopy(trainer.student_zea_adapter.state_dict())
    teacher = copy.deepcopy(trainer.teacher_zea_adapter.state_dict())
    storage = None
    if args.ratio == '05':
        state = alias_preserving_cpu_state(trainer.student)
        buffer = io.BytesIO()
        torch.save(state, buffer)
        size = buffer.tell()
        buffer.seek(0)
        restored = torch.load(buffer, map_location='cpu')
        original = trainer.student.state_dict()
        assert original.keys() == restored.keys()
        assert all((torch.equal(t.detach().cpu(), restored[k]) for k, t in original.items()))
        storage = dict(status='PASS', all_keys_values_bitwise_equal=True, entries=len(state), unique_tensor_objects=len({id(v) for v in state.values()}), serialized_bytes=size, serialized_gib=size / 1024 ** 3, method='preserve exact registered aliases, no quantization', wrote_checkpoint_to_disk=False)
        del state, restored, original, buffer
    names = [Path(p).name.removesuffix('.nii.gz') for p in trainer.lab_set.image_paths]
    test_names = {Path(p).name.removesuffix('.nii.gz') for p in trainer.test_set.image_paths}
    assert not set(names) & test_names and len(names) == int(args.ratio)
    dataset = CaseListDataset(trainer.lab_set.fold_cv_dir, names, build_eval_transform(128), trainer.lab_set.threshold, split='train')
    records = []
    with torch.no_grad():
        for index in range(len(dataset)):
            with scoped_rng(stream_seed(trainer.pair_seed, 'zea-teacher-quality', index)):
                item = dataset[index]
            image, label = trainer._prepare_batch({k: v.unsqueeze(0) for k, v in item.items() if torch.is_tensor(v)})
            gt = label[:, 0] > 0
            for variant in ['student_adapter', 'ema_adapter', 'no_adapter']:
                trainer._zea_active = variant != 'no_adapter'
                trainer.student_zea_adapter.load_state_dict(teacher if variant == 'ema_adapter' else student)
                with torch.cuda.amp.autocast():
                    logits = trainer._forward_main_logits(trainer.student, image)
                probability = logits.float().softmax(1)[:, 1]
                mask = logits.argmax(1) > 0
                tp = int((mask & gt).sum())
                fp = int((mask & ~gt).sum())
                fn = int((~mask & gt).sum())
                core = probability >= 0.92
                core_tp = int((core & gt).sum())
                core_fp = int((core & ~gt).sum())
                records.append(dict(case=names[index], variant=variant, dice=2 * tp / max(1, 2 * tp + fp + fn), tp=tp, fp=fp, fn=fn, core_tp=core_tp, core_fp=core_fp, core_precision=core_tp / max(1, core_tp + core_fp), core_recall=core_tp / max(1, int(gt.sum()))))
            print('case_complete', index + 1, flush=True)
    trainer._zea_active = True
    trainer.student_zea_adapter.load_state_dict(student)
    aggregates = {}
    for variant in ['student_adapter', 'ema_adapter', 'no_adapter']:
        rr = [r for r in records if r['variant'] == variant]
        tp = sum((r['core_tp'] for r in rr))
        fp = sum((r['core_fp'] for r in rr))
        aggregates[variant] = dict(mean_dice=sum((r['dice'] for r in rr)) / len(rr), core_precision=tp / max(1, tp + fp), core_tp=tp, core_fp=fp, mean_core_recall=sum((r['core_recall'] for r in rr)) / len(rr))
    result = dict(status='COMPLETE', ratio=args.ratio, source=str(source / 'best.pth'), scope='all training-labeled cases; fixed student backbone+HR head; swapping saved adapter only', historical_full_teacher=False, optimizer_updates=0, test_iterations=0, unlabeled_iterations=0, storage_roundtrip=storage, aggregates=aggregates, records=records)
    (output / 'audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2))
if __name__ == '__main__':
    main()
