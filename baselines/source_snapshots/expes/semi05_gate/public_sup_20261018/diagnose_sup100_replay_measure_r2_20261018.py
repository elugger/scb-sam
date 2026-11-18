"""Authorized, bounded TRAIN-only replay; no checkpoint writes or test evaluation.

Two independent restorations replay the 52 already-recorded successful steps.
Measurement revision after r1 exact-loss mismatch: retain every mismatch; never qualify a continuation.
At the next failing batch, compare original scaling with the original init scale.
Never call optimizer.step on that failed batch; all diagnostic updates are discarded.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import traceback

ROOT = Path('__RECORDED_PROJECT_ROOT__')
BASE = ROOT/'expes/work_dir/semi05_gate/PUBLIC_BASIC_SUP_FOLD2_20261018'
OUT = BASE/'control/sup100_replay_diagnostic_20261018_r2'
CKPT = ROOT/'expes/artifacts/FCD_A_MATRIX_20261006/PUBLIC_BASIC_SUP_FOLD2_20261018/sup100/latest.pth'
CHECKPOINT_SHA = 'e96bb38fec82c83b1a5e17ce722d21e0bedea084dc4f0218d71919cc94ad51be'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def restore_rng(state, torch, np, generator):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    torch.cuda.set_rng_state_all(state['torch_cuda_all'])
    generator.set_state(state['data_generator'])


def exact_tree_hash(value):
    import numpy as np
    import torch
    h = hashlib.sha256()

    def add(x):
        if isinstance(x, torch.Tensor):
            t = x.detach().cpu().contiguous()
            h.update(str(('tensor', str(t.dtype), tuple(t.shape))).encode())
            h.update(memoryview(t.reshape(-1).view(torch.uint8).numpy()))
        elif isinstance(x, np.ndarray):
            h.update(str(('numpy', str(x.dtype), x.shape)).encode())
            h.update(x.tobytes())
        elif isinstance(x, dict):
            h.update(b'dict')
            for key in sorted(x, key=lambda k: (type(k).__name__, repr(k))):
                add(key)
                add(x[key])
        elif isinstance(x, (list, tuple)):
            h.update(type(x).__name__.encode())
            for item in x:
                add(item)
        else:
            h.update(str((type(x).__name__, repr(x))).encode())
        h.update(b'\x00')
    add(value)
    return h.hexdigest()


def main(args):
    from expes.semi05_gate.public_sup_20261018 import train_public_basic_sup_20261018 as original
    from expes.semi05_gate.public_sup_20261018 import public_basic_sup_guard_20261018 as guard
    require(original.sha256_file(__file__) == args.source_sha, 'Diagnostic source SHA changed')
    cfg_path = ROOT/'expes/semi05_gate/public_sup_20261018/sup100.json'
    cfg = json.loads(cfg_path.read_text())
    require(cfg == json.loads((BASE/'sup100/config.json').read_text()), 'Original config contents changed')
    original.validate_config(cfg)
    checked = original.verify_manifests(cfg)
    claim = original.validate_gpu_claim(cfg, 1, args.claim_fd, args.claim_device, args.claim_inode)
    require(original.sha256_file(CKPT) == CHECKPOINT_SHA, 'Boundary checkpoint SHA changed')
    previous_failure = json.loads((BASE/'sup100/failure.json').read_text())
    require(previous_failure['successful_updates'] == 14930, 'Unexpected failure boundary')
    steps_path = BASE/'sup100/training_steps.jsonl'
    steps_sha = original.sha256_file(steps_path)
    steps = [json.loads(line) for line in steps_path.read_text().splitlines()]
    require(len(steps) == 14930, 'Original ledger changed')
    expected = steps[14878:]
    require(len(expected) == 52, 'Wrong replay length')
    for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[variable] = '4'
    import numpy as np
    import torch
    import torchio as tio
    from torch.cuda import amp
    from torch.utils.data import DataLoader
    from segment_anything.auto_sam_seg import build_auto_sam_seg
    from utils.auto_sam_loss import build_supervised_loss_fns, cascade_supervised_loss
    from utils.cascade_train_utils import normalize_batch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    require(torch.cuda.device_count() == 1, 'Exactly one visible GPU required')
    report = {'status': 'RUNNING', 'source_sha256': args.source_sha, 'checkpoint_sha256': CHECKPOINT_SHA,
              'original_ledger_sha256': steps_sha, 'claim': claim, 'passes': [],
              'test_evaluations': 0, 'formal_training_updates': 0, 'weights_written': 0,
              'diagnostic_optimizer_updates': 0, 'root_cause_proven': False}
    original.atomic_json(OUT/'started.json', {'pid': os.getpid(), 'time': original.now_iso(), **report})
    main_loss, coarse_loss = build_supervised_loss_fns()
    norm = tio.ZNormalization(masking_method=lambda tensor: tensor > 0)
    ids = original.read_case_ids(original.resolve_path(cfg, cfg['train_list']))
    require(len(ids) == 346, 'Not the authorized supervised train split')
    for pass_index in range(2):
        with guard.original_lock():
            capacity = guard.capacity_snapshot(required_ram=14*guard.GIB+guard.MARGIN)
        checkpoint = original.load_trusted_checkpoint(CKPT, map_location='cpu')
        require(checkpoint['successful_updates'] == 14878 and checkpoint['completed_epochs'] == 43
                and checkpoint['next_epoch'] == 43 and checkpoint['test_evaluations'] == 0,
                'Unexpected checkpoint boundary')
        require(checkpoint['config_sha256'] == original.sha256_file(cfg_path), 'Checkpoint config identity mismatch')
        require(checkpoint['source_manifest'] == checked, 'Checkpoint source manifest mismatch')
        student = build_auto_sam_seg(model_type='vit_b_ori', sam_checkpoint=str(original.resolve_path(cfg, cfg['sam_checkpoint'])),
                    coarse_checkpoint=None, img_size=128, num_classes=2, deep_supervision=False,
                    device='cuda:0', mc_dropout_p=0.0).to('cuda:0')
        student.set_train_phase('e2e')
        student.load_state_dict(checkpoint['student'], strict=True)
        sam = student.cascade.sam
        optimizer = torch.optim.AdamW([
            {'params': sam.image_encoder.parameters(), 'lr': 1e-4},
            {'params': sam.prompt_encoder.parameters(), 'lr': 1e-5},
            {'params': sam.mask_decoder.parameters(), 'lr': 1e-5},
            {'params': student.cascade.coarse_head.parameters(), 'lr': 1.25e-4}],
            betas=(0.9, 0.999), weight_decay=0.1)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler = amp.GradScaler(enabled=True, init_scale=1024.0)
        scaler.load_state_dict(checkpoint['scaler'])
        generator = torch.Generator().manual_seed(2027)
        dataset = original.make_strict_dataset(cfg, ids, 'train')
        loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0,
                            pin_memory=True, drop_last=False, generator=generator)
        calls = {'count': 0}
        step_method = original.install_optimizer_step_counter(optimizer, calls)
        student.train()
        restore_rng(checkpoint['rng'], torch, np, generator)
        del checkpoint
        result = {'pass_index': pass_index, 'capacity': capacity, 'matched_steps': 0, 'exact_loss_matches': 0, 'max_absolute_loss_difference': 0.0}
        iterator = iter(loader)
        for position in range(53):
            batch = next(iterator)
            case_id = batch['case_id'][0]
            image, label = original.prepare_batch(batch, 'cuda:0', norm, normalize_batch)
            if position < 52:
                require(case_id == expected[position]['case_id'], 'Actual case replay mismatch')
            before_rng = original.capture_rng(torch, np, generator)
            before_buffers = {k: v.detach().clone() for k, v in student.named_buffers()}
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(enabled=True):
                logits, coarse = student(image, return_aux=True)
                loss = cascade_supervised_loss(logits, coarse, label.float(), main_loss, coarse_loss, aux_weight=0.3)
            loss_value = float(loss.detach().item())
            require(bool(torch.isfinite(loss)), 'Non-finite replay forward loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nonfinite = [name for name, p in student.named_parameters()
                         if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
            if position < 52:
                require(not nonfinite, 'Replay diverged before the recorded failure')
                loss_difference = abs(loss_value - expected[position]['loss'])
                result['exact_loss_matches'] += int(loss_difference == 0)
                result['max_absolute_loss_difference'] = max(result['max_absolute_loss_difference'], loss_difference)
                scaler.step(optimizer)
                scaler.update()
                require(calls['count'] == position+1, 'Diagnostic optimizer call count mismatch')
                require(float(scaler.get_scale()) == expected[position]['amp_scale'], 'Scale replay mismatch')
                result['matched_steps'] += 1
                report['diagnostic_optimizer_updates'] += 1
                original.append_jsonl(OUT/'diagnostic_steps.jsonl', {'pass': pass_index, 'position': position,
                      'case_id': case_id, 'loss': loss_value, 'amp_scale': float(scaler.get_scale()),
                      'matches_original_exactly': loss_difference == 0, 'original_loss': expected[position]['loss'],
                      'absolute_loss_difference': loss_difference, 'time': original.now_iso()})
                if position == 51:
                    result['boundary_model_sha256'] = original.state_dict_sha256(student.state_dict())
                    result['boundary_optimizer_sha256'] = exact_tree_hash(optimizer.state_dict())
                    result['boundary_rng_sha256'] = exact_tree_hash(original.capture_rng(torch, np, generator))
                    result['boundary_scaler_sha256'] = exact_tree_hash(scaler.state_dict())
            else:
                result['original_failure_reproduced'] = bool(nonfinite)
                result['failed_case_id'] = case_id
                result['failed_loss'] = loss_value
                result['failed_scale'] = float(scaler.get_scale())
                result['nonfinite_gradient_names'] = nonfinite
                result['failed_step_optimizer_calls'] = 0
                del logits, coarse, loss
                optimizer.zero_grad(set_to_none=True)
                restore_rng(before_rng, torch, np, generator)
                with torch.no_grad():
                    for name, value in student.named_buffers():
                        value.copy_(before_buffers[name])
                low_scaler = amp.GradScaler(enabled=True, init_scale=1024.0)
                with amp.autocast(enabled=True):
                    logits, coarse = student(image, return_aux=True)
                    loss = cascade_supervised_loss(logits, coarse, label.float(), main_loss, coarse_loss, aux_weight=0.3)
                require(float(loss.detach().item()) == loss_value, 'Same-batch forward changed before scaling comparison')
                low_scaler.scale(loss).backward()
                low_scaler.unscale_(optimizer)
                lower_nonfinite = [name for name, p in student.named_parameters()
                                  if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
                result['scale1024_nonfinite_gradient_names'] = lower_nonfinite
                result['scale1024_all_gradients_finite'] = not lower_nonfinite
                require(calls['count'] == 52, 'Failure diagnostic updated optimizer')
            del image, label, logits, coarse, loss, batch, before_buffers, before_rng
        report['passes'].append(result)
        original.atomic_json(OUT/'progress.json', report)
        optimizer.step = step_method
        del step_method, iterator, loader, dataset, student, sam, optimizer, scaler, low_scaler
        torch.cuda.empty_cache()
    a, b = report['passes']
    keys = ['boundary_model_sha256', 'boundary_optimizer_sha256', 'boundary_rng_sha256',
            'boundary_scaler_sha256', 'failed_case_id', 'failed_loss', 'failed_scale',
            'nonfinite_gradient_names', 'scale1024_all_gradients_finite']
    independent_exact = all(a[k] == b[k] for k in keys)
    scale_effect = all(p['original_failure_reproduced'] and p['scale1024_all_gradients_finite'] for p in report['passes'])
    require(original.sha256_file(steps_path) == steps_sha and original.sha256_file(CKPT) == CHECKPOINT_SHA,
            'Original evidence changed')
    report.update(status='DIAGNOSTIC_ONLY_NOT_CONTINUATION_APPROVAL',
                  identical_restored_model_optimizer_rng_scaler_after_52_steps=independent_exact,
                  original_52_steps_case_loss_scale_exact_match=all(p['exact_loss_matches'] == 52 for p in report['passes']),
                  scaling_dependent_overflow_reproduced=scale_effect,
                  root_cause_proven=scale_effect,
                  continuation_authorized_by_this_report=False,
                  time=original.now_iso())
    original.atomic_json(OUT/'report.json', report)
    print(json.dumps({'status': report['status'], 'diagnostic_optimizer_updates':104,
                      'test_evaluations':0, 'formal_training_updates':0}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--claim-fd', type=int, required=True)
    parser.add_argument('--claim-device', type=int, required=True)
    parser.add_argument('--claim-inode', type=int, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except BaseException as error:
        failure = {'status':'FAILED_NO_AUTO_RETRY','error':str(error),'traceback':traceback.format_exc(),
                   'test_evaluations':0,'formal_training_updates':0,'weights_written':0}
        with (OUT/'diagnostic_failure.json').open('x') as f:
            json.dump(failure, f, indent=2)
        raise
