from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import re
import socket
import stat
import sys
import time
import traceback
from datetime import datetime, timezone

def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat()

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def read_case_ids(path):
    values = [line.strip() for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not values:
        raise ValueError('Empty case list')
    if len(set(values)) != len(values):
        raise ValueError('Duplicate case IDs')
    for value in values:
        if value in ('.', '..') or '/' in value or '\\' in value or ('\x00' in value) or (':' in value):
            raise ValueError(f'Unsafe case ID: {value!r}')
        if value.endswith('.nii') or value.endswith('.nii.gz'):
            raise ValueError('Case lists must contain stems, not filenames')
    return values

def validate_case_sets(train_ids, test_ids, expected_train_count, expected_test_count):
    if len(train_ids) != int(expected_train_count) or len(test_ids) != int(expected_test_count):
        raise ValueError('Case-list counts disagree with the frozen configuration')
    if not train_ids or not test_ids or len(set(train_ids)) != len(train_ids) or (len(set(test_ids)) != len(test_ids)):
        raise ValueError('Empty or duplicate case sets')
    if set(train_ids) & set(test_ids):
        raise ValueError('Train/test case overlap')
    return {'train_count': len(train_ids), 'test_count': len(test_ids), 'case_sets_disjoint': True}

def training_budget(train_count, epochs=60):
    if isinstance(train_count, bool) or not isinstance(train_count, int) or train_count <= 0:
        raise ValueError('train_count must be a positive integer')
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError('epochs must be a positive integer')
    return {'epochs': epochs, 'train_count': train_count, 'updates_per_epoch': train_count, 'total_updates': train_count * epochs, 'batch_size': 1, 'replay_per_epoch': 0}

def validate_config(cfg):
    required = ('project_root', 'output_dir', 'artifact_dir', 'fold_cv_dir', 'train_list', 'test_list', 'sam_checkpoint', 'sam_checkpoint_sha256', 'source_sha256', 'input_sha256', 'resource_guard_module', 'train_count', 'test_count', 'labeled_cases', 'fold', 'seed', 'epochs')
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f'Missing configuration keys: {missing}')
    for key, expected in (('fold', 2), ('seed', 2027), ('epochs', 60)):
        if type(cfg[key]) is not int or cfg[key] != expected:
            raise ValueError(f'Frozen {key} must equal {expected}')
    if 'label_percent' in cfg:
        raise ValueError('Case-count experiment must not be labeled a percentage arm')
    if type(cfg['labeled_cases']) is not int or cfg['labeled_cases'] not in (5, 10):
        raise ValueError('Only five/ten labeled cases are designed')
    if type(cfg['train_count']) is not int or cfg['train_count'] != cfg['labeled_cases']:
        raise ValueError('Actual labeled count differs')
    if cfg.get('checkpoint_policy') != 'final_student_only_no_resume':
        raise ValueError('Frozen final-only checkpoint policy required')
    if cfg.get('amp_policy') != 'fixed_1024':
        raise ValueError('Fixed AMP1024 required')
    from ncase_protocol import ROOT, BASE, artifact_dir
    if cfg['project_root'] != str(ROOT) or cfg['output_dir'] != str(BASE / ('sup_n' + str(cfg['labeled_cases']).zfill(3))) or cfg['artifact_dir'] != str(artifact_dir(cfg['labeled_cases'])) or (cfg['resource_guard_module'] != 'ncase_resources'):
        raise ValueError('Noncanonical independent experiment paths')
    if type(cfg['test_count']) is not int or cfg['test_count'] != 87:
        raise ValueError('Public fold2 must have exactly 87 test case IDs')
    budget = training_budget(cfg['train_count'], cfg['epochs'])
    for key, expected in (('batch_size', 1), ('num_workers', 0), ('img_size', 128), ('model_type', 'vit_b_ori'), ('num_classes', 2), ('deep_supervision', False), ('test_policy', 'fixed_last_epoch_once'), ('mode', 'sup'), ('use_hpq', False), ('use_unlabeled', False), ('resume', False)):
        if key in cfg and cfg[key] != expected:
            raise ValueError(f'Unsupported {key}: only {expected!r} is allowed')
    for key in ('init_from', 'coarse_checkpoint', 'teacher_checkpoint', 'best_checkpoint', 'resume_from'):
        if cfg.get(key) is not None:
            raise ValueError(f'{key} is not permitted in a fresh basic-supervised run')
    for key, expected in (('lr', 0.0001), ('lr_encoder_scale', 1.0), ('lr_decoder_scale', 0.1), ('lr_coarse_scale', 1.25), ('weight_decay', 0.1), ('aux_weight', 0.3), ('beta1', 0.9), ('beta2', 0.999), ('amp_init_scale', 1024.0)):
        if key in cfg and (not isinstance(cfg[key], (int, float)) or float(cfg[key]) != expected):
            raise ValueError(f'Frozen supervised optimizer/loss parameter {key} must equal {expected}')
    if not isinstance(cfg['source_sha256'], dict) or not cfg['source_sha256']:
        raise ValueError('The source hash manifest is required')
    if not isinstance(cfg['input_sha256'], dict) or not cfg['input_sha256']:
        raise ValueError('The input hash manifest is required')
    for value in [cfg['sam_checkpoint_sha256'], *cfg['source_sha256'].values(), *cfg['input_sha256'].values()]:
        if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
            raise ValueError('Every SHA256 must be exactly 64 lowercase hex characters')
    return budget

def resolve_path(cfg, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(cfg['project_root']) / path).resolve()

def atomic_json(path, payload):
    path = Path(path)
    temp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temp.open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)

def append_jsonl(path, payload):
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())

def verify_manifests(cfg):
    checked = {}
    own_path = Path(__file__).resolve()
    own_bound = False
    for group in ('source_sha256', 'input_sha256'):
        for name, expected in cfg[group].items():
            path = resolve_path(cfg, name)
            actual = sha256_file(path)
            if actual != expected:
                raise RuntimeError(f'SHA mismatch: {path}')
            checked[str(path)] = {'sha256': actual, 'bytes': path.stat().st_size, 'kind': group}
            own_bound = own_bound or (group == 'source_sha256' and path == own_path)
    if not own_bound:
        raise RuntimeError('Runner source is not bound in source_sha256')
    for key in ('train_list', 'test_list'):
        path = resolve_path(cfg, cfg[key])
        if str(path) not in checked or checked[str(path)]['kind'] != 'input_sha256':
            raise RuntimeError(f'{key} is not bound in input_sha256')
    checkpoint = resolve_path(cfg, cfg['sam_checkpoint'])
    if sha256_file(checkpoint) != cfg['sam_checkpoint_sha256']:
        raise RuntimeError('Generic SAM pretraining checkpoint SHA mismatch')
    checked[str(checkpoint)] = {'sha256': cfg['sam_checkpoint_sha256'], 'bytes': checkpoint.stat().st_size, 'kind': 'generic_sam_pretraining_not_public_test_selected'}
    return checked

def make_strict_dataset(cfg, case_ids, split):
    import SimpleITK as sitk
    import torchio as tio
    from expes.common.semi_data import CaseListDataset, build_eval_transform, build_train_transform

    class StrictCaseListDataset(CaseListDataset):

        def __init__(self):
            transform = build_train_transform(128, random_flip=True) if split == 'train' else build_eval_transform(128)
            super().__init__(resolve_path(cfg, cfg['fold_cv_dir']), case_ids, transform=transform, threshold=1000, split=split)
            self.actual_case_ids = tuple(case_ids)

        def __getitem__(self, index):
            from ncase_inputs import check_case
            check_case(cfg, self.actual_case_ids[index], split, self.image_paths[index], self.label_paths[index])
            image = sitk.ReadImage(self.image_paths[index])
            label = sitk.ReadImage(self.label_paths[index])
            origin_adjusted = image.GetOrigin() != label.GetOrigin()
            direction_adjusted = image.GetDirection() != label.GetDirection()
            if origin_adjusted:
                image.SetOrigin(label.GetOrigin())
            if direction_adjusted:
                image.SetDirection(label.GetDirection())
            subject = tio.Subject(image=tio.ScalarImage.from_sitk(image), label=tio.LabelMap.from_sitk(label))
            subject = self.transform(subject)
            label_tensor = subject.label.data.clone().detach().long()
            if label_tensor.dim() == 4 and label_tensor.shape[0] == 1:
                label_tensor = label_tensor.squeeze(0)
            if not bool(((label_tensor == 0) | (label_tensor == 1)).all()):
                raise RuntimeError(f'Non-binary label in actual case {self.actual_case_ids[index]}')
            return {'image': subject.image.data.clone().detach(), 'label': label_tensor, 'case_id': self.actual_case_ids[index], 'requested_index': index, 'foreground_voxels': int(label_tensor.sum().item()), 'origin_adjusted': origin_adjusted, 'direction_adjusted': direction_adjusted}
    return StrictCaseListDataset()

def capture_rng(torch, np, data_generator):
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch_cpu': torch.get_rng_state(), 'torch_cuda_all': torch.cuda.get_rng_state_all(), 'data_generator': data_generator.get_state()}

def state_dict_sha256(state_dict):
    import torch
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        descriptor = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(',', ':')).encode('utf-8')
        digest.update(len(descriptor).to_bytes(8, 'little'))
        digest.update(descriptor)
        raw = tensor.reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(raw))
        del tensor, raw
    return digest.hexdigest()

def load_trusted_checkpoint(path, map_location='cpu'):
    import inspect
    import torch
    kwargs = {'map_location': map_location}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = False
    return torch.load(path, **kwargs)

def install_optimizer_step_counter(optimizer, counter):
    from functools import wraps
    original_step = optimizer.step

    @wraps(original_step)
    def counted_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        counter['count'] += 1
        return result
    optimizer.step = counted_step
    return original_step

def audit_sam_initialization(sam, checkpoint_path, expected_sha256, extractor):
    import torch
    checkpoint_path = Path(checkpoint_path)
    if sha256_file(checkpoint_path) != expected_sha256:
        raise RuntimeError('SAM checkpoint changed before its initialization load audit')
    checkpoint = load_trusted_checkpoint(checkpoint_path, map_location='cpu')
    expected = extractor(checkpoint)
    actual = sam.state_dict()
    if not isinstance(expected, dict) or not expected:
        raise RuntimeError('The actual builder extractor did not return a nonempty state dictionary')
    missing = sorted(set(actual) - set(expected))
    unexpected = sorted(set(expected) - set(actual))
    shape_mismatches = []
    value_mismatches = []
    dtype_conversions = []
    invalid_entries = []
    checked_count = 0
    for key in sorted(set(actual) & set(expected)):
        source = expected[key]
        target = actual[key]
        if not isinstance(source, torch.Tensor) or not isinstance(target, torch.Tensor):
            invalid_entries.append(key)
            continue
        if tuple(source.shape) != tuple(target.shape):
            shape_mismatches.append({'key': key, 'checkpoint_shape': list(source.shape), 'model_shape': list(target.shape)})
            continue
        target_cpu = target.detach().cpu()
        source_cpu = source.detach().cpu()
        if source_cpu.dtype != target_cpu.dtype:
            dtype_conversions.append({'key': key, 'checkpoint_dtype': str(source_cpu.dtype), 'model_dtype': str(target_cpu.dtype)})
            source_cpu = source_cpu.to(dtype=target_cpu.dtype)
        if not torch.equal(target_cpu, source_cpu):
            value_mismatches.append(key)
        checked_count += 1
        del target_cpu, source_cpu
    if sha256_file(checkpoint_path) != expected_sha256:
        raise RuntimeError('SAM checkpoint changed during its initialization load audit')
    passed = not (missing or unexpected or shape_mismatches or value_mismatches or invalid_entries)
    return {'time': now_iso(), 'status': 'PASS' if passed else 'FAIL', 'checkpoint_sha256': expected_sha256, 'extractor': 'segment_anything.build_sam3D_cascade._extract_state_dict', 'actual_model_key_count': len(actual), 'checkpoint_key_count': len(expected), 'tensor_values_checked': checked_count, 'all_values_checked': checked_count == len(actual) == len(expected), 'missing_keys': missing, 'unexpected_keys': unexpected, 'shape_mismatches': shape_mismatches, 'value_mismatches': value_mismatches, 'invalid_entries': invalid_entries, 'dtype_conversions': dtype_conversions, 'model_reloaded_or_repaired': False, 'model_forward_calls': 0, 'training_updates': 0, 'test_evaluations': 0}

def validate_gpu_claim(cfg, gpu, fd, expected_device, expected_inode):
    import fcntl
    if fd <= 2:
        raise RuntimeError('GPU claim must be an inherited dedicated file descriptor')
    output_path = Path(cfg['output_dir'])
    if not output_path.is_absolute():
        output_path = Path(cfg['project_root']) / output_path
    path = output_path.parent / 'control' / f'gpu{gpu}.lock'
    if any((item.is_symlink() for item in (path, *path.parents))):
        raise RuntimeError('GPU claim path contains a symlink')
    current = path.stat()
    inherited = os.fstat(fd)
    for value in (current, inherited):
        if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
            raise RuntimeError('GPU claim must be a regular single-link file')
        if (value.st_dev, value.st_ino) != (expected_device, expected_inode):
            raise RuntimeError("GPU claim identity differs from the launcher's recorded identity")
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return {'path': str(path), 'fd': fd, 'device': inherited.st_dev, 'inode': inherited.st_ino, 'nlink': inherited.st_nlink, 'exclusive_claim_retained': True}

def prepare_batch(batch, device, norm_transform, normalize_batch):
    image = normalize_batch(batch['image'], norm_transform).to(device, non_blocking=True)
    label = batch['label'].to(device, non_blocking=True)
    if label.dim() == 4:
        label = label.unsqueeze(1)
    return (image, label)

def run(cfg, config_sha256, gpu, gpu_claim=None):
    import fcntl
    for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[variable] = '4'
    if gpu != 0:
        raise RuntimeError('This task owns GPU0 only')
    budget = validate_config(cfg)
    checked = verify_manifests(cfg)
    train_ids = read_case_ids(resolve_path(cfg, cfg['train_list']))
    test_ids = read_case_ids(resolve_path(cfg, cfg['test_list']))
    validate_case_sets(train_ids, test_ids, cfg['train_count'], cfg['test_count'])
    output_dir = resolve_path(cfg, cfg['output_dir'])
    artifact_dir = resolve_path(cfg, cfg['artifact_dir'])
    if output_dir == artifact_dir or output_dir.exists() or artifact_dir.exists():
        raise RuntimeError('Fresh output and artifact directories must be distinct and absent; no automatic resume')
    output_dir.mkdir(parents=True, exist_ok=False)
    run_lock = (output_dir / 'run.lock').open('x')
    fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    successful_updates = 0
    completed_epochs = 0
    try:
        artifact_dir.mkdir(parents=True, exist_ok=False)
        atomic_json(output_dir / 'config.json', cfg)
        manifest = {'created_at': now_iso(), 'pid': os.getpid(), 'hostname': socket.gethostname(), 'gpu': gpu, 'inherited_gpu_claim': gpu_claim, 'config_sha256': config_sha256, 'files': checked, 'budget': budget, 'labeled_cases': cfg['labeled_cases'], 'fold': 2, 'seed': 2027, 'train_ids': train_ids, 'test_ids': test_ids, 'architecture': 'coarse_unet_mask_prompt_sam_basic_cascade', 'hpq_enabled': False, 'unlabeled_data_used': False, 'test_evaluations_before_final': 0, 'selection_rule': 'fixed_epoch_60_student', 'gt_roi_used': True, 'distance_metric_spacing': [1.0, 1.0, 1.0], 'distance_metric_unit': 'voxel_not_mm', 'empty_single_side_distance_policy': 'historical_binary_case_metrics_see_bound_source', 'undefined_or_failed_surface_distance_sentinel': 373.12866, 'patient_alias_boundary_verified': False, 'unique_patient_count': None, 'case_replacement_enabled': False, 'foreground_threshold_rejection_enabled': False, 'amp_init_scale': 1024.0, 'amp_scale_policy': 'fixed_1024_no_growth; finite_and_single_step_gates_unchanged', 'continuation_state_boundary': 'none_final_student_only_no_optimizer_rng', 'continuation_replay_verified': False}
        if Path('/proc/self/stat').exists():
            manifest['proc_start_ticks'] = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]
        atomic_json(output_dir / 'started.json', manifest)
        guard = importlib.import_module(cfg['resource_guard_module'])
        guard_file = Path(guard.__file__).resolve()
        if str(guard_file) not in checked or checked[str(guard_file)]['kind'] != 'source_sha256':
            raise RuntimeError('Resource guard source is not bound in source_sha256')
        if not callable(getattr(guard, 'guarded_torch_save', None)) or not callable(getattr(guard, 'check_load_capacity', None)):
            raise RuntimeError('Resource guard API is incomplete')
        with guard.original_lock():
            startup_capacity = guard.capacity_snapshot(config=cfg)
        atomic_json(output_dir / 'startup_capacity.json', startup_capacity)
        import numpy as np
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        manifest['threading'] = {'torch_intraop': torch.get_num_threads(), 'torch_interop': torch.get_num_interop_threads(), 'environment': {key: os.environ[key] for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}, 'cpu_affinity_count': len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None, 'policy': 'uniform_engineering_cpu_thread_cap_not_per_arm_tuning'}
        manifest['torch_version'] = str(torch.__version__)
        manifest['checkpoint_load_policy'] = 'inspect_signature_then_single_trusted_load; weights_only_false_only_when_explicitly_supported'
        manifest['optimizer_step_count_policy'] = 'unchanged_bound_AdamW_step_called_once; count_only_after_successful_return'
        atomic_json(output_dir / 'started.json', manifest)
        import torchio as tio
        from torch.cuda import amp
        from torch.utils.data import DataLoader
        from segment_anything.auto_sam_seg import build_auto_sam_seg
        from segment_anything.build_sam3D_cascade import _extract_state_dict
        from utils.auto_sam_loss import build_supervised_loss_fns, cascade_supervised_loss
        from utils.cascade_train_utils import normalize_batch
        from expes.common.seg_metrics import aggregate_metrics, binary_case_metrics, logits_to_binary
        for module_name in ('segment_anything.auto_sam_seg', 'segment_anything.build_sam3D_cascade', 'segment_anything.modeling.sam3D_cascade', 'utils.auto_sam_loss', 'utils.cascade_train_utils', 'expes.common.semi_data', 'expes.common.seg_metrics'):
            module_path = Path(importlib.import_module(module_name).__file__).resolve()
            if str(module_path) not in checked or checked[str(module_path)]['kind'] != 'source_sha256':
                raise RuntimeError(f'Scientific dependency is not SHA-bound: {module_name}')
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Exactly one visible CUDA GPU is required')
        random.seed(2027)
        np.random.seed(2027)
        torch.manual_seed(2027)
        torch.cuda.manual_seed_all(2027)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        device = torch.device('cuda:0')
        student = build_auto_sam_seg(model_type='vit_b_ori', sam_checkpoint=str(resolve_path(cfg, cfg['sam_checkpoint'])), coarse_checkpoint=None, img_size=128, num_classes=2, deep_supervision=False, device=str(device), mc_dropout_p=0.0).to(device)
        student.set_train_phase('e2e')
        if student.gate_mask_prompt or student.mc_dropout_p != 0.0:
            raise RuntimeError('Unexpected basic-cascade gate or MC dropout')
        try:
            initialization_audit = audit_sam_initialization(student.cascade.sam, resolve_path(cfg, cfg['sam_checkpoint']), cfg['sam_checkpoint_sha256'], _extract_state_dict)
        except BaseException as error:
            atomic_json(output_dir / 'initialization_load_audit.json', {'time': now_iso(), 'status': 'ERROR', 'error': str(error), 'training_updates': 0, 'test_evaluations': 0, 'model_reloaded_or_repaired': False})
            raise
        atomic_json(output_dir / 'initialization_load_audit.json', initialization_audit)
        if initialization_audit['status'] != 'PASS':
            raise RuntimeError('Generic SAM initialization did not fully match; see initialization_load_audit.json')
        atomic_json(output_dir / 'initialization.json', {'time': now_iso(), 'seed': 2027, 'sam_checkpoint_sha256': cfg['sam_checkpoint_sha256'], 'full_student_state_sha256': state_dict_sha256(student.state_dict()), 'coarse_state_sha256': state_dict_sha256(student.cascade.coarse_head.state_dict()), 'data_batches_read_before_hash': 0, 'public_checkpoint_loaded': False, 'test_evaluations': 0, 'coarse_checkpoint': None})
        sam = student.cascade.sam
        optimizer = torch.optim.AdamW([{'params': sam.image_encoder.parameters(), 'lr': 0.0001}, {'params': sam.prompt_encoder.parameters(), 'lr': 1e-05}, {'params': sam.mask_decoder.parameters(), 'lr': 1e-05}, {'params': student.cascade.coarse_head.parameters(), 'lr': 0.000125}], betas=(0.9, 0.999), weight_decay=0.1)
        scaler = amp.GradScaler(enabled=True, init_scale=1024.0)
        main_loss, coarse_loss = build_supervised_loss_fns()
        norm_transform = tio.ZNormalization(masking_method=lambda tensor: tensor > 0)
        data_generator = torch.Generator().manual_seed(2027)
        train_dataset = make_strict_dataset(cfg, train_ids, 'train')
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True, drop_last=False, generator=data_generator)
        step_calls = {'count': 0}
        original_optimizer_step = install_optimizer_step_counter(optimizer, step_calls)
        for epoch in range(60):
            student.train()
            epoch_begin = time.monotonic()
            seen = []
            losses = []
            for position, batch in enumerate(train_loader):
                case_id = batch['case_id'][0]
                if case_id in seen or case_id not in train_ids:
                    raise RuntimeError('Duplicate or illegal actual training case')
                tick = time.monotonic()
                image, label = prepare_batch(batch, device, norm_transform, normalize_batch)
                optimizer.zero_grad(set_to_none=True)
                with amp.autocast(enabled=True):
                    logits, coarse_logits = student(image, return_aux=True)
                    loss = cascade_supervised_loss(logits, coarse_logits, label.float(), main_loss, coarse_loss, aux_weight=0.3)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('Non-finite supervised loss; no replay or automatic restart')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nonfinite_grad = any((parameter.grad is not None and (not bool(torch.isfinite(parameter.grad).all())) for parameter in student.parameters()))
                if nonfinite_grad:
                    raise RuntimeError('Non-finite gradients; no skip/replay or automatic restart')
                before_steps = step_calls['count']
                scaler.step(optimizer)
                scaler.update(new_scale=1024.0)
                if step_calls['count'] != before_steps + 1:
                    raise RuntimeError('AMP did not execute exactly one optimizer step')
                successful_updates += 1
                seen.append(case_id)
                losses.append(float(loss.detach().item()))
                append_jsonl(output_dir / 'training_steps.jsonl', {'time': now_iso(), 'epoch': epoch, 'epoch_number': epoch + 1, 'epoch_position': position, 'successful_update': successful_updates, 'case_id': case_id, 'requested_index': int(batch['requested_index'][0]), 'loss': losses[-1], 'foreground_voxels': int(batch['foreground_voxels'][0]), 'origin_adjusted': bool(batch['origin_adjusted'][0]), 'direction_adjusted': bool(batch['direction_adjusted'][0]), 'optimizer_steps': 1, 'skipped': False, 'gradient_finite': True, 'amp_scale': float(scaler.get_scale()), 'elapsed_seconds': time.monotonic() - tick})
                atomic_json(output_dir / 'progress.json', {'time': now_iso(), 'state': 'TRAINING', 'epoch': epoch, 'completed_epochs': completed_epochs, 'successful_updates': successful_updates, 'total_updates': budget['total_updates'], 'last_actual_case_id': case_id, 'test_evaluations': 0})
                del image, label, logits, coarse_logits, loss
            if sorted(seen) != sorted(train_ids) or len(seen) != cfg['train_count']:
                raise RuntimeError('Epoch must visit every declared labeled case exactly once')
            completed_epochs = epoch + 1
            if successful_updates != completed_epochs * cfg['train_count']:
                raise RuntimeError('Complete-epoch update accounting failed')
            append_jsonl(output_dir / 'training_epochs.jsonl', {'time': now_iso(), 'epoch': epoch, 'epoch_number': completed_epochs, 'successful_updates': successful_updates, 'actual_case_ids': seen, 'mean_supervised_loss': sum(losses) / len(losses), 'test_evaluations': 0, 'elapsed_seconds': time.monotonic() - epoch_begin})
            atomic_json(output_dir / 'progress.json', {'time': now_iso(), 'state': 'EPOCH_COMPLETE', 'completed_epochs': completed_epochs, 'successful_updates': successful_updates, 'total_updates': budget['total_updates'], 'test_evaluations': 0, 'checkpoint_policy': 'final_student_only_no_resume'})
        if successful_updates != budget['total_updates'] or completed_epochs != 60:
            raise RuntimeError('Fixed last-epoch budget has not completed')
        final_path = artifact_dir / 'final.pth'
        if final_path.exists():
            raise RuntimeError('Refusing to overwrite final.pth')
        final_payload = {'format': 'public_basic_sup_fixed_final_student_v1', 'student': student.state_dict(), 'completed_epochs': 60, 'epoch': 59, 'successful_updates': successful_updates, 'config_sha256': config_sha256, 'source_manifest': checked, 'selection_rule': 'fixed_epoch_60_student', 'test_evaluations_before_save': 0}
        final_receipt = guard.guarded_torch_save(payload=final_payload, target=final_path, config=cfg, save_kind='final')
        append_jsonl(output_dir / 'save_receipts.jsonl', {'time': now_iso(), 'kind': 'final', 'receipt': final_receipt})
        final_sha256 = sha256_file(final_path)
        optimizer.step = original_optimizer_step
        del original_optimizer_step
        del final_payload, optimizer, scaler, main_loss, coarse_loss, train_loader, train_dataset
        torch.cuda.empty_cache()
        load_receipt = guard.check_load_capacity(target=final_path, config=cfg)
        reloaded = load_trusted_checkpoint(final_path, map_location='cpu')
        if reloaded['completed_epochs'] != 60 or reloaded['successful_updates'] != successful_updates:
            raise RuntimeError('Reloaded final checkpoint budget mismatch')
        student.load_state_dict(reloaded['student'], strict=True)
        del reloaded
        if sha256_file(final_path) != final_sha256:
            raise RuntimeError('Final checkpoint changed during reload')
        atomic_json(output_dir / 'final_reload_verified.json', {'time': now_iso(), 'sha256': final_sha256, 'path': str(final_path), 'load_capacity_receipt': load_receipt, 'strict_load': True})
        evaluation_marker = output_dir / 'test_evaluation_started.json'
        with evaluation_marker.open('x', encoding='utf-8') as handle:
            json.dump({'time': now_iso(), 'final_sha256': final_sha256, 'test_count': len(test_ids)}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        student.eval()
        atomic_json(output_dir / 'progress.json', {'time': now_iso(), 'state': 'FINAL_TEST_ONCE', 'completed_epochs': 60, 'successful_updates': successful_updates, 'total_updates': budget['total_updates'], 'test_evaluations_started': 1, 'final_sha256': final_sha256})
        test_dataset = make_strict_dataset(cfg, test_ids, 'test')
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
        case_metrics = []
        evaluated_ids = []
        with torch.no_grad():
            for batch in test_loader:
                case_id = batch['case_id'][0]
                if case_id != test_ids[len(evaluated_ids)] or case_id in evaluated_ids:
                    raise RuntimeError('Unexpected test actual-case order or duplicate')
                image, label = prepare_batch(batch, device, norm_transform, normalize_batch)
                with amp.autocast(enabled=True):
                    logits = student(image)
                pred = logits_to_binary(logits)
                gt = (label > 0).squeeze(1).detach().cpu().numpy().astype(bool)
                metrics = binary_case_metrics(pred[0], gt[0], spacing=(1.0, 1.0, 1.0))
                if not all((math.isfinite(float(value)) for value in metrics.values())):
                    raise RuntimeError(f'Non-finite final test metric: {case_id}')
                row = {'time': now_iso(), 'actual_case_id': case_id, 'requested_case_id': case_id, 'foreground_voxels': int(batch['foreground_voxels'][0]), 'metrics': metrics, 'predicted_foreground_voxels': int(pred[0].sum()), 'surface_metric_sentinel_used': bool(metrics['hd95'] == 373.12866 or metrics['asd'] == 373.12866), 'final_sha256': final_sha256, 'case_replaced': False}
                append_jsonl(output_dir / 'final_test_cases.jsonl', row)
                case_metrics.append(metrics)
                evaluated_ids.append(case_id)
        if evaluated_ids != test_ids or len(case_metrics) != cfg['test_count']:
            raise RuntimeError('Final test coverage mismatch')
        final_metrics = aggregate_metrics(case_metrics)
        result = {'time': now_iso(), 'status': 'COMPLETE', 'fold': 2, 'labeled_cases': cfg['labeled_cases'], 'seed': 2027, 'completed_epochs': 60, 'successful_updates': successful_updates, 'final_sha256': final_sha256, 'final_checkpoint': str(final_path), 'selection_rule': 'fixed_epoch_60_student', 'test_evaluations': 1, 'test_actual_case_ids': evaluated_ids, 'metrics': final_metrics, 'elapsed_seconds': time.monotonic() - started, 'gt_roi_used': True, 'distance_metric_spacing': [1.0, 1.0, 1.0], 'distance_metric_unit': 'voxel_not_mm', 'undefined_or_failed_surface_distance_sentinel': 373.12866, 'patient_alias_boundary_verified': False, 'unique_patient_count': None, 'no_hpq': True, 'no_unlabeled': True}
        atomic_json(output_dir / 'final_metrics.json', result)
        atomic_json(output_dir / 'progress.json', result)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    except BaseException as error:
        atomic_json(output_dir / 'failure.json', {'time': now_iso(), 'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc(), 'successful_updates': successful_updates, 'completed_epochs': completed_epochs, 'automatic_restart_allowed': False, 'partial_test_present': (output_dir / 'test_evaluation_started.json').exists()})
        raise
    finally:
        run_lock.close()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--gpu', required=True, type=int, choices=(0,))
    parser.add_argument('--gpu-claim-fd', required=True, type=int)
    parser.add_argument('--gpu-claim-device', required=True, type=int)
    parser.add_argument('--gpu-claim-inode', required=True, type=int)
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    if config_sha256 != args.config_sha256:
        raise RuntimeError('Configuration bytes do not match the authorized launcher SHA')
    cfg = json.loads(config_bytes.decode('utf-8'))
    validate_config(cfg)
    gpu_claim = validate_gpu_claim(cfg, args.gpu, args.gpu_claim_fd, args.gpu_claim_device, args.gpu_claim_inode)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    project_root = str(Path(cfg['project_root']).resolve())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        run(cfg, config_sha256, args.gpu, gpu_claim=gpu_claim)
    finally:
        os.close(args.gpu_claim_fd)
if __name__ == '__main__':
    main()
