from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from expes.common.losses import pseudo_loss
from expes.semi05_gate.weak_strong_view_ucp_trainer import WeakStrongViewUCPTrainer
from utils.auto_sam_loss import cascade_supervised_loss

class FinalLogitContextualResidualHead(nn.Module):

    def __init__(self, coarse_channels: int, sam_channels: int, projection_channels: int=8, hidden_channels: int=16, delta_max: float=0.5) -> None:
        super().__init__()
        if min(coarse_channels, sam_channels, projection_channels, hidden_channels) <= 0:
            raise ValueError('FCLR channel counts must be positive')
        if delta_max <= 0.0:
            raise ValueError('fclr_delta_max must be positive')
        projection_groups = 4 if projection_channels % 4 == 0 else 1
        hidden_groups = 4 if hidden_channels % 4 == 0 else 1
        self.coarse_projection = nn.Sequential(nn.Conv3d(coarse_channels, projection_channels, 1, bias=False), nn.GroupNorm(projection_groups, projection_channels), nn.GELU())
        self.sam_projection = nn.Sequential(nn.Conv3d(sam_channels, projection_channels, 1, bias=False), nn.GroupNorm(projection_groups, projection_channels), nn.GELU())
        self.mix = nn.Sequential(nn.Conv3d(2 * projection_channels + 1, hidden_channels, 3, padding=1, bias=False), nn.GroupNorm(hidden_groups, hidden_channels), nn.GELU())
        self.out = nn.Conv3d(hidden_channels, 1, 1, bias=True)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.delta_max = float(delta_max)

    def forward(self, coarse_feature: torch.Tensor, sam_embedding: torch.Tensor, low_res_logit: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if low_res_logit.ndim != 5 or low_res_logit.shape[1] != 1:
            raise ValueError(f'low_res_logit must have shape [B,1,D,H,W], got {tuple(low_res_logit.shape)}')
        target_size = tuple(low_res_logit.shape[-3:])
        coarse = self.coarse_projection(coarse_feature.detach())
        sam = self.sam_projection(sam_embedding.detach())
        if tuple(coarse.shape[-3:]) != target_size:
            coarse = F.interpolate(coarse, target_size, mode='trilinear', align_corners=False)
        if tuple(sam.shape[-3:]) != target_size:
            sam = F.interpolate(sam, target_size, mode='trilinear', align_corners=False)
        conditioning = torch.cat([coarse, sam, low_res_logit.detach().float()], dim=1)
        delta = self.delta_max * torch.tanh(self.out(self.mix(conditioning)))
        return (low_res_logit.float() + delta.float(), delta.float())

class FinalLogitContextualResidualWSVUCPTrainer(WeakStrongViewUCPTrainer):
    _COARSE_DECODER_STAGE = 2

    def __init__(self, cfg) -> None:
        self._fclr_stats: List[Dict[str, float]] = []
        self._fclr_grad_stats: Dict[str, List[float]] = {}
        self._fclr_clip_norms: List[float] = []
        self._fclr_identity_checked = False
        self._fclr_gradient_checked = False
        self._fclr_loaded_head = False
        self._fclr_last_student_delta: torch.Tensor | None = None
        super().__init__(cfg)
        scfg = self.cfg['semi']
        coarse_channels = int(scfg.get('fclr_coarse_channels', 128))
        sam_channels = int(self.student.cascade.prompt_encoder.embed_dim)
        projection_channels = int(scfg.get('fclr_projection_channels', 8))
        hidden_channels = int(scfg.get('fclr_hidden_channels', 16))
        delta_max = float(scfg.get('fclr_delta_max', 0.5))
        kwargs = dict(coarse_channels=coarse_channels, sam_channels=sam_channels, projection_channels=projection_channels, hidden_channels=hidden_channels, delta_max=delta_max)
        self.student_fclr_head = FinalLogitContextualResidualHead(**kwargs).to(self.device)
        self.teacher_fclr_head = FinalLogitContextualResidualHead(**kwargs).to(self.device)
        self.teacher_fclr_head.requires_grad_(False)
        self.teacher_fclr_head.eval()
        self._update_teacher_fclr(alpha=0.0)
        self._register_fclr_gradient_hooks()
        logging.info('FCLR head coarse=%d sam=%d projection=%d hidden=%d delta_max=%.4f parameters=%d', coarse_channels, sam_channels, projection_channels, hidden_channels, delta_max, sum((p.numel() for p in self.student_fclr_head.parameters())))

    def _register_fclr_gradient_hooks(self) -> None:
        watched = {'coarse_projection': self.student_fclr_head.coarse_projection[0].weight, 'sam_projection': self.student_fclr_head.sam_projection[0].weight, 'mix': self.student_fclr_head.mix[0].weight, 'out': self.student_fclr_head.out.weight}
        self._fclr_grad_stats = {name: [] for name in watched}
        for name, parameter in watched.items():

            def record(gradient: torch.Tensor, key: str=name) -> torch.Tensor:
                values = self._fclr_grad_stats[key]
                if len(values) < 16:
                    scale = 1.0
                    if hasattr(self, 'scaler') and self.scaler.is_enabled():
                        scale = max(1.0, float(self.scaler.get_scale()))
                    values.append(float(gradient.detach().float().norm().item()) / scale)
                return gradient
            parameter.register_hook(record)

    @torch.no_grad()
    def _update_teacher_fclr(self, alpha: float) -> None:
        for teacher, student in zip(self.teacher_fclr_head.parameters(), self.student_fclr_head.parameters()):
            teacher.data.mul_(alpha).add_(student.data, alpha=1.0 - alpha)
        for teacher, student in zip(self.teacher_fclr_head.buffers(), self.student_fclr_head.buffers()):
            teacher.copy_(student)

    def _update_teacher(self, alpha: float) -> None:
        super()._update_teacher(alpha)
        if hasattr(self, 'student_fclr_head'):
            self._update_teacher_fclr(alpha)

    def _build_ucp_optimizer(self) -> None:
        super()._build_ucp_optimizer()
        self.optimizer.add_param_group({'params': self.student_fclr_head.parameters(), 'lr': float(self.cfg['semi'].get('fclr_head_lr', 0.0001)), 'weight_decay': 0.0})
        ids = [id(p) for group in self.optimizer.param_groups for p in group['params']]
        head_ids = {id(p) for p in self.student_fclr_head.parameters()}
        if len(ids) != len(set(ids)) or not head_ids.issubset(set(ids)):
            raise RuntimeError('FCLR optimizer parameter registration failed')
        if any((p.requires_grad for p in self.teacher_fclr_head.parameters())):
            raise RuntimeError('EMA FCLR head must not require gradients')

        def clip_head(_optimizer, _args, _kwargs):
            norm = torch.nn.utils.clip_grad_norm_(self.student_fclr_head.parameters(), 1.0)
            self._fclr_clip_norms.append(float(norm.detach().float().item()))
        if hasattr(self.optimizer, 'register_step_pre_hook'):
            self._fclr_step_hook = self.optimizer.register_step_pre_hook(clip_head)
        else:
            original_step = self.optimizer.step

            def clipped_step(*args, **kwargs):
                clip_head(self.optimizer, args, kwargs)
                return original_step(*args, **kwargs)
            self._fclr_original_optimizer_step = original_step
            self.optimizer.step = clipped_step
        logging.info('FCLR optimizer_check=pass head_lr=%.2e head_parameters=%d total_trainable=%d unique_parameters=%d', float(self.cfg['semi'].get('fclr_head_lr', 0.0001)), sum((p.numel() for p in self.student_fclr_head.parameters())), sum((p.numel() for group in self.optimizer.param_groups for p in group['params'])), len(ids))

    def _coarse_forward_with_feature(self, model, image: torch.Tensor):
        network = model.cascade.coarse_head.network
        stage = network.decoder.stages[self._COARSE_DECODER_STAGE]
        captured: Dict[str, torch.Tensor] = {}

        def capture(_module, _inputs, output) -> None:
            if not torch.is_tensor(output):
                raise TypeError('coarse decoder stage 2 must return a tensor')
            captured['feature'] = output
        handle = stage.register_forward_hook(capture)
        try:
            prompt, coarse_logits = model.cascade.coarse_head(image, return_logits=True)
        finally:
            handle.remove()
        if 'feature' not in captured:
            raise RuntimeError('FCLR coarse stage-2 hook did not execute')
        feature = captured['feature']
        expected = int(self.cfg['semi'].get('fclr_coarse_channels', 128))
        if feature.shape[1] != expected:
            raise RuntimeError(f'FCLR expected {expected} coarse channels, got {feature.shape[1]}')
        return (prompt, coarse_logits, feature)

    def _forward_with_fclr(self, model, head, image: torch.Tensor):
        if bool(getattr(model, 'gate_mask_prompt', False)):
            raise RuntimeError('FCLR expects the locked native continuous prompt')
        head.train(model.training and model is self.student)
        prompt, coarse_logits, coarse_feature = self._coarse_forward_with_feature(model, image)
        cascade = model.cascade
        image_embedding = cascade.image_encoder(image)
        sparse, dense = cascade.prompt_encoder(points=None, boxes=None, masks=prompt)
        low_res_mask, _iou = cascade.mask_decoder(image_embeddings=image_embedding, image_pe=cascade.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=False)
        refined_low_res, delta = head(coarse_feature, image_embedding, low_res_mask)
        base_full = F.interpolate(low_res_mask, size=image.shape[-3:], mode='trilinear', align_corners=False)
        refined_full = F.interpolate(refined_low_res.to(low_res_mask.dtype), size=image.shape[-3:], mode='trilinear', align_corners=False)
        base_logits = model._fg_to_logits(base_full)
        logits = model._fg_to_logits(refined_full)
        if model is self.student and model.training:
            self._fclr_last_student_delta = delta
            self._fclr_stats.append({'delta_mean': float(delta.detach().abs().mean().item()), 'delta_max': float(delta.detach().abs().max().item()), 'delta_active': float((delta.detach().abs() >= 0.001).float().mean().item()), 'base_logit_mean': float(low_res_mask.detach().float().abs().mean().item())})
        return (logits, coarse_logits, base_logits, delta)

    def _forward_main_logits(self, model, image: torch.Tensor) -> torch.Tensor:
        if model is self.student:
            head = self.student_fclr_head
        elif model is self.teacher:
            head = self.teacher_fclr_head
        else:
            raise ValueError('FCLR forward received an unknown model')
        with amp.autocast(enabled=self.device.type == 'cuda'):
            logits, *_ = self._forward_with_fclr(model, head, image)
        return logits

    @staticmethod
    def _soft_dice(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(1, probability.ndim))
        intersection = (probability * target).sum(dim=dims)
        denominator = probability.sum(dim=dims) + target.sum(dim=dims)
        return ((2.0 * intersection + 1e-06) / (denominator + 1e-06)).mean()

    def _supervised_step(self, image: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        with amp.autocast(enabled=self.device.type == 'cuda'):
            logits, coarse_logits, base_logits, delta = self._forward_with_fclr(self.student, self.student_fclr_head, image)
            segmentation = cascade_supervised_loss(logits, coarse_logits, label.float(), self.main_loss_fn, self.coarse_loss_fn, aux_weight=float(self.cfg['train'].get('aux_weight', 0.3)))
            sparse = delta.float().abs().mean()
            total = segmentation + float(self.cfg['semi'].get('fclr_sparse_weight', 0.001)) * sparse
        if self._fclr_stats:
            with torch.no_grad():
                target = (label > 0).float()
                base_dice = self._soft_dice(torch.softmax(base_logits.float(), 1)[:, 1:2], target)
                refined_dice = self._soft_dice(torch.softmax(logits.float(), 1)[:, 1:2], target)
            self._fclr_stats[-1].update({'segmentation_loss': float(segmentation.detach().item()), 'sparse_loss': float(sparse.detach().item()), 'base_dice': float(base_dice.item()), 'refined_dice': float(refined_dice.item()), 'dice_delta': float((refined_dice - base_dice).item())})
        return total

    def _extra_unlabeled_loss(self, student_logits: torch.Tensor, epoch: int) -> torch.Tensor:
        view_loss = super()._extra_unlabeled_loss(student_logits, epoch)
        if self._fclr_last_student_delta is None:
            return view_loss
        sparse = self._fclr_last_student_delta.float().abs().mean()
        return view_loss + float(self.cfg['semi'].get('fclr_sparse_weight', 0.001)) * sparse

    @torch.no_grad()
    def _identity_check(self) -> None:
        if self._fclr_identity_checked or self._fclr_loaded_head:
            return
        batch = next(iter(self.unlab_loader))
        image, _ = self._prepare_batch(batch)
        self.student.eval()
        self.student_fclr_head.eval()
        with amp.autocast(enabled=self.device.type == 'cuda'):
            base = self.student(image)
            refined, _coarse, _base_logits, delta = self._forward_with_fclr(self.student, self.student_fclr_head, image)
        difference = float((refined.float() - base.float()).abs().max().item())
        delta_max = float(delta.float().abs().max().item())
        mask_equal = bool(torch.equal(refined.argmax(1), base.argmax(1)))
        logging.info('FCLR identity delta_max=%.9f logit_max_diff=%.9f mask_equal=%s', delta_max, difference, mask_equal)
        if delta_max != 0.0 or difference > 1e-05 or (not mask_equal):
            raise RuntimeError('FCLR zero-initialized head failed identity check')
        self._fclr_identity_checked = True

    def _flush_fclr_stats(self) -> None:
        if not self._fclr_stats and (not any(self._fclr_grad_stats.values())):
            return

        def mean(key: str) -> float:
            values = [row[key] for row in self._fclr_stats if key in row]
            return float(np.mean(values)) if values else float('nan')

        def maximum(key: str) -> float:
            values = [row[key] for row in self._fclr_stats if key in row]
            return float(np.max(values)) if values else float('nan')
        if not self._fclr_gradient_checked:
            missing = [key for key, values in self._fclr_grad_stats.items() if not values or not np.isfinite(max(values)) or max(values) <= 0.0]
            if missing:
                raise RuntimeError('FCLR gradient check failed: ' + ', '.join(missing))
            self._fclr_gradient_checked = True
            logging.info('FCLR gradient_check=pass')
        with torch.no_grad():
            distances = [(s.detach().float() - t.detach().float()).abs().mean() for s, t in zip(self.student_fclr_head.parameters(), self.teacher_fclr_head.parameters())]
            ema_distance = float(torch.stack(distances).mean().item())
        logging.info('FCLR train forwards=%d seg=%.6f sparse=%.8f delta_mean=%.9f delta_max=%.9f active=%.6f base_dice=%.6f refined_dice=%.6f dice_delta=%+.6f grad_out=%.6e grad_mix=%.6e grad_coarse=%.6e grad_sam=%.6e clip_mean=%.6e clip_max=%.6e ema_head_distance=%.9e', len(self._fclr_stats), mean('segmentation_loss'), mean('sparse_loss'), mean('delta_mean'), maximum('delta_max'), mean('delta_active'), mean('base_dice'), mean('refined_dice'), mean('dice_delta'), float(np.mean(self._fclr_grad_stats['out'])), float(np.mean(self._fclr_grad_stats['mix'])), float(np.mean(self._fclr_grad_stats['coarse_projection'])), float(np.mean(self._fclr_grad_stats['sam_projection'])), float(np.mean(self._fclr_clip_norms)) if self._fclr_clip_norms else float('nan'), float(np.max(self._fclr_clip_norms)) if self._fclr_clip_norms else float('nan'), ema_distance)
        self._fclr_stats.clear()
        for values in self._fclr_grad_stats.values():
            values.clear()
        self._fclr_clip_norms.clear()

    def _evaluate(self) -> Dict[str, float]:
        self._identity_check()
        self._flush_fclr_stats()
        self.student_fclr_head.eval()
        return super()._evaluate()

    def _load(self, path: str | Path) -> None:
        state = torch.load(str(path), map_location=self.device)
        payload = state if isinstance(state, dict) else {}
        if isinstance(state, dict) and 'student' in state:
            student_state = state['student']
        elif isinstance(state, dict) and 'model_state_dict' in state:
            student_state = state['model_state_dict']
        elif isinstance(state, dict) and 'cascade_state_dict' in state:
            student_state = state['cascade_state_dict']
        else:
            student_state = state
        self.student.load_state_dict(student_state)
        if 'teacher' in payload:
            self.teacher.load_state_dict(payload['teacher'])
        else:
            for teacher, student in zip(self.teacher.parameters(), self.student.parameters()):
                teacher.data.copy_(student.data)
        self._fclr_loaded_head = 'student_fclr_head' in payload
        if self._fclr_loaded_head:
            self.student_fclr_head.load_state_dict(payload['student_fclr_head'])
            if 'teacher_fclr_head' in payload:
                self.teacher_fclr_head.load_state_dict(payload['teacher_fclr_head'])
            else:
                self._update_teacher_fclr(0.0)
        else:
            self._update_teacher_fclr(0.0)
        logging.info('FCLR load path=%s head_present=%s', str(path), self._fclr_loaded_head)

    @staticmethod
    def _compact_state(module: nn.Module) -> Dict[str, torch.Tensor]:
        return {key: value.detach().cpu() for key, value in module.state_dict().items()}

    def _save_ucp(self, name: str, epoch: int) -> Path:
        path = self.output_dir / name
        torch.save({'student': self._compact_state(self.student), 'student_fclr_head': self._compact_state(self.student_fclr_head), 'teacher_fclr_head': self._compact_state(self.teacher_fclr_head), 'fclr_format_version': 1, 'best_dice': self.best_dice, 'best_epoch': epoch, 'cfg': self.cfg}, str(path))
        return path

    def run_fclr(self, init_from: str):
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        return self.run_ucp(init_from)

    def run_preflight(self, init_from: str) -> Dict[str, float]:
        self._load(init_from)
        self._update_teacher(alpha=0.0)
        self._build_ucp_optimizer()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
        self._identity_check()
        batch_l = next(iter(self.lab_loader))
        batch_u = next(iter(self.unlab_loader))
        image_l, label_l = self._prepare_batch(batch_l)
        image_u, _ = self._prepare_batch(batch_u)
        losses: List[float] = []
        for step in range(2):
            self.student.train()
            self.student_fclr_head.train()
            self.teacher.eval()
            self.teacher_fclr_head.eval()
            self.optimizer.zero_grad(set_to_none=True)
            pseudo, pseudo_mask, _ = self._reliable_pseudo(image_u, epoch=0)
            student_u, view_state = self._student_unlabeled_view(image_u, float(self.cfg['semi'].get('ucp_noise_std', 0.05)))
            with amp.autocast(enabled=self.device.type == 'cuda'):
                loss_sup = self._supervised_step(image_l, label_l)
                logits_view = self._forward_main_logits(self.student, student_u)
                logits_u = self._restore_student_logits(logits_view, view_state)
                loss_pseudo = self._pseudo_weight(0) * pseudo_loss(logits_u, pseudo, pseudo_mask, bool(self.cfg['train'].get('use_dice', True)), float(self.cfg['train'].get('dice_weight', 0.5)))
                loss = loss_sup + loss_pseudo + self._extra_unlabeled_loss(logits_u, 0)
            if not bool(torch.isfinite(loss.detach()).item()):
                raise RuntimeError('FCLR preflight produced non-finite loss')
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self._update_teacher(float(self.cfg['semi'].get('ucp_ema_alpha', 0.999)))
            losses.append(float(loss.detach().item()))
        self._flush_fclr_stats()
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            peak_gb = torch.cuda.max_memory_allocated(self.device) / float(1024 ** 3)
        else:
            peak_gb = 0.0
        logging.info('FCLR preflight=pass losses=%s peak_gb=%.3f', ','.join((f'{value:.6f}' for value in losses)), peak_gb)
        return {'loss_step0': losses[0], 'loss_step1': losses[1], 'peak_gb': peak_gb}
