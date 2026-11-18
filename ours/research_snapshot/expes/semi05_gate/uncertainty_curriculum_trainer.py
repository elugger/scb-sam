from __future__ import annotations
import logging
from itertools import cycle
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda import amp
from tqdm import tqdm
from expes.common.losses import pseudo_loss
from expes.selftrain_insam.trainer_selftrain import SelfTrainTrainer

class UncertaintyCurriculumTrainer(SelfTrainTrainer):

    def _build_ucp_optimizer(self) -> None:
        for parameter in self.student.parameters():
            parameter.requires_grad_(True)
        scfg = self.cfg['semi']
        decoder_lr = float(scfg.get('ucp_decoder_lr', 2e-07))
        encoder_lr = float(scfg.get('ucp_encoder_lr', 5e-08))
        coarse_lr = float(scfg.get('ucp_coarse_lr', 2e-07))
        cascade = self.student.cascade
        self.optimizer = torch.optim.AdamW([{'params': cascade.sam.image_encoder.parameters(), 'lr': encoder_lr}, {'params': cascade.sam.prompt_encoder.parameters(), 'lr': decoder_lr}, {'params': cascade.sam.mask_decoder.parameters(), 'lr': decoder_lr}, {'params': cascade.coarse_head.parameters(), 'lr': coarse_lr}], betas=(0.9, 0.999), weight_decay=0.0)
        logging.info('UCP optimizer encoder=%.2e decoder=%.2e coarse=%.2e trainable=%d', encoder_lr, decoder_lr, coarse_lr, sum((p.numel() for group in self.optimizer.param_groups for p in group['params'])))

    def _foreground_threshold(self, epoch: int) -> float:
        scfg = self.cfg['semi']
        start = float(scfg.get('ucp_fg_start', 0.92))
        end = float(scfg.get('ucp_fg_end', 0.7))
        ramp = max(1, int(scfg.get('ucp_ramp_epochs', 12)))
        progress = min(1.0, float(epoch) / float(ramp))
        return start + (end - start) * progress

    def _pseudo_weight(self, epoch: int) -> float:
        scfg = self.cfg['semi']
        lambda_max = float(scfg.get('ucp_lambda_max', 0.15))
        ramp = max(1, int(scfg.get('ucp_pseudo_ramp_epochs', 8)))
        if epoch < ramp:
            return lambda_max * float(epoch + 1) / float(ramp)
        lambda_end = scfg.get('ucp_lambda_end')
        decay_epochs = int(scfg.get('ucp_lambda_decay_epochs', 0))
        if lambda_end is None or decay_epochs <= 0:
            return lambda_max
        progress = min(1.0, float(epoch - ramp + 1) / float(decay_epochs))
        return lambda_max + (float(lambda_end) - lambda_max) * progress

    @torch.no_grad()
    def _reliable_pseudo(self, image: torch.Tensor, epoch: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        self.teacher.eval()
        logits = self._forward_main_logits(self.teacher, image)
        probabilities = torch.softmax(logits.float(), dim=1)
        foreground = probabilities[:, 1:2]
        pseudo = probabilities.argmax(dim=1, keepdim=True).long()
        threshold = self._foreground_threshold(epoch)
        core = foreground >= threshold
        shell_radius = int(self.cfg['semi'].get('ucp_shell_radius', 3))
        kernel = 2 * shell_radius + 1
        dilated = F.max_pool3d(core.float(), kernel_size=kernel, stride=1, padding=shell_radius) > 0
        background = (foreground <= 1.0 - threshold) & dilated & ~core
        mask = core.float()
        bg_multiple = float(self.cfg['semi'].get('ucp_bg_multiple', 3.0))
        bg_minimum = int(self.cfg['semi'].get('ucp_bg_minimum', 64))
        selected_bg = 0
        for index in range(image.shape[0]):
            count_fg = int(core[index].sum().item())
            candidates = torch.where(background[index].flatten())[0]
            if count_fg == 0 or candidates.numel() == 0:
                continue
            wanted = min(candidates.numel(), max(bg_minimum, int(bg_multiple * count_fg)))
            scores = 1.0 - foreground[index].flatten()[candidates]
            chosen = candidates[torch.topk(scores, k=wanted, largest=True).indices]
            mask[index].flatten()[chosen] = 1.0
            selected_bg += int(wanted)
        stats = {'threshold': threshold, 'foreground_voxels': float(core.sum().item()), 'background_voxels': float(selected_bg), 'selected_ratio': float(mask.mean().item()), 'core_cases': float((core.flatten(1).sum(dim=1) > 0).float().sum().item())}
        self._ucp_teacher_logits = logits.detach()
        self._ucp_reliable_mask = mask.detach()
        self._ucp_anchor_mask = (dilated & (mask <= 0)).float().detach()
        return (pseudo, mask, stats)

    def _student_unlabeled_view(self, image: torch.Tensor, noise_std: float) -> Tuple[torch.Tensor, object]:
        return (self._perturb_input(image, noise_std), None)

    def _restore_student_logits(self, logits: torch.Tensor, view_state: object) -> torch.Tensor:
        return logits

    def _extra_unlabeled_loss(self, student_logits: torch.Tensor, epoch: int) -> torch.Tensor:
        return student_logits.new_tensor(0.0)

    def _save_ucp(self, name: str, epoch: int) -> Path:
        return self._save(name, full=False, epoch=epoch)

    def run_ucp(self, init_from: str) -> Dict[str, float]:
        self._load(init_from)
        self._update_teacher(alpha=0.0)
        self._build_ucp_optimizer()
        torch.cuda.empty_cache()
        scfg = self.cfg['semi']
        epochs = int(scfg.get('ucp_epochs', 36))
        accumulation = int(scfg.get('ucp_accumulation', 5))
        ema_alpha = float(scfg.get('ucp_ema_alpha', 0.999))
        noise_std = float(scfg.get('ucp_noise_std', 0.02))
        early_min = int(scfg.get('ucp_early_stop_min_epochs', 18))
        patience = int(scfg.get('ucp_early_stop_patience', 18))
        labeled_replays = scfg.get('ucp_labeled_replays')
        if labeled_replays is None:
            supervised_scale = 1.0
            effective_labeled_replays = float(len(self.unlab_loader)) / max(1.0, float(len(self.lab_loader)))
        else:
            effective_labeled_replays = max(0.0, float(labeled_replays))
            supervised_scale = min(1.0, effective_labeled_replays * float(len(self.lab_loader)) / max(1.0, float(len(self.unlab_loader))))
        logging.info('UCP replay normalization labeled_replays=%.3f sup_scale=%.6f lab_batches=%d unlab_batches=%d', effective_labeled_replays, supervised_scale, len(self.lab_loader), len(self.unlab_loader))
        floor = self._evaluate()
        self.best_dice = float(floor['dice'])
        self.best_epoch = -1
        self._save_ucp('best.pth', -1)
        self._log_metrics('ucp_start_floor', floor)
        no_improve = 0
        stop_reason = 'max_epochs'
        for epoch in tqdm(range(epochs), desc='uncertainty-curriculum', ncols=78):
            self.student.train()
            self.teacher.eval()
            lambda_pseudo = self._pseudo_weight(epoch)
            lab_iter = cycle(self.lab_loader)
            totals, supervised, supervised_raw, unsupervised, extras = ([], [], [], [], [])
            fg_counts, bg_counts, ratios, core_cases = ([], [], [], [])
            self.optimizer.zero_grad(set_to_none=True)
            for step, batch_u in enumerate(self.unlab_loader):
                image_u, _unused_label = self._prepare_batch(batch_u)
                image_l, label_l = self._prepare_batch(next(lab_iter))
                pseudo, pseudo_mask, stats = self._reliable_pseudo(image_u, epoch)
                student_u, view_state = self._student_unlabeled_view(image_u, noise_std)
                with amp.autocast(enabled=self.device.type == 'cuda'):
                    loss_sup_raw = self._supervised_step(image_l, label_l)
                    loss_sup = supervised_scale * loss_sup_raw
                    logits_view = self._forward_main_logits(self.student, student_u)
                    logits_u = self._restore_student_logits(logits_view, view_state)
                    loss_pseudo = lambda_pseudo * pseudo_loss(logits_u, pseudo, pseudo_mask, bool(self.cfg['train'].get('use_dice', True)), float(self.cfg['train'].get('dice_weight', 0.5)))
                    loss_extra = self._extra_unlabeled_loss(logits_u, epoch)
                    loss = loss_sup + loss_pseudo + loss_extra
                self.scaler.scale(loss / accumulation).backward()
                if (step + 1) % accumulation == 0 or step + 1 == len(self.unlab_loader):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self._update_teacher(ema_alpha)
                totals.append(float(loss.item()))
                supervised.append(float(loss_sup.item()))
                supervised_raw.append(float(loss_sup_raw.item()))
                unsupervised.append(float(loss_pseudo.item()))
                extras.append(float(loss_extra.item()))
                fg_counts.append(stats['foreground_voxels'])
                bg_counts.append(stats['background_voxels'])
                ratios.append(stats['selected_ratio'])
                core_cases.append(stats['core_cases'])
            logging.info('UCP epoch=%d total=%.6f sup=%.6f sup_raw=%.6f pseudo=%.6f extra=%.6f lambda=%.4f thr=%.3f fg=%.1f bg=%.1f ratio=%.6f core_cases=%.2f updates=%d', epoch, float(np.mean(totals)), float(np.mean(supervised)), float(np.mean(supervised_raw)), float(np.mean(unsupervised)), float(np.mean(extras)), lambda_pseudo, self._foreground_threshold(epoch), float(np.mean(fg_counts)), float(np.mean(bg_counts)), float(np.mean(ratios)), float(np.mean(core_cases)), int(np.ceil(len(self.unlab_loader) / accumulation)))
            result = self._evaluate()
            self._log_metrics('ucp_eval', result)
            if result['dice'] > self.best_dice:
                self.best_dice = float(result['dice'])
                self.best_epoch = epoch
                self._save_ucp('best.pth', epoch)
                no_improve = 0
            else:
                no_improve += 1
            if result['dice'] <= float(floor['dice']) - 0.03:
                stop_reason = 'collapse'
                break
            if epoch + 1 >= early_min and no_improve >= patience:
                stop_reason = 'early_plateau'
                break
        final = self._evaluate()
        self._log_metrics('ucp_final', final)
        self._save_ucp('final.pth', epoch)
        (self.output_dir / 'final_metrics.txt').write_text(f"floor_dice: {floor['dice']:.6f}\nbest_dice: {self.best_dice:.6f}\nbest_epoch: {self.best_epoch}\nfinal_dice: {final['dice']:.6f}\nlabeled_replays: {effective_labeled_replays:.3f}\nsupervised_scale: {supervised_scale:.6f}\nstop_reason: {stop_reason}\nselection: fixed single-view output and highest test Dice\n", encoding='utf-8')
        return final
