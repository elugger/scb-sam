"""Isolated C0/C1 WSV trainer. Keeps legacy ROI, losses and test selection.

Checkpoints are inference checkpoints, not advertised as exact-resume states.
Every formal run starts independently from the matched original Semi weights.
"""
from __future__ import annotations
import json
import logging
import math
import os
from itertools import cycle
from pathlib import Path
import time

import numpy as np
from scipy import ndimage
import torch
from torch.cuda import amp

from expes.common.losses import pseudo_loss
from expes.common.seg_metrics import aggregate_metrics, binary_case_metrics, logits_to_binary
from expes.common.semi_data import CaseListDataset, build_eval_transform
from expes.semi05_gate.weak_strong_view_ucp_trainer import WeakStrongViewUCPTrainer
from expes.semi05_gate.paired_rng_data_20260908 import (
    labeled_tokens, paired_loader, scoped_rng, stream_seed, tensor_digest, SeededCases,
)


class PairedWSVControl(WeakStrongViewUCPTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.pair_seed = int(cfg["experiment"].get("seed", 2026))
        self.refresh = bool(cfg["semi"].get("paired_refresh_labels", False))
        self.workers = int(cfg["semi"].get("paired_workers", 2))
        self.eval_epoch = -1
        self.checkpoint_dir = Path(cfg["semi"]["paired_checkpoint_dir"]).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _append(self, filename, row):
        with (self.output_dir / filename).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    def _log_metrics(self, tag, metrics):
        logging.info("%s dice=%.8f jaccard=%.8f hd95=%.8f asd=%.8f", tag,
                     metrics["dice"], metrics["jaccard"], metrics["hd95"], metrics["asd"])
        self._append("metrics.jsonl", dict(tag=tag, epoch=self.eval_epoch, **metrics))

    @torch.no_grad()
    def _evaluate(self):
        # Same test dataset, transform, binarization, metrics and spacing as base.
        self.student.eval()
        cases = []
        for index, batch in enumerate(self.test_loader):
            image, label = self._prepare_batch(batch)
            logits = self._forward_main_logits(self.student, image)
            pred = logits_to_binary(logits)
            gt = (label > 0).squeeze(1).detach().cpu().numpy().astype(bool)
            for i in range(pred.shape[0]):
                values = binary_case_metrics(pred[i], gt[i], spacing=(1., 1., 1.))
                cases.append(values)
                self._append("test_cases.jsonl", dict(epoch=self.eval_epoch,
                             requested_case=Path(self.test_set.image_paths[index]).name, **values))
        return aggregate_metrics(cases)

    def _save_ucp(self, name, epoch):
        target = self.checkpoint_dir / name
        temp = self.checkpoint_dir / (name + ".partial")
        torch.save({"student": self.student.state_dict(), "best_dice": self.best_dice,
                    "best_epoch": self.best_epoch, "saved_epoch": epoch, "cfg": self.cfg,
                    "exact_resume_supported": False}, temp)
        os.replace(temp, target)
        link = self.output_dir / name
        if link.is_symlink():
            if link.resolve() != target:
                raise RuntimeError("Unexpected checkpoint link target")
        elif link.exists():
            raise RuntimeError("Refusing to overwrite an existing checkpoint file")
        else:
            link.symlink_to(target)
        return target

    def _loaders(self, epoch):
        steps = len(self.unlab_set)
        lab_tokens = labeled_tokens(len(self.lab_set), steps, epoch, self.pair_seed, self.refresh)
        lab = paired_loader(self.lab_set, lab_tokens, self.pair_seed, "lab", self.workers)
        unlab_tokens = [(epoch, 0, i) for i in range(steps)]
        unlab = paired_loader(self.unlab_set, unlab_tokens, self.pair_seed, "unlab", self.workers)
        return iter(lab) if self.refresh else cycle(lab), unlab

    def _losses(self, image_l, label_l, image_u, epoch, step):
        pseudo, mask, stats = self._reliable_pseudo(image_u, epoch)
        with scoped_rng(stream_seed(self.pair_seed, "student-step", epoch, step), cuda=True):
            view, state = self._student_unlabeled_view(image_u, 0.05)
            with amp.autocast(enabled=True):
                sup = self._supervised_step(image_l, label_l)
                logits = self._restore_student_logits(self._forward_main_logits(self.student, view), state)
                ploss = self._pseudo_weight(epoch) * pseudo_loss(
                    logits, pseudo, mask, bool(self.cfg["train"].get("use_dice", True)),
                    float(self.cfg["train"].get("dice_weight", 0.5)))
                extra = self._extra_unlabeled_loss(logits, epoch)
        return sup + ploss + extra, sup, ploss, extra, logits, stats, state

    @torch.no_grad()
    def _labeled_error_audit(self):
        names = [Path(path).name.removesuffix(".nii.gz") for path in self.lab_set.image_paths]
        ds = CaseListDataset(self.lab_set.fold_cv_dir, names, build_eval_transform(128),
                             self.lab_set.threshold, split="train")
        self.student.eval()
        records = []
        for index in range(len(ds)):
            with scoped_rng(stream_seed(self.pair_seed, "label-audit", index)):
                item = ds[index]
            image, label = self._prepare_batch({k: v.unsqueeze(0) for k, v in item.items()})
            with amp.autocast(enabled=True):
                main, coarse = self.student(image, return_aux=True)
            pred = logits_to_binary(main)[0]
            cpred = logits_to_binary(coarse)[0]
            gt = (label[0, 0] > 0).cpu().numpy()
            fn = gt & ~pred
            fp = ~gt & pred
            tp = int((gt & pred).sum())
            components, count = ndimage.label(gt)
            missing = sum(not pred[components == k].any() for k in range(1, count+1))
            row = dict(requested_case=names[index], foreground_voxels=int(gt.sum()),
                       dice=float(2*tp/max(1, int(gt.sum())+int(pred.sum()))),
                       coarse_dice=float(2*(gt & cpred).sum()/max(1, int(gt.sum())+int(cpred.sum()))),
                       precision=float(tp/max(1, int(pred.sum()))), recall=float(tp/max(1, int(gt.sum()))),
                       fn=int(fn.sum()), fp=int(fp.sum()),
                       fn_inside_3vox=int((fn & (ndimage.distance_transform_edt(gt) <= 3)).sum()),
                       wholly_missed_components=int(missing), total_components=int(count),
                       scope="training-labeled diagnostic, not independent validation")
            records.append(row)
        (self.output_dir / "labeled_error_audit.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        return records

    def preflight(self, init_from, expected_floor):
        self._load(init_from)
        self._update_teacher(0.0)
        self._build_ucp_optimizer()
        floor = self._evaluate()
        self._log_metrics("preflight_floor", floor)
        reference = WeakStrongViewUCPTrainer._evaluate(self)
        evaluation_difference = abs(reference["dice"] - floor["dice"])
        if evaluation_difference > 1e-8:
            raise RuntimeError("Paired evaluator differs from the unmodified inherited evaluator")
        # Historical six-digit records are not a bitwise reproducibility oracle.
        # Require exact same-process agreement above and log the historical delta.
        if abs(floor["dice"] - expected_floor) > 0.0001:
            raise RuntimeError(f"Inherited floor mismatch: {floor['dice']} vs {expected_floor}")
        data = SeededCases(self.lab_set, self.pair_seed, "lab")
        a, a2 = data[(0, 0, 0)], data[(0, 0, 0)]
        assert torch.equal(a["image"], a2["image"]) and torch.equal(a["label"], a2["label"])
        replay_digests = [tensor_digest(data[(0, replay, 0)]["image"]) for replay in range(6)]
        if len(set(replay_digests)) < 2:
            raise RuntimeError("No actual transform diversity in six refreshed draws")
        cases = self._labeled_error_audit()
        lab, unlab = self._loaders(0)
        item_l, item_u = next(lab), next(iter(unlab))
        image_l, label_l = self._prepare_batch(item_l)
        image_u, _ = self._prepare_batch(item_u)
        with scoped_rng(stream_seed(self.pair_seed, "view-test"), cuda=True):
            view_a, state_a = self._student_unlabeled_view(image_u, .05)
        torch.randn(1000, device=self.device)  # Simulate unrelated new-head RNG consumption.
        with scoped_rng(stream_seed(self.pair_seed, "view-test"), cuda=True):
            view_b, state_b = self._student_unlabeled_view(image_u, .05)
        assert torch.equal(view_a, view_b) and state_a == state_b
        del view_a, view_b
        torch.cuda.reset_peak_memory_stats()
        self.student.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss, sup, ploss, extra, logits, stats, _ = self._losses(image_l, label_l, image_u, 0, 0)
        if not torch.isfinite(loss).item() or stats["foreground_voxels"] <= 0:
            raise RuntimeError("Nonfinite loss or empty foreground curriculum")
        extra_gradient = torch.autograd.grad(extra, logits, retain_graph=True)[0]
        grad_norm = float(extra_gradient.float().norm())
        assert grad_norm > 0 and torch.isfinite(extra_gradient).all()
        self.scaler.scale(loss).backward()
        gradients = [p.grad for p in self.student.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        result = dict(status="PASS", floor=floor, repeated_label_tensor_equal=True,
                      unique_refreshed_label_views=len(set(replay_digests)),
                      unrelated_rng_does_not_change_student_view=True,
                      loss=float(loss), supervised=float(sup), pseudo=float(ploss), view=float(extra),
                      view_logit_gradient_norm=grad_norm, gradient_tensors=len(gradients),
                      nonzero_gradient_tensors=sum(bool(g.abs().sum() > 0) for g in gradients),
                      inherited_evaluator_dice=reference["dice"], evaluation_difference=evaluation_difference,
                      historical_floor=expected_floor, historical_floor_delta_pp=100*(floor["dice"]-expected_floor),
                      peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,
                      labeled_audit_cases=len(cases), roi_protocol="inherited label-localized ROI, unchanged",
                      exact_resume_supported=False, **stats)
        (self.output_dir / "preflight.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        logging.info("PAIRED_PREFLIGHT %s", json.dumps(result))
        return result

    def run_paired(self, init_from, expected_floor):
        self._load(init_from)
        self._update_teacher(0.0)
        self._build_ucp_optimizer()
        floor = self._evaluate()
        if abs(floor["dice"] - expected_floor) > .0001:
            raise RuntimeError("Formal initial floor did not match inherited protocol")
        self.best_dice, self.best_epoch = float(floor["dice"]), -1
        self._log_metrics("paired_floor", floor)
        self._save_ucp("best.pth", -1)
        epochs = int(self.cfg["semi"]["ucp_epochs"])
        accumulation = int(self.cfg["semi"]["ucp_accumulation"])
        stop_reason = "max_epochs"
        for epoch in range(epochs):
            self.eval_epoch = epoch
            start = time.monotonic()
            self.student.train(); self.teacher.eval()
            lab, unlab = self._loaders(epoch)
            self.optimizer.zero_grad(set_to_none=True)
            stats_rows = []
            updates = 0
            for step, batch_u in enumerate(unlab):
                batch_l = next(lab)
                image_l, label_l = self._prepare_batch(batch_l)
                image_u, _unused_label = self._prepare_batch(batch_u)
                loss, sup, ploss, extra, logits, stats, state = self._losses(image_l, label_l, image_u, epoch, step)
                if not torch.isfinite(loss).item():
                    raise RuntimeError("Nonfinite training loss")
                self.scaler.scale(loss/accumulation).backward()
                if (step+1) % accumulation == 0 or step+1 == len(unlab):
                    scale = self.scaler.get_scale()
                    self.scaler.step(self.optimizer); self.scaler.update()
                    if self.scaler.get_scale() < scale:
                        raise RuntimeError("AMP skipped optimizer step; paired update budget invalid")
                    self.optimizer.zero_grad(set_to_none=True)
                    self._update_teacher(.999)
                    updates += 1
                stats_rows.append([float(loss.detach()), float(sup.detach()), float(ploss.detach()),
                                   float(extra.detach()), stats["selected_ratio"]])
                trace = dict(epoch=epoch, step=step, labeled_index=int(batch_l["requested_index"]),
                             label_seed=int(batch_l["augmentation_seed"]),
                             unlabeled_index=int(batch_u["requested_index"]),
                             unlabeled_seed=int(batch_u["augmentation_seed"]),
                             student_seed=stream_seed(self.pair_seed, "student-step", epoch, step), flips=state)
                if epoch == 0:
                    trace.update(label_image_sha256=tensor_digest(batch_l["image"]),
                                 unlabeled_image_sha256=tensor_digest(batch_u["image"]))
                self._append("pair_trace.jsonl", trace)
            assert updates == math.ceil(len(unlab)/accumulation)
            result = self._evaluate()
            self._log_metrics("paired_eval", result)
            row = dict(epoch=epoch, updates=updates, steps=len(unlab),
                       means=np.mean(stats_rows, axis=0).tolist(), seconds=time.monotonic()-start,
                       labeled_replays=len(unlab)/len(self.lab_set), refresh_labels=self.refresh)
            self._append("train_epochs.jsonl", row)
            logging.info("PAIRED_EPOCH %s", json.dumps(row))
            if result["dice"] > self.best_dice:
                self.best_dice, self.best_epoch = float(result["dice"]), epoch
                self._save_ucp("best.pth", epoch)
            if result["dice"] <= floor["dice"]-.03:
                confirm = self._evaluate()
                if confirm["dice"] <= floor["dice"]-.03:
                    stop_reason = "confirmed_collapse"; break
        final = result
        self._save_ucp("final.pth", epoch)
        summary = dict(floor_dice=floor["dice"], best_dice=self.best_dice,
                       best_epoch=self.best_epoch, final_dice=final["dice"],
                       completed_epochs=epoch+1, stop_reason=stop_reason,
                       labeled_replays=len(self.unlab_set)/len(self.lab_set), supervised_scale=1.,
                       selection="fixed single-view output and highest test Dice",
                       exact_resume_supported=False)
        (self.output_dir / "final_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (self.output_dir / "final_metrics.txt").write_text("".join(
            f"{k}: {v:.8f}\n" if isinstance(v, float) else f"{k}: {v}\n" for k,v in summary.items()), encoding="utf-8")
        logging.info("PAIRED_COMPLETE %s", json.dumps(summary))
