"""Cross-style complementary strong views on the full AutoSAM decoder path.

The EMA teacher and labeled supervision remain native. Two distinct image-only
unlabeled donors provide FDA appearance views. Complementary 50% channel masks
are applied to the adapted SAM image embedding immediately before the full mask
decoder; their pseudo and consistency losses are averaged, replacing the old
single FVC/WSV perturbation objective rather than being added on top.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.cuda import amp
from torch.utils.checkpoint import checkpoint

from expes.common.losses import pseudo_loss
from expes.common.semi_data import read_case_list
from expes.semi05_gate.fcd_l0_fda_lesion_integrity_20260925 import read_donor
from expes.semi05_gate.paired_rng_data_20260908 import SeededCases, scoped_rng, stream_seed
from expes.semi05_gate.paired_wsv_control_20260908 import PairedWSVControl
from expes.semi05_gate.paired_zea_cross_scale_half_20260911 import PairedCrossScaleHalf


class PairedCrossStyleComplement(PairedCrossScaleHalf):
    """Native supervision plus two complementary full-decoder style views."""

    def __init__(self, cfg):
        assert cfg["semi"]["cross_style_complement"] == "two-donor-full-decoder"
        super().__init__(cfg)
        split = Path(cfg["dataset"]["split_root"]) / f"fold{cfg['dataset']['fold']}"
        self._style_donors = read_case_list(split / "train_unlab.txt")
        assert len(self._style_donors) + len(read_case_list(split / "train_lab.txt")) == 100
        assert len(self._style_donors) >= 2
        self._style_root = Path(cfg["dataset"]["fold_cv_dir_template"].format(
            fold=cfg["dataset"]["fold"])) / "train/imagesTr"
        self._style_amp_cache = {}
        self._b_mask = None
        self._b_branch = None
        self._b_capture = False
        self._b_captured_embeddings = []
        self._b_last = None

        # CrossScale's adapter hook is registered during super().__init__. This
        # later hook therefore masks the adapted embedding delivered to the
        # complete main mask decoder and residual refinement path.
        def complementary_mask(_module, _inputs, embedding):
            if self._b_mask is None:
                return embedding
            mask = self._b_mask.to(device=embedding.device, dtype=embedding.dtype)
            if tuple(mask.shape) != (1, embedding.shape[1], 1, 1, 1):
                raise RuntimeError(f"Invalid complementary mask {tuple(mask.shape)}")
            # Reentrant activation checkpointing executes the first pass under
            # no_grad and recomputes with gradients during backward. Capture
            # only the latter so the audit observes the true routed gradient.
            if self._b_capture and embedding.requires_grad:
                embedding.retain_grad()
                self._b_captured_embeddings.append(
                    dict(branch=int(self._b_branch), embedding=embedding, mask=mask.detach().clone()))
            return embedding * mask

        self._b_mask_handle = self.student.cascade.image_encoder.register_forward_hook(
            complementary_mask)

    def _donor_amplitude(self, name):
        if name not in self._style_amp_cache:
            image = read_donor(self._style_root / f"{name}.nii.gz")
            spectrum = np.fft.fftshift(np.fft.fftn(image))
            self._style_amp_cache[name] = torch.from_numpy(
                np.abs(spectrum[58:71, 58:71, 58:71]).astype(np.float32))
        return self._style_amp_cache[name].to(self.device)

    @staticmethod
    def _style(source, donor_amplitude):
        if tuple(source.shape) != (1, 1, 128, 128, 128):
            raise RuntimeError(f"Unexpected image shape: {tuple(source.shape)}")
        with torch.no_grad():
            original = source.float()
            spectrum = torch.fft.fftshift(
                torch.fft.fftn(original, dim=(-3, -2, -1)), dim=(-3, -2, -1))
            center = spectrum[..., 58:71, 58:71, 58:71]
            spectrum[..., 58:71, 58:71, 58:71] = torch.polar(
                .5 * center.abs() + .5 * donor_amplitude, torch.angle(center))
            styled = torch.fft.ifftn(
                torch.fft.ifftshift(spectrum, dim=(-3, -2, -1)),
                dim=(-3, -2, -1)).real.float()
            valid = original > original.amin() + .1
            styled = torch.where(valid, styled, original)
            if not torch.isfinite(styled).all():
                raise RuntimeError("Non-finite FDA output")
            return styled, float((styled - original).abs()[valid].mean())

    def _donor_pair(self, epoch, step):
        count = len(self._style_donors)
        first = stream_seed(self.pair_seed, "B-style-donor-1", epoch, step) % count
        second = stream_seed(self.pair_seed, "B-style-donor-2", epoch, step) % count
        if second == first:
            second = (second + 1) % count
        assert first != second
        return self._style_donors[first], self._style_donors[second]

    def _complementary_masks(self, epoch, step, channels):
        if channels % 2:
            raise RuntimeError(f"Complementary masking requires even channels, got {channels}")
        generator = torch.Generator(device=self.device).manual_seed(
            stream_seed(self.pair_seed, "B-complement-mask", epoch, step))
        order = torch.randperm(channels, device=self.device, generator=generator)
        first = torch.zeros((1, channels, 1, 1, 1), device=self.device, dtype=torch.float32)
        first[:, order[:channels // 2]] = 2.0
        second = 2.0 - first
        assert int((first > 0).sum()) == channels // 2
        assert int((second > 0).sum()) == channels // 2
        assert torch.equal(first + second, torch.full_like(first, 2.0))
        return first, second

    def _masked_main_forward(self, image, mask, branch):
        if self._b_mask is not None:
            raise RuntimeError("Nested complementary mask forward")
        self._b_mask, self._b_branch = mask, branch
        try:
            return super()._forward_main_logits(self.student, image)
        finally:
            self._b_mask, self._b_branch = None, None

    def _checkpointed_masked_forward(self, image, mask, branch):
        # Two simultaneous full 3D SAM activation graphs exceed a 24 GiB card.
        # Recompute each complete branch during backward; no model/loss/path is
        # removed. The detached augmentation input is marked differentiable
        # solely to satisfy reentrant checkpointing in the project Torch build.
        value = image.detach().requires_grad_(True)
        return checkpoint(lambda item: self._masked_main_forward(item, mask, branch), value)

    def _losses(self, image_l, label_l, image_u, epoch, step):
        # One native teacher target; no donor label or test data are accessed.
        pseudo, reliable, stats = self._reliable_pseudo(image_u, epoch)
        donor1, donor2 = self._donor_pair(epoch, step)
        style1, delta1 = self._style(image_u, self._donor_amplitude(donor1))
        style2, delta2 = self._style(image_u, self._donor_amplitude(donor2))
        channels = int(self.student.cascade.prompt_encoder.embed_dim)
        mask1, mask2 = self._complementary_masks(epoch, step, channels)

        with scoped_rng(stream_seed(self.pair_seed, "B-strong-view-1", epoch, step), cuda=True):
            view1, state1 = self._student_unlabeled_view(style1, .05)
        with scoped_rng(stream_seed(self.pair_seed, "B-strong-view-2", epoch, step), cuda=True):
            view2, state2 = self._student_unlabeled_view(style2, .05)

        with amp.autocast(enabled=True):
            # Labeled supervision is deliberately native, unlike historical L1.
            supervised = self._supervised_step(image_l, label_l)
            logits1 = self._restore_student_logits(
                self._checkpointed_masked_forward(view1, mask1, 1), state1)
            logits2 = self._restore_student_logits(
                self._checkpointed_masked_forward(view2, mask2, 2), state2)
            weight = self._pseudo_weight(epoch)
            pseudo1 = weight * pseudo_loss(
                logits1, pseudo, reliable,
                bool(self.cfg["train"].get("use_dice", True)),
                float(self.cfg["train"].get("dice_weight", .5)))
            pseudo2 = weight * pseudo_loss(
                logits2, pseudo, reliable,
                bool(self.cfg["train"].get("use_dice", True)),
                float(self.cfg["train"].get("dice_weight", .5)))
            extra1 = self._extra_unlabeled_loss(logits1, epoch)
            extra2 = self._extra_unlabeled_loss(logits2, epoch)
            pseudo_mean = .5 * (pseudo1 + pseudo2)
            extra_mean = .5 * (extra1 + extra2)
            total = supervised + pseudo_mean + extra_mean

        self._b_last = dict(
            epoch=int(epoch), step=int(step), donor1=donor1, donor2=donor2,
            donor_paths=[str(self._style_root / f"{donor1}.nii.gz"),
                         str(self._style_root / f"{donor2}.nii.gz")],
            donor_labels_used=False, style_delta1=delta1, style_delta2=delta2,
            kept_channels_each=channels // 2, complementary_sum=2.0,
            pseudo1=float(pseudo1.detach()), pseudo2=float(pseudo2.detach()),
            extra1=float(extra1.detach()), extra2=float(extra2.detach()),
            native_labeled_supervision=True, native_teacher=True,
            old_fvc_active=False)
        self._append("cross_style_steps.jsonl", self._b_last)
        state = dict(view1=list(state1), view2=list(state2), donor1=donor1,
                     donor2=donor2,
                     mask_seed=stream_seed(self.pair_seed, "B-complement-mask", epoch, step))
        # The paired preflight differentiates the extra objective with respect
        # to the returned tensor. Return branch 1 (which is genuinely an input
        # to extra_mean) rather than a downstream arithmetic summary.
        return total, supervised, pseudo_mean, extra_mean, logits1, stats, state

    @staticmethod
    def _group_gradient(module):
        gradients = [p.grad.detach().float() for p in module.parameters() if p.grad is not None]
        return dict(
            tensors=len(gradients),
            nonzero=sum(bool(g.abs().sum() > 0) for g in gradients),
            norm=float(sum(g.pow(2).sum() for g in gradients).sqrt()) if gradients else 0.0)

    def preflight(self, init_from, expected_floor):
        self._b_capture = True
        self._b_captured_embeddings.clear()
        # Call the clean paired preflight directly: inherited FVC preflight
        # assumes the old one-view perturbation, which this method replaces.
        result = PairedWSVControl.preflight(self, init_from, expected_floor)
        self._b_capture = False
        if len(self._b_captured_embeddings) != 2:
            raise RuntimeError(f"Expected two captured strong embeddings, got {len(self._b_captured_embeddings)}")
        branch_gradients = []
        for record in self._b_captured_embeddings:
            gradient = record["embedding"].grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise RuntimeError("Missing/non-finite strong-view embedding gradient")
            mask = record["mask"].expand_as(gradient)
            kept = gradient[mask > 0]
            dropped = gradient[mask == 0]
            branch_gradients.append(dict(
                branch=record["branch"], kept_norm=float(kept.float().norm()),
                dropped_max_abs=float(dropped.float().abs().max()),
                kept_channels=int((record["mask"] > 0).sum())))
        if not all(row["kept_norm"] > 0 and row["dropped_max_abs"] == 0
                   for row in branch_gradients):
            raise RuntimeError(f"Complementary gradient routing failed: {branch_gradients}")

        groups = dict(
            sam_encoder=self._group_gradient(self.student.cascade.sam.image_encoder),
            mask_decoder=self._group_gradient(self.student.cascade.sam.mask_decoder),
            coarse=self._group_gradient(self.student.cascade.coarse_head),
            refinement=self._group_gradient(self.student_fclr_head),
            cross_scale_adapter=self._group_gradient(self.student_zea_adapter),
        )
        if not all(value["norm"] > 0 for value in groups.values()):
            raise RuntimeError(f"Missing full-path gradient: {groups}")
        if not self._b_last or self._b_last["donor1"] == self._b_last["donor2"]:
            raise RuntimeError("Distinct donor audit failed")
        if self._b_last["style_delta1"] <= 1e-3 or self._b_last["style_delta2"] <= 1e-3:
            raise RuntimeError(f"Style views did not change image: {self._b_last}")
        if self._fvc_seed is not None:
            raise RuntimeError("Old FVC unexpectedly active")

        # Execute the disposable first update left by PairedWSVControl, then a
        # second full update. Zero-initialized residual/adapter paths need this
        # to prove that upstream parameters, not merely final zero layers,
        # receive the new objective.
        scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scaler.get_scale() < scale:
            raise RuntimeError("AMP skipped first preflight update")
        self.optimizer.zero_grad(set_to_none=True)
        self._update_teacher(.999)
        lab, unlab = self._loaders(0)
        image_l, label_l = self._prepare_batch(next(lab))
        image_u, _ = self._prepare_batch(next(iter(unlab)))
        self.student.train(); self.teacher.eval()
        second, *_ = self._losses(image_l, label_l, image_u, 0, 1)
        self.scaler.scale(second).backward()
        second_groups = dict(
            sam_encoder=self._group_gradient(self.student.cascade.sam.image_encoder),
            mask_decoder=self._group_gradient(self.student.cascade.sam.mask_decoder),
            coarse=self._group_gradient(self.student.cascade.coarse_head),
            refinement=self._group_gradient(self.student_fclr_head),
            cross_scale_adapter=self._group_gradient(self.student_zea_adapter),
        )
        if not all(value["norm"] > 0 for value in second_groups.values()):
            raise RuntimeError(f"Missing second-update gradient: {second_groups}")
        adapter_named = {name: max(values) if values else 0.0
                         for name, values in self._zea_gradient_stats.items()}
        refinement_named = {name: max(values) if values else 0.0
                            for name, values in self._fclr_grad_stats.items()}
        if not all(value > 0 for value in adapter_named.values()):
            raise RuntimeError(f"Inactive adapter parameters: {adapter_named}")
        if not all(value > 0 for value in refinement_named.values()):
            raise RuntimeError(f"Inactive refinement parameters: {refinement_named}")
        if any(parameter.grad is not None for parameter in self.teacher.parameters()):
            raise RuntimeError("EMA teacher received gradients")
        scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scaler.get_scale() < scale:
            raise RuntimeError("AMP skipped second preflight update")
        self.optimizer.zero_grad(set_to_none=True)
        self._update_teacher(.999)

        # Full checkpoint roundtrip for each ratio after the disposable update.
        data = SeededCases(self.lab_set, self.pair_seed, "B-roundtrip")
        batch = data[(0, 0, 0)]
        image, _ = self._prepare_batch({k: v.unsqueeze(0) for k, v in batch.items()
                                        if torch.is_tensor(v)})
        self.student.eval(); self.student_fclr_head.eval(); self.student_zea_adapter.eval()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), amp.autocast(enabled=True):
            before = self._forward_main_logits(self.student, image)
        target = self._save_ucp("storage_preflight.pth", -1)
        self._load(target)
        with torch.no_grad(), amp.autocast(enabled=True):
            after = self._forward_main_logits(self.student, image)
        if not torch.equal(before, after):
            raise RuntimeError("Checkpoint roundtrip changed native logits")

        result.update(
            control="L-CROSS-STYLE-COMPLEMENT-R1",
            paired_base="HALF-R2",
            source_idea="FDA image-only donors plus UniMatch-V2 complementary dropout adaptation",
            labeled_path="native single supervised forward",
            teacher_path="native EMA single forward",
            strong_paths=2,
            donors_distinct=True,
            donor_gt_used=False,
            mask_location="adapted SAM image embedding before full mask decoder",
            mask_kept_fraction=.5,
            masks_exactly_complementary=True,
            branch_embedding_gradients=branch_gradients,
            full_path_gradients=groups,
            second_update_gradients=second_groups,
            adapter_named_gradient_norms=adapter_named,
            refinement_named_gradient_norms=refinement_named,
            disposable_optimizer_updates=2,
            teacher_gradient_free=True,
            pseudo_and_consistency_aggregation="mean; original total weights preserved",
            replaces_old_fvc_and_single_wsv=True,
            activation_checkpointing="both complete strong branches; recomputed during backward",
            checkpoint_roundtrip=dict(path=str(target), logits_bitwise_equal=True,
                                      bytes=target.stat().st_size),
            peak_gpu_gib=torch.cuda.max_memory_allocated() / 1024**3,
            mechanism_status="PASS")
        (self.output_dir / "cross_style_preflight.json").write_text(
            __import__("json").dumps(result, indent=2))
        return result
