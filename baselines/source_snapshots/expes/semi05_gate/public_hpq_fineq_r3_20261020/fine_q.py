"""Development candidate: full-resolution Q confidence/target, same support.

Only Q pseudo-target construction changes. Not a launcher or readiness receipt.
No GT input. Thresholds, context geometry, coefficient and reduction are fixed.
"""
import hashlib
import math
import torch
import torch.nn.functional as F

SCHEMA = "PUBLIC_N005_FINE_Q_V1"


def context_targets(pa, pb, signal32, axis):
    if (tuple(pa.shape) != (1, 1, 128, 128, 128) or pa.shape != pb.shape
            or signal32.shape != (1, 1, 32, 32, 32) or signal32.dtype != torch.bool
            or type(axis) is not int or axis not in (0, 1, 2)):
        raise ValueError("Invalid single-context input geometry")
    if pa.requires_grad or pb.requires_grad:
        raise ValueError("Teacher targets must be detached")
    if not (bool(torch.isfinite(pa).all()) and bool(torch.isfinite(pb).all())
            and bool(((pa >= 0) & (pa <= 1)).all())
            and bool(((pb >= 0) & (pb <= 1)).all())):
        raise ValueError("Invalid teacher probabilities")
    overlap = torch.zeros_like(signal32)
    region = [slice(None)] * 5
    region[axis + 2] = slice(8, 24)
    overlap[tuple(region)] = True
    support32 = signal32 & overlap
    support = F.interpolate(support32.float(), size=(128,) * 3, mode="nearest").bool()[:, 0]
    reliable = (((pa >= .7) & (pb >= .7)) | ((pa <= .3) & (pb <= .3)))[:, 0]
    mask = support & reliable
    target = (pb[:, 0] >= .5).long()
    return mask, target, support32


def context_objectives(logits, mask, target):
    if logits.shape != (1, 2, 128, 128, 128) or mask.shape != target.shape:
        raise ValueError("Invalid Q logits or mask")
    ce = F.cross_entropy(logits.float(), target, reduction="none")
    counts = [int((mask & (target == c)).sum()) for c in (0, 1)]
    terms = [ce[mask & (target == c)].mean() for c in (0, 1) if counts[c]]
    balanced = torch.stack(terms).mean() if terms else logits.float().sum() * 0.
    voxel = ce[mask].mean() if bool(mask.any()) else logits.float().sum() * 0.
    return balanced, voxel, counts, terms


def validate_context_row(row, reduction):
    """New voxel-count gate; never pretend fine voxels are coarse blocks."""
    if (row.get("schema") != SCHEMA or row.get("target_resolution") != [128, 128, 128]
            or row.get("support_resolution") != [32, 32, 32]
            or row.get("reduction") != reduction or row.get("coefficient") != 1.
            or row.get("applied") != (reduction != "0")
            or not row.get("declared_state_restored") or not row.get("no_gt_argument")
            or row.get("relation_computed") is not False):
        raise ValueError("Fine Q recipe or isolation mismatch")
    fields = ("foreground_voxels", "background_voxels", "reliable_voxels", "support_blocks")
    if not all(type(row.get(k)) is int and row[k] >= 0 for k in fields):
        raise ValueError("Invalid fine Q counts")
    n = row["foreground_voxels"] + row["background_voxels"]
    if n != row["reliable_voxels"] or not n <= 64 * row["support_blocks"] <= 128 ** 3 // 2:
        raise ValueError("Fine Q voxel/support count mismatch")
    norms = row.get("raw_gradient_norms", {})
    if len(norms) != 6 or not all(math.isfinite(v) and v >= 0 for v in norms.values()):
        raise ValueError("Nonfinite/missing Q gradients")
    if n and not all(v > 0 for v in norms.values()):
        raise ValueError("Selected Q context has vanishing parameter group")


class FineQMixin:
    """Insert before the frozen qualified trainer; H/P are inherited unchanged."""

    def contextual_gradients(self, image, epoch, step, parameters):
        from expes.semi05_gate.fcd_cac_fg_20260928 import context_views
        from expes.semi05_gate.fcd_dycon_r2_losses_20260927 import _image_signal
        self._collect_routes()
        assert image.shape == (1, 1, 128, 128, 128) and not image.requires_grad
        with self.isolated_context(epoch, step):
            axis = (epoch + step) % 3
            a, b = context_views(image, axis)
            with torch.no_grad(), torch.cuda.amp.autocast():
                ta = self._forward_main_logits(self.teacher, a)
                tb = self._forward_main_logits(self.teacher, b)
                pa = ta.float().softmax(1)[:, 1:2]
                pb = tb.float().softmax(1)[:, 1:2]
            del ta, tb, b
            mask, target, support32 = context_targets(
                pa, pb, _image_signal(image, (32, 32, 32)), axis)
            del pa, pb
            with torch.cuda.amp.autocast():
                logits = self._forward_main_logits(self.student, a)
            balanced, voxel, counts, terms = context_objectives(logits, mask, target)
            objective = voxel if self.matrix_c == "V" else balanced
            scale = float(self.scaler.get_scale())
            grads = torch.autograd.grad(self.scaler.scale(objective / 5.), parameters,
                                        allow_unused=True)
            ids = {id(p): i for i, p in enumerate(parameters)}
            norms = {name: self.norm(grads[ids[id(p)]] / scale for p in group
                                    if grads[ids[id(p)]] is not None)
                     for name, group in self.gradient_groups().items()}
            row = dict(schema=SCHEMA, axis=axis, reduction=self.matrix_c, coefficient=1.,
                       target_resolution=[128, 128, 128], support_resolution=[32, 32, 32],
                       support_blocks=int(support32.sum()), reliable_voxels=int(mask.sum()),
                       background_voxels=counts[0], foreground_voxels=counts[1],
                       class_losses=[float(t.detach()) for t in terms],
                       balanced=float(balanced.detach()), voxel=float(voxel.detach()),
                       raw_gradient_norms=norms, applied=self.matrix_c != "0",
                       foreground_multiplier=(sum(counts) / sum(n > 0 for n in counts) / counts[1])
                       if counts[1] else 0., no_gt_argument=True, relation_computed=False)
            if self.qualification_routing:
                row["mask_sha256"] = hashlib.sha256(mask.cpu().numpy().tobytes()).hexdigest()
                row["target_sha256"] = hashlib.sha256(target.cpu().numpy().tobytes()).hexdigest()
        row["declared_state_restored"] = True
        validate_context_row(row, self.matrix_c)
        return grads, row

    def probe_context_reductions(self, image, epoch, step):
        """Own candidate qualification; do not claim equality to coarse Q."""
        from expes.semi05_gate.fcd_cac_fg_20260928 import context_views
        from expes.semi05_gate.fcd_dycon_r2_losses_20260927 import _image_signal
        assert self.qualification_routing
        parameters = [p for g in self.optimizer.param_groups for p in g["params"]]
        original = self.matrix_c
        rows = {}
        try:
            for reduction in ("V", "B"):
                self.matrix_c = reduction
                grads, row = self.contextual_gradients(image, epoch, step, parameters)
                measured = [None if g is None else g.detach().cpu() for g in grads]
                del grads
                # Independent expression, no call to context_targets/objectives.
                with self.isolated_context(epoch, step):
                    axis = (epoch + step) % 3
                    a, b = context_views(image, axis)
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        p = self._forward_main_logits(self.teacher, a).float().softmax(1)[:, 1]
                        q = self._forward_main_logits(self.teacher, b).float().softmax(1)[:, 1]
                    support = F.interpolate(_image_signal(image, (32,) * 3).float(),
                                            size=(128,) * 3, mode="nearest").bool()[:, 0]
                    coordinate = torch.arange(128, device=image.device)
                    central = ((coordinate >= 32) & (coordinate < 96))
                    shape = [1] * 4
                    shape[axis + 1] = 128
                    mask = support & central.reshape(shape)
                    mask &= ((p >= .7) & (q >= .7)) | ((p <= .3) & (q <= .3))
                    target = (q >= .5).long()
                    with torch.cuda.amp.autocast():
                        logits = self._forward_main_logits(self.student, a)
                    ce = F.nll_loss(F.log_softmax(logits.float(), dim=1), target, reduction="none")
                    terms = [ce[mask & (target == c)].mean() for c in (0, 1)
                             if bool((mask & (target == c)).any())]
                    objective = (ce[mask].mean() if reduction == "V" else torch.stack(terms).mean()) \
                        if terms else logits.float().sum() * 0.
                    ref = torch.autograd.grad(self.scaler.scale(objective / 5.), parameters,
                                              allow_unused=True)
                    assert all((x is None and y is None) or
                               (x is not None and y is not None and torch.equal(x, y.detach().cpu()))
                               for x, y in zip(measured, ref)), "Fine Q independent gradient mismatch"
                    for key, value in (("mask_sha256", mask), ("target_sha256", target)):
                        assert row[key] == hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest()
                    del ref, measured
                rows[reduction] = row
            for key in ("mask_sha256", "target_sha256", "support_blocks", "reliable_voxels",
                        "background_voxels", "foreground_voxels", "class_losses", "balanced", "voxel"):
                assert rows["V"][key] == rows["B"][key]
            return dict(reductions=rows, candidate_formula_gradient_exact=True,
                        same_target_mask=True, optimizer_updates=0, coarse_Q_equivalence_claimed=False)
        finally:
            self.matrix_c = original
