import hashlib
import math
import torch.nn.functional as F
from torch.cuda import amp
from expes.semi05_gate.fcd_cac_fg_20260928 import context_views
from expes.semi05_gate.fcd_dycon_r2_losses_20260927 import _image_signal
import torch
from expes.semi05_gate.paired_fcd_matrix_20261006 import MatrixStyle, MatrixMixin
from expes.semi05_gate.paired_l_cross_style_complement_20261001 import PairedCrossStyleComplement

class QualifiedStyleBalanced(MatrixStyle):

    def __init__(self, cfg):
        super().__init__(cfg)
        assert self.matrix_group == 'A1B' and self.matrix_c == 'B'
        self.qualification_routing = False
        self._a1b_styles = []
        self._a1b_routes = []

    def _collect_routes(self):
        if not self._b_captured_embeddings:
            return
        records = self._b_captured_embeddings
        assert len(records) == 2 and sorted((r['branch'] for r in records)) == [1, 2]
        masks = [r['mask'] for r in records]
        assert torch.equal(masks[0] + masks[1], torch.full_like(masks[0], 2.0))
        rows = []
        scale = float(self.scaler.get_scale())
        for record in records:
            gradient = record['embedding'].grad
            assert gradient is not None and bool(torch.isfinite(gradient).all())
            mask = record['mask'].expand_as(gradient)
            kept = float((gradient[mask > 0].float() / scale).norm())
            dropped = float(gradient[mask == 0].float().abs().max())
            channels = int((record['mask'] > 0).sum())
            assert kept > 0 and dropped == 0 and (channels * 2 == record['mask'].shape[1])
            rows.append(dict(branch=record['branch'], kept_norm=kept, dropped_max_abs=dropped, kept_channels=channels))
        self._a1b_routes.append(rows)
        self._b_captured_embeddings = []
        self._b_capture = False

    def _losses(self, image_l, label_l, image_u, epoch, step):
        self._collect_routes()
        self._b_capture = bool(self.qualification_routing)
        result = PairedCrossStyleComplement._losses(self, image_l, label_l, image_u, epoch, step)
        style = dict(self._b_last)
        assert style['donor1'] != style['donor2']
        assert style['donor1'] in self._style_donors and style['donor2'] in self._style_donors
        assert not style['donor_labels_used'] and style['native_labeled_supervision'] and style['native_teacher']
        assert not style['old_fvc_active'] and self._fvc_seed is None
        assert style['complementary_sum'] == 2.0 and style['style_delta1'] > 0.001 and (style['style_delta2'] > 0.001)
        for index, keys in ((2, ('pseudo1', 'pseudo2')), (3, ('extra1', 'extra2'))):
            actual = float(result[index].detach())
            mean = 0.5 * (style[keys[0]] + style[keys[1]])
            assert math.isclose(actual, mean, rel_tol=0.002, abs_tol=1e-05)
        self._a1b_styles.append(style)
        return result

    def contextual_gradients(self, image, epoch, step, parameters):
        self._collect_routes()
        assert image.shape == (1, 1, 128, 128, 128)
        assert not image.requires_grad
        with self.isolated_context(epoch, step):
            axis = (epoch + step) % 3
            a, b = context_views(image, axis)
            with torch.no_grad(), amp.autocast():
                ta = self._forward_main_logits(self.teacher, a)
                tb = self._forward_main_logits(self.teacher, b)
                qa = F.avg_pool3d(ta.float().softmax(1)[:, 1:2], 4, 4)
                qb = F.avg_pool3d(tb.float().softmax(1)[:, 1:2], 4, 4)
            del ta, tb, b
            signal = _image_signal(image, (32, 32, 32))
            overlap = torch.zeros_like(signal)
            region = [slice(None)] * 5
            region[axis + 2] = slice(8, 24)
            overlap[tuple(region)] = True
            reliable = signal & overlap & ((qa >= 0.7) & (qb >= 0.7) | (qa <= 0.3) & (qb <= 0.3))
            mask = F.interpolate(reliable.float(), size=(128,) * 3, mode='nearest').bool()[:, 0]
            target = F.interpolate((qb >= 0.5).float(), size=(128,) * 3, mode='nearest').long()[:, 0]
            with amp.autocast():
                logits = self._forward_main_logits(self.student, a)
            ce = F.cross_entropy(logits.float(), target, reduction='none')
            counts = [int((mask & (target == c)).sum()) for c in (0, 1)]
            terms = [ce[mask & (target == c)].mean() for c in (0, 1) if counts[c]]
            balanced = torch.stack(terms).mean() if terms else logits.float().sum() * 0.0
            voxel = ce[mask].mean() if mask.any() else logits.float().sum() * 0.0
            objective = voxel if self.matrix_c == 'V' else balanced
            scale = float(self.scaler.get_scale())
            grads = torch.autograd.grad(self.scaler.scale(objective / 5.0), parameters, allow_unused=True)
            ids = {id(p): i for i, p in enumerate(parameters)}
            norms = {name: self.norm((grads[ids[id(p)]] / scale for p in group if grads[ids[id(p)]] is not None)) for name, group in self.gradient_groups().items()}
            assert all((__import__('math').isfinite(v) for v in norms.values()))
            row = dict(axis=axis, reduction=self.matrix_c, coefficient=1.0, reliable_blocks=int(reliable.sum()), background_voxels=counts[0], foreground_voxels=counts[1], class_losses=[float(term.detach()) for term in terms], balanced=float(balanced.detach()), voxel=float(voxel.detach()), raw_gradient_norms=norms, applied=self.matrix_c != '0', foreground_multiplier=sum(counts) / sum((n > 0 for n in counts)) / counts[1] if counts[1] else 0.0, no_gt_argument=True, relation_computed=False)
            if self.qualification_routing:
                row['mask_sha256'] = hashlib.sha256(mask.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                row['target_sha256'] = hashlib.sha256(target.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        row['declared_state_restored'] = True
        return (grads, row)

    def probe_context_reductions(self, image, epoch, step):
        assert self.qualification_routing
        parameters = [p for g in self.optimizer.param_groups for p in g['params']]
        original_c = self.matrix_c
        rows = {}
        try:
            for reduction in ('V', 'B'):
                self.matrix_c = reduction
                grads, measured = self.contextual_gradients(image, epoch, step, parameters)
                cpu_grads = [None if g is None else g.detach().cpu().clone() for g in grads]
                del grads
                reference_grads, reference = MatrixMixin.contextual_gradients(self, image, epoch, step, parameters)
                assert {k: v for k, v in measured.items() if not k.endswith('_sha256')} == reference
                assert all((x is None and y is None or (x is not None and y is not None and torch.equal(x, y.detach().cpu())) for x, y in zip(cpu_grads, reference_grads))), reduction
                if measured['reliable_blocks']:
                    assert all((v > 0 for v in measured['raw_gradient_norms'].values()))
                rows[reduction] = measured
                del cpu_grads, reference_grads
            for key in ('mask_sha256', 'target_sha256', 'reliable_blocks', 'background_voxels', 'foreground_voxels', 'class_losses', 'balanced', 'voxel'):
                assert rows['V'][key] == rows['B'][key], key
            return dict(reductions=rows, frozen_formula_gradient_exact=True, same_target_mask=True, optimizer_updates=0)
        finally:
            self.matrix_c = original_c

    def step_matrix(self, raw, compute_context=True, inject_overflow=False):
        self._a1b_styles = []
        self._a1b_routes = []
        ema_calls = []
        original_ema = self._update_teacher

        def measured_ema(momentum):
            ema_calls.append(momentum)
            return original_ema(momentum)
        self._update_teacher = measured_ema
        try:
            result = super().step_matrix(raw, compute_context=compute_context, inject_overflow=inject_overflow)
        finally:
            self._update_teacher = original_ema
        assert ema_calls == ([] if result['skipped'] else [0.999])
        result['ema_calls'] = len(ema_calls)
        self._collect_routes()
        assert len(self._a1b_styles) == 5
        if self.qualification_routing:
            assert len(self._a1b_routes) == 5
        for index, micro in enumerate(result['microbatches']):
            micro['style'] = self._a1b_styles[index]
            if self.qualification_routing:
                micro['branch_routes'] = self._a1b_routes[index]
        result['branch_routing_measured'] = bool(self.qualification_routing)
        return result
