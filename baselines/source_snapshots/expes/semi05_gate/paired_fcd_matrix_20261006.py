"""Independent clean A-stage factors. Draft until real qualification passes.

No relation objective. The C=0 arm still computes the identical contextual CE
gradient and discards it; declared forward state is restored for every C arm.
Inherited HALF/LB loss weights, views and native single-T1 deployment remain.
"""
import contextlib
import gc
import time
import torch
import torch.nn.functional as F
from torch.cuda import amp
from expes.semi05_gate.paired_zea_cross_scale_half_20260911 import PairedCrossScaleHalf
from expes.semi05_gate.paired_l_cross_style_complement_20261001 import PairedCrossStyleComplement
from expes.semi05_gate.paired_hard_relations_20261005 import PairedHardRelations
from expes.semi05_gate.fcd_cac_fg_20260928 import context_views
from expes.semi05_gate.fcd_dycon_r2_losses_20260927 import _image_signal
from expes.semi05_gate.paired_rng_data_20260908 import scoped_rng, stream_seed
from expes.semi05_gate.fcd_mpl_device_transaction_20261003 import DeviceTree


class MatrixMixin:
    roots = PairedHardRelations.roots
    gradient_groups = PairedHardRelations.gradient_groups
    clear_relation_graph = PairedHardRelations.clear_relation_graph
    norm = staticmethod(PairedHardRelations.norm)

    def initialize_matrix(self, path):
        if not hasattr(self, '_matrix_initial_head_state'):
            self._matrix_initial_head_state={k:v.detach().cpu().clone() for k,v in self.student_fclr_head.state_dict().items()}
        self._load(path)
        # Frozen inherited _load intentionally leaves a missing head untouched;
        # independent cold replays must restore its original construction state.
        if not self._fclr_loaded_head:
            self.student_fclr_head.load_state_dict(self._matrix_initial_head_state,strict=True)
        self._update_teacher(0.); self._build_ucp_optimizer()
        # Each independent cold replay restores the original scaler too;
        # do not inherit growth/overflow state from a previous branch.
        self.scaler = amp.GradScaler()
        self.matrix_updates = 0
        # Patched optimizers may hold reference cycles. Repeated independent
        # replay must release abandoned moment tensors, not accumulate them.
        before = torch.cuda.memory_allocated()
        gc.collect(); torch.cuda.empty_cache()
        self.initialization_memory = dict(before_gc=before, after_gc=torch.cuda.memory_allocated())

    @contextlib.contextmanager
    def isolated_context(self, epoch, step):
        # These are the forward/backward mutable caches actually used by HALF.
        names = ('_xsa_context', '_fclr_stats', '_fclr_grad_stats', '_fclr_clip_norms',
                 '_fclr_last_student_delta', '_ucp_teacher_logits', '_ucp_reliable_mask',
                 '_ucp_anchor_mask', '_zea_gradient_stats', '_fvc_seed', '_fvc_capture',
                 '_fvc_feature', '_fvc_drop_fraction', '_fclr_identity_checked',
                 '_fclr_gradient_checked', '_b_last', '_b_mask', '_b_branch',
                 '_b_capture', '_b_captured_embeddings')
        tree = DeviceTree([p for m in self.roots().values() for p in m.parameters()])
        targets, absent = [], []
        for owner, attr in [(self, n) for n in names] + [(h, 'last_gate_stats') for h in (self.student_fclr_head, self.teacher_fclr_head)]:
            if hasattr(owner, attr): targets.append((owner, attr, tree.capture(getattr(owner, attr))))
            else: absent.append((owner, attr))
        buffers = [tree.capture(b) for m in self.roots().values() for b in m.buffers()]
        modes = [(m, m.training) for root in self.roots().values() for m in root.modules()]
        try:
            # Independent scope preserves the subsequent inherited data/RNG path.
            with scoped_rng(stream_seed(self.pair_seed, 'student-step', epoch, step), cuda=True):
                yield
        finally:
            restored = set()
            for node in buffers: tree.restore(node, restored)
            for owner, attr, node in targets: setattr(owner, attr, tree.restore(node, restored))
            for owner, attr in absent:
                if hasattr(owner, attr): delattr(owner, attr)
            for module, training in modes: module.training = training
            checked = set()
            for node in buffers: tree.assert_restored(node, node[1], checked)
            for owner, attr, node in targets: tree.assert_restored(node, getattr(owner, attr), checked)

    def contextual_gradients(self, image, epoch, step, parameters):
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
            region = [slice(None)]*5; region[axis+2] = slice(8, 24); overlap[tuple(region)] = True
            reliable = signal & overlap & (((qa >= .7) & (qb >= .7)) | ((qa <= .3) & (qb <= .3)))
            mask = F.interpolate(reliable.float(), size=(128,)*3, mode='nearest').bool()[:, 0]
            target = F.interpolate((qb >= .5).float(), size=(128,)*3, mode='nearest').long()[:, 0]
            with amp.autocast(): logits = self._forward_main_logits(self.student, a)
            ce = F.cross_entropy(logits.float(), target, reduction='none')
            counts = [int((mask & (target == c)).sum()) for c in (0, 1)]
            terms = [ce[mask & (target == c)].mean() for c in (0, 1) if counts[c]]
            balanced = torch.stack(terms).mean() if terms else logits.float().sum()*0.
            voxel = ce[mask].mean() if mask.any() else logits.float().sum()*0.
            objective = voxel if self.matrix_c == 'V' else balanced
            scale = float(self.scaler.get_scale())
            grads = torch.autograd.grad(self.scaler.scale(objective/5.), parameters, allow_unused=True)
            ids = {id(p): i for i, p in enumerate(parameters)}
            norms = {name: self.norm(grads[ids[id(p)]]/scale for p in group if grads[ids[id(p)]] is not None)
                     for name, group in self.gradient_groups().items()}
            assert all(__import__('math').isfinite(v) for v in norms.values())
            row = dict(axis=axis, reduction=self.matrix_c, coefficient=1., reliable_blocks=int(reliable.sum()),
                       background_voxels=counts[0], foreground_voxels=counts[1],
                       class_losses=[float(term.detach()) for term in terms], balanced=float(balanced.detach()),
                       voxel=float(voxel.detach()), raw_gradient_norms=norms, applied=self.matrix_c != '0',
                       foreground_multiplier=(sum(counts)/sum(n > 0 for n in counts)/counts[1]) if counts[1] else 0.,
                       no_gt_argument=True, relation_computed=False)
        row['declared_state_restored'] = True
        return grads, row

    def step_matrix(self, raw, compute_context=True, inject_overflow=False):
        assert len(raw) == 5
        start = time.monotonic()
        for name, module in self.roots().items(): module.train(not name.startswith('teacher'))
        self.optimizer.zero_grad(set_to_none=True); self.clear_relation_graph()
        parameters = [p for g in self.optimizer.param_groups for p in g['params']]
        rows = []
        for item in raw:
            epoch, step = item['epoch'], item['step']
            with scoped_rng(stream_seed(self.pair_seed, 'student-step', epoch, step), cuda=True):
                base = self._losses(item['image_l'], item['label_l'], item['image_u'], epoch, step)
                loss, sup, pseudo, wsv, logits, stats, views = base
                values = dict(step=step, supervised=float(sup.detach()), pseudo=float(pseudo.detach()),
                              wsv=float(wsv.detach()), baseline_total=float(loss.detach()), views=views,
                              native_reliable_ratio=float(stats['selected_ratio']))
                self.scaler.scale(loss/5.).backward()
                del base, loss, sup, pseudo, wsv, logits
            self._ucp_teacher_logits = None; self._ucp_reliable_mask = None; self._ucp_anchor_mask = None
            self.clear_relation_graph()
            if compute_context:
                grads, row = self.contextual_gradients(item['image_u'], epoch, step, parameters)
                if self.matrix_c != '0':
                    for parameter, gradient in zip(parameters, grads):
                        if gradient is not None:
                            if parameter.grad is None: parameter.grad = gradient.detach().clone()
                            else: parameter.grad.add_(gradient)
                # For C=0 do not create zero gradients: that could activate AdamW
                # decay on a parameter that the original path did not update.
                values['context'] = row; del grads
            rows.append(values)
        assert all(p.grad is None for name, m in self.roots().items() if name.startswith('teacher') for p in m.parameters())
        if inject_overflow: next(p for p in parameters if p.grad is not None).grad.flatten()[0] = float('inf')
        before = float(self.scaler.get_scale()); clips = len(self._fclr_clip_norms)
        self.scaler.unscale_(self.optimizer)
        finite = all(bool(torch.isfinite(p.grad).all()) for p in parameters if p.grad is not None)
        norms = {name: self.norm(p.grad for p in group) for name, group in self.gradient_groups().items()}
        self.scaler.step(self.optimizer); self.scaler.update()
        skipped = float(self.scaler.get_scale()) < before
        assert skipped == (not finite) and len(self._fclr_clip_norms)-clips == (0 if skipped else 1)
        if not skipped: self._update_teacher(.999); self.matrix_updates += 1
        self.optimizer.zero_grad(set_to_none=True); self.clear_relation_graph()
        return dict(group=self.matrix_group, update=self.matrix_updates, skipped=skipped,
                    scale_before=before, scale_after=float(self.scaler.get_scale()), teacher_no_grad=True,
                    clip_calls=len(self._fclr_clip_norms)-clips,
                    gradient_groups={k: v if __import__('math').isfinite(v) else None for k, v in norms.items()},
                    microbatches=rows, context_computed=compute_context, seconds=time.monotonic()-start)


class MatrixHalf(MatrixMixin, PairedCrossScaleHalf):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.matrix_group = cfg['semi']['matrix_group']; self.matrix_c = cfg['semi']['matrix_c']
        assert self.matrix_group in ('A00', 'A0V', 'A0B') and self.matrix_c in ('0', 'V', 'B')
        assert dict(A00='0', A0V='V', A0B='B')[self.matrix_group] == self.matrix_c


class MatrixStyle(MatrixMixin, PairedCrossStyleComplement):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.matrix_group = cfg['semi']['matrix_group']; self.matrix_c = cfg['semi']['matrix_c']
        assert self.matrix_group in ('A10', 'A1V', 'A1B') and self.matrix_c in ('0', 'V', 'B')
        assert dict(A10='0', A1V='V', A1B='B')[self.matrix_group] == self.matrix_c


def build_matrix(cfg):
    return (MatrixHalf if cfg['semi']['matrix_group'].startswith('A0') else MatrixStyle)(cfg)
