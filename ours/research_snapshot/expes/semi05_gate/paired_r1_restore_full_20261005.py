import time
import torch
import torch.nn.functional as F
from expes.semi05_gate.fcd_cac_fg_20260928 import context_views
from expes.semi05_gate.fcd_dycon_r2_losses_20260927 import _image_signal
from expes.semi05_gate.fcd_hard_spatial_relations_20261004 import pick_regions, reliable_class_loss
from expes.semi05_gate.fcd_deterministic_losses_r2_20261005 import relation_loss
from expes.semi05_gate.paired_hard_relations_20261005 import PairedHardRelations
from expes.semi05_gate.paired_zea_cross_scale_half_20260911 import PairedCrossScaleHalf
from expes.semi05_gate.paired_rng_data_20260908 import scoped_rng, stream_seed
from expes.semi05_gate.fcd_mpl_device_transaction_20261003 import DeviceTree

class PairedR1RestoreFull(PairedHardRelations):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.restore_arm = cfg['semi']['restore_arm']
        assert self.restore_arm in ('reference', 'restore') and self.relation_arm == 'off'

    def original_gradients(self, item, pars):
        names = ('_xsa_context', '_fclr_stats', '_fclr_grad_stats', '_fclr_clip_norms', '_fclr_last_student_delta', '_ucp_teacher_logits', '_ucp_reliable_mask', '_ucp_anchor_mask', '_zea_gradient_stats', '_fvc_seed', '_fvc_capture', '_fvc_feature', '_fvc_drop_fraction', '_fclr_identity_checked', '_fclr_gradient_checked')
        tree = DeviceTree([p for m in self.roots().values() for p in m.parameters()])
        targets, absent = ([], [])
        for owner, attr in [(self, n) for n in names] + [(h, 'last_gate_stats') for h in (self.student_fclr_head, self.teacher_fclr_head)]:
            if hasattr(owner, attr):
                targets.append((owner, attr, tree.capture(getattr(owner, attr))))
            else:
                absent.append((owner, attr))
        buffers = [tree.capture(b) for m in self.roots().values() for b in m.buffers()]
        modes = [(m, m.training) for root in self.roots().values() for m in root.modules()]
        try:
            with scoped_rng(stream_seed(self.pair_seed, 'student-step', item['epoch'], item['step']), cuda=True):
                baseline = PairedCrossScaleHalf._losses(self, item['image_l'], item['label_l'], item['image_u'], item['epoch'], item['step'])
                total, sup, pseudo, extra, logits, stats, view = baseline
                original = pseudo + extra
                scale = float(self.scaler.get_scale())
                grads = torch.autograd.grad(self.scaler.scale(original / 5.0), pars, allow_unused=True)
                index = {id(p): i for i, p in enumerate(pars)}
                norms = {name: self.norm((grads[index[id(p)]] / scale for p in group if grads[index[id(p)]] is not None)) for name, group in self.gradient_groups().items()}
                row = dict(epoch=item['epoch'], step=item['step'], pseudo=float(pseudo.detach()), wsv=float(extra.detach()), supervised_discarded=float(sup.detach()), supervised_added=0.0, selected_ratio=float(stats['selected_ratio']), fvc_drop_fraction=float(self._fvc_drop_fraction), view_state=view, raw_gradient_norms=norms, applied=self.restore_arm == 'restore')
                del baseline, total, sup, pseudo, extra, logits, original
        finally:
            restored = set()
            for b in buffers:
                tree.restore(b, restored)
            for owner, attr, node in targets:
                setattr(owner, attr, tree.restore(node, restored))
            for owner, attr in absent:
                if hasattr(owner, attr):
                    delattr(owner, attr)
            for m, training in modes:
                m.training = training
            checked = set()
            for b in buffers:
                tree.assert_restored(b, b[1], checked)
            for owner, attr, node in targets:
                tree.assert_restored(node, getattr(owner, attr), checked)
        row['declared_state_restored'] = True
        return (grads, row)

    @staticmethod
    def norm(grads):
        value = PairedHardRelations.norm(grads)
        return value if __import__('math').isfinite(value) else None

    def step_relations(self, raw, arm=None, inject_overflow=False):
        assert len(raw) == 5
        arm = arm or self.relation_arm
        assert arm == 'off'
        begin = time.monotonic()
        self.set_relation_train()
        self.optimizer.zero_grad(set_to_none=True)
        self.clear_relation_graph()
        pars = [p for group in self.optimizer.param_groups for p in group['params']]
        index = {id(p): i for i, p in enumerate(pars)}
        rows = []
        for item in raw:
            xl, yl, xu = (item['image_l'], item['label_l'], item['image_u'])
            epoch, step = (item['epoch'], item['step'])
            axis = (epoch + step) % 3
            with scoped_rng(stream_seed(self.pair_seed, 'student-step', epoch, step), cuda=True):
                sup = self._supervised_step(xl, yl)
                self.scaler.scale(sup / 5.0).backward()
                sup_value = float(sup.detach())
                del sup
                self.clear_relation_graph()
                a, b = context_views(xu, axis)
                with torch.no_grad():
                    ta, _ = self.relation_forward(self.teacher, a)
                    tb, tf = self.relation_forward(self.teacher, b)
                    qa = F.avg_pool3d(ta.float().softmax(1)[:, 1:2], 4, 4)
                    qb = F.avg_pool3d(tb.float().softmax(1)[:, 1:2], 4, 4)
                hi, ei, mask, counts = pick_regions(qa, qb, _image_signal(xu, (32, 32, 32)), axis)
                del ta, b
                logits, feature = self.relation_forward(self.student, a)
                ids = ei if arm == 'easy' else hi
                if len(ids) >= 2:
                    rel, parts = relation_loss(feature, tf, logits, tb, ids)
                    energies = dict(feature=float(parts['feature_energy']), output=float(parts['output_energy']))
                else:
                    rel = feature.float().sum() * 0.0 + logits.float().sum() * 0.0
                    energies = dict(feature=None, output=None)
                ce = reliable_class_loss(logits, qa, qb, mask)
                scale = float(self.scaler.get_scale())
                grads = torch.autograd.grad(self.scaler.scale(self.relation_coefficient * rel / 5.0), pars, allow_unused=True, retain_graph=True)
                raw_norms = {name: self.norm((grads[index[id(p)]] / scale for p in group if grads[index[id(p)]] is not None)) for name, group in self.gradient_groups().items()}
                self.scaler.scale(ce / 5.0).backward()
                for p, grad in zip(pars, grads):
                    if grad is not None:
                        applied = grad if arm != 'off' else grad * 0.0
                        if p.grad is None:
                            p.grad = applied.detach().clone()
                        else:
                            p.grad.add_(applied)
                rows.append(dict(step=step, axis=axis, supervised=sup_value, reliable=float(ce.detach()), relation=float(rel.detach()), **counts, reliable_blocks=int(mask.sum()), energies=energies, raw_relation_gradient_norms=raw_norms, applied_relation_gradient_norms=raw_norms if arm != 'off' else {k: 0.0 for k in raw_norms}, student_variance=float(feature.detach().float().flatten(2).var(2, unbiased=False).mean()), teacher_variance=float(tf.detach().float().flatten(2).var(2, unbiased=False).mean())))
            del a, tb, tf, qa, qb, hi, ei, ids, mask, logits, feature, rel, ce, grads
            if len(raw) and 'parts' in locals():
                del parts
            if 'applied' in locals():
                del applied
            self.clear_relation_graph()
        assert all((p.grad is None for name, m in self.roots().items() if name.startswith('teacher') for p in m.parameters()))
        original_rows = []
        for item in raw:
            extra_grads, extra_row = self.original_gradients(item, pars)
            if self.restore_arm == 'restore':
                for p, g in zip(pars, extra_grads):
                    if g is not None:
                        if p.grad is None:
                            p.grad = g.detach().clone()
                        else:
                            p.grad.add_(g)
            original_rows.append(extra_row)
            del extra_grads
        if inject_overflow:
            next((p for p in pars if p.grad is not None)).grad.flatten()[0] = float('inf')
        scale = float(self.scaler.get_scale())
        clips = len(self._fclr_clip_norms)
        self.scaler.unscale_(self.optimizer)
        norms = {name: self.norm((p.grad for p in group)) for name, group in self.gradient_groups().items()}
        gradients_finite = all((bool(torch.isfinite(p.grad).all()) for p in pars if p.grad is not None))
        self.scaler.step(self.optimizer)
        self.scaler.update()
        skipped = float(self.scaler.get_scale()) < scale
        assert skipped == (not gradients_finite)
        assert len(self._fclr_clip_norms) - clips == (0 if skipped else 1)
        if not skipped:
            self._update_teacher(0.999)
            self.relation_updates += 1
        self.optimizer.zero_grad(set_to_none=True)
        self.clear_relation_graph()
        return dict(restore_arm=self.restore_arm, original_unlabeled=original_rows, gradients_finite=gradients_finite, arm=arm, update=self.relation_updates, skipped=skipped, scale_before=scale, scale_after=float(self.scaler.get_scale()), clip_calls=len(self._fclr_clip_norms) - clips, gradient_groups=norms, microbatches=rows, teacher_no_grad=True, seconds=time.monotonic() - begin)
