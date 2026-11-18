"""Original AutoSAM data path with fixed spatial displacement objectives.

All arms calculate the raw relation gradient. Off discards it, easy changes
only the matched query positions. Six-root archives are research snapshots.
"""
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import time
import torch
import torch.nn.functional as F
from expes.semi05_gate.paired_zea_cross_scale_half_20260911 import PairedCrossScaleHalf
from expes.semi05_gate.fcd_cac_fg_20260928 import context_views
from expes.semi05_gate.fcd_dycon_r2_losses_20260927 import _image_signal
from expes.semi05_gate.fcd_hard_spatial_relations_20261004 import pick_regions, reliable_class_loss
from expes.semi05_gate.fcd_deterministic_losses_r2_20261005 import relation_loss
from expes.semi05_gate.paired_rng_data_20260908 import scoped_rng, stream_seed


def tensor_state_digest(state):
    h = hashlib.sha256()
    for root, values in sorted(state.items()):
        for name, value in sorted(values.items()):
            h.update((root + '.' + name + str(value.dtype) + str(tuple(value.shape))).encode())
            h.update(value.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class PairedHardRelations(PairedCrossScaleHalf):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.relation_arm = cfg['semi']['relation_arm']
        assert self.relation_arm in ('hard', 'off', 'easy')
        self.relation_coefficient = cfg['semi']['relation_coefficient']
        self.relation_updates = 0
        self.relation_storage_root = Path(cfg['semi']['relation_storage_root']).resolve()

    def roots(self):
        return dict(student=self.student, head=self.student_fclr_head,
                    adapter=self.student_zea_adapter, teacher=self.teacher,
                    teacher_head=self.teacher_fclr_head, teacher_adapter=self.teacher_zea_adapter)

    def initialize_relation(self, path):
        self._load(path)
        self._update_teacher(0.)
        self._build_ucp_optimizer()

    def clear_relation_graph(self):
        self._fclr_last_student_delta = None
        self._xsa_context.clear()

    def relation_forward(self, model, image):
        captured = []
        handle = model.cascade.mask_decoder.output_upscaling.register_forward_hook(
            lambda module, inputs, value: captured.append(value))
        try:
            with torch.cuda.amp.autocast():
                logits = self._forward_main_logits(model, image)
        finally:
            handle.remove()
        assert len(captured) == 1
        return logits, captured[0]

    @staticmethod
    def norm(grads):
        return sum(float(g.detach().float().square().sum()) for g in grads if g is not None) ** .5

    def gradient_groups(self):
        return dict(encoder=list(self.student.cascade.image_encoder.parameters()),
                    decoder=list(self.student.cascade.mask_decoder.parameters()),
                    prompt=list(self.student.cascade.prompt_encoder.parameters()),
                    coarse=list(self.student.cascade.coarse_head.parameters()),
                    head=list(self.student_fclr_head.parameters()),
                    adapter=list(self.student_zea_adapter.parameters()))

    def set_relation_train(self):
        for name, module in self.roots().items():
            module.train(not name.startswith('teacher'))

    def step_relations(self, raw, arm=None):
        assert len(raw) == 5
        arm = arm or self.relation_arm
        assert arm in ('hard', 'easy', 'off')
        begin = time.monotonic()
        self.set_relation_train()
        self.optimizer.zero_grad(set_to_none=True)
        self.clear_relation_graph()
        pars = [p for group in self.optimizer.param_groups for p in group['params']]
        index = {id(p): i for i, p in enumerate(pars)}
        rows = []
        for item in raw:
            xl, yl, xu = item['image_l'], item['label_l'], item['image_u']
            epoch, step = item['epoch'], item['step']
            axis = (epoch + step) % 3
            with scoped_rng(stream_seed(self.pair_seed, 'student-step', epoch, step), cuda=True):
                sup = self._supervised_step(xl, yl)
                self.scaler.scale(sup / 5.).backward()
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
                # Empty support is recorded; do not pick a different case/axis.
                if len(ids) >= 2:
                    rel, parts = relation_loss(feature, tf, logits, tb, ids)
                    energies = dict(feature=float(parts['feature_energy']), output=float(parts['output_energy']))
                else:
                    rel = feature.float().sum() * 0. + logits.float().sum() * 0.
                    energies = dict(feature=None, output=None)
                ce = reliable_class_loss(logits, qa, qb, mask)
                scale = float(self.scaler.get_scale())
                grads = torch.autograd.grad(self.scaler.scale(self.relation_coefficient * rel / 5.),
                                            pars, allow_unused=True, retain_graph=True)
                raw_norms = {name: self.norm(grads[index[id(p)]] / scale
                             for p in group if grads[index[id(p)]] is not None)
                             for name, group in self.gradient_groups().items()}
                self.scaler.scale(ce / 5.).backward()
                for p, grad in zip(pars, grads):
                    if grad is not None:
                        applied = grad if arm != 'off' else grad * 0.
                        if p.grad is None:
                            p.grad = applied.detach().clone()
                        else:
                            p.grad.add_(applied)
                rows.append(dict(step=step, axis=axis, supervised=sup_value,
                    reliable=float(ce.detach()), relation=float(rel.detach()), **counts,
                    reliable_blocks=int(mask.sum()), energies=energies,
                    raw_relation_gradient_norms=raw_norms,
                    applied_relation_gradient_norms=raw_norms if arm != 'off' else {k: 0. for k in raw_norms},
                    student_variance=float(feature.detach().float().flatten(2).var(2, unbiased=False).mean()),
                    teacher_variance=float(tf.detach().float().flatten(2).var(2, unbiased=False).mean())))
            del a, tb, tf, qa, qb, hi, ei, ids, mask, logits, feature, rel, ce, grads
            if len(raw) and 'parts' in locals():
                del parts
            if 'applied' in locals():
                del applied
            self.clear_relation_graph()
        assert all(p.grad is None for name, m in self.roots().items()
                   if name.startswith('teacher') for p in m.parameters())
        scale = float(self.scaler.get_scale())
        clips = len(self._fclr_clip_norms)
        self.scaler.unscale_(self.optimizer)
        norms = {name: self.norm(p.grad for p in group) for name, group in self.gradient_groups().items()}
        assert all(torch.isfinite(p.grad).all() for p in pars if p.grad is not None)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        skipped = float(self.scaler.get_scale()) < scale
        assert len(self._fclr_clip_norms) - clips == (0 if skipped else 1)
        if not skipped:
            self._update_teacher(.999)
            self.relation_updates += 1
        self.optimizer.zero_grad(set_to_none=True)
        self.clear_relation_graph()
        return dict(arm=arm, update=self.relation_updates, skipped=skipped,
                    scale_before=scale, scale_after=float(self.scaler.get_scale()),
                    clip_calls=len(self._fclr_clip_norms) - clips,
                    gradient_groups=norms, microbatches=rows,
                    teacher_no_grad=True, seconds=time.monotonic() - begin)

    def relation_state(self):
        return {name: self._compact_state(module) for name, module in self.roots().items()}

    def save_relation(self, name, epoch):
        assert name in ('best.pth', 'final.pth', 'qualification.pth')
        target = self.checkpoint_dir / name
        temporary = self.checkpoint_dir / (name + '.partial')
        assert target.parent.resolve().is_relative_to(self.relation_storage_root)
        with (self.relation_storage_root / 'save.lock').open('a+') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            assert not temporary.exists()
            state = self.relation_state()
            unique = {v.storage().data_ptr(): v.storage().nbytes() for values in state.values() for v in values.values()}
            estimate = sum(unique.values()) + 8 * 1024**2
            assert shutil.disk_usage(self.relation_storage_root).free > estimate + 1024**3
            payload = dict(roots=state, cfg=self.cfg, saved_epoch=epoch, arm=self.relation_arm,
                           coefficient=self.relation_coefficient, updates=self.relation_updates,
                           best_dice=self.best_dice, best_epoch=self.best_epoch,
                           exact_resume_supported=False)
            with temporary.open('wb') as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        link = self.output_dir / name
        if link.is_symlink():
            assert link.resolve() == target.resolve()
        else:
            assert not link.exists()
            link.symlink_to(target.resolve())
        return target

    def load_relation(self, path):
        payload = torch.load(str(path), map_location='cpu')
        assert set(payload['roots']) == set(self.roots())
        assert payload['coefficient'] == self.relation_coefficient
        for name, module in self.roots().items():
            module.load_state_dict(payload['roots'][name], strict=True)
        self.relation_updates = payload['updates']
        return {k: payload[k] for k in ('saved_epoch', 'updates', 'arm', 'exact_resume_supported')}
