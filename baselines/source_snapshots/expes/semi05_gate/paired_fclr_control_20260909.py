"""C2: historical FCLR mechanism on the exact C0 paired training loop."""
import json
import os
import torch
from expes.semi05_gate.paired_wsv_control_20260908 import PairedWSVControl
from expes.semi05_gate.final_logit_contextual_residual_wsv_ucp import (
    FinalLogitContextualResidualWSVUCPTrainer as FCLR,
    FinalLogitContextualResidualHead,
)
from expes.semi05_gate.paired_rng_data_20260908 import scoped_rng, stream_seed


class PairedFCLRControl(PairedWSVControl, FCLR):
    def __init__(self, cfg):
        super().__init__(cfg)
        # Recreate only the new heads under their own RNG. Base model loading,
        # data streams and per-step stochastic forwards remain those of C0.
        with scoped_rng(stream_seed(self.pair_seed, "fclr-head-init"), cuda=True):
            kwargs = dict(coarse_channels=128,
                          sam_channels=int(self.student.cascade.prompt_encoder.embed_dim),
                          projection_channels=8, hidden_channels=16, delta_max=.5)
            self.student_fclr_head = FinalLogitContextualResidualHead(**kwargs).to(self.device)
            self.teacher_fclr_head = FinalLogitContextualResidualHead(**kwargs).to(self.device)
        self.teacher_fclr_head.requires_grad_(False)
        self.teacher_fclr_head.eval()
        self._update_teacher_fclr(0.)
        self._register_fclr_gradient_hooks()

    def _evaluate(self):
        # Paired evaluator supplies per-case, full-precision records. Do not
        # invoke the old identity check which draws from an unrelated loader.
        self._flush_fclr_stats()
        self.student_fclr_head.eval()
        return PairedWSVControl._evaluate(self)

    def _save_ucp(self, name, epoch):
        target = self.checkpoint_dir / name
        temp = self.checkpoint_dir / (name + ".partial")
        torch.save(dict(student=self._compact_state(self.student),
                        student_fclr_head=self._compact_state(self.student_fclr_head),
                        teacher_fclr_head=self._compact_state(self.teacher_fclr_head),
                        fclr_format_version=1, best_dice=self.best_dice,
                        best_epoch=self.best_epoch, saved_epoch=epoch, cfg=self.cfg,
                        exact_resume_supported=False), temp)
        os.replace(temp, target)
        link = self.output_dir / name
        if link.is_symlink():
            if link.resolve() != target:
                raise RuntimeError("Unexpected checkpoint symlink")
        elif link.exists():
            raise RuntimeError("Refusing checkpoint overwrite")
        else:
            link.symlink_to(target)
        return target

    def preflight(self, init_from, expected_floor):
        result = super().preflight(init_from, expected_floor)
        # Zero output blocks upstream head gradients on the very first backward;
        # two actual updates are required before judging all head paths.
        lab, unlab = self._loaders(0)
        image_l, label_l = self._prepare_batch(next(lab))
        image_u, _ = self._prepare_batch(next(iter(unlab)))
        self.student.eval()
        with torch.no_grad(), torch.cuda.amp.autocast():
            base = self.student(image_u)
            refined, _, _, delta = self._forward_with_fclr(
                self.student, self.student_fclr_head, image_u)
        difference = float((base.float()-refined.float()).abs().max())
        assert difference <= 1e-5 and float(delta.abs().max()) == 0.
        assert torch.equal(base.argmax(1), refined.argmax(1))
        del base, refined, delta
        losses = []
        for step in range(2):
            self.student.train()
            self.teacher.eval()
            self.optimizer.zero_grad(set_to_none=True)
            loss, *_ = self._losses(image_l, label_l, image_u, 0, step)
            assert torch.isfinite(loss)
            self.scaler.scale(loss).backward()
            parameters = list(self.student.parameters()) + list(self.student_fclr_head.parameters())
            assert all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None)
            scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            assert self.scaler.get_scale() >= scale
            self._update_teacher(.999)
            losses.append(float(loss.detach()))
        norms = {key: max(values) for key, values in self._fclr_grad_stats.items()}
        self._flush_fclr_stats()
        assert all(v > 0 for v in norms.values())
        result.update(control="C2", paired_base="C0", fclr_identity_max_diff=difference,
                      fclr_gradient_norms=norms, fclr_two_step_losses=losses,
                      peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3)
        (self.output_dir / "preflight.json").write_text(json.dumps(result, indent=2))
        return result
