import json
import torch
from expes.semi05_gate.paired_wsv_control_20260908 import PairedWSVControl
from expes.semi05_gate.paired_rng_data_20260908 import scoped_rng, stream_seed, SeededCases

class PairedFVCControl(PairedWSVControl):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.fvc_probability = float(cfg['semi']['fvc_probability'])
        assert 0 < self.fvc_probability < 1
        self._fvc_seed = None
        self._fvc_capture = False
        self._fvc_feature = None

    def _perturbed_forward(self, image, seed):
        calls = []

        def perturb(_module, _inputs, embedding):
            assert torch.is_tensor(embedding) and embedding.ndim == 5
            generator = torch.Generator(device=embedding.device).manual_seed(seed)
            keep = torch.rand((embedding.shape[0], embedding.shape[1], 1, 1, 1), device=embedding.device, generator=generator) >= self.fvc_probability
            output = embedding * keep.to(embedding.dtype) / (1 - self.fvc_probability)
            calls.append(float((~keep).float().mean()))
            if self._fvc_capture:
                self._fvc_feature = output
            return output
        handle = self.student.cascade.image_encoder.register_forward_hook(perturb)
        try:
            result = super()._forward_main_logits(self.student, image)
        finally:
            handle.remove()
        assert len(calls) == 1, 'Expected exactly one image encoder forward'
        self._fvc_drop_fraction = calls[0]
        return result

    def _forward_main_logits(self, model, image):
        if model is self.student and self._fvc_seed is not None:
            return self._perturbed_forward(image, self._fvc_seed)
        return super()._forward_main_logits(model, image)

    def _losses(self, image_l, label_l, image_u, epoch, step):
        self._fvc_seed = stream_seed(self.pair_seed, 'fvc-channel', epoch, step)
        try:
            result = super()._losses(image_l, label_l, image_u, epoch, step)
        finally:
            self._fvc_seed = None
        if self._fvc_capture:
            gradient = torch.autograd.grad(result[3], self._fvc_feature, retain_graph=True)[0]
            self._fvc_feature_gradient_norm = float(gradient.float().norm())
            assert self._fvc_feature_gradient_norm > 0 and torch.isfinite(gradient).all()
            self._fvc_feature = None
        self._append('fvc_steps.jsonl', dict(epoch=epoch, step=step, dropout_fraction=self._fvc_drop_fraction, dropout_seed=stream_seed(self.pair_seed, 'fvc-channel', epoch, step)))
        return result

    def preflight(self, init_from, expected_floor):
        self._fvc_capture = True
        result = super().preflight(init_from, expected_floor)
        self._fvc_capture = False
        self.student.eval()
        records = []
        ds = SeededCases(self.lab_set, self.pair_seed, 'lab')
        with torch.no_grad():
            for index in range(min(4, len(ds))):
                batch = ds[0, 0, index]
                image, label = self._prepare_batch({k: v.unsqueeze(0) for k, v in batch.items() if torch.is_tensor(v)})
                with scoped_rng(stream_seed(self.pair_seed, 'fvc-audit-view', index), cuda=True):
                    view, state = self._student_unlabeled_view(image, 0.05)
                base = self._restore_student_logits(super()._forward_main_logits(self.student, view), state)
                pert = self._restore_student_logits(self._perturbed_forward(view, stream_seed(self.pair_seed, 'fvc-audit-drop', index)), state)
                clean_again = self._restore_student_logits(self._forward_main_logits(self.student, view), state)
                assert torch.equal(base, clean_again), 'Perturbation leaked into clean inference'
                fg = label[:, 0] > 0
                pb, pp = (base.softmax(1)[:, 1], pert.softmax(1)[:, 1])
                mb, mp = (base.argmax(1) > 0, pert.argmax(1) > 0)
                dice = lambda m: float(2 * (m & fg).sum() / (m.sum() + fg.sum()).clamp_min(1))
                records.append(dict(requested_index=index, base_dice=dice(mb), perturbed_dice=dice(mp), fg_probability_change=float((pp - pb).abs()[fg].mean()), fg_prediction_retention=float((mp & mb & fg).sum() / (mb & fg).sum().clamp_min(1)), dropped_channels_fraction=self._fvc_drop_fraction))
        mean = lambda k: sum((r[k] for r in records)) / len(records)
        mechanism_pass = 1e-06 < mean('fg_probability_change') < 0.15 and mean('fg_prediction_retention') >= 0.5 and (mean('perturbed_dice') >= mean('base_dice') - 0.1)
        result.update(control='FVC-R1', paired_base='C0', probability=self.fvc_probability, feature_location='SAM image encoder output', mechanism_status='PASS' if mechanism_pass else 'FAIL_CONFIG', feature_consistency_gradient_norm=self._fvc_feature_gradient_norm, clean_inference_bitwise_equal=True, labeled_perturbation_audit=records)
        (self.output_dir / 'preflight.json').write_text(json.dumps(result, indent=2))
        if not mechanism_pass:
            raise RuntimeError('FVC perturbation gate failed; inspect audit before adjusting')
        return result
