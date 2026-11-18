import copy
import json
import os
import torch
import torch.nn as nn
from expes.semi05_gate.paired_ghr_fvc_20260909 import PairedGHRFVC
from expes.semi05_gate.paired_rng_data_20260908 import SeededCases, scoped_rng, stream_seed

class ZeroEmbeddingAdapter(nn.Module):

    def __init__(self, channels=384, bottleneck=16):
        super().__init__()
        self.down = nn.Conv3d(channels, bottleneck, 1)
        self.activation = nn.GELU()
        self.up = nn.Conv3d(bottleneck, channels, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def residual(self, embedding):
        return self.up(self.activation(self.down(embedding)))

    def forward(self, embedding):
        return embedding + self.residual(embedding)

class PairedZEASAM(PairedGHRFVC):

    def __init__(self, cfg):
        super().__init__(cfg)
        scfg = cfg['semi']
        self.zea_bottleneck = int(scfg.get('zea_bottleneck', 16))
        self.zea_lr = float(scfg.get('zea_lr', 0.0001))
        assert self.zea_bottleneck == 16 and self.zea_lr == 0.0001
        with scoped_rng(stream_seed(self.pair_seed, 'zea-adapter-init'), cuda=True):
            self.student_zea_adapter = ZeroEmbeddingAdapter(channels=int(self.student.cascade.prompt_encoder.embed_dim), bottleneck=self.zea_bottleneck).to(self.device)
        self.teacher_zea_adapter = copy.deepcopy(self.student_zea_adapter).requires_grad_(False).eval()
        assert sum((p.numel() for p in self.student_zea_adapter.parameters())) == 12688
        self._zea_initial_state = self._compact_state(self.student_zea_adapter)
        self._zea_active = True
        self._zea_gradient_stats = {name: [] for name, _ in self.student_zea_adapter.named_parameters()}

        def student_hook(_module, _inputs, embedding):
            return self.student_zea_adapter(embedding) if self._zea_active else embedding

        def teacher_hook(_module, _inputs, embedding):
            return self.teacher_zea_adapter(embedding) if self._zea_active else embedding
        self._zea_student_hook = self.student.cascade.image_encoder.register_forward_hook(student_hook)
        self._zea_teacher_hook = self.teacher.cascade.image_encoder.register_forward_hook(teacher_hook)
        for name, parameter in self.student_zea_adapter.named_parameters():

            def record(gradient, name=name):
                values = self._zea_gradient_stats[name]
                if len(values) < 16:
                    scale = max(1.0, float(self.scaler.get_scale())) if hasattr(self, 'scaler') else 1.0
                    values.append(float(gradient.detach().float().norm()) / scale)
                return gradient
            parameter.register_hook(record)

    @torch.no_grad()
    def _update_teacher_zea(self, alpha):
        for teacher, student in zip(self.teacher_zea_adapter.parameters(), self.student_zea_adapter.parameters()):
            teacher.data.mul_(alpha).add_(student.data, alpha=1 - alpha)
        for teacher, student in zip(self.teacher_zea_adapter.buffers(), self.student_zea_adapter.buffers()):
            teacher.copy_(student)

    def _update_teacher(self, alpha):
        super()._update_teacher(alpha)
        if hasattr(self, 'student_zea_adapter'):
            self._update_teacher_zea(alpha)

    def _build_ucp_optimizer(self):
        super()._build_ucp_optimizer()
        self.optimizer.add_param_group(dict(params=self.student_zea_adapter.parameters(), lr=self.zea_lr, weight_decay=0.0))
        ids = [id(p) for group in self.optimizer.param_groups for p in group['params']]
        adapter_ids = {id(p) for p in self.student_zea_adapter.parameters()}
        assert len(ids) == len(set(ids)) and adapter_ids.issubset(set(ids))
        assert not any((p.requires_grad for p in self.teacher_zea_adapter.parameters()))

    def _load(self, path):
        super()._load(path)
        payload = torch.load(str(path), map_location='cpu')
        if isinstance(payload, dict) and 'student_zea_adapter' in payload:
            self.student_zea_adapter.load_state_dict(payload['student_zea_adapter'])
            if 'teacher_zea_adapter' in payload:
                self.teacher_zea_adapter.load_state_dict(payload['teacher_zea_adapter'])
            else:
                self._update_teacher_zea(0.0)
        else:
            self.student_zea_adapter.load_state_dict(self._zea_initial_state)
            self._update_teacher_zea(0.0)
        del payload

    def _save_ucp(self, name, epoch):
        target = self.checkpoint_dir / name
        temp = self.checkpoint_dir / (name + '.partial')
        torch.save(dict(student=self._compact_state(self.student), student_fclr_head=self._compact_state(self.student_fclr_head), teacher_fclr_head=self._compact_state(self.teacher_fclr_head), student_zea_adapter=self._compact_state(self.student_zea_adapter), teacher_zea_adapter=self._compact_state(self.teacher_zea_adapter), zea_format_version=1, best_dice=self.best_dice, best_epoch=self.best_epoch, saved_epoch=epoch, cfg=self.cfg, exact_resume_supported=False), temp)
        os.replace(temp, target)
        link = self.output_dir / name
        if link.is_symlink():
            if link.resolve() != target:
                raise RuntimeError('Unexpected ZEA checkpoint symlink')
        elif link.exists():
            raise RuntimeError('Refusing ZEA checkpoint overwrite')
        else:
            link.symlink_to(target)
        return target

    def preflight(self, init_from, expected_floor):
        self._load(init_from)
        data = SeededCases(self.lab_set, self.pair_seed, 'zea-identity')
        batch = data[0, 0, 0]
        image, _ = self._prepare_batch({k: v.unsqueeze(0) for k, v in batch.items() if torch.is_tensor(v)})
        self.student.eval()
        self.student_fclr_head.eval()
        self.student_zea_adapter.eval()
        with torch.no_grad(), torch.cuda.amp.autocast():
            active = self._forward_main_logits(self.student, image)
        self._zea_active = False
        try:
            with torch.no_grad(), torch.cuda.amp.autocast():
                disabled = self._forward_main_logits(self.student, image)
        finally:
            self._zea_active = True
        identity_difference = float((active.float() - disabled.float()).abs().max())
        assert torch.equal(active, disabled) and identity_difference == 0.0
        result = super().preflight(init_from, expected_floor)
        norms = {name: max(values) if values else 0.0 for name, values in self._zea_gradient_stats.items()}
        assert all((value > 0 for value in norms.values()))
        up_norm = float(self.student_zea_adapter.up.weight.detach().float().norm())
        assert up_norm > 0
        restored = ZeroEmbeddingAdapter(384, self.zea_bottleneck).to(self.device)
        restored.load_state_dict(self.student_zea_adapter.state_dict())
        assert all((torch.equal(a, b) for a, b in zip(restored.state_dict().values(), self.student_zea_adapter.state_dict().values())))
        ema_distance = sum((float((a - b).float().pow(2).sum()) for a, b in zip(self.student_zea_adapter.parameters(), self.teacher_zea_adapter.parameters()))) ** 0.5
        result.update(control='ZEA-SAM-R1', paired_base='GHR-FVC-R1', adapter_location='ImageEncoderViT3D neck output, before FVC perturbation', adapter='384->16->384/GELU residual', adapter_parameters=12688, adapter_lr=self.zea_lr, initial_logit_max_difference=identity_difference, initial_logits_bitwise_equal=True, adapter_gradient_norms=norms, adapter_up_weight_norm_after_preflight=up_norm, adapter_ema_distance=ema_distance, adapter_roundtrip_bitwise_equal=True, adapter_in_test=True, unlabeled_gt_extra=False, zea_status='PASS')
        (self.output_dir / 'preflight.json').write_text(json.dumps(result, indent=2))
        return result
