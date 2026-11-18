import copy
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from expes.semi05_gate.paired_hr_fvc_20260909 import PairedHRFVC
from expes.semi05_gate.paired_hr_fclr_20260909 import HighResolutionResidualHead
from expes.semi05_gate.paired_rng_data_20260908 import scoped_rng, stream_seed

class GatedHighResolutionResidualHead(HighResolutionResidualHead):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gate = nn.Conv3d(kwargs.get('hidden_channels', 16) + 1, 1, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.last_gate_stats = None

    def forward(self, coarse_feature, sam_embedding, low_res_logit):
        assert tuple(coarse_feature.shape[-3:]) == (64, 64, 64)
        assert tuple(low_res_logit.shape[-3:]) == (32, 32, 32)
        size = (64, 64, 64)
        low = F.interpolate(low_res_logit.detach().float(), size, mode='trilinear', align_corners=False)
        coarse = self.coarse_projection(coarse_feature.detach())
        sam = self.sam_projection(sam_embedding.detach())
        sam = F.interpolate(sam, size, mode='trilinear', align_corners=False)
        hidden = self.mix(torch.cat([coarse, sam, low], dim=1))
        gate = 2.0 * torch.sigmoid(self.gate(torch.cat([hidden, low.sigmoid()], 1)).float())
        activation = self.out(hidden)
        delta = (self.delta_max * torch.tanh(activation * gate.to(activation.dtype))).float()
        if self.training:
            with torch.no_grad():
                self.last_gate_stats = dict(mean=float(gate.mean()), minimum=float(gate.min()), maximum=float(gate.max()), std=float(gate.std()), delta_max=float(delta.abs().max()))
        return delta

class PairedGHRFVC(PairedHRFVC):

    def __init__(self, cfg):
        super().__init__(cfg)
        previous = self.student_fclr_head
        with scoped_rng(stream_seed(self.pair_seed, 'ghr-gate-init'), cuda=True):
            head = GatedHighResolutionResidualHead(coarse_channels=64, sam_channels=int(self.student.cascade.prompt_encoder.embed_dim), projection_channels=8, hidden_channels=16, delta_max=0.5).to(self.device)
        missing, unexpected = head.load_state_dict(previous.state_dict(), strict=False)
        assert set(missing) == {'gate.weight', 'gate.bias'} and (not unexpected)
        assert all((torch.equal(v, head.state_dict()[k]) for k, v in previous.state_dict().items()))
        self.student_fclr_head = head
        self.teacher_fclr_head = copy.deepcopy(head).requires_grad_(False).eval()
        self._register_fclr_gradient_hooks()

    def _register_fclr_gradient_hooks(self):
        super()._register_fclr_gradient_hooks()
        if not hasattr(self.student_fclr_head, 'gate'):
            return
        for name, parameter in self.student_fclr_head.gate.named_parameters():
            key = 'gate_' + name
            self._fclr_grad_stats[key] = []

            def record(gradient, key=key):
                values = self._fclr_grad_stats[key]
                if len(values) < 16:
                    scale = max(1.0, float(self.scaler.get_scale())) if hasattr(self, 'scaler') else 1.0
                    values.append(float(gradient.detach().float().norm()) / scale)
                return gradient
            parameter.register_hook(record)

    def _evaluate(self):
        head = self.student_fclr_head
        if hasattr(head, 'gate') and head.last_gate_stats is not None:
            self._append('gate_epochs.jsonl', dict(last_training_forward=head.last_gate_stats, gate_weight_norm=float(head.gate.weight.detach().float().norm()), gate_bias=float(head.gate.bias.detach().float().item())))
            head.last_gate_stats = None
        return super()._evaluate()

    def preflight(self, init_from, expected_floor):
        result = super().preflight(init_from, expected_floor)
        assert result['fclr_gradient_norms']['gate_weight'] > 0
        assert result['fclr_gradient_norms']['gate_bias'] > 0
        result.update(control='GHR-FVC-R1', paired_base='HR-FVC-R1', gate_initial_value=1.0, gate_range=[0.0, 2.0], residual_cap=0.5, gate_parameters=sum((p.numel() for p in self.student_fclr_head.gate.parameters())), gate_extra_loss=False, gate_status='PASS')
        (self.output_dir / 'preflight.json').write_text(json.dumps(result, indent=2))
        return result
