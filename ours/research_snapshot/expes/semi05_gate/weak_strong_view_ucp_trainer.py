from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn.functional as F
from expes.semi05_gate.uncertainty_curriculum_trainer import UncertaintyCurriculumTrainer

class WeakStrongViewUCPTrainer(UncertaintyCurriculumTrainer):

    def _student_unlabeled_view(self, image: torch.Tensor, noise_std: float) -> Tuple[torch.Tensor, object]:
        scfg = self.cfg['semi']
        mode = str(scfg.get('ws_flip_mode', 'single'))
        spatial_dims: List[int] = []
        if mode == 'single':
            axis = int(torch.randint(0, 3, (1,), device=image.device).item())
            spatial_dims = [2 + axis]
        elif mode == 'independent':
            spatial_dims = [dim for dim in (2, 3, 4) if bool((torch.rand((), device=image.device) < 0.5).item())]
        elif mode != 'none':
            raise ValueError(f'Unsupported ws_flip_mode: {mode}')
        strong = torch.flip(image, dims=spatial_dims) if spatial_dims else image
        scale_jitter = float(scfg.get('ws_intensity_scale', 0.0))
        shift_jitter = float(scfg.get('ws_intensity_shift', 0.0))
        if scale_jitter > 0:
            shape = (strong.shape[0], 1, 1, 1, 1)
            scale = 1.0 + (2.0 * torch.rand(shape, device=strong.device) - 1.0) * scale_jitter
            strong = strong * scale
        if shift_jitter > 0:
            shape = (strong.shape[0], 1, 1, 1, 1)
            shift = (2.0 * torch.rand(shape, device=strong.device) - 1.0) * shift_jitter
            strong = strong + shift
        strong = self._perturb_input(strong, noise_std)
        return (strong, tuple(spatial_dims))

    def _restore_student_logits(self, logits: torch.Tensor, view_state: object) -> torch.Tensor:
        spatial_dims = tuple(view_state or ())
        return torch.flip(logits, dims=spatial_dims) if spatial_dims else logits

    def _extra_unlabeled_loss(self, student_logits: torch.Tensor, epoch: int) -> torch.Tensor:
        weight = float(self.cfg['semi'].get('ws_consistency_weight', 0.0))
        if weight <= 0:
            return student_logits.new_tensor(0.0)
        teacher_prob = torch.softmax(self._ucp_teacher_logits.float(), dim=1)
        student_prob = torch.softmax(student_logits.float(), dim=1)
        per_voxel = F.mse_loss(student_prob, teacher_prob, reduction='none').mean(dim=1, keepdim=True)
        mask = self._ucp_reliable_mask.float()
        denominator = mask.sum().clamp_min(1.0)
        ramp = min(1.0, float(epoch + 1) / float(max(1, int(self.cfg['semi'].get('ucp_pseudo_ramp_epochs', 8)))))
        return weight * ramp * (per_voxel * mask).sum() / denominator
