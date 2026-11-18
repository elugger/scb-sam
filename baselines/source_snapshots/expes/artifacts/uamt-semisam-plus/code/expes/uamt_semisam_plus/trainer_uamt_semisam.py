"""UA-MT + SemiSAM+: uncertainty-aware MT specialist + frozen SAM-Med3D."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict

import torch
import torch.nn.functional as F

from expes.common.ramps import sigmoid_rampup
from expes.mt_semisam_plus.trainer_mt_semisam import MTSemiSAMTrainer


class UAMTSemiSAMTrainer(MTSemiSAMTrainer):
    method_tag = "UA-MT-SemiSAM+"

    @torch.no_grad()
    def _teacher_uncertainty(self, unlabeled: torch.Tensor, t_samples: int) -> torch.Tensor:
        """Epistemic uncertainty via T noisy teacher forwards (online mean)."""
        mean = None
        for _ in range(t_samples):
            pred = torch.softmax(self.teacher(self._add_noise(unlabeled)), dim=1)
            mean = pred if mean is None else mean + pred
            del pred
        assert mean is not None
        mean = mean / float(t_samples)
        return -torch.sum(mean * torch.log(mean + 1e-6), dim=1, keepdim=True)

    def _specialist_consistency_loss(
        self,
        outputs: torch.Tensor,
        outputs_soft: torch.Tensor,
        ema_output: torch.Tensor,
        ema_output_soft: torch.Tensor,
        unlabeled: torch.Tensor,
        labeled_bs: int,
        max_iterations: int,
    ) -> torch.Tensor:
        """UA-MT: softmax-MSE masked by teacher entropy threshold."""
        del outputs_soft, ema_output_soft
        scfg = self.cfg.get("semi", {})
        t_samples = int(scfg.get("uamt_mc_samples", 8))

        student_soft = F.softmax(outputs[labeled_bs:], dim=1)
        teacher_soft = F.softmax(ema_output, dim=1)
        consistency_dist = (student_soft - teacher_soft) ** 2

        uncertainty = self._teacher_uncertainty(unlabeled, t_samples)
        thr = (0.75 + 0.25 * sigmoid_rampup(self.global_step, max_iterations)) * math.log(2.0)
        mask = (uncertainty < thr).float()
        return torch.sum(mask * consistency_dist) / (2.0 * torch.sum(mask) + 1e-16)

    def _final_metrics_extra(self) -> Dict[str, Any]:
        return {
            "uamt_mc_samples": int(self.cfg.get("semi", {}).get("uamt_mc_samples", 8)),
        }

    def run(self) -> Dict[str, float]:
        logging.info(
            "UA-MT specialist consistency (mc_samples=%d)",
            int(self.cfg.get("semi", {}).get("uamt_mc_samples", 8)),
        )
        return super().run()
