"""AutoSAMSeg: coarse nnU-Net -> mask prompt -> SAM decoder (no clicks)."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from segment_anything.build_sam3D_cascade import build_sam3d_cascade
from segment_anything.modeling.sam3D_cascade import Sam3DCascade


class AutoSAMSeg(nn.Module):
    """Single-optimizer cascade segmenter with V-Net-like forward API."""

    def __init__(
        self,
        cascade: Sam3DCascade,
        num_classes: int = 2,
        mc_dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        if num_classes not in (1, 2):
            raise ValueError("num_classes must be 1 (sigmoid) or 2 (softmax)")
        self.cascade = cascade
        self.num_classes = int(num_classes)
        self.mc_dropout_p = float(mc_dropout_p)
                                                                                       
        self.gate_mask_prompt = False
        self.uncertainty_keep = 0.85

    @property
    def img_size(self) -> int:
        return int(self.cascade.img_size)

    def set_train_phase(self, phase: str) -> None:
        self.cascade.set_train_phase(phase)

    def set_mc_dropout_p(self, p: float) -> None:
        self.mc_dropout_p = float(p)
        if hasattr(self.cascade, "mc_dropout_p"):
            self.cascade.mc_dropout_p = float(p)

    def _fg_to_logits(self, fg_logits: torch.Tensor) -> torch.Tensor:
        if self.num_classes == 1:
            return fg_logits
        bg = torch.zeros_like(fg_logits)
        return torch.cat([bg, fg_logits], dim=1)

    @staticmethod
    def _entropy_gate_from_coarse_logits(
        coarse_logits: Union[torch.Tensor, list, tuple],
        prompt_size: Tuple[int, int, int],
        keep: float,
    ) -> torch.Tensor:
        """U2: zero high-entropy (uncertain) prompt voxels.

        Keep the lowest-entropy ``keep`` fraction; drop the uncertain tail.
        (Unlike U1-on-pseudo, this does NOT prefer overconfident errors into
        the loss — it only blanks unreliable coarse regions as SAM mask prompt.)
        """
        import torch.nn.functional as F

        if isinstance(coarse_logits, (list, tuple)):
            main = coarse_logits[0]
        else:
            main = coarse_logits
        probs = torch.softmax(main.float(), dim=1)
        u = -torch.sum(probs * torch.log(probs + 1e-6), dim=1, keepdim=True)
        u_lr = F.interpolate(u, size=prompt_size, mode="trilinear", align_corners=False)
        keep = float(max(0.05, min(1.0, keep)))
        flat = u_lr.reshape(u_lr.shape[0], -1)
        q = torch.quantile(flat, keep, dim=1, keepdim=True)
        gate = (flat <= q).float().reshape_as(u_lr)
        return gate

    def forward(
        self,
        images: torch.Tensor,
        return_aux: bool = False,
        detach_coarse_prompt: bool = False,
    ):
        if self.gate_mask_prompt:
            low_res_prompt, coarse_logits = self.cascade.coarse_head(
                images, return_logits=True
            )
            if detach_coarse_prompt:
                low_res_prompt = low_res_prompt.detach()
            gate = self._entropy_gate_from_coarse_logits(
                coarse_logits,
                tuple(low_res_prompt.shape[-3:]),
                self.uncertainty_keep,
            )
            low_res_prompt = low_res_prompt * gate
            low_res_sam, full_res_sam, iou_predictions = self.cascade.forward_sam(
                images=images,
                mask_prompt=low_res_prompt,
            )
            out = {
                "coarse_logits": coarse_logits,
                "mask_prompt": low_res_prompt,
                "low_res_sam": low_res_sam,
                "full_res_sam": full_res_sam,
                "iou_predictions": iou_predictions,
            }
        else:
            out = self.cascade(
                images,
                detach_coarse_prompt=detach_coarse_prompt,
            )

        logits = self._fg_to_logits(out["full_res_sam"])
        if return_aux:
            return logits, out["coarse_logits"]
        return logits

    def predict_proba(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.forward(images, return_aux=False)
        if self.num_classes == 1:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=1)


def build_auto_sam_seg(
    model_type: str = "vit_b_ori",
    sam_checkpoint: Optional[str] = None,
    coarse_checkpoint: Optional[str] = None,
    img_size: int = 128,
    num_classes: int = 2,
    deep_supervision: bool = False,
    device: str = "cuda",
    mc_dropout_p: float = 0.0,
) -> AutoSAMSeg:
    cascade = build_sam3d_cascade(
        model_type=model_type,
        sam_checkpoint=sam_checkpoint,
        coarse_checkpoint=coarse_checkpoint,
        img_size=img_size,
        deep_supervision=deep_supervision,
        device=device,
        mc_dropout_p=mc_dropout_p,
    )
    return AutoSAMSeg(cascade=cascade, num_classes=num_classes, mc_dropout_p=mc_dropout_p)


def load_auto_sam_checkpoint(
    model: AutoSAMSeg,
    checkpoint_path: str,
    device: str = "cpu",
    strict: bool = True,
) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "student" in ckpt:
            state = ckpt["student"]
        elif "cascade_state_dict" in ckpt:
            state = ckpt["cascade_state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt
    model.load_state_dict(state, strict=strict)
