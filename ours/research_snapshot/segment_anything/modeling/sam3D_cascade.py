"""SAM-Med3D cascade: coarse U-Net mask prompt + frozen/unfrozen SAM refinement."""

from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .coarse_nnunet3d import CoarseNNUNet3D
from .sam3D import Sam3D

TrainPhase = Literal["coarse_only", "sam_refine", "e2e"]


class Sam3DCascade(nn.Module):
    """
    Fully automatic cascade model:
      image -> coarse U-Net -> low-res mask prompt -> SAM prompt encoder -> SAM mask decoder
    No interactive point/box prompts are used.
    """

    def __init__(
        self,
        sam: Sam3D,
        coarse_head: Optional[CoarseNNUNet3D] = None,
        img_size: int = 128,
    ) -> None:
        super().__init__()
        self.sam = sam
        self.img_size = img_size
        mask_prompt_size = img_size // 4
        self.coarse_head = coarse_head or CoarseNNUNet3D(
            mask_prompt_size=(mask_prompt_size, mask_prompt_size, mask_prompt_size)
        )

    @property
    def image_encoder(self) -> nn.Module:
        return self.sam.image_encoder

    @property
    def prompt_encoder(self) -> nn.Module:
        return self.sam.prompt_encoder

    @property
    def mask_decoder(self) -> nn.Module:
        return self.sam.mask_decoder

    def set_train_phase(self, phase: TrainPhase) -> None:
        """Configure which submodules are trainable for staged optimization."""
        if phase == "coarse_only":
            for p in self.sam.parameters():
                p.requires_grad = False
            for p in self.coarse_head.parameters():
                p.requires_grad = True
        elif phase == "sam_refine":
            for p in self.coarse_head.parameters():
                p.requires_grad = False
            for p in self.sam.parameters():
                p.requires_grad = True
        elif phase == "e2e":
            for p in self.parameters():
                p.requires_grad = True
        else:
            raise ValueError(f"Unknown train phase: {phase}")

    def forward_sam(
        self,
        images: torch.Tensor,
        mask_prompt: torch.Tensor,
        multimask_output: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        SAM refinement given a mask prompt (no point/box prompts).

        Returns:
            low_res_masks: decoder logits at 4x image-embedding resolution
            full_res_masks: trilinear upsampled to input volume size
            iou_predictions: mask quality scores
        """
        image_embeddings = self.sam.image_encoder(images)
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=None,
            masks=mask_prompt,
        )
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        full_res_masks = F.interpolate(
            low_res_masks,
            size=images.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )
        return low_res_masks, full_res_masks, iou_predictions

    def forward(
        self,
        images: torch.Tensor,
        multimask_output: bool = False,
        detach_coarse_prompt: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        End-to-end cascade forward pass.

        Args:
            images: (B, 1, D, H, W), already normalized like SAM-Med3D training data
            detach_coarse_prompt: stop gradients through coarse head output (phase-2 warm start)
        """
        low_res_prompt, coarse_logits = self.coarse_head(images, return_logits=True)
        if detach_coarse_prompt:
            low_res_prompt = low_res_prompt.detach()

        low_res_sam, full_res_sam, iou_predictions = self.forward_sam(
            images=images,
            mask_prompt=low_res_prompt,
            multimask_output=multimask_output,
        )
        coarse_prob = torch.softmax(
            self.coarse_head._select_main_logits(coarse_logits), dim=1
        )[:, 1:2]
        return {
            "coarse_logits": coarse_logits,
            "coarse_prob": coarse_prob,
            "mask_prompt": low_res_prompt,
            "low_res_sam": low_res_sam,
            "full_res_sam": full_res_sam,
            "iou_predictions": iou_predictions,
        }
