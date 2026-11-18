"""Full nnU-Net v2 3D coarse segmentation head (ResidualEncoderUNet)."""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet


def build_nnunet3d_fullres(
    input_channels: int = 1,
    num_classes: int = 2,
    deep_supervision: bool = False,
) -> ResidualEncoderUNet:
    """
    Build nnU-Net v2 3d_fullres ResidualEncoderUNet (ResEnc topology).

    Uses residual encoder blocks + InstanceNorm3d + LeakyReLU, which captures
    global context via deep bottleneck and preserves boundary detail via skip
    connections — the default strong 3D configuration in nnU-Net v2.
    """
    n_stages = 6
    network = ResidualEncoderUNet(
        input_channels=input_channels,
        n_stages=n_stages,
        features_per_stage=[32, 64, 128, 256, 320, 320],
        conv_op=nn.Conv3d,
        kernel_sizes=[[3, 3, 3]] * n_stages,
        strides=[[1, 1, 1]] + [[2, 2, 2]] * (n_stages - 1),
        n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
        num_classes=num_classes,
        n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=deep_supervision,
    )
    network.initialize(network)
    return network


class CoarseNNUNet3D(nn.Module):
    """Full nnU-Net 3D head that produces SAM mask-prompt tensors."""

    def __init__(
        self,
        network: Optional[ResidualEncoderUNet] = None,
        mask_prompt_size: Tuple[int, int, int] = (32, 32, 32),
        num_classes: int = 2,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        self.network = network or build_nnunet3d_fullres(
            num_classes=num_classes,
            deep_supervision=deep_supervision,
        )
        self.mask_prompt_size = mask_prompt_size
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision

    @staticmethod
    def _select_main_logits(
        logits: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]],
    ) -> torch.Tensor:
        if isinstance(logits, (list, tuple)):
            return logits[0]
        return logits

    def forward_logits(
        self,
        images: torch.Tensor,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Return full-resolution logits; list when deep_supervision is enabled."""
        return self.network(images)

    def forward(
        self,
        images: torch.Tensor,
        return_logits: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            images: (B, 1, D, H, W)
        Returns:
            low_res_mask_prompt: (B, 1, mask_D, mask_H, mask_W), values in [0, 1]
            optionally full-res logits when return_logits=True
        """
        logits = self.forward_logits(images)
        main_logits = self._select_main_logits(logits)
        if self.num_classes == 1:
            prob = torch.sigmoid(main_logits)
        else:
            prob = torch.softmax(main_logits, dim=1)[:, 1:2]

        low_res = F.interpolate(
            prob,
            size=self.mask_prompt_size,
            mode="trilinear",
            align_corners=False,
        )
        if return_logits:
            return low_res, logits
        return low_res
