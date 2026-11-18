"""Build SAM-Med3D + coarse nnU-Net cascade."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from segment_anything.build_sam3D import sam_model_registry3D
from segment_anything.modeling.coarse_nnunet3d import CoarseNNUNet3D, build_nnunet3d_fullres
from segment_anything.modeling.sam3D_cascade import Sam3DCascade


def _extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a state_dict or a dict containing model weights.")
    for key in ("network_weights", "model_state_dict", "state_dict"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]

    if all(isinstance(k, str) for k in checkpoint.keys()):
        return checkpoint
    raise ValueError("Checkpoint must be a state_dict or a dict containing model weights.")


def _strip_coarse_prefix(key: str) -> str:
    for prefix in (
        "coarse_head.",
        "module.coarse_head.",
        "network.",
        "module.network.",
        "unet.",
    ):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def build_sam3d_cascade(
    model_type: str = "vit_b_ori",
    sam_checkpoint: Optional[str] = None,
    coarse_checkpoint: Optional[str] = None,
    img_size: int = 128,
    deep_supervision: bool = False,
    device: str = "cuda",
    mc_dropout_p: float = 0.0,
) -> Sam3DCascade:
    """
    Build a cascade model and optionally load SAM / coarse-head weights.
    """
    build_fn = sam_model_registry3D[model_type]
    sam = build_fn(checkpoint=None)
    if sam_checkpoint:
        ckpt = torch.load(sam_checkpoint, map_location="cpu")
        state = _extract_state_dict(ckpt) if isinstance(ckpt, dict) else ckpt
        sam.load_state_dict(state, strict=False)

    mask_prompt_size = img_size // 4
    network = build_nnunet3d_fullres(
        num_classes=2,
        deep_supervision=deep_supervision,
    )
    coarse_head = CoarseNNUNet3D(
        network=network,
        mask_prompt_size=(mask_prompt_size, mask_prompt_size, mask_prompt_size),
        deep_supervision=deep_supervision,
    )
    if coarse_checkpoint:
        cckpt = torch.load(coarse_checkpoint, map_location="cpu")
        cstate = _extract_state_dict(cckpt) if isinstance(cckpt, dict) else cckpt
        mapped = {_strip_coarse_prefix(k): v for k, v in cstate.items()}
        coarse_head.load_state_dict(mapped, strict=False)

    model = Sam3DCascade(sam=sam, coarse_head=coarse_head, img_size=img_size)

    model.mc_dropout_p = float(mc_dropout_p)
    return model.to(device)
