"""SDCL helpers adapted from SDCL-master for dynamic patch size."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
SDCL_CODE = ROOT / "SDCL-master" / "code"
if str(SDCL_CODE) not in sys.path:
    sys.path.insert(0, str(SDCL_CODE))

from pancreas.losses import mix_loss, mix_max_kl_loss, mix_mse_loss  # noqa: E402


def forward_logits(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if isinstance(out, (list, tuple)):
        return out[0]
    return out


def context_mask(img: torch.Tensor, mask_ratio: float = 2 / 3) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, _, img_x, img_y, img_z = img.shape
    device = img.device
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z, device=device)
    mask = torch.ones(img_x, img_y, img_z, device=device)
    patch_x = max(1, int(img_x * mask_ratio))
    patch_y = max(1, int(img_y * mask_ratio))
    patch_z = max(1, int(img_z * mask_ratio))
    w = np.random.randint(0, max(1, img_x - patch_x + 1))
    h = np.random.randint(0, max(1, img_y - patch_y + 1))
    z = np.random.randint(0, max(1, img_z - patch_z + 1))
    mask[w : w + patch_x, h : h + patch_y, z : z + patch_z] = 0
    loss_mask[:, w : w + patch_x, h : h + patch_y, z : z + patch_z] = 0
    return mask.long(), loss_mask.long()


def get_cut_mask(out: torch.Tensor, thres: float = 0.5) -> torch.Tensor:
    probs = F.softmax(out, dim=1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :, :].contiguous()
    return largest_cc_batch(masks)


def largest_cc_batch(segmentation: torch.Tensor) -> torch.Tensor:
    batch_list = []
    for n in range(segmentation.shape[0]):
        n_prob = segmentation[n].detach().cpu().numpy().astype(np.uint8)
        labels, nlab = ndimage.label(n_prob)
        if nlab > 0:
            counts = np.bincount(labels.flat)
            counts[0] = 0
            largest = labels == counts.argmax()
        else:
            largest = n_prob.astype(bool)
        batch_list.append(largest)
    return torch.tensor(np.stack(batch_list), device=segmentation.device, dtype=torch.long)


def get_xor_region(out1: torch.Tensor, out2: torch.Tensor) -> torch.Tensor:
    s1 = torch.softmax(out1, dim=1)
    l1 = torch.argmax(s1, dim=1)
    s2 = torch.softmax(out2, dim=1)
    l2 = torch.argmax(s2, dim=1)
    return l1 != l2


@torch.no_grad()
def update_ema(model: torch.nn.Module, ema_model: torch.nn.Module, alpha: float) -> None:
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(alpha).add_((1.0 - alpha) * p.data)


__all__ = [
    "context_mask",
    "forward_logits",
    "get_cut_mask",
    "get_xor_region",
    "mix_loss",
    "mix_max_kl_loss",
    "mix_mse_loss",
    "update_ema",
]
