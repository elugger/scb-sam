"""Segmentation metrics for training-time test evaluation (dice / jaccard / hd95 / asd)."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from surface_distance import (
    compute_average_surface_distance,
    compute_robust_hausdorff,
    compute_surface_distances,
)


def logits_to_binary(logits: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """Convert model logits (B,C,...) to binary mask ndarray (B, D, H, W)."""
    if logits.shape[1] == 1:
        pred = (torch.sigmoid(logits) > threshold).float()
    else:
        pred = (logits.argmax(dim=1, keepdim=True) > 0).float()
    pred = pred.squeeze(1).detach().cpu().numpy().astype(bool)
    return pred


def _to_bool_volume(mask: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    mask = np.squeeze(mask)
    return (mask > 0).astype(bool)


def binary_case_metrics(
    pred: np.ndarray | torch.Tensor,
    gt: np.ndarray | torch.Tensor,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> Dict[str, float]:
    """Compute dice / jaccard / hd95 / asd for one binary 3D volume."""
    pred_b = _to_bool_volume(pred)
    gt_b = _to_bool_volume(gt)
    if pred_b.ndim != 3 or gt_b.ndim != 3:
        raise ValueError(f"Expected 3D masks, got pred={pred_b.shape} gt={gt_b.shape}")

    inter = np.logical_and(pred_b, gt_b).sum()
    pred_sum = pred_b.sum()
    gt_sum = gt_b.sum()
    union = np.logical_or(pred_b, gt_b).sum()

    dice = float(2.0 * inter / (pred_sum + gt_sum)) if (pred_sum + gt_sum) > 0 else 1.0
    jaccard = float(inter / union) if union > 0 else 1.0

    if pred_sum == 0 and gt_sum == 0:
        return {"dice": dice, "jaccard": jaccard, "hd95": 0.0, "asd": 0.0}
    if pred_sum == 0 or gt_sum == 0:
        # One side empty: surface metrics undefined; use large sentinel like medpy failure fallbacks.
        return {"dice": 0.0 if pred_sum + gt_sum > 0 else dice, "jaccard": 0.0, "hd95": 373.12866, "asd": 373.12866}

    spacing_mm: Tuple[float, float, float] = (
        float(spacing[0]),
        float(spacing[1]),
        float(spacing[2]),
    )
    try:
        surf = compute_surface_distances(gt_b, pred_b, spacing_mm)
        hd95 = float(compute_robust_hausdorff(surf, 95))
        asd_gt2pred, asd_pred2gt = compute_average_surface_distance(surf)
        asd = float(0.5 * (asd_gt2pred + asd_pred2gt))
    except Exception:
        hd95, asd = 373.12866, 373.12866

    return {"dice": dice, "jaccard": jaccard, "hd95": hd95, "asd": asd}


def aggregate_metrics(case_metrics: list[Dict[str, float]]) -> Dict[str, float]:
    if not case_metrics:
        return {"dice": 0.0, "jaccard": 0.0, "hd95": 0.0, "asd": 0.0}
    keys = ("dice", "jaccard", "hd95", "asd")
    return {k: float(np.mean([m[k] for m in case_metrics])) for k in keys}
