"""Frozen generalist branch (SemiSAM+): SAM-Med3D promptable segmentation.

Adapted from SemiSAM-main/code_semisam+/semisam_plus.py with two fixes:
1) load SAM once and reuse (original reloads every call);
2) checkpoint path points at this repo's ckpt/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SEMISAM_CODE = ROOT / "SemiSAM-main" / "code_semisam+"

# Click helpers live in SemiSAM; keep import local after sys.path inject.
_CLICK_READY = False


def _ensure_click_imports():
    global _CLICK_READY, get_next_click3D_torch_ritm, get_next_click3D_torch_2
    if _CLICK_READY:
        return
    import sys

    p = str(SEMISAM_CODE)
    if p not in sys.path:
        sys.path.insert(0, p)
    from utils.click_method import (  # type: ignore
        get_next_click3D_torch_2,
        get_next_click3D_torch_ritm,
    )

    _CLICK_READY = True


def compute_epistemic_uncertainty(all_preds):
    predictions = torch.stack(all_preds)
    return torch.var(predictions, dim=0)


def _online_var_update(n, mean, m2, x):
    """Welford online variance; avoids stacking all click volumes in RAM."""
    n = n + 1
    if mean is None:
        mean = x.detach().clone()
        m2 = torch.zeros_like(mean)
        return n, mean, m2
    delta = x - mean
    mean = mean + delta / n
    delta2 = x - mean
    m2 = m2 + delta * delta2
    return n, mean, m2


@torch.inference_mode()
def predict3d_unc(img3d, gt3d, sam, device, num_clicks=10):
    _ensure_click_imports()
    click_method = get_next_click3D_torch_ritm
    bsz, ch, w, h, d = img3d.shape
    pred_batch = torch.zeros((bsz, ch, w, h, d), device=device)
    unc_batch = torch.zeros((bsz, ch, w, h, d), device=device)
    crop_size = 128

    for b in range(bsz):
        img = img3d[b : b + 1]
        gt = gt3d[b : b + 1]
        prev_masks = torch.zeros_like(img, device=device)
        low_res = F.interpolate(prev_masks.float(), size=(crop_size // 4,) * 3)
        image_embedding = sam.image_encoder(img.to(device))

        # Online stats over [gt, click_1, ..., click_K] (same as original stack+var)
        n, mean, m2 = 0, None, None
        n, mean, m2 = _online_var_update(n, mean, m2, gt.float())
        prev_seg_prob = torch.zeros_like(img, device=device)
        medsam_seg = torch.zeros_like(img, dtype=torch.uint8, device=device)

        for num_click in range(num_clicks):
            method = get_next_click3D_torch_2 if num_click > 1 else click_method
            batch_points, batch_labels = method(prev_seg_prob.to(device), gt.to(device))
            points_co = torch.cat(batch_points, dim=0).to(device)
            points_la = torch.cat(batch_labels, dim=0).to(device)
            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                points=[points_co, points_la], boxes=None, masks=low_res.to(device)
            )
            low_res, _ = sam.mask_decoder(
                image_embeddings=image_embedding.to(device),
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            prev_masks = F.interpolate(
                low_res, size=img.shape[-3:], mode="trilinear", align_corners=False
            )
            medsam_seg_prob = torch.sigmoid(prev_masks)
            prev_seg_prob = medsam_seg_prob
            medsam_seg = (medsam_seg_prob > 0.5).to(torch.uint8)
            n, mean, m2 = _online_var_update(n, mean, m2, medsam_seg_prob)
            low_res = F.interpolate(prev_masks.float(), size=(crop_size // 4,) * 3)
            del sparse_embeddings, dense_embeddings, points_co, points_la

        if n > 1:
            unc_batch[b : b + 1] = m2 / (n - 1)
        pred_batch[b : b + 1] = medsam_seg
        del image_embedding, mean, m2, prev_masks, prev_seg_prob, low_res

    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pred_batch, unc_batch


@torch.inference_mode()
def predict3d_mask(img3d, gt3d, sam, device):
    bsz, ch, w, h, d = img3d.shape
    pred_batch = torch.zeros((bsz, ch, w, h, d), device=device)
    crop_size = 128
    for b in range(bsz):
        img = img3d[b : b + 1]
        gt = gt3d[b : b + 1]
        low_res_gt = F.interpolate(gt.float(), size=(crop_size // 4,) * 3)
        image_embedding = sam.image_encoder(img.to(device))
        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            points=None, boxes=None, masks=low_res_gt.to(device)
        )
        low_res, _ = sam.mask_decoder(
            image_embeddings=image_embedding.to(device),
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        masks = F.interpolate(
            low_res, size=img.shape[-3:], mode="trilinear", align_corners=False
        )
        pred_batch[b : b + 1] = (torch.sigmoid(masks) > 0.5).to(torch.uint8)
    return pred_batch


@torch.inference_mode()
def predict3d_point(img3d, gt3d, sam, device, num_clicks=10):
    _ensure_click_imports()
    click_method = get_next_click3D_torch_ritm
    bsz, ch, w, h, d = img3d.shape
    pred_batch = torch.zeros((bsz, ch, w, h, d), device=device)
    crop_size = 128
    for b in range(bsz):
        img = img3d[b : b + 1]
        gt = gt3d[b : b + 1]
        prev_masks = torch.zeros_like(img, device=device)
        low_res = F.interpolate(prev_masks.float(), size=(crop_size // 4,) * 3)
        image_embedding = sam.image_encoder(img.to(device))
        prev_seg_prob = torch.zeros_like(img, device=device)
        medsam_seg = torch.zeros_like(img, dtype=torch.uint8, device=device)
        for num_click in range(num_clicks):
            method = get_next_click3D_torch_2 if num_click > 1 else click_method
            batch_points, batch_labels = method(prev_seg_prob.to(device), gt.to(device))
            points_co = torch.cat(batch_points, dim=0).to(device)
            points_la = torch.cat(batch_labels, dim=0).to(device)
            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                points=[points_co, points_la], boxes=None, masks=low_res.to(device)
            )
            low_res, _ = sam.mask_decoder(
                image_embeddings=image_embedding.to(device),
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            prev_masks = F.interpolate(
                low_res, size=img.shape[-3:], mode="trilinear", align_corners=False
            )
            medsam_seg_prob = torch.sigmoid(prev_masks)
            prev_seg_prob = medsam_seg_prob
            medsam_seg = (medsam_seg_prob > 0.5).to(torch.uint8)
            low_res = F.interpolate(prev_masks.float(), size=(crop_size // 4,) * 3)
        pred_batch[b : b + 1] = medsam_seg
    return pred_batch


class SemiSAMGeneralist:
    """Frozen SAM-Med3D used as SemiSAM+ generalist."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: torch.device,
        prompt: str = "unc",
        num_clicks: int = 10,
    ):
        from segment_anything.build_sam3D import sam_model_registry3D

        ckpt = Path(checkpoint)
        if not ckpt.is_absolute():
            ckpt = ROOT / ckpt
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)

        self.device = device
        self.prompt = str(prompt)
        self.num_clicks = int(num_clicks)
        self.sam = sam_model_registry3D["vit_b_ori"](checkpoint=None).to(device)
        state = torch.load(str(ckpt), map_location=device)
        sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
        self.sam.load_state_dict(sd)
        self.sam.eval()
        for p in self.sam.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def __call__(
        self, volume_batch: torch.Tensor, soft_fg: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """volume: (B,1,D,H,W); soft_fg: specialist P(fg) (B,1,...). Returns hard mask + optional unc."""
        mask = soft_fg
        if self.prompt == "mask":
            out = predict3d_mask(volume_batch, mask, self.sam, self.device)
            return out.float(), None
        if self.prompt == "point":
            out = predict3d_point(
                volume_batch, mask, self.sam, self.device, num_clicks=self.num_clicks
            )
            return out.float(), None
        out, unc = predict3d_unc(
            volume_batch, mask, self.sam, self.device, num_clicks=self.num_clicks
        )
        return out.float(), unc
