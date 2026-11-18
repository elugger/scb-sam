import os
import random
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import torchio as tio
from monai.losses import DiceCELoss
from torch.backends import cudnn
from utils.data_loader import Dataset_Union_ALL, Union_Dataloader

def init_seeds(seed: int=2026, cuda_deterministic: bool=True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True

def get_train_dataloader(data_paths: List[str], img_size: int=128, batch_size: int=1, num_workers: int=2, threshold: int=1000):
    dataset = Dataset_Union_ALL(paths=data_paths, transform=tio.Compose([tio.ToCanonical(), tio.CropOrPad(mask_name='label', target_shape=(img_size, img_size, img_size)), tio.RandomFlip(axes=(0, 1, 2))]), threshold=threshold)
    return Union_Dataloader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

def normalize_batch(image3d: torch.Tensor, norm_transform: tio.ZNormalization) -> torch.Tensor:
    image3d = norm_transform(image3d.squeeze(dim=1))
    return image3d.unsqueeze(dim=1)

def compute_batch_dice(pred_logits: torch.Tensor, gt3d: torch.Tensor, threshold: float=0.5) -> float:
    pred = torch.sigmoid(pred_logits) if pred_logits.shape[1] == 1 else pred_logits.argmax(dim=1, keepdim=True).float()
    if pred.shape[1] > 1:
        pred = (pred > 0).float()
    else:
        pred = (pred > threshold).float()
    gt = (gt3d > 0).float()
    dice_list = []
    for i in range(gt.shape[0]):
        inter = (pred[i] * gt[i]).sum()
        denom = pred[i].sum() + gt[i].sum()
        dice_list.append((2 * inter / denom).item() if denom > 0 else float('nan'))
    valid = [d for d in dice_list if not np.isnan(d)]
    return float(np.mean(valid)) if valid else 0.0

def deep_supervision_loss(logits_list, gt3d: torch.Tensor, loss_fn: DiceCELoss, weights: Optional[List[float]]=None) -> torch.Tensor:
    gt_long = gt3d.long()
    if weights is None:
        weights = [0.5 ** i for i in range(len(logits_list))]
    total = 0.0
    for logit, w in zip(logits_list, weights):
        target = gt_long
        if logit.shape[-3:] != gt_long.shape[-3:]:
            target = F.interpolate(gt_long.float(), size=logit.shape[-3:], mode='nearest').long()
        total = total + w * loss_fn(logit, target)
    return total / sum(weights)

def save_checkpoint(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)
