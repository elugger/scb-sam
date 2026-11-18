import torch
import torch.nn.functional as F

def align_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.shape[-3:] == target.shape[-3:]:
        return logits
    return F.interpolate(logits, size=target.shape[-3:], mode='trilinear', align_corners=False)

def dice_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, eps: float=1e-06) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)[:, 1]
    target_f = (target == 1).float()
    inter = (probs * target_f).sum()
    return 1.0 - (2 * inter + eps) / (probs.sum() + target_f.sum() + eps)

def supervised_loss(logits: torch.Tensor, target: torch.Tensor, use_dice: bool=True, dice_weight: float=0.5) -> torch.Tensor:
    logits = align_logits(logits, target)
    ce = F.cross_entropy(logits, target.long().squeeze(1))
    if not use_dice:
        return ce
    dice = dice_loss_with_logits(logits, target.squeeze(1))
    return (1.0 - dice_weight) * ce + dice_weight * dice

def softmax_mse_loss_map(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float=1.0) -> torch.Tensor:
    student_logits = student_logits.float()
    teacher_logits = teacher_logits.float()
    if student_logits.shape != teacher_logits.shape:
        teacher_logits = align_logits(teacher_logits, student_logits[:, :1])
        student_logits = align_logits(student_logits, teacher_logits[:, :1])
    t = max(float(temperature), 1e-06)
    s = F.softmax(student_logits / t, dim=1)
    tgt = F.softmax(teacher_logits / t, dim=1)
    return (s - tgt) ** 2

def softmax_mse_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float=1.0) -> torch.Tensor:
    if student_logits.shape != teacher_logits.shape:
        teacher_logits = align_logits(teacher_logits, student_logits[:, :1])
        student_logits = align_logits(student_logits, teacher_logits[:, :1])
    t = max(float(temperature), 1e-06)
    s = F.softmax(student_logits / t, dim=1)
    tgt = F.softmax(teacher_logits / t, dim=1)
    return ((s - tgt) ** 2).mean() * float(s.shape[1])

def uncertainty_entropy_map(probs: torch.Tensor) -> torch.Tensor:
    return -torch.sum(probs * torch.log(probs + 1e-06), dim=1, keepdim=True)

def masked_consistency_mse(student_logits: torch.Tensor, teacher_logits: torch.Tensor, mask: torch.Tensor, temperature: float=1.0) -> torch.Tensor:
    mse = softmax_mse_loss_map(student_logits, teacher_logits, temperature=temperature)
    mask = mask.float()
    if mask.shape[1] == 1 and mse.shape[1] > 1:
        m = mask.expand_as(mse)
    else:
        m = mask
    return torch.sum(m * mse) / (2.0 * torch.sum(mask) + 1e-16)

def confidence_mask_from_logits(logits: torch.Tensor, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1, keepdim=True)
    mask = (conf >= threshold).float()
    return (mask, pred)

def pseudo_loss(logits: torch.Tensor, pseudo_label: torch.Tensor, mask: torch.Tensor, use_dice: bool=True, dice_weight: float=0.5) -> torch.Tensor:
    logits = align_logits(logits, pseudo_label)
    mask_sum = mask.sum()
    if mask_sum < 1:
        return logits.new_tensor(0.0)
    ce = (F.cross_entropy(logits, pseudo_label.long().squeeze(1), reduction='none') * mask.squeeze(1)).sum() / (mask_sum + 1e-06)
    if not use_dice:
        return ce
    probs = F.softmax(logits, dim=1)[:, 1]
    tgt = (pseudo_label.squeeze(1) == 1).float()
    inter = (probs * tgt * mask.squeeze(1)).sum()
    dice = 1.0 - (2 * inter + 1e-06) / ((probs * mask.squeeze(1)).sum() + (tgt * mask.squeeze(1)).sum() + 1e-06)
    return (1.0 - dice_weight) * ce + dice_weight * dice
