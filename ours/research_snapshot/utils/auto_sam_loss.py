from typing import List, Union
import torch
from monai.losses import DiceCELoss
from utils.cascade_train_utils import deep_supervision_loss

def build_supervised_loss_fns():
    main_loss_fn = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    coarse_loss_fn = DiceCELoss(softmax=True, to_onehot_y=True, include_background=True)
    return (main_loss_fn, coarse_loss_fn)

def cascade_supervised_loss(main_logits: torch.Tensor, coarse_logits: Union[torch.Tensor, List[torch.Tensor]], gt3d: torch.Tensor, main_loss_fn: DiceCELoss, coarse_loss_fn: DiceCELoss, aux_weight: float=0.3) -> torch.Tensor:
    gt_float = gt3d.float()
    if main_logits.shape[1] == 2:
        main_loss = DiceCELoss(softmax=True, to_onehot_y=True, include_background=True)(main_logits, gt3d.long())
    else:
        main_loss = main_loss_fn(main_logits, gt_float)
    if aux_weight <= 0:
        return main_loss
    if isinstance(coarse_logits, (list, tuple)):
        aux_loss = deep_supervision_loss(coarse_logits, gt_float, coarse_loss_fn)
    else:
        aux_loss = coarse_loss_fn(coarse_logits, gt3d.long())
    return main_loss + aux_weight * aux_loss
