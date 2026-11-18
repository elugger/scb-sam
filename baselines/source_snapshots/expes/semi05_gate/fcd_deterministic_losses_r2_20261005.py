"""Exact hard-target CE via log-softmax/one-hot multiplication; opt-in deterministic context."""
from contextlib import contextmanager
import torch
import torch.nn.functional as F
from expes.semi05_gate.fcd_deterministic_spatial_ops_r3_20261005 import deterministic_interpolation as spatial_context,relation_loss


def cross_entropy_exact(input,target,weight=None,size_average=None,ignore_index=-100,reduce=None,reduction='mean',label_smoothing=0.):
    assert size_average is None and reduce is None and label_smoothing==0.
    assert target.dtype==torch.long and target.shape==input.shape[:1]+input.shape[2:]
    with torch.cuda.amp.autocast(enabled=False):
        x=input.double() if input.dtype==torch.float64 else input.float()
        valid=target!=ignore_index
        safe=torch.where(valid,target,torch.zeros_like(target))
        classes=torch.arange(x.shape[1],device=x.device).reshape(1,x.shape[1],*([1]*(x.ndim-2)))
        selector=(safe.unsqueeze(1)==classes).to(x.dtype)
        value=-(x.log_softmax(1)*selector).sum(1)
        weights=valid.to(x.dtype)
        if weight is not None:weights=weights*weight.to(x.dtype)[safe]
        loss=value*weights
        if reduction=='none':return loss
        if reduction=='sum':return loss.sum()
        assert reduction=='mean'
        return loss.sum()/weights.sum()


def check_ce():
    rows=[]
    for dtype in (torch.float64,torch.float32,torch.float16):
        for weighted in (False,True):
            for reduction in ('none','mean','sum'):
                x=torch.randn(1,2,8,8,8,device='cuda',dtype=dtype,requires_grad=True)
                target=torch.randint(0,2,(1,8,8,8),device='cuda');target[:,0,0,0]=-100
                weight=torch.tensor([.7,1.3],device='cuda',dtype=dtype) if weighted else None
                cpu=x.detach().cpu().double().requires_grad_(True)
                ref=F.cross_entropy(cpu,target.cpu(),weight=None if weight is None else weight.cpu().double(),reduction=reduction)
                rg,=torch.autograd.grad(ref.sum(),cpu)
                ys=[];gs=[]
                for _ in range(2):
                    y=cross_entropy_exact(x,target,weight=weight,reduction=reduction)
                    g,=torch.autograd.grad(y.sum(),x)
                    ys.append(y.detach());gs.append(g.detach())
                ferror=float((ys[0].cpu().double()-ref.detach()).norm()/ref.detach().norm())
                gerror=float((gs[0].cpu().double()-rg).norm()/rg.norm())
                row=dict(dtype=str(dtype),weighted=weighted,reduction=reduction,forward_relative_l2=ferror,gradient_relative_l2=gerror,repeat_forward=torch.equal(*ys),repeat_gradient=torch.equal(*gs))
                rows.append(row)
                tolerance={torch.float64:1e-12,torch.float32:2e-6,torch.float16:.002}[dtype]
                assert row['repeat_forward'] and row['repeat_gradient'] and max(ferror,gerror)<tolerance,row
    return rows


@contextmanager
def deterministic_training(calls=None):
    original=F.cross_entropy
    with spatial_context(calls):
        F.cross_entropy=cross_entropy_exact
        try:yield
        finally:F.cross_entropy=original
