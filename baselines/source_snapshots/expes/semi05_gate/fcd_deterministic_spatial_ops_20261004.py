"""Opt-in equivalent linear operators; never patches frozen source on disk."""
from contextlib import contextmanager
import torch
import torch.nn.functional as F


def block_mean4(x):
    b,c,d,h,w = x.shape
    assert d % 4 == h % 4 == w % 4 == 0
    return x.reshape(b,c,d//4,4,h//4,4,w//4,4).mean((3,5,7))


def axis_weights(n, m, device, dtype):
    # PyTorch align_corners=False half-pixel coordinates, edge replication.
    coordinate = ((torch.arange(m,device=device,dtype=dtype)+.5)*(n/m)-.5).clamp(0,n-1)
    lower = coordinate.floor().long()
    upper = (lower+1).clamp_max(n-1)
    fraction = coordinate-lower
    index = torch.arange(n,device=device)
    return (index[None,:] == lower[:,None]).to(dtype)*(1-fraction[:,None]) + (index[None,:] == upper[:,None]).to(dtype)*fraction[:,None]


def trilinear_exact(x, size):
    assert x.ndim == 5 and len(size) == 3
    dtype = x.dtype
    # Accumulate fp16 inputs in fp32 without autocast/TF32 approximation.
    with torch.cuda.amp.autocast(enabled=False):
        z = x.double() if dtype == torch.float64 else x.float()
        for dim, m in zip((4,3,2), reversed(tuple(size))):
            n = z.shape[dim]
            if n == m:
                continue
            weight = axis_weights(n,m,z.device,z.dtype)
            z = torch.matmul(z.movedim(dim,-1), weight.T).movedim(-1,dim)
        return z.to(dtype)


@contextmanager
def deterministic_interpolation(calls=None):
    original = F.interpolate
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    def wrapped(input,size=None,scale_factor=None,mode='nearest',align_corners=None,recompute_scale_factor=None,**kwargs):
        if mode != 'trilinear':
            return original(input,size=size,scale_factor=scale_factor,mode=mode,align_corners=align_corners,recompute_scale_factor=recompute_scale_factor,**kwargs)
        assert align_corners is False and size is not None and scale_factor is None
        assert recompute_scale_factor in (None,False) and not kwargs.get('antialias',False)
        shape = (size,size,size) if isinstance(size,int) else tuple(size)
        if calls is not None:
            key = dict(source=list(input.shape),target=list(shape),dtype=str(input.dtype),requires_grad=input.requires_grad)
            if key not in calls:
                calls.append(key)
        return trilinear_exact(input,shape)
    F.interpolate = wrapped
    try:
        yield
    finally:
        F.interpolate = original
        torch.backends.cuda.matmul.allow_tf32 = original_tf32


def relation_loss(sf,tf,sl,tl,ids):
    from expes.semi05_gate.fcd_hard_spatial_relations_20261004 import displacement_loss,vectors
    feature,ef = displacement_loss(vectors(sf),vectors(tf),ids)
    student = vectors(block_mean4((sl[:,1:2]-sl[:,0:1]).float()))
    teacher = vectors(block_mean4((tl[:,1:2]-tl[:,0:1]).float()))
    output,eo = displacement_loss(student,teacher,ids)
    return (feature+output)/2,dict(feature=feature,output=output,feature_energy=ef,output_energy=eo)
