from contextlib import contextmanager
import torch
import torch.nn.functional as F
from expes.semi05_gate.fcd_deterministic_spatial_ops_20261004 import trilinear_exact, relation_loss, deterministic_interpolation as interpolation_only

def triple(v):
    return (v, v, v) if isinstance(v, int) else tuple(v)

def avg_pool_exact(x, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    assert x.ndim == 5 and (not ceil_mode)
    kernel = triple(kernel_size)
    stride = triple(stride if stride is not None else kernel_size)
    padding = triple(padding)
    dtype = x.dtype
    with torch.cuda.amp.autocast(enabled=False), torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True, allow_tf32=False):
        z = x.double() if dtype == torch.float64 else x.float()
        weight = torch.ones(z.shape[1], 1, *kernel, device=z.device, dtype=z.dtype)
        total = F.conv3d(z, weight, stride=stride, padding=padding, groups=z.shape[1])
        if divisor_override is not None:
            denominator = divisor_override
        elif count_include_pad:
            denominator = kernel[0] * kernel[1] * kernel[2]
        else:
            denominator = F.conv3d(torch.ones_like(z[:, :1]), weight[:1], stride=stride, padding=padding)
        return (total / denominator).to(dtype)

@contextmanager
def deterministic_interpolation(calls=None):
    original = F.avg_pool3d

    def wrapped(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
        if not (input.requires_grad and torch.is_grad_enabled()):
            return original(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
        if calls is not None:
            row = dict(operator='avg_pool3d', source=list(input.shape), kernel=triple(kernel_size), stride=triple(stride or kernel_size), padding=triple(padding), dtype=str(input.dtype))
            if row not in calls:
                calls.append(row)
        return avg_pool_exact(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
    with interpolation_only(calls):
        F.avg_pool3d = wrapped
        try:
            yield
        finally:
            F.avg_pool3d = original

def check_pooling():
    rows = []
    for kernel, stride, padding, count in ((3, 1, 1, True), (3, 1, 1, False), (4, 4, 0, True), (2, 2, 0, True)):
        for dtype in (torch.float64, torch.float32, torch.float16):
            x = torch.randn(1, 1, 16, 16, 16, device='cuda', dtype=dtype, requires_grad=True)
            cpu = x.detach().cpu().double().requires_grad_(True)
            ref = F.avg_pool3d(cpu, kernel, stride, padding, False, count)
            cot = torch.randn_like(ref).to('cuda', dtype=dtype)
            rg, = torch.autograd.grad(ref, cpu, cot.cpu().double())
            ys = []
            gs = []
            for _ in range(2):
                y = avg_pool_exact(x, kernel, stride, padding, False, count)
                g, = torch.autograd.grad(y, x, cot)
                ys.append(y.detach())
                gs.append(g.detach())
            ferror = float((ys[0].cpu().double() - ref.detach()).norm() / ref.detach().norm())
            gerror = float((gs[0].cpu().double() - rg).norm() / rg.norm())
            row = dict(kernel=kernel, stride=stride, padding=padding, count_include_pad=count, dtype=str(dtype), forward_relative_l2=ferror, gradient_relative_l2=gerror, repeat_forward=torch.equal(*ys), repeat_gradient=torch.equal(*gs))
            rows.append(row)
            tolerance = {torch.float64: 1e-12, torch.float32: 2e-06, torch.float16: 0.002}[dtype]
            assert row['repeat_forward'] and row['repeat_gradient'] and (max(ferror, gerror) < tolerance), row
    return rows
