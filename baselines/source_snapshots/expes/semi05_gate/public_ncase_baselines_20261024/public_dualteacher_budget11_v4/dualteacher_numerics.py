"""Process-local strict reproducibility policy; original ATen backward on CPU.

Torch 2.0.1 has no deterministic CUDA trilinear/max-pool3d backward. Preserve
the original CUDA resize/pool forward and native derivatives, executing those
two ATen backward kernels plus original cross-entropy/NLL and floating cumsum on CPU. No changed loss equation, resize convention,
model, batch, optimizer or clock. This changes device/rounding/runtime cost.
"""
import hashlib,os
from pathlib import Path

POLICY='STRICT_FP32_CPU_ATEN_UNSUPPORTED_KERNELS_V3'
_installed=None

def need(ok,message):
    if not ok:raise ValueError(message)

def install(torch):
    global _installed
    if _installed is not None:
        snapshot(torch);return
    need(torch.__version__=='2.0.1+cu117','Unreviewed Torch numeric implementation')
    need(not torch.cuda.is_initialized(),'Numeric policy must precede CUDA initialization')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'
    torch.set_num_threads(2)
    torch.backends.cudnn.enabled=True;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True,warn_only=False)
    import torch.nn.functional as F
    original_interpolate,original_pool=F.interpolate,F.max_pool3d
    original_ce,original_nll=F.cross_entropy,F.nll_loss
    original_cumsum,original_tensor_cumsum=torch.cumsum,torch.Tensor.cumsum

    class Trilinear(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x,size,align):
            ctx.input_size=list(x.shape);ctx.output_size=list(size);ctx.align=align;ctx.device=x.device
            return torch.ops.aten.upsample_trilinear3d.default(x,list(size),align,None,None,None)
        @staticmethod
        def backward(ctx,grad):
            need(not torch.is_grad_enabled(),'Higher order derivative not reviewed')
            g=torch.ops.aten.upsample_trilinear3d_backward.default(
                grad.detach().to('cpu'),ctx.output_size,ctx.input_size,ctx.align,None,None,None)
            return g.to(ctx.device),None,None

    class Pool(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x,kernel,stride,padding,dilation,ceil):
            y,indices=torch.ops.aten.max_pool3d_with_indices.default(x,kernel,stride,padding,dilation,ceil)
            ctx.save_for_backward(x,indices);ctx.arguments=(kernel,stride,padding,dilation,ceil)
            ctx.mark_non_differentiable(indices)
            return y,indices
        @staticmethod
        def backward(ctx,grad,unused):
            need(not torch.is_grad_enabled(),'Higher order derivative not reviewed')
            x,indices=ctx.saved_tensors
            g=torch.ops.aten.max_pool3d_with_indices_backward.default(
                grad.detach().to('cpu'),x.detach().to('cpu'),*ctx.arguments,indices.to('cpu'))
            return g.to(x.device),None,None,None,None,None

    def interpolate(input,size=None,scale_factor=None,mode='nearest',align_corners=None,recompute_scale_factor=None,antialias=False):
        if input.is_cuda and input.requires_grad and torch.is_grad_enabled() and mode=='trilinear':
            need(input.ndim==5 and input.dtype==torch.float32 and scale_factor is None
                and recompute_scale_factor in (None,False) and not antialias and type(align_corners) is bool,
                'Unreviewed differentiable trilinear invocation')
            need(isinstance(size,(list,tuple)) and len(size)==3 and all(type(s) is int and s>0 for s in size),
                'Unreviewed trilinear output size')
            return Trilinear.apply(input,tuple(size),align_corners)
        return original_interpolate(input,size,scale_factor,mode,align_corners,recompute_scale_factor,antialias)

    def triple(x,default=None):
        if x is None:x=default
        if isinstance(x,int):return [x]*3
        need(isinstance(x,(tuple,list)) and len(x)==3,'Unreviewed pool argument')
        return list(x)

    def max_pool3d(input,kernel_size,stride=None,padding=0,dilation=1,ceil_mode=False,return_indices=False):
        if input.is_cuda and input.requires_grad and torch.is_grad_enabled():
            need(input.ndim==5 and input.dtype==torch.float32,'Unreviewed differentiable max-pool input')
            k=triple(kernel_size);args=(k,triple(stride,k),triple(padding),triple(dilation),ceil_mode)
            y,indices=Pool.apply(input,*args)
            return (y,indices) if return_indices else y
        return original_pool(input,kernel_size,stride,padding,dilation,ceil_mode,return_indices)

    def loss_cpu(input,target,weight):
        need(input.dtype==torch.float32 and target.dtype==torch.int64,'Unreviewed native loss dtype')
        need(weight is None or (not weight.requires_grad and weight.dtype==torch.float32),'Unreviewed native loss weight')
        return input.to('cpu'),target.to('cpu'),None if weight is None else weight.to('cpu')

    def cross_entropy(input,target,weight=None,size_average=None,ignore_index=-100,reduce=None,reduction='mean',label_smoothing=0.0):
        if input.is_cuda:
            x,y,w=loss_cpu(input,target,weight)
            return original_ce(x,y,w,size_average,ignore_index,reduce,reduction,label_smoothing).to(input.device)
        return original_ce(input,target,weight,size_average,ignore_index,reduce,reduction,label_smoothing)

    def nll_loss(input,target,weight=None,size_average=None,ignore_index=-100,reduce=None,reduction='mean'):
        if input.is_cuda:
            x,y,w=loss_cpu(input,target,weight)
            return original_nll(x,y,w,size_average,ignore_index,reduce,reduction).to(input.device)
        return original_nll(input,target,weight,size_average,ignore_index,reduce,reduction)

    def cumsum(input,dim,*,dtype=None,out=None):
        if input.is_cuda and input.is_floating_point():
            need(input.dtype==torch.float32 and dtype in (None,torch.float32) and out is None,'Unreviewed CUDA floating cumsum')
            return original_cumsum(input.to('cpu'),dim,dtype=dtype).to(input.device)
        return original_cumsum(input,dim,dtype=dtype,out=out)

    def tensor_cumsum(input,dim,dtype=None):
        if input.is_cuda and input.is_floating_point():return cumsum(input,dim,dtype=dtype)
        return original_tensor_cumsum(input,dim,dtype=dtype)

    torch.cumsum=cumsum;torch.Tensor.cumsum=tensor_cumsum
    F.interpolate=interpolate;F.max_pool3d=max_pool3d
    F.cross_entropy=cross_entropy;F.nll_loss=nll_loss
    _installed=dict(interpolate=interpolate,max_pool3d=max_pool3d,original_interpolate=original_interpolate,
                    original_pool=original_pool,Trilinear=Trilinear,Pool=Pool,
                    cross_entropy=cross_entropy,nll_loss=nll_loss,original_ce=original_ce,original_nll=original_nll,cumsum=cumsum,tensor_cumsum=tensor_cumsum,
                    original_cumsum=original_cumsum,original_tensor_cumsum=original_tensor_cumsum)
    snapshot(torch)

def snapshot(torch):
    import torch.nn.functional as F
    need(_installed is not None and F.interpolate is _installed['interpolate'] and F.max_pool3d is _installed['max_pool3d']
        and F.cross_entropy is _installed['cross_entropy'] and F.nll_loss is _installed['nll_loss']
        and torch.cumsum is _installed['cumsum'] and torch.Tensor.cumsum is _installed['tensor_cumsum'],
        'Numeric operator policy absent/replaced')
    flags=dict(policy=POLICY,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),torch=torch.__version__,
        deterministic=torch.are_deterministic_algorithms_enabled(),warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_enabled=torch.backends.cudnn.enabled,cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_tf32=torch.backends.cudnn.allow_tf32,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,matmul_precision=torch.get_float32_matmul_precision(),
        cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),threads=torch.get_num_threads())
    need(flags['torch']=='2.0.1+cu117' and flags['deterministic'] and not flags['warn_only']
        and flags['cudnn_enabled'] and flags['cudnn_deterministic'] and not flags['cudnn_benchmark']
        and not flags['cudnn_tf32'] and not flags['matmul_tf32'] and flags['matmul_precision']=='highest'
        and flags['cublas_workspace']==':16:8' and flags['threads']==2,'Numeric runtime flags changed')
    return flags

def state_layout(core,models,torch):
    def layout(v):
        if torch.is_tensor(v):
            return dict(dtype=str(v.dtype),shape=list(v.shape),stride=list(v.stride()),offset=v.storage_offset())
        if isinstance(v,dict):return {k:layout(x) for k,x in v.items()}
        if isinstance(v,(tuple,list)):return type(v)(layout(x) for x in v)
        return None
    return dict(models={name:dict(parameters={k:layout(p) for k,p in m.named_parameters()},
        buffers={k:layout(b) for k,b in m.named_buffers()},gradients={k:layout(p.grad) for k,p in m.named_parameters()})
        for name,m in models.items()},optimizer=layout(core.optimizer.state_dict()),optimizer_op=layout(core.optimizer_op.state_dict()))
