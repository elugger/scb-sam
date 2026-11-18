"""CAC-inspired foreground cross-context identities replace unlabeled losses."""
import torch
import torch.nn.functional as F
from torch.cuda import amp
from expes.semi05_gate.paired_zea_cross_scale_half_20260911 import PairedCrossScaleHalf
from expes.semi05_gate.paired_rng_data_20260908 import scoped_rng, stream_seed


def context_views(x,axis):
    assert tuple(x.shape[-3:])==(128,128,128)
    a,b=x.clone(),x.clone()
    sa,sb=[slice(None)]*5,[slice(None)]*5
    sa[axis+2],sb[axis+2]=slice(96,128),slice(0,32)
    background=x[...,0,0,0].reshape(x.shape[0],x.shape[1],1,1,1)
    a[tuple(sa)],b[tuple(sb)]=background,background
    return a,b


class PairedCACForeground(PairedCrossScaleHalf):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.cac_debug=None

    def feature_forward(self,model,image):
        capture=[]
        handle=model.cascade.mask_decoder.output_upscaling.register_forward_hook(lambda m,i,v:capture.append(v))
        try:
            with amp.autocast(enabled=self.device.type=='cuda'):
                logits=self._forward_main_logits(model,image)
        finally:
            handle.remove()
        assert len(capture)==1 and tuple(capture[0].shape[1:])==(48,32,32,32)
        return logits,capture[0]

    def _losses(self,image_l,label_l,image_u,epoch,step):
        axis,reverse=step%3,(epoch+step)%2
        a,b=context_views(image_u,axis)
        if reverse:
            a,b=b,a
        with torch.no_grad():
            ta,_=self.feature_forward(self.teacher,a)
            tb,k=self.feature_forward(self.teacher,b)
            qa=F.avg_pool3d(ta.float().softmax(1)[:,1:2],4,4)
            qb=F.avg_pool3d(tb.float().softmax(1)[:,1:2],4,4)
        with scoped_rng(stream_seed(self.pair_seed,'student-step',epoch,step),cuda=True):
            with amp.autocast(enabled=self.device.type=='cuda'):
                supervised=self._supervised_step(image_l,label_l)
            logits,z=self.feature_forward(self.student,a)
        ca,cb=torch.maximum(qa,1-qa),torch.maximum(qb,1-qb)
        mask=torch.zeros_like(qa,dtype=torch.bool)
        sl=[slice(None)]*5
        sl[axis+2]=slice(8,24)
        mask[tuple(sl)]=True
        positives=mask & (qb>=.92) & (cb>ca)
        negatives=mask & (qa<=.08) & (qb<=.08)
        generator=torch.Generator(device=image_u.device).manual_seed(stream_seed(self.pair_seed,'cac-pairs',epoch,step))
        terms=[]
        pcount=ncount=conflict=0
        for batch in range(z.shape[0]):
            pos=positives[batch].flatten().nonzero()[:,0]
            neg=negatives[batch].flatten().nonzero()[:,0]
            pcount+=len(pos)
            ncount+=len(neg)
            conflict+=int((positives[batch] & (qa[batch]<.5)).sum())
            if not len(pos) or not len(neg):
                continue
            pos=pos[torch.randperm(len(pos),device=pos.device,generator=generator)[:128]]
            neg=neg[torch.randperm(len(neg),device=neg.device,generator=generator)[:512]]
            queries=F.normalize(z[batch].float().flatten(1).T[pos],dim=1)
            keys=F.normalize(k[batch].detach().float().flatten(1).T,dim=1)
            positive=(queries*keys[pos]).sum(1,keepdim=True)/.2
            negative=queries@keys[neg].T/.2
            scores=torch.cat((positive,negative),1)
            terms.append(F.cross_entropy(scores,torch.zeros(len(pos),device=pos.device,dtype=torch.long)))
        raw=torch.stack(terms).mean() if terms else z.float().sum()*0.
        weight=self.student.cascade.mask_decoder.output_upscaling[0].weight
        def gradient(loss):
            g=torch.autograd.grad(loss*256.,weight,retain_graph=True,allow_unused=False)[0].detach().float()/256.
            if not torch.isfinite(g).all():
                raise RuntimeError('Nonfinite CAC gradient calibration')
            return g
        gs,gu=gradient(supervised),gradient(raw)
        ns,nu=gs.norm(),gu.norm()
        if terms:
            if nu<=1e-10 or ns<=1e-10:
                raise RuntimeError('Vanishing CAC/supervised gradient')
            scale=(ns/nu).detach()
            if scale>1e6:
                raise RuntimeError('CAC calibration beyond engineering bound')
        else:
            scale=raw.new_tensor(0.)
        extra=raw*scale
        cosine=float((gs*gu).sum()/(ns*nu).clamp_min(1e-20))
        self.cac_debug=dict(supervised=supervised,raw=raw,extra=extra,used=bool(terms),
            positives=pcount,negatives=ncount,conflict=conflict,sup_norm=float(ns),raw_norm=float(nu),
            scale=float(scale),cosine=cosine,axis=axis,reverse=reverse)
        self._append('cac_steps.jsonl',dict(epoch=epoch,step=step,**{k:v for k,v in self.cac_debug.items() if k not in ('supervised','raw','extra')},
            supervised=float(supervised.detach()),raw=float(raw.detach()),extra=float(extra.detach()),
            inherited_unlabeled_weight=0.,background_anchor_weight=0.))
        # A teacher target is retained only for logging compatibility; not optimized.
        self._ucp_teacher_logits=tb.detach()
        stats=dict(selected_ratio=float(positives.float().mean()))
        return supervised+extra,supervised,raw*0.,extra,logits,stats,[]
