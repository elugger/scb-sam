"""Paper-aligned ReLiF interface repairs, not an unchanged-official-run claim.

Basis: MICCAI 2026 paper 1855, Sec.2.2/2.3, Eq.(3)-(6), pp.4-5.
These components still require a pinned original SAM checkpoint and real GPU
qualification before they can be used in a main-table run.
"""
import torch
import torch.nn.functional as F


def foundation_probabilities(single_foreground_logits):
    if (single_foreground_logits.ndim!=5 or single_foreground_logits.shape[1]!=1
            or not single_foreground_logits.is_floating_point()):
        raise ValueError('Require continuous single-channel SAM logits, not thresholded masks')
    if not bool(torch.isfinite(single_foreground_logits).all()):
        raise ValueError('Nonfinite SAM prior')
    fg=single_foreground_logits.detach().sigmoid()
    return torch.cat((1-fg,fg),dim=1)


def case_local_prompt(labeled_targets,student_probabilities,labeled_count):
    # Concatenation across batch roles, not addition/broadcast of a labeled
    # patient's target to a different unlabeled patient's prompt.
    if (student_probabilities.ndim!=5 or student_probabilities.shape[1]!=2
            or labeled_targets.shape[0]!=labeled_count
            or not 0<labeled_count<student_probabilities.shape[0]):
        raise ValueError('Actual labeled/unlabeled prompt batch differs')
    lab=labeled_targets.unsqueeze(1).to(student_probabilities.dtype)
    if lab.shape[2:]!=student_probabilities.shape[2:]:
        raise ValueError('Prompt spatial identity differs')
    return torch.cat((lab,student_probabilities[labeled_count:,1:2].detach()),dim=0)


def gated_co_regularization(clean_logits,augmented_logits,threshold):
    clean,aug=clean_logits.softmax(1),augmented_logits.softmax(1)
    conf_c,idx_c=clean.max(1,keepdim=True)
    conf_a,idx_a=aug.max(1,keepdim=True)
    gate=((idx_c==idx_a)&(conf_c>=threshold)&(conf_a>=threshold)).detach()
    return (((clean-aug).square().sum(1,keepdim=True)*gate).sum()
            /(gate.sum()+1e-6)),gate


def larc_loss(augmented_encoder_features,clean_encoder_features,prior,projection):
    if prior.ndim!=5 or prior.shape[1]!=2:
        raise ValueError('LARC needs real background/foreground channels')
    # The original official code detached both feature branches. Keep the
    # augmented encoder path live, stopping only the clean projected target.
    if not augmented_encoder_features.requires_grad:
        raise ValueError('Augmented encoder graph detached')
    weight=prior[:,1:2].detach().pow(1.5)
    valid=weight.sum((2,3,4)).squeeze(1)>50
    if not bool(valid.any()):
        return augmented_encoder_features.sum()*0,valid
    weight=F.interpolate(weight[valid],size=augmented_encoder_features.shape[2:],
                         mode='trilinear',align_corners=False)
    def pooled(features):
        return (features[valid]*weight).sum((2,3,4))/(weight.sum((2,3,4))+1e-6)
    za=F.normalize(projection(pooled(augmented_encoder_features)),dim=1)
    zc=F.normalize(projection(pooled(clean_encoder_features)),dim=1).detach()
    return (1-F.cosine_similarity(za,zc,dim=1)).mean(),valid


def verify_projection_optimizer(optimizer,projection):
    ids={id(p) for group in optimizer.param_groups for p in group['params']}
    if any(id(p) not in ids for p in projection.parameters() if p.requires_grad):
        raise ValueError('LARC projection parameters missing from actual optimizer')


def register_encoder_hook(model,buffer):
    if not hasattr(model,'center'):
        raise ValueError('Pinned ReLiF U-Net encoder center missing')
    return model.center.register_forward_hook(lambda module,inputs,output:buffer.append(output))
