import torch
import torch.nn.functional as F

def _image_signal(image, size):
    corner = image[:, :, :1, :1, :1]
    return F.adaptive_avg_pool3d((image - corner).abs().gt(1e-06).float(), size) >= 0.5

def local_uncertainty(student_logits, teacher_logits, image, epoch):
    size = (32, 32, 32)
    p = F.adaptive_avg_pool3d(student_logits.float().softmax(1), size)
    q = F.adaptive_avg_pool3d(teacher_logits.detach().float().softmax(1), size)
    hp = -(p * p.clamp_min(1e-07).log()).sum(1, keepdim=True)
    hq = -(q * q.clamp_min(1e-07).log()).sum(1, keepdim=True)
    beta = 0.1 ** (min(max(epoch, 0), 35) / 35.0)
    disagreement = (p - q).square().sum(1, keepdim=True) / ((beta * hp).exp() + (beta * hq).exp())
    per_voxel = disagreement + beta * (hp + hq)
    core = q[:, 1:2].ge(0.1)
    region = F.max_pool3d(core.float(), 7, stride=1, padding=3).bool()
    region &= _image_signal(image, size)
    count = region.sum()
    loss = (per_voxel * region).sum() / count.clamp_min(1)
    return (loss, dict(beta=beta, region_count=int(count), region_fraction=float(region.float().mean()), local_mse=float((disagreement * region).sum().detach() / count.clamp_min(1))))

def anchored_feature_contrast(labeled_feature, unlabeled_feature, teacher_logits, student_logits, labeled_gt, labeled_image, unlabeled_image):
    size = (32, 32, 32)
    assert labeled_feature.shape[:2] == unlabeled_feature.shape[:2] and labeled_feature.shape[0] == 1, (tuple(labeled_feature.shape), tuple(unlabeled_feature.shape))
    labeled_feature = F.adaptive_avg_pool3d(labeled_feature, size)
    unlabeled_feature = F.adaptive_avg_pool3d(unlabeled_feature, size)
    q = F.adaptive_avg_pool3d(teacher_logits.detach().float().softmax(1)[:, 1:2], size)
    p = F.adaptive_avg_pool3d(student_logits.float().softmax(1)[:, 1:2], size)
    gt = F.adaptive_avg_pool3d(labeled_gt.float(), size)
    labeled_signal = gt.ge(0.5)
    valid = _image_signal(unlabeled_image, size)
    threshold = 0.7
    teacher_neighborhood = F.max_pool3d(q.ge(0.1).float(), 7, stride=1, padding=3).bool()
    query_masks = (q.le(1 - threshold) & valid & teacher_neighborhood, q.ge(threshold) & valid)
    labeled_valid = _image_signal(labeled_image, size)
    labeled_neighborhood = F.max_pool3d(labeled_signal.float(), 7, stride=1, padding=3).bool()
    key_masks = (gt.le(0.01) & labeled_valid & labeled_neighborhood, labeled_signal & labeled_valid)
    query_vectors, key_vectors, labels = ([], [], [])
    counts = {}
    for cls in (0, 1):
        key_index = torch.where(key_masks[cls].flatten())[0]
        query_index = torch.where(query_masks[cls].flatten())[0]
        counts[f'key_{cls}'] = int(key_index.numel())
        counts[f'query_{cls}'] = int(query_index.numel())
        if min(key_index.numel(), query_index.numel()) < 4:
            return (unlabeled_feature.float().sum() * 0, dict(**counts, used=False))
        query_rank = (p - q).abs().flatten()[query_index]
        qi = query_index[torch.topk(query_rank, min(64, len(query_index))).indices]
        key_rank = gt.flatten()[key_index] if cls else 1 - gt.flatten()[key_index]
        ki = key_index[torch.topk(key_rank, min(64, len(key_index))).indices]
        query_vectors.append(unlabeled_feature.float().flatten(2)[0].T[qi])
        key_vectors.append(labeled_feature.detach().float().flatten(2)[0].T[ki])
        labels.append(cls)
    keys = [F.normalize(v, dim=1, eps=1e-06) for v in key_vectors]
    losses = []
    for cls, vectors in enumerate(query_vectors):
        query = F.normalize(vectors, dim=1, eps=1e-06)
        positive = (query @ keys[cls].T).clamp(-1, 1)
        negative = (query @ keys[1 - cls].T).clamp(-1, 1)
        wp = ((1 - positive.detach()) / 2).clamp_min(0.0001).pow(0.8)
        wn = ((1 + negative.detach()) / 2).clamp_min(0.0001).pow(0.8)
        log_neg = torch.logsumexp(negative / 0.6 + wn.log(), dim=1, keepdim=True)
        pair = torch.logaddexp(positive / 0.6, log_neg) - positive / 0.6
        losses.append((pair * wp).sum() / wp.sum().clamp_min(1e-06))
    return (0.5 * (losses[0] + losses[1]), dict(**counts, used=True, query_total=sum((len(v) for v in query_vectors))))

def self_test():
    torch.manual_seed(27)
    s = torch.randn(1, 2, 16, 16, 16, requires_grad=True)
    t = torch.randn_like(s, requires_grad=True)
    image = torch.randn(1, 1, 16, 16, 16)
    u, _ = local_uncertainty(s, t, image, 0)
    u.backward()
    assert torch.isfinite(u) and s.grad is not None and (t.grad is None)
    return dict(status='UNIT_PASS', teacher_detached=True, finite=True)
