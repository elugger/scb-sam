import torch
import torch.nn.functional as F
OFFSETS = (1024, -1024, 32, -32, 1, -1)

def pick_regions(qa, qb, signal, axis):
    qa, qb, signal = (qa.detach(), qb.detach(), signal.detach())
    overlap = torch.zeros_like(signal)
    sl = [slice(None)] * 5
    sl[axis + 2] = slice(8, 24)
    overlap[tuple(sl)] = True
    support = signal & overlap
    hard = support & ((qa > 0.3) & (qa < 0.7) | (qb > 0.3) & (qb < 0.7) | ((qa >= 0.5) != (qb >= 0.5)))
    easy = support & ~hard & ((qa >= 0.5) == (qb >= 0.5))
    interior = F.avg_pool3d(support.float(), 3, 1, 1) >= 1.0 - 1e-06
    hi = (hard & interior).flatten().nonzero()[:, 0]
    ei = (easy & interior).flatten().nonzero()[:, 0]
    count = min(128, len(hi), len(ei))

    def take(ids):
        return ids[torch.linspace(0, len(ids) - 1, count, device=ids.device).long()] if count else ids[:0]
    reliable = support & ((qa >= 0.7) & (qb >= 0.7) | (qa <= 0.3) & (qb <= 0.3))
    return (take(hi), take(ei), reliable, dict(hard_available=len(hi), easy_available=len(ei), matched_count=count))

def vectors(feature):
    assert tuple(feature.shape[-3:]) == (32, 32, 32) and feature.shape[0] == 1
    return feature.float()[0].flatten(1).T

def edges(z, ids):
    offsets = torch.tensor(OFFSETS, device=ids.device)
    return z[ids, None, :] - z[ids[:, None] + offsets[None, :]]

def displacement_loss(student, teacher, ids):
    assert len(ids) >= 2
    target = edges(teacher.detach(), ids)
    energy = target.square().mean().detach()
    if not torch.isfinite(energy) or energy <= 1e-08:
        raise RuntimeError('Degenerate teacher spatial targets; no automatic rescaling/retry')
    loss = (edges(student, ids) - target).square().mean() / energy
    return (loss, energy)

def relation_loss(student_feature, teacher_feature, student_logits, teacher_logits, ids):
    sf, tf = (vectors(student_feature), vectors(teacher_feature))
    sl = vectors(F.avg_pool3d((student_logits[:, 1:2] - student_logits[:, 0:1]).float(), 4, 4))
    tl = vectors(F.avg_pool3d((teacher_logits[:, 1:2] - teacher_logits[:, 0:1]).float(), 4, 4))
    feature, ef = displacement_loss(sf, tf, ids)
    output, eo = displacement_loss(sl, tl, ids)
    return ((feature + output) / 2, dict(feature=feature, output=output, feature_energy=ef, output_energy=eo))

def reliable_class_loss(logits, qa, qb, reliable):
    assert not qa.requires_grad and (not qb.requires_grad) and (not reliable.requires_grad)
    mask = F.interpolate(reliable.float(), size=(128, 128, 128), mode='nearest').bool()[:, 0]
    target = F.interpolate((qb >= 0.5).float(), size=(128, 128, 128), mode='nearest').long()[:, 0]
    ce = F.cross_entropy(logits.float(), target, reduction='none')
    terms = [ce[mask & (target == c)].mean() for c in (0, 1) if (mask & (target == c)).any()]
    return torch.stack(terms).mean() if terms else logits.float().sum() * 0.0
