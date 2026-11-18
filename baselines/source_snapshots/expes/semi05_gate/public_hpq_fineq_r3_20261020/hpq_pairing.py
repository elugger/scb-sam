"""Pair actual strict cases while retaining the frozen RNG/token helper.

Only data orchestration changes: the frozen loss/gradient/EMA step is untouched.
This module is not a qualified training launcher.
"""
from itertools import cycle
from hpq_contract import (require, epoch_tokens, validate_loaded_case,
                          validate_splits, validate_micro_trace, validate_epoch_trace)


def _scalar(value):
    if isinstance(value, (list, tuple)):
        require(len(value) == 1, 'Only batch1 is permitted')
        return value[0]
    if hasattr(value, 'numel'):
        require(value.numel() == 1, 'Metadata must be scalar')
        return value.item()
    return value


def validate_collated(batch, expected_id, index, augmentation_seed, labeled):
    require(set(batch) == {'image', 'case_id', 'actual_index', 'gt_roi_used', 'origin_adjusted',
                           'direction_adjusted', 'requested_index', 'augmentation_seed'} |
            ({'label'} if labeled else set()), 'Unexpected collated fields')
    require(_scalar(batch['requested_index']) == index, 'Requested index changed')
    require(_scalar(batch['augmentation_seed']) == augmentation_seed, 'Frozen augmentation seed changed')
    sample = {k: v if k in ('image', 'label') else _scalar(v)
              for k, v in batch.items() if k not in ('requested_index', 'augmentation_seed')}
    validate_loaded_case(sample, expected_id, index, labeled)


def _iter_groups(engine, epoch, record, helpers):
    labeled_tokens, paired_loader, stream_seed, tensor_digest = helpers
    t = engine.t
    splits = engine.input_splits
    validate_splits(splits)
    require(t.pair_seed == 2027 and t.refresh is False and t.workers == 0,
            'Frozen paired RNG/label-cycle/worker rule changed')
    require(tuple(t.lab_set.actual_case_ids) == tuple(splits['train_lab']) and
            tuple(t.unlab_set.actual_case_ids) == tuple(splits['train_unlab']), 'Dataset order differs from bound splits')
    tokens = epoch_tokens(epoch)
    labs = labeled_tokens(5, 345, epoch, t.pair_seed, t.refresh)
    require(len(labs) == 5 and all(token[:2] == (epoch, 0) for token in labs), 'Unexpected labeled replay schedule')
    order = [token[2] for token in labs]
    lab = cycle(paired_loader(t.lab_set, labs, t.pair_seed, 'lab', t.workers))
    unlab = paired_loader(t.unlab_set, tokens, t.pair_seed, 'unlab', t.workers)
    rows, group = [], []
    for step, bu in enumerate(unlab):
        require(step < 345, 'Unexpected extra microbatch')
        bl = next(lab)
        lt, ut = labs[step % 5], tokens[step]
        ls, us = stream_seed(2027, 'lab', *lt), stream_seed(2027, 'unlab', *ut)
        validate_collated(bl, splits['train_lab'][lt[2]], lt[2], ls, True)
        validate_collated(bu, splits['train_unlab'][ut[2]], ut[2], us, False)
        xl, yl = t._prepare_batch(bl)
        xu = engine.normalize_batch(bu['image'], t.norm_transform).to(t.device)
        # Exact frozen loss signature: no unlab label/GT volume is present.
        group.append(dict(epoch=epoch, step=step, image_l=xl, label_l=yl, image_u=xu))
        row = dict(epoch=epoch, step=step, labeled_index=lt[2], unlabeled_index=ut[2],
                   labeled_case_id=_scalar(bl['case_id']), unlabeled_case_id=_scalar(bu['case_id']),
                   label_seed=ls, unlabeled_seed=us, student_seed=stream_seed(2027, 'student-step', epoch, step),
                   unlabeled_replay=ut[1], padding=bool(ut[1]),
                   padding_case_id=_scalar(bu['case_id']) if ut[1] else None,
                   unlabeled_label_returned=False, gt_roi_used=True)
        validate_micro_trace(row, splits, epoch, step, order)
        if record and epoch == 0:
            row.update(label_image_sha=tensor_digest(bl['image']), unlab_image_sha=tensor_digest(bu['image']))
        rows.append(row)
        if record:
            engine.append('pair_trace.jsonl', row)
        if len(group) == 5:
            yield group
            group = []
    require(not group, 'Incomplete accumulation group')
    receipt = validate_epoch_trace(rows, splits, epoch, order)
    if record:
        engine.append('epoch_case_coverage.jsonl', receipt)


def raw_groups(engine, epoch, record=False):
    from expes.semi05_gate.paired_rng_data_20260908 import (
        labeled_tokens, paired_loader, stream_seed, tensor_digest)
    yield from _iter_groups(engine, epoch, record,
                            (labeled_tokens, paired_loader, stream_seed, tensor_digest))
