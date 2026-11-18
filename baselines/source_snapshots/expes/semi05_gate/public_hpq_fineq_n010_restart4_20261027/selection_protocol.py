"""Prospective ten-case FineQ selection logic; NOT a trainer or GPU admission.

No CUDA, filesystem writes, model loading, launch, automatic retry or resume.
Integration must bind full evaluation provenance and checkpoint receipts before
using these decisions. A CPU PASS does not qualify a runnable experiment.
"""
from collections import Counter
from decimal import Decimal
import hashlib
import math
from pathlib import Path

MAX_EPOCHS = 36
MIN_EPOCHS = 11
PATIENCE = 5
MIN_DELTA_PP = Decimal('0.05')
LABELED = 10
UNLABELED = 336
DEVELOPMENT = 87
ACCUMULATION = 5
MICROS_PER_EPOCH = 340
UPDATES_PER_EPOCH = 68
FINE_Q_SHA256 = 'fb78c11ea08e2d5ffd2e85c61a8e90b2c2b0e3db8248b139d10c62119919b308'
POLICY = dict(maximum_cycles=36, warmup_cycles=11, patience=5, min_delta_pp='0.05',
              labeled_cases=10, unlabeled_cases=336, microbatches_per_cycle=340,
              updates_per_cycle=68, selection='strict_raw_mean_Dice_exact_ties_earlier')

def policy_sha():
    import json
    return hashlib.sha256(json.dumps(POLICY, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

SPLIT_PINS = {
    'n005_lab.txt': '1ac1f18841b6d46134c71e75ddcfe83202b94c175a7f1c24a9aef8087e3eb808',
    'n005_unlab.txt': '027156909bd08ed658480129eee5e6903faa86e32ee872f69e7c2211eef92a25',
    'n010_lab.txt': '0359692a4d81ff32d92a4e00a2bc2a1aa65b6041c82672813c7544aa0b631121',
    'n010_unlab.txt': 'ee6322f4e0e910bd2cc7cb8f18ed7844157ce05fc5cea3b00450e6e56eb8b402',
    'original_lab17.txt': '34f4f47574cccb6fb6b09d44e284bb75a2a53b3779a10628dd6e4f331d9fe002',
    'test.txt': 'f50e39909a88adf23550d165d4538f832d9b7e012c12af957acf3393c11000a6',
    'train_whole.txt': '9fdf84b9a66f1e73a8c2b791b7d9c81d72a43bab5f42a59613dfce5a0b3be639',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def integer(value, low, high):
    return type(value) is int and low <= value <= high


def verify_inputs(split_directory, fine_q_source):
    """Read only fixed public metadata, not image/label volumes."""
    splits = {}
    for name, digest in SPLIT_PINS.items():
        raw = (Path(split_directory) / name).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == digest, 'Split bytes changed: ' + name)
        splits[name] = raw.decode('utf-8').splitlines()
        require(len(set(splits[name])) == len(splits[name]), 'Duplicate IDs')
    lab, unlab, whole, dev = [splits[n] for n in
        ('n010_lab.txt', 'n010_unlab.txt', 'train_whole.txt', 'test.txt')]
    require(tuple(map(len, (lab, unlab, whole, dev))) == (10, 336, 346, 87), 'Wrong counts')
    ranked = sorted(splits['original_lab17.txt'], key=lambda c:
        hashlib.sha256(('Public433/fold2/ncase/2027/' + c).encode()).hexdigest())
    require(lab == ranked[:10] and splits['n005_lab.txt'] == ranked[:5], 'Wrong nested ranking')
    require(unlab == [c for c in whole if c not in lab], 'Unlabeled order/membership changed')
    require(not set(lab) & set(unlab) and set(lab + unlab) == set(whole), 'Training union differs')
    require(not set(whole) & set(dev), 'Train/development overlap')
    require(hashlib.sha256(Path(fine_q_source).read_bytes()).hexdigest() == FINE_Q_SHA256,
            'Not accepted FineQ science')
    return dict(labeled=lab, unlabeled=unlab, development=dev,
                patient_alias_verified=False, gt_roi_used=True)


def budget(completed_epochs):
    require(integer(completed_epochs, 0, MAX_EPOCHS), 'Epoch budget invalid')
    return dict(completed_epochs=completed_epochs,
                successful_updates=UPDATES_PER_EPOCH * completed_epochs,
                primary_labeled_inputs=MICROS_PER_EPOCH * completed_epochs,
                primary_unlabeled_inputs=MICROS_PER_EPOCH * completed_epochs,
                ssl_inputs_per_labeled_case=34 * completed_epochs,
                supervised_prefix_updates=600,
                supervised_prefix_inputs_per_labeled_case=60,
                legal_unlabeled_replays=4 * completed_epochs,
                total_compute_matched=False)


def cycle_indices(epoch, labeled_order):
    """Expected coverage only; order must come from the bound original RNG helper."""
    require(integer(epoch, 0, MAX_EPOCHS - 1), 'Epoch index invalid')
    require(len(labeled_order) == 10 and all(type(i) is int for i in labeled_order)
            and set(labeled_order) == set(range(10)), 'Invalid labeled permutation')
    unlab = [(i, False) for i in range(336)] + [((epoch * 4 + j) % 336, True) for j in range(4)]
    result = [(labeled_order[i % 10], ui, replay) for i, (ui, replay) in enumerate(unlab)]
    require(set(Counter(li for li, _, _ in result).values()) == {34}, 'Label coverage imbalance')
    return result


def evaluation_record(completed_epochs, successful_updates, case_rows, expected_case_ids):
    """Aggregate exactly one complete ordered 87-case evaluation, Dice in [0,1]."""
    require(integer(completed_epochs, 1, MAX_EPOCHS), 'Invalid completed epoch')
    require(type(successful_updates) is int and successful_updates == completed_epochs * 68,
            'Evaluation not at completed cycle')
    require(len(expected_case_ids) == 87 and len(set(expected_case_ids)) == 87, 'Invalid development manifest')
    require([row['case_id'] for row in case_rows] == list(expected_case_ids), 'Incomplete/reordered evaluation')
    values = [row['dice'] for row in case_rows]
    require(all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in values),
            'Invalid Dice fraction')
    dice_pp = sum((Decimal(str(v)) for v in values), Decimal(0)) * 100 / 87
    return dict(completed_epochs=completed_epochs, successful_updates=successful_updates,
                case_count=87, mean_dice_percent=str(dice_pp))


def decision(history):
    """Strict-best selection is separate from the plateau counter.

    At epoch11 set the significant-improvement anchor to the best score so far.
    Starting epoch12 reset only on >=0.05pp improvement over that anchor;
    otherwise count one stale epoch. Stop at five stale epochs (earliest16)
    or epoch36. Exact raw-score ties retain the earlier checkpoint. Incomplete
    evaluations cannot be admitted by evaluation_record and never tick patience.
    """
    require(isinstance(history, list) and len(history) <= 36, 'Invalid history')
    best = anchor = None
    best_epoch = None
    stale = 0
    stopped = False
    reason = None
    for epoch, row in enumerate(history, start=1):
        require(not stopped, 'Evidence contains training past the frozen stopping boundary')
        require(type(row['completed_epochs']) is int and row['completed_epochs'] == epoch,
                'Missing/duplicate/out-of-order cycle')
        require(type(row['successful_updates']) is int and row['successful_updates'] == epoch * 68,
                'Wrong successful update count')
        require(type(row['case_count']) is int and row['case_count'] == 87, 'Partial evaluation')
        score = Decimal(str(row['mean_dice_percent']))
        require(score.is_finite() and 0 <= score <= 100, 'Invalid aggregate Dice')
        if best is None or score > best:
            best, best_epoch = score, epoch
        if epoch == MIN_EPOCHS:
            anchor = best
        elif epoch > MIN_EPOCHS:
            if score >= anchor + MIN_DELTA_PP:
                anchor, stale = score, 0
            else:
                stale += 1
        stopped = epoch == MAX_EPOCHS or stale >= PATIENCE
        reason = 'MAX_36_CYCLES' if epoch == MAX_EPOCHS else ('PLATEAU_5_CYCLES' if stopped else None)
    return dict(stop=stopped, stop_reason=reason, completed_epochs=len(history),
                best_completed_epoch=best_epoch,
                best_dice_percent=None if best is None else str(best),
                stale_epochs_after_warmup=stale,
                current_is_strict_best=bool(history) and best_epoch == len(history),
                checkpoint_receipts_verified=False, trainer_integrated=False)
