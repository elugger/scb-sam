"""Prospective SFR stage/cost mapping, with no training or dispatch side effects.

The source is the pinned base SFR+MT/VNet implementation, not SFR+ confidence
selection. The native SAM prefix remains 6000 updates. Its batch-2 DataLoader
keeps a final one-case batch when the labeled set is odd. SSL retains 2+2 input
batching; only the original 6000-step LR/ramp clock is mapped to the fixed budget.
"""
import math
import random

from budget_protocol import BUDGETS

OFFICIAL_COMMIT = "dc32304b6ba7a6f9ecbcea5a30e9eebe98510ad6"
PREFIX_ITERATIONS = 6000
NATIVE_REFERENCE_ITERATIONS = 6000
PATCH = (112, 112, 80)
CANVAS = 1024


def stage_plan(dataset, ratio):
    budget = next((b for b in BUDGETS if b.dataset == dataset and b.ratio == ratio), None)
    if budget is None or budget.primary_pairs % 2:
        raise ValueError("SFR budget is outside the frozen scope or breaks native 2+2 batching")
    batches_per_epoch = (budget.labeled_cases + 1) // 2
    epochs, remaining_batches = divmod(PREFIX_ITERATIONS, batches_per_epoch)
    prefix_inputs = epochs * budget.labeled_cases + min(2 * remaining_batches, budget.labeled_cases)
    return dict(
        method="sfr", variant="base SFR with MT and default VNet", source_commit=OFFICIAL_COMMIT,
        dataset=dataset, ratio=ratio, seed=2027, patch=list(PATCH), sam_canvas=CANVAS,
        prefix_iterations=PREFIX_ITERATIONS, prefix_labeled_inputs=prefix_inputs,
        prefix_optimizer_updates=PREFIX_ITERATIONS, prefix_batch_size=2,
        prefix_keeps_native_one_case_epoch_tail=bool(budget.labeled_cases % 2),
        prefix_test_evaluations=0, prefix_selection="fixed_terminal_6000",
        prefix_optimizer=dict(name="AdamW", lr=.001, betas=[.9, .999], weight_decay=.1),
        prefix_augmentation=dict(native_nt=2, nonlinear_rate=.5, rotation_flip_2d=True),
        prefix_lora_rank=4, prefix_warmup=False,
        ssl_iterations=budget.primary_pairs // 2,
        ssl_labeled_inputs=budget.primary_pairs, ssl_unlabeled_inputs=budget.primary_pairs,
        ssl_optimizer_updates=budget.primary_pairs // 2,
        ssl_primary_labeled_per_iteration=2, ssl_primary_unlabeled_per_iteration=2,
        ssl_optimizer=dict(name="SGD", lr=.01, momentum=.9, weight_decay=.0001),
        ssl_reference_iterations=NATIVE_REFERENCE_ITERATIONS,
        ssl_lr_rule=".01 * .1**floor(6000*completed/(2500*ssl_iterations))",
        ssl_consistency_rule=".1 * sigmoid_rampup(floor(6000*completed/(150*ssl_iterations)),40)",
        ssl_ema_rule="min(1-1/(actual_zero_based_update+1),.99); parameters only, as source",
        ssl_teacher_input_forwards=2 * budget.primary_pairs,
        ssl_sam_stitched_board_forwards=budget.primary_pairs,
        prefix_sam_stitched_board_forwards=prefix_inputs,
        inference=dict(patch=list(PATCH), stride_xy=18, stride_z=4, aggregation="mean_softmax"),
        selection_uses_test=False, total_compute_matched=False, launch_ready=False,
    )


def finetune_batches(labeled_case_ids, seed=2027):
    """6000 finite batches, retaining official batch-2 epoch tails.

    The explicit case order is an adaptation for auditable identities. It does
    not claim the same random trajectory as the original multi-worker loader.
    """
    ids = tuple(labeled_case_ids)
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError("Unique labeled identities and at least two cases required")
    rng = random.Random(seed)
    completed = 0
    while completed < PREFIX_ITERATIONS:
        order = list(ids)
        rng.shuffle(order)
        for start in range(0, len(order), 2):
            yield tuple(order[start:start + 2])
            completed += 1
            if completed == PREFIX_ITERATIONS:
                return


def prefix_lr(completed):
    if type(completed) is not int or not 0 <= completed <= PREFIX_ITERATIONS:
        raise ValueError("Invalid completed prefix count")
    return .001 * (1 - completed / PREFIX_ITERATIONS) ** .9


def ssl_clock(completed, target_iterations):
    if (type(completed) is not int or type(target_iterations) is not int
            or target_iterations <= 0 or not 0 <= completed <= target_iterations):
        raise ValueError("Invalid SFR SSL clock")
    # Integer rational comparisons avoid boundary errors at the mapped drops.
    lr_decays = (NATIVE_REFERENCE_ITERATIONS * completed) // (2500 * target_iterations)
    ramp_epoch = (NATIVE_REFERENCE_ITERATIONS * completed) // (150 * target_iterations)
    phase = 1 - min(40, ramp_epoch) / 40
    return dict(lr=.01 * .1 ** lr_decays, consistency_weight=.1 * math.exp(-5 * phase * phase),
                reference_ramp_epoch=ramp_epoch, lr_decays=lr_decays)


def inference_windows(shape):
    """Original LA sliding-window coordinates, covering the complete ROI."""
    if len(shape) != 3 or any(type(v) is not int or v <= 0 for v in shape):
        raise ValueError("Positive three-dimensional shape required")
    padding = tuple((max(0, p - n) // 2, max(0, p - n) - max(0, p - n) // 2)
                    for n, p in zip(shape, PATCH))
    padded = tuple(n + a + b for n, (a, b) in zip(shape, padding))
    axes = [tuple(min(stride * i, n - patch) for i in range(math.ceil((n - patch) / stride) + 1))
            for n, patch, stride in zip(padded, PATCH, (18, 18, 4))]
    return dict(shape=list(shape), padded_shape=list(padded), padding=[list(p) for p in padding],
                starts=[list(x) for x in axes], window_count=math.prod(len(a) for a in axes),
                patch=list(PATCH), stride=[18, 18, 4])
