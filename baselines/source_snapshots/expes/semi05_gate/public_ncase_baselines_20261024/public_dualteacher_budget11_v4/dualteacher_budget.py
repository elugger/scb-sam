"""Prospective Public cutoff and counters; no models, jobs or dispatch."""
import math


def require(ok, message):
    if not ok:
        raise ValueError(message)


def plan(labels):
    require(type(labels) is int and labels in (5, 10), 'Five or ten labeled CASES required')
    per_cycle = 5 * math.ceil((346-labels)/5)
    iterations = math.ceil(11*per_cycle/2)
    original = 36*per_cycle//2
    return dict(labeled_cases=labels, unlabeled_cases=346-labels,
                primary_pairs_per_cycle=per_cycle, reference_primary_inputs=11*per_cycle,
                ssl_iterations=iterations, primary_inputs_per_stream=2*iterations,
                overshoot_per_stream=2*iterations-11*per_cycle,
                original_clock_iterations=original, original_sam_start_index=math.ceil(6*original/200),
                prefix_updates=0, effective_optimizer_updates=2*iterations,
                student_optimizer_updates=iterations, op_effective_updates=iterations,
                op_step_calls=2*iterations, terminal_selection='fixed_terminal_EMA_plus_OP')


def counts(iterations):
    require(type(iterations) is int and iterations >= 0, 'Integral nonnegative iteration count required')
    return dict(iterations=iterations, labeled_inputs=2*iterations, unlabeled_inputs=2*iterations,
                successful_optimizer_updates=2*iterations, student_optimizer_updates=iterations,
                op_effective_updates=iterations, op_step_calls=2*iterations)


def validate(counters, recipe):
    require(recipe == plan(recipe['labeled_cases']), 'Prospective recipe changed')
    require(isinstance(counters, dict) and set(counters) == set(counts(0)), 'Incomplete counters')
    require(all(type(x) is int for x in counters.values()), 'Nonintegral/boolean counter')
    n = counters['iterations']
    require(counters == counts(n) and n <= recipe['ssl_iterations'], 'Counts or fixed cutoff differ')
    return counters


def validate_clock(clock, index, recipe):
    """Check metadata against the unchanged native clock, never compress it."""
    require(recipe == plan(recipe['labeled_cases']), 'Prospective recipe changed')
    require(type(index) is int and 0 <= index < recipe['ssl_iterations'], 'Outside cutoff')
    total = recipe['original_clock_iterations']
    epoch = min(199, 200*index//total)
    expected = dict(iteration=index, total_iterations=total, native_epoch=epoch,
                    native_reference_iteration=40000*index/total, primary_progress=index/total,
                    learning_rate=max(1e-6, .0025*(1-epoch/200)**.9), sam_enabled=epoch>5)
    require(clock == expected, 'Native original clock changed or compressed')
    return clock
