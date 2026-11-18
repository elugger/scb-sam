"""SFR phase accounting; native 6000 SAM prefix and separate 36-cycle clock."""
import copy
import math


def plan(labels):
    if type(labels) is not int or labels not in (5, 10):
        raise ValueError('Expected five or ten labeled cases')
    reference = 11 * 5 * math.ceil((346-labels)/5)
    actual = 2*math.ceil(reference/2)
    return dict(labeled_cases=labels, reference_inputs=reference, primary_inputs=actual,
                rounding_extra_inputs=actual-reference, prefix_iterations=6000,
                prefix_labeled_inputs=prefix_inputs(6000, labels), prefix_updates=6000,
                ssl_iterations=actual//2, ssl_updates=actual//2,
                original_clock_inputs=36*5*math.ceil((346-labels)/5),
                original_clock_ssl_iterations=36*5*math.ceil((346-labels)/5)//2)


def prefix_inputs(iterations, labels):
    full, remainder = divmod(iterations, (labels+1)//2)
    return full*labels+min(2*remainder, labels)


def counts(n, labeled, unlabeled):
    return dict(iterations=n, labeled_inputs=labeled, unlabeled_inputs=unlabeled,
                successful_optimizer_updates=n)


def initial():
    return dict(phase='prefix', prefix=counts(0, 0, 0), ssl=counts(0, 0, 0))


def validate(progress, recipe):
    if set(progress) != {'phase', 'prefix', 'ssl'} or progress['phase'] not in ('prefix', 'ssl'):
        raise ValueError('Unknown SFR phase/progress')
    for phase in ('prefix', 'ssl'):
        c = progress[phase]
        if set(c) != set(counts(0, 0, 0)) or any(type(x) is not int or x < 0 for x in c.values()):
            raise ValueError('Invalid integral SFR counters')
        n = c['iterations']
        expected = counts(n, prefix_inputs(n, recipe['labeled_cases']), 0) if phase == 'prefix' else counts(n, 2*n, 2*n)
        if c != expected or n > recipe[phase+'_iterations']:
            raise ValueError('SFR native input/update count or endpoint differs')
    if progress['phase'] == 'prefix' and progress['ssl']['iterations']:
        raise ValueError('SFR SSL before mandatory prefix')
    if progress['phase'] == 'ssl' and progress['prefix']['iterations'] != recipe['prefix_iterations']:
        raise ValueError('SFR incomplete prefix at phase transition')
    return progress


def accept(progress, step, recipe):
    validate(progress, recipe)
    phase = progress['phase']; n = progress[phase]['iterations']
    size = prefix_inputs(n+1, recipe['labeled_cases'])-prefix_inputs(n, recipe['labeled_cases']) if phase == 'prefix' else 2
    expected = dict(labeled_inputs=size, unlabeled_inputs=0 if phase == 'prefix' else 2, successful_optimizer_updates=1)
    if step.get('phase') != phase or step.get('iteration') != n+1:
        raise ValueError('Noncontiguous SFR phase step')
    if any(type(step.get(k)) is not int or step[k] != v for k, v in expected.items()):
        raise ValueError('SFR native batch or optimizer increment differs')
    required = ('finite_loss', 'finite_nonzero_gradients', 'actual_case_ids_verified')
    if any(step.get(k) is not True for k in required) or step.get('unlabeled_gt_in_loss') is not False or step.get('skip_count') != 0:
        raise ValueError('Incomplete actual SFR update proof')
    if phase == 'ssl' and (step.get('teacher_no_grad') is not True or step.get('sam_no_grad') is not True):
        raise ValueError('SFR frozen auxiliary received gradients')
    for key in ('loss', 'gradient_norm'):
        x = step.get(key)
        if type(x) not in (float, int) or not math.isfinite(x) or key == 'gradient_norm' and x <= 0:
            raise ValueError('Missing/nonfinite SFR loss or gradient')
    out = copy.deepcopy(progress)
    out[phase] = {k: v+(1 if k == 'iterations' else expected[k]) for k, v in progress[phase].items()}
    return validate(out, recipe)


def transition(progress, recipe):
    validate(progress, recipe)
    if progress['phase'] != 'prefix' or progress['prefix']['iterations'] != recipe['prefix_iterations']:
        raise ValueError('SFR prefix transition at wrong position')
    out = copy.deepcopy(progress); out['phase'] = 'ssl'
    return validate(out, recipe)


def ready_for_test(progress, recipe):
    validate(progress, recipe)
    return progress['phase'] == 'ssl' and progress['ssl']['iterations'] == recipe['ssl_iterations']
