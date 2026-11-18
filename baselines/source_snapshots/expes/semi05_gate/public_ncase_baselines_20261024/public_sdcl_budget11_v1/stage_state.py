"""Pure SDCL phase accounting; no model, data, CUDA, or filesystem access."""
import math


def plan(labels):
    if type(labels) is not int or labels not in (5, 10):
        raise ValueError('Expected five or ten labeled cases')
    reference = 11 * 5 * math.ceil((346 - labels) / 5)
    actual = 4 * math.ceil(reference / 4)
    prefix = 81 * 4 * (labels // 2)
    return dict(labeled_cases=labels, reference_inputs=reference,
                primary_inputs=actual, rounding_extra_inputs=actual-reference,
                prefix_iterations=prefix, prefix_labeled_inputs=4*prefix,
                prefix_updates=2*prefix, ssl_iterations=actual//4,
                ssl_updates=actual//2, original_clock_inputs=36*5*math.ceil((346-labels)/5))


def counts(iterations, labeled, unlabeled, updates):
    return dict(iterations=iterations, labeled_inputs=labeled,
                unlabeled_inputs=unlabeled, successful_optimizer_updates=updates)


def initial():
    return dict(phase='prefix', prefix=counts(0, 0, 0, 0), ssl=counts(0, 0, 0, 0))


def validate(progress, recipe):
    if set(progress) != {'phase', 'prefix', 'ssl'} or progress['phase'] not in ('prefix', 'ssl'):
        raise ValueError('Unknown phase/progress fields')
    for phase in ('prefix', 'ssl'):
        c = progress[phase]
        if set(c) != set(counts(0, 0, 0, 0)) or any(type(x) is not int or x < 0 for x in c.values()):
            raise ValueError('Non-integral or invalid phase counters')
        n = c['iterations']
        if c != counts(n, 4*n, 0 if phase == 'prefix' else 4*n, 2*n):
            raise ValueError('Native SDCL input/update counters differ')
        if n > recipe[phase+'_iterations']:
            raise ValueError('Phase budget exceeded')
    if progress['phase'] == 'prefix' and progress['ssl']['iterations'] != 0:
        raise ValueError('SSL before prefix transition')
    if progress['phase'] == 'ssl' and progress['prefix']['iterations'] != recipe['prefix_iterations']:
        raise ValueError('Incomplete mandatory prefix')
    return progress


def accept(progress, step, recipe):
    import copy
    validate(progress, recipe)
    phase = progress['phase']
    if step.get('phase') != phase:
        raise ValueError('Step belongs to another phase')
    c = progress[phase]
    expected = dict(labeled_inputs=4, unlabeled_inputs=0 if phase == 'prefix' else 4,
                    successful_optimizer_updates=2)
    if any(type(step.get(k)) is not int or step[k] != v for k, v in expected.items()):
        raise ValueError('Actual native batch/update receipt differs')
    if step.get('iteration') != c['iterations']+1:
        raise ValueError('Non-contiguous phase iteration')
    required = ('finite_loss', 'finite_nonzero_gradients', 'actual_case_ids_verified', 'teacher_no_grad')
    if any(step.get(k) is not True for k in required) or step.get('skip_count') != 0 or step.get('unlabeled_gt_in_loss') is not False:
        raise ValueError('Invalid actual update proof')
    losses = step.get('losses')
    if not isinstance(losses, dict) or set(losses) != {'loss_v', 'loss_r'} or any(type(x) not in (int, float) or not math.isfinite(x) for x in losses.values()):
        raise ValueError('Nonfinite or missing dual losses')
    norms = step.get('gradient_norms')
    if not isinstance(norms, list) or len(norms) != 2 or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in norms):
        raise ValueError('Missing successful dual optimizer gradients')
    out = copy.deepcopy(progress)
    out[phase] = {k: v+(1 if k == 'iterations' else expected[k]) for k, v in c.items()}
    return validate(out, recipe)


def transition(progress, recipe):
    import copy
    validate(progress, recipe)
    if progress['phase'] != 'prefix' or progress['prefix']['iterations'] != recipe['prefix_iterations']:
        raise ValueError('Prefix transition at wrong position')
    out = copy.deepcopy(progress)
    out['phase'] = 'ssl'
    return validate(out, recipe)


def ready_for_test(progress, recipe):
    validate(progress, recipe)
    return progress['phase'] == 'ssl' and progress['ssl']['iterations'] == recipe['ssl_iterations']
