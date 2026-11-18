"""ReLiF cutoff/recovery metadata. No model imports, dispatch or test execution."""
import copy
import math

FORMAT = 'PUBLIC_RELIF_COMPLETE_RECOVERY_DRAFT_V1'
ROLES = ('formal', 'engineering_warmup', 'engineering_larc')


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def plan(labels):
    require(type(labels) is int and labels in (5, 10), 'Expected five or ten CASES')
    per_cycle = 5 * math.ceil((346 - labels) / 5)
    total = 36 * per_cycle
    return dict(labeled_cases=labels, unlabeled_cases=346-labels,
                ssl_iterations=11*per_cycle, primary_inputs_per_stream=11*per_cycle,
                optimizer_updates=11*per_cycle, original_clock_iterations=total,
                primary_pairs_per_cycle=per_cycle, prefix_updates=0,
                original_larc_start_index=math.ceil(1000*total/30000),
                original_larc_full_weight_index=math.ceil(2000*total/30000),
                overshoot_per_stream=0)


def role_window(recipe, role):
    require(role in ROLES, 'Unknown execution role')
    if role == 'formal':
        return 0, recipe['ssl_iterations']
    start = 0 if role == 'engineering_warmup' else recipe['original_larc_start_index']
    # Only a prospective bounded qualification window. It is never formal work.
    return start, start + 4


def counts(n):
    require(type(n) is int and n >= 0, 'Invalid integral update count')
    return dict(iterations=n, labeled_inputs=n, unlabeled_inputs=n,
                successful_optimizer_updates=n)


def validate(counters, recipe, role='formal'):
    start, stop = role_window(recipe, role)
    require(isinstance(counters, dict) and set(counters) == set(counts(0)), 'Counter fields differ')
    require(all(type(v) is int and v >= 0 for v in counters.values()), 'Nonintegral/negative counters')
    n = counters['iterations']
    require(counters == counts(n) and n <= stop-start, 'Native input/update accounting or cutoff differs')
    return counters


def accept(counters, row, recipe, role, expected_pair, expected_clock):
    validate(counters, recipe, role)
    start, stop = role_window(recipe, role)
    i = start + counters['iterations']
    require(i < stop, 'Update after cutoff')
    require(row.get('actual_pair') == expected_pair and row.get('completed_primary_pairs') == i+1,
            'Actual schedule row/cursor differs')
    for k in ('labeled_inputs', 'unlabeled_inputs', 'successful_optimizer_updates'):
        require(type(row.get(k)) is int and row[k] == 1, 'ReLiF requires one input per stream and one update')
    for k in ('finite_loss', 'finite_nonzero_gradients', 'actual_case_ids_verified', 'teacher_no_grad'):
        require(row.get(k) is True, 'Missing actual update proof: '+k)
    require(row.get('skip_count') == 0 and row.get('unlabeled_gt_in_loss') is False, 'Skipped update or unlabeled GT')
    native = row.get('native_receipt', {})
    require(native.get('step') == i+1 and native.get('clock') == expected_clock,
            'Native schedule clock compressed or changed')
    require(expected_clock['total'] == recipe['original_clock_iterations'] and expected_clock['step'] == i,
            'Expected clock is not original 36-cycle clock')
    for k, v in dict(primary_labeled=1, primary_unlabeled=1, optimizer_updates=1,
                     student_case_forwards=4, foundation_case_forwards=2, sobf_views=2,
                     teacher_case_forwards=0, extra_donor_inputs=0, test_evaluations=0).items():
        require(type(native.get(k)) is int and native[k] == v, 'Native compute accounting differs: '+k)
    require(native.get('unlab_gt_in_loss') is False and native.get('case_ids') ==
            dict(labeled=expected_pair['labeled'], unlabeled=expected_pair['unlabeled']), 'Native case identities differ')
    for k, v in row.get('losses', {}).items():
        require(type(v) in (float, int) and math.isfinite(v), 'Nonfinite loss: '+k)
    require(set(row.get('losses', {})) == {'loss','supervised_ce','supervised_dice','distillation','cgcr','larc'},
            'Incomplete loss receipt')
    require(type(row.get('gradient_norm')) in (float, int) and math.isfinite(row['gradient_norm'])
            and row['gradient_norm'] > 0, 'Missing finite nonzero joint gradient')
    return validate(counts(counters['iterations']+1), recipe, role)


def validate_payload(state, identity, recipe, role, pairs):
    """Validate role, clocks and actual row ledger before touching model tensors.

    The tensor/RNG value roundtrip is a separate future GPU qualification; these
    metadata checks never claim it has happened.
    """
    require(state.get('format') == FORMAT and state.get('contract_sha256') == identity, 'Recovery format/contract differs')
    require(state.get('recipe') == recipe and state.get('execution_role') == role, 'Recovery role/recipe differs')
    c = validate(state['counts'], recipe, role)
    start, stop = role_window(recipe, role)
    position = start + c['iterations']
    require(type(state['position']) is int and state['position'] == position, 'Recovery cursor reset/shift')
    core = state['core_state']
    require(type(core['step']) is int and core['step'] == position
            and core['total'] == recipe['original_clock_iterations'], 'Native recovery clock differs')
    require(tuple(core['shape']) == (128,128,128), 'Native crop changed')
    require(set(core) == {'student','projection','optimizer','step','total','shape','implementation'},
            'Incomplete native student/projection/optimizer state')
    require(state['actual_rows'] == pairs[start:position], 'Actual recovery rows differ')
    require(state['cycles_completed'] == (position // recipe['primary_pairs_per_cycle'] if role == 'formal' else 0),
            'Recovered cycle count differs')
    require(set(state['module_modes']) == set(state['requires_grad']) == {'student','projection','prior'},
            'Missing module modes/gradient flags')
    for group in state['module_modes'].values():
        require(isinstance(group,dict) and '' in group and all(type(v) is bool for v in group.values()), 'Invalid module modes')
    for group in state['requires_grad'].values():
        require(isinstance(group,dict) and group and all(type(v) is bool for v in group.values()), 'Invalid gradient flags')
    require(state['module_modes']['student'][''] and state['module_modes']['projection'][''], 'Trainable modules not training')
    require(not any(state['module_modes']['prior'].values()) and not any(state['requires_grad']['prior'].values()),
            'Frozen original SAM mode/gradients differ')
    require(state['foundation']['variant'] == 'vit_b_ori' and state['foundation']['model_family'] == 'SAM-Med3D_original',
            'Original SAM replaced')
    require(isinstance(state['frozen_prior_signature'],str) and len(state['frozen_prior_signature']) == 64,
            'Frozen SAM tensor signature missing')
    require(isinstance(state['rng'],(list,tuple)) and len(state['rng']) == 4, 'Incomplete Python/NumPy/CPU/CUDA RNG')
    return copy.deepcopy(c)
