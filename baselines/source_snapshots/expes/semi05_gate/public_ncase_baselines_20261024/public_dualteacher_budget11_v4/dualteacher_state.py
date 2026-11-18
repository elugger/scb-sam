"""Draft complete-boundary metadata and native receipt checks. No models."""
import copy
import math
from dualteacher_budget import plan,counts,require,validate_clock

FORMAT='PUBLIC_DUALTEACHER_COMPLETE_RECOVERY_NUMERIC_V2'
ROLES=('formal','engineering_warmup','engineering_sam')
MODELS={'student','ema','op','criterion','sam'}
CORE_KEYS={'student','ema','op','criterion','optimizer','optimizer_op','ema_step','position','dynamic_threshold_fg','dynamic_threshold_bg'}


def role_window(recipe,role):
    require(recipe==plan(recipe['labeled_cases']) and role in ROLES,'Recipe/role changed')
    start=0 if role!='engineering_sam' else recipe['original_sam_start_index']
    return start,recipe['ssl_iterations'] if role=='formal' else start+4


def validate_counts(value,recipe,role):
    start,stop=role_window(recipe,role)
    require(isinstance(value,dict) and set(value)==set(counts(0)),'Missing native counters')
    require(all(type(v) is int for v in value.values()),'Nonintegral counters')
    n=value['iterations'];require(value==counts(n) and n<=stop-start,'Native iteration/update/input accounting differs')
    return value


def finite_number(v):
    return type(v) in (int,float) and math.isfinite(v)


def accept(value,row,recipe,role,expected_pairs,expected_clock):
    validate_counts(value,recipe,role);start,stop=role_window(recipe,role);i=start+value['iterations']
    require(i<stop and len(expected_pairs)==2,'Outside cutoff or short native batch')
    validate_clock(expected_clock,i,recipe)
    require(row.get('actual_pairs')==expected_pairs and row.get('completed_primary_pairs')==2*(i+1),'Actual pair cursor differs')
    # The unchanged BY adapter calls the student update primary; native receipt
    # separately proves OP. New cumulative total counts both effective updates.
    for k,v in dict(labeled_inputs=2,unlabeled_inputs=2,successful_optimizer_updates=1,skip_count=0).items():
        require(type(row.get(k)) is int and row[k]==v,'Native primary adapter field differs: '+k)
    for k in ('finite_loss','finite_nonzero_gradients','actual_case_ids_verified','teacher_no_grad'):
        require(row.get(k) is True,'Missing update proof: '+k)
    require(row.get('unlabeled_gt_in_loss') is False,'Unlabeled GT crossed boundary')
    native=row.get('native_receipt',{})
    require(type(native.get('step')) is int and native['step']==i+1 and native.get('clock')==expected_clock,'Native cursor/clock changed')
    require(native.get('case_ids')=={k:[p[k] for p in expected_pairs] for k in ('labeled','unlabeled')},'Native case identities differ')
    for k,v in dict(primary_labeled=2,primary_unlabeled=2,optimizer_updates=2,primary_optimizer_updates=1,
        student_optimizer_calls=1,op_optimizer_calls=2,effective_student_updates=1,effective_op_updates=1,
        student_case_forwards=6,op_case_forwards=4,teacher_case_forwards=4 if expected_clock['sam_enabled'] else 2,test_evaluations=0).items():
        require(type(native.get(k)) is int and native[k]==v,'Native accounting differs: '+k)
    require(native.get('unlab_gt_in_loss') is False and native.get('criterion_projection_optimizer_owned') is False,'Native parameter/GT ownership changed')
    valid=native.get('sam_valid_cases'); require(type(valid) is int and 0<=valid<=2,'SAM valid count differs')
    require(expected_clock['sam_enabled'] or valid==0,'Premature SAM branch')
    require(type(native.get('foundation_case_forwards')) is int and native['foundation_case_forwards']==(2 if expected_clock['sam_enabled'] and valid else 0),'SAM forward count differs')
    if valid:require(native.get('sam_loss_requires_grad') is True,'Valid SAM loss detached')
    events=native.get('op_step_events');require(isinstance(events,list) and len(events)==2,'OP calls missing')
    require(events[0].get('effective_update') is False and events[0].get('gradient_l1')==0 and events[0].get('parameter_delta_l1')==0,'OP first call must remain ineffective')
    require(events[1].get('effective_update') is True and finite_number(events[1].get('parameter_delta_l1')) and events[1]['parameter_delta_l1']>0,'OP effective update missing')
    for v in (row.get('gradient_norm'),native.get('student_gradient_l1'),native.get('op_gradient_l1')):
        require(finite_number(v) and v>0,'Nonfinite/zero actual gradient')
    loss=row.get('losses',{});require(set(loss)=={'loss','sup_loss','unsup_loss','bc_loss','sam_loss','op_loss'},'Incomplete losses')
    require(all(finite_number(v) and native.get(k)==v for k,v in loss.items()),'Nonfinite or inconsistent loss')
    require(finite_number(native.get('dynamic_threshold_fg')),'Nonfinite native threshold')
    return validate_counts(counts(value['iterations']+1),recipe,role)


def validate_payload(state,identity,recipe,role,pairs):
    require(state.get('format')==FORMAT and state.get('contract_sha256')==identity,'Recovery format/identity differs')
    require(state.get('recipe')==recipe and state.get('execution_role')==role,'Recovery recipe/role differs')
    from dualteacher_numerics import POLICY
    require(state.get('numerical_runtime',{}).get('policy')==POLICY,'Missing/foreign numeric runtime')
    require(set(state.get('tensor_layout',{}))=={'models','optimizer','optimizer_op'} and set(state['tensor_layout']['models'])==MODELS,'Missing tensor layout closure')
    c=validate_counts(state['counts'],recipe,role);start,stop=role_window(recipe,role);pos=start+c['iterations']
    require(type(state['position']) is int and type(state['primary_position']) is int and state['position']==pos and state['primary_position']==2*pos,'Native/primary cursor differs')
    require(state['original_clock_iterations']==recipe['original_clock_iterations'],'Clock compressed')
    core=state['core_state'];require(set(core)==CORE_KEYS,'Incomplete native state closure')
    require(type(core['position']) is int and core['position']==pos and type(core['ema_step']) is int and core['ema_step']==c['iterations'],'EMA actual updates must not include skipped engineering clock')
    require(all(finite_number(core[k]) for k in ('dynamic_threshold_bg','dynamic_threshold_fg')),'Nonfinite threshold')
    require(state['actual_pairs']==pairs[2*start:2*pos],'Actual sampling prefix differs')
    require(len(state['native_rows'])==c['iterations'],'Missing native receipts')
    for j,row in enumerate(state['native_rows']):
        i=start+j;expected=pairs[2*i:2*i+2]
        require(type(row.get('step')) is int and row['step']==i+1 and row.get('case_ids')=={k:[p[k] for p in expected] for k in ('labeled','unlabeled')},'Native row identity differs')
        validate_clock(row['clock'],i,recipe)
        require(type(row.get('effective_student_updates')) is int and type(row.get('effective_op_updates')) is int and row['effective_student_updates']==row['effective_op_updates']==1,'Missing actual optimizer update')
    require(type(state['cycles_completed']) is int and state['cycles_completed']==(2*pos//recipe['primary_pairs_per_cycle'] if role=='formal' else 0),'Cycle count differs')
    for key in ('module_modes','requires_grad','parameter_gradients'):
        require(set(state[key])==MODELS,'Missing mode/flag/gradient closure')
    for name in MODELS:
        modes=state['module_modes'][name];flags=state['requires_grad'][name];grads=state['parameter_gradients'][name]
        require(isinstance(modes,dict) and '' in modes and all(type(v) is bool for v in modes.values()),'Invalid module modes')
        require(isinstance(flags,dict) and flags and all(type(v) is bool for v in flags.values()),'Invalid parameter flags')
        require(isinstance(grads,dict) and set(grads)==set(flags),'Gradient buffer closure differs')
    require(all(state['module_modes'][n][''] for n in ('student','ema','op','criterion')),'Native train/teacher mode differs')
    require(not any(state['module_modes']['sam'].values()),'Frozen SAM left eval')
    for name in ('ema','sam'):
        require(not any(state['requires_grad'][name].values()) and all(v is None for v in state['parameter_gradients'][name].values()),'Teacher/foundation gradients changed')
    require(state['foundation']['model_family']=='SAM-Med3D_turbo' and state['foundation']['bytes']==402163626 and state['foundation']['sha256']=='899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c','Frozen Turbo identity differs')
    require(isinstance(state['frozen_sam_signature'],str) and len(state['frozen_sam_signature'])==64 and all(c in '0123456789abcdef' for c in state['frozen_sam_signature']),'Missing full Turbo tensor signature')
    require(isinstance(state['rng'],(list,tuple)) and len(state['rng'])==4,'Incomplete RNG')
    require(set(state['instrumentation'])=={'forward_cases','student_events','op_events'},'Missing native instruments')
    require(set(state['immutable_native_config'])=={'config','ema_alpha','criterion_scalars','data_proxy','extraction'},'Missing native construction metadata')
    return copy.deepcopy(c)
