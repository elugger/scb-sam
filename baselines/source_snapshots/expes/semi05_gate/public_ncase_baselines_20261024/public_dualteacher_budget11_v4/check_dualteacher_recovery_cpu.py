"""New draft checks only: AST, synthetic receipts/metadata, mock boundary.

No model, tensor, CUDA, patient, native optimizer, test, job or reservation.
This does not qualify tensor restoration or shared storage/resource gates.
"""
import argparse
import ast
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from dualteacher_budget import plan,counts
from dualteacher_native import public_core_ast,CORE_SHA
from dualteacher_state import FORMAT,MODELS,CORE_KEYS,role_window,accept,validate_payload
from dualteacher_recovery import RecoveryMixin

checks=[]
def yes(name,ok):
    if not ok:raise AssertionError(name)
    checks.append(name)
def reject(name,fn):
    try:fn()
    except (ValueError,KeyError,TypeError):checks.append(name);return
    raise AssertionError('Accepted invalid state: '+name)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def native_clock(i,r):
    t=r['original_clock_iterations'];e=min(199,200*i//t)
    return dict(iteration=i,total_iterations=t,native_epoch=e,native_reference_iteration=40000*i/t,
        primary_progress=i/t,learning_rate=max(1e-6,.0025*(1-e/200)**.9),sam_enabled=e>5)
def synthetic_pairs(r):
    return [dict(labeled='L'+str(i%r['labeled_cases']),unlabeled='U'+str(i%r['unlabeled_cases']),position=i)
            for i in range(2*r['ssl_iterations'])]
def receipt(i,r,pairs):
    c=native_clock(i,r);p=pairs[2*i:2*i+2]
    loss={k:.2 for k in ('loss','sup_loss','unsup_loss','bc_loss','sam_loss','op_loss')}
    n=dict(step=i+1,clock=c,case_ids={k:[x[k] for x in p] for k in ('labeled','unlabeled')},
        primary_labeled=2,primary_unlabeled=2,optimizer_updates=2,primary_optimizer_updates=1,
        student_optimizer_calls=1,op_optimizer_calls=2,effective_student_updates=1,effective_op_updates=1,
        student_case_forwards=6,op_case_forwards=4,teacher_case_forwards=4 if c['sam_enabled'] else 2,
        foundation_case_forwards=2 if c['sam_enabled'] else 0,sam_valid_cases=1 if c['sam_enabled'] else 0,
        test_evaluations=0,unlab_gt_in_loss=False,criterion_projection_optimizer_owned=False,
        sam_loss_requires_grad=c['sam_enabled'],student_gradient_l1=.5,op_gradient_l1=.4,dynamic_threshold_fg=.5,
        op_step_events=[dict(effective_update=False,gradient_l1=0,parameter_delta_l1=0),
                        dict(effective_update=True,gradient_l1=.4,parameter_delta_l1=.1)],**loss)
    return dict(actual_pairs=p,completed_primary_pairs=2*(i+1),labeled_inputs=2,unlabeled_inputs=2,
        successful_optimizer_updates=1,skip_count=0,finite_loss=True,finite_nonzero_gradients=True,
        actual_case_ids_verified=True,teacher_no_grad=True,unlabeled_gt_in_loss=False,
        losses=loss,gradient_norm=.7,native_receipt=n)
def payload(r,role,pairs,n):
    start,_=role_window(r,role);pos=start+n
    core={k:{} for k in CORE_KEYS};core.update(position=pos,ema_step=n,dynamic_threshold_fg=.5,dynamic_threshold_bg=.5)
    return dict(format=FORMAT,contract_sha256='a'*64,recipe=r,execution_role=role,position=pos,primary_position=2*pos,
        original_clock_iterations=r['original_clock_iterations'],counts=counts(n),core_state=core,
        actual_pairs=pairs[2*start:2*pos],native_rows=[receipt(i,r,pairs)['native_receipt'] for i in range(start,pos)],
        cycles_completed=2*pos//r['primary_pairs_per_cycle'] if role=='formal' else 0,
        module_modes={k:{'':k!='sam','child':k!='sam'} for k in MODELS},
        requires_grad={k:{'w':k not in ('sam','ema')} for k in MODELS},
        parameter_gradients={k:{'w':None} for k in MODELS},foundation=dict(model_family='SAM-Med3D_turbo',bytes=402163626,
        sha256='899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c'),
        frozen_sam_signature='b'*64,rng=('python','numpy','cpu','cuda'),initial_signatures={},
        instrumentation=dict(forward_cases={},student_events=[],op_events=[]),
        immutable_native_config=dict(config={},ema_alpha=.99,criterion_scalars={},data_proxy={},extraction={}))
def setpath(value,path,new):
    target=value
    for key in path[:-1]:target=target[key]
    target[path[-1]]=new

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--core',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    original=ast.parse(Path(a.core).read_bytes());adapted,proof=public_core_ast(a.core)
    oc=next(n for n in original.body if isinstance(n,ast.ClassDef) and n.name=='DualteacherCore')
    ac=next(n for n in adapted.body if isinstance(n,ast.ClassDef) and n.name=='DualteacherCore')
    for name in ('update','state_dict','load_state_dict','close'):
        left=next(n for n in oc.body if isinstance(n,ast.FunctionDef) and n.name==name)
        right=next(n for n in ac.body if isinstance(n,ast.FunctionDef) and n.name==name)
        yes('native_'+name+'_AST_identical',ast.dump(left)==ast.dump(right))
    yes('native_disk_SHA_unchanged',sha(a.core)==CORE_SHA)
    yes('all_other_native_AST_identical',proof['all_other_ast_identical'])
    guard=next(n for n in ac.body if isinstance(n,ast.FunctionDef) and n.name=='__init__').body[0].value.args[0]
    code=compile(ast.Expression(guard),'<guard-only>','eval')
    for value in (6210,6120,1898,1870,1710,1620,True,6210.):
        yes('clock_constructor_guard_'+repr(value),eval(code,{'total':value}) is (type(value) is int and value in (6210,6120)))
    for labels in (5,10):
        r=plan(labels);pairs=synthetic_pairs(r)
        for role in ('formal','engineering_warmup','engineering_sam'):
            start,stop=role_window(r,role)
            for n in (0,2,4):
                s=payload(r,role,pairs,n)
                yes(f'payload_{labels}_{role}_{n}',validate_payload(s,'a'*64,r,role,pairs)==counts(n))
            c=counts(0)
            for i in range(start,start+4):c=accept(c,receipt(i,r,pairs),r,role,pairs[2*i:2*i+2],native_clock(i,r))
            yes(f'four_receipts_{labels}_{role}',c==counts(4))
            if role!='formal':reject(f'engineering_cutoff_{labels}_{role}',lambda:accept(c,receipt(start+4,r,pairs),r,role,pairs[2*(start+4):2*(start+4)+2],native_clock(start+4,r)))
    r=plan(5);pairs=synthetic_pairs(r);role='engineering_sam';start,_=role_window(r,role)
    s=payload(r,role,pairs,2)
    mutations=[
        ('foreign_identity',('contract_sha256',),'c'*64),('engineering_as_formal',('execution_role',),'formal'),
        ('compressed_clock',('original_clock_iterations',),1898),('missing_optimizer',('core_state',),{}),
        ('wrong_EMA_step',('core_state','ema_step'),start+2),('wrong_native_cursor',('position',),1),
        ('wrong_primary_cursor',('primary_position',),1),('wrong_success_updates',('counts','successful_optimizer_updates'),6),
        ('bool_counter',('counts','iterations'),True),('sampling_replaced',('actual_pairs',),[]),
        ('missing_native_rows',('native_rows',),[]),('wrong_native_row',('native_rows',0,'step'),0),
        ('compressed_row_clock',('native_rows',0,'clock','total_iterations'),1898),
        ('missing_OP_update',('native_rows',0,'effective_op_updates'),0),('bool_OP_update',('native_rows',0,'effective_op_updates'),True),
        ('hidden_engineering_cycles',('cycles_completed',),1),('missing_modes',('module_modes',),{}),
        ('EMA_eval',('module_modes','ema',''),False),('SAM_train',('module_modes','sam','child'),True),
        ('EMA_grad_flag',('requires_grad','ema','w'),True),('SAM_grad_flag',('requires_grad','sam','w'),True),
        ('EMA_grad_buffer',('parameter_gradients','ema','w'),'nonempty'),('SAM_grad_buffer',('parameter_gradients','sam','w'),'nonempty'),
        ('criterion_gradient_missing',('parameter_gradients','criterion'),{}),('student_flags_missing',('requires_grad','student'),{}),
        ('wrong_foundation_family',('foundation','model_family'),'original'),('wrong_foundation_SHA',('foundation','sha256'),'x'*64),
        ('wrong_foundation_size',('foundation','bytes'),0),('missing_tensor_signature',('frozen_sam_signature',),'z'*64),
        ('missing_RNG',('rng',),[]),('missing_instruments',('instrumentation',),{}),('missing_config',('immutable_native_config',),{}),
        ('nonfinite_threshold',('core_state','dynamic_threshold_fg'),float('nan'))]
    for name,path,value in mutations:
        bad=copy.deepcopy(s);setpath(bad,path,value)
        reject(name,lambda:validate_payload(bad,'a'*64,r,role,pairs))
    row=receipt(start,r,pairs)
    badrows=[('OP_not_third_update',('native_receipt','optimizer_updates'),3),
        ('primary_student_not_two',('successful_optimizer_updates',),2),('first_OP_changed',('native_receipt','op_step_events',0,'effective_update'),True),
        ('SAM_detached',('native_receipt','sam_loss_requires_grad'),False),('changed_criterion_optimizer',('native_receipt','criterion_projection_optimizer_owned'),True),
        ('unlabeled_GT',('unlabeled_gt_in_loss',),True),('teacher_grad',('teacher_no_grad',),False),
        ('extra_test',('native_receipt','test_evaluations'),1),('wrong_actual_pairs',('actual_pairs',),[]),
        ('zero_gradient',('gradient_norm',),0),('nonfinite_loss',('losses','loss'),float('nan')),
        ('wrong_SAM_count',('native_receipt','foundation_case_forwards'),1),('wrong_teacher_forwards',('native_receipt','teacher_case_forwards'),2)]
    for name,path,value in badrows:
        bad=copy.deepcopy(row);setpath(bad,path,value)
        reject(name,lambda:accept(counts(0),bad,r,role,pairs[2*start:2*start+2],native_clock(start,r)))
    class Model:
        def __init__(self,frozen=False):self.p=[SimpleNamespace(requires_grad=not frozen,grad=None)];self.training=not frozen
        def parameters(self):return iter(self.p)
    mock=RecoveryMixin();mock.initialized=True;mock.test_started=False;mock.reloaded_sha=None
    mock.recipe=r;mock.execution_role='formal';mock.position=0;mock.actual_pairs=[];mock.native_rows=[];mock.pairs=pairs
    student,op,sam,ema=Model(),Model(),Model(True),Model(True)
    mock.core=SimpleNamespace(model=student,op=op,sam=sam,ema=SimpleNamespace(model=ema,step=0),step_count=0,total=r['original_clock_iterations'],
        optimizer=SimpleNamespace(param_groups=[{'params':student.p}]),optimizer_op=SimpleNamespace(param_groups=[{'params':op.p}]))
    mock.check_boundary();checks.append('mock_optimizer_parameter_order_boundary')
    mock.core.optimizer.param_groups[0]['params']=op.p
    reject('mock_optimizer_membership_rejected',mock.check_boundary)
    yes('no_heavy_runtime_imported',not {'torch','numpy','medim'}&set(sys.modules))
    result=dict(status='PASS_DRAFT_METADATA_AND_AST_ONLY',checks=len(checks),check_names=checks,native_adaptation=proof,
        source_files={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        measurements=dict(model_loads=0,CUDA_loads=0,patient_loads=0,native_optimizer_updates=0,actual_tests=0,new_jobs=0,new_reservations=0),
        limitations=['synthetic receipts and metadata; no tensor/GPU restoration proof','mock optimizer boundary; no actual optimizer execution',
                    'backend/storage/operator/admission/observer integration still pending','no source or qualification borrowed from another method'])
    with Path(a.out).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=result['status'],checks=len(checks),out=str(a.out),sha256=sha(a.out))))
if __name__=='__main__':main()
