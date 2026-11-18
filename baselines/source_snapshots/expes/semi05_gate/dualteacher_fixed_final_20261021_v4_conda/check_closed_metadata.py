"""No model imports: closed-scope negative tests and independent ledger gates."""
import copy,json,subprocess,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
import run_dualteacher as control
import supervisor_verify_dualteacher as audit
import dualteacher_backend as backend
checks=[]
for method,dataset,ratio in [('relif3d','BY',5),('dualteacher_sam','Public433',5),('dualteacher_sam','BY',20)]:
    try:control.job_paths(method,dataset,ratio,'r1')
    except ValueError:checks.append('reject scope '+method+'/'+dataset+'/'+str(ratio))
    else:raise AssertionError('Unsupported scope accepted')
try:control.admission_review()
except ValueError as e:assert 'not admitted' in str(e);checks.append('admission stays closed before prepare')
else:raise AssertionError('Draft admitted')
base=['--action','prepare','--method','dualteacher_sam','--dataset','BY','--ratio','5','--gpu','1']
for flag,value in [('--method','relif3d'),('--dataset','Public433'),('--ratio','20'),('--gpu','0')]:
    args=list(base);args[args.index(flag)+1]=value
    r=subprocess.run([sys.executable,str(BASE/'run_dualteacher.py')]+args,capture_output=True,text=True)
    assert r.returncode==2 and 'invalid choice' in r.stderr
    checks.append('CLI rejects '+flag+' '+value)
pairs=[dict(epoch=0,within=i,labeled='l'+str(i),unlabeled='u'+str(i),unlabeled_replay=False) for i in range(2)]
losses=dict(loss=1.25,sup_loss=1.,unsup_loss=.4,bc_loss=.5,sam_loss=0.,op_loss=.8)
native=dict(step=1,clock=dict(iteration=0,total_iterations=1710,native_epoch=0,native_reference_iteration=0.,primary_progress=0.,learning_rate=.0025,sam_enabled=False),
    case_ids=dict(labeled=['l0','l1'],unlabeled=['u0','u1']),primary_labeled=2,primary_unlabeled=2,optimizer_updates=2,primary_optimizer_updates=1,
    student_optimizer_calls=1,op_optimizer_calls=2,effective_student_updates=1,effective_op_updates=1,
    op_step_events=[dict(gradient_l1=0.,effective_update=False,parameter_delta_l1=0.),dict(gradient_l1=5.,effective_update=True,parameter_delta_l1=.05)],
    student_gradient_l1=3.,op_gradient_l1=5.,student_case_forwards=6,teacher_case_forwards=2,op_case_forwards=4,foundation_case_forwards=0,
    sam_valid_cases=0,sam_loss_requires_grad=False,unlab_gt_in_loss=False,test_evaluations=0,criterion_projection_optimizer_owned=False,dynamic_threshold_fg=.5,**losses)
row=dict(finite_loss=True,finite_nonzero_gradients=True,teacher_no_grad=True,actual_case_ids_verified=True,skip_count=0,unlabeled_gt_in_loss=False,
    labeled_inputs=2,unlabeled_inputs=2,successful_optimizer_updates=1,completed_primary_pairs=2,actual_pairs=pairs,gradient_norm=2.,native_receipt=native,losses=losses)
audit.validate_step(row,pairs,0,1710);checks.append('independent initial native ledger fixture accepted')
def reject(label,change):
    r=copy.deepcopy(row);change(r)
    try:audit.validate_step(r,pairs,0,1710)
    except (ValueError,KeyError):checks.append('reject '+label)
    else:raise AssertionError('Tampered evidence accepted: '+label)
reject('unlabeled GT',lambda x:x.update(unlabeled_gt_in_loss=True))
reject('extra primary update',lambda x:x.update(successful_optimizer_updates=2))
reject('substituted case',lambda x:x['actual_pairs'][0].update(unlabeled='replacement'))
reject('nonfinite gradient',lambda x:x.update(gradient_norm=float('inf')))
reject('wrong native epoch',lambda x:x['native_receipt']['clock'].update(native_epoch=6))
reject('hidden OP first update',lambda x:x['native_receipt']['op_step_events'][0].update(effective_update=True,parameter_delta_l1=.1))
reject('unreported SAM compute',lambda x:x['native_receipt'].update(foundation_case_forwards=2))
reject('test during training',lambda x:x['native_receipt'].update(test_evaluations=1))
reject('wrong total loss',lambda x:(x['native_receipt'].update(loss=1.5),x['losses'].update(loss=1.5)))
apairs=[dict(epoch=0,within=i,labeled='l'+str(i%2),unlabeled='u'+str(i),unlabeled_replay=False) for i in range(106)]
ar=copy.deepcopy(row);ar.update(completed_primary_pairs=106,actual_pairs=apairs[104:106]);nr=ar['native_receipt']
nr.update(step=53,clock=audit.clock(52,1710),case_ids=dict(labeled=['l0','l1'],unlabeled=['u104','u105']),optimizer_updates=2,effective_op_updates=1,
    teacher_case_forwards=4,foundation_case_forwards=2,sam_valid_cases=2,sam_loss_requires_grad=True,sam_loss=.2,loss=1.27)
nr['op_step_events'][0].update(effective_update=False,parameter_delta_l1=0.)
ar['losses'].update(sam_loss=.2,loss=1.27);audit.validate_step(ar,apairs,52,1710);checks.append('independent SAM-active ledger fixture accepted')
nr['sam_loss_requires_grad']=False
try:audit.validate_step(ar,apairs,52,1710)
except ValueError:checks.append('reject detached active SAM loss')
else:raise AssertionError('Detached SAM branch accepted')
assert not audit.clock(51,1710)['sam_enabled'] and audit.clock(52,1710)['sam_enabled'];checks.append('native SAM activation boundary')
assert 'torch' not in sys.modules and not (BASE/'formal').exists()
v=dict(status='CLOSED_SCOPE_AND_INDEPENDENT_LEDGER_GATES_PASS_NOT_MODEL_OR_GPU_QUALIFICATION',checks=checks,actual_models_loaded=0,patient_inputs=0,gpu_used=False,formal_dispatch=False,
    independent_complete_pipeline_test=False,admission_closed=True)
with (BASE/'closed_metadata_checks_conda_v3_final.json').open('x',encoding='utf8') as f:json.dump(v,f,indent=2)
print(json.dumps(dict(status=v['status'],checks=len(checks))))
