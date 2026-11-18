"""New numeric/state/lineage wiring only. No rerun of earlier CPU suites."""
import ast,copy,json,time
from pathlib import Path
import public_dualteacher_protocol as p
from dualteacher_numerics import POLICY
from dualteacher_state import validate_payload,FORMAT
from dualteacher_budget import plan,counts
from check_dualteacher_recovery_cpu import payload,synthetic_pairs
from dualteacher_native import public_core_ast

def main():
    checks=[]
    def check(ok,name):p.require(ok,name);checks.append(name)
    def reject(fn,name):
        try:fn()
        except (KeyError,ValueError,TypeError):checks.append(name)
        else:raise AssertionError('Accepted '+name)
    for labels in (5,10):
        r=plan(labels);pairs=synthetic_pairs(r)
        for role in ('formal','engineering_warmup','engineering_sam'):
            s=payload(r,role,pairs,2)
            s.update(numerical_runtime=dict(policy=POLICY),tensor_layout=dict(models={n:{} for n in s['module_modes']},optimizer={},optimizer_op={}))
            check(validate_payload(s,'a'*64,r,role,pairs)==counts(2),f'new_state_{labels}_{role}')
            for key,value in (('format','PUBLIC_DUALTEACHER_COMPLETE_RECOVERY_DRAFT_V1'),('numerical_runtime',{}),('tensor_layout',{})):
                bad=copy.deepcopy(s);bad[key]=value
                reject(lambda:validate_payload(bad,'a'*64,r,role,pairs),f'refuse_{labels}_{role}_{key}')
    tree,proof=public_core_ast(p.OLD/'dualteacher_core.py')
    check(proof['all_other_ast_identical'],'native_scientific_ast_still_identical')
    old=p.read(p.BASE/'dualteacher_sam/Public433/lab005/fold2/r1/contract.json');c=copy.deepcopy(old)
    c['tag']='r2';j,target=p.paths(5,'r2');c.update(target=str(target),output=str(j/'formal'),
        numerical_runtime_policy=POLICY,repair_lineage=p.read(p.HERE/'template.json')['repair_lineage'])
    source={x['path']:x for x in p.read(p.read(p.HERE/'template.json')['origin_contract']['path'])['sources']+
            p.read(p.read(p.HERE/'template.json')['origin_contract']['path'])['environment']['files']}
    source.update({str(f):dict(path=str(f),sha256=p.sha(f)) for f in [*p.HERE.glob('*.py'),p.HERE/'template.json',p.HERE/'budget11.json']})
    c['sources']=list(source.values());c['dualteacher_recipe']=p.public_recipe(c)
    p.verify(c);check(True,'complete_new_contract_projection_verified_no_job_created')
    bad=copy.deepcopy(c);bad['numerical_runtime_policy']='unbound'
    reject(lambda:p.verify(bad),'foreign_contract_numeric_policy_refused')
    bad=copy.deepcopy(c);bad['repair_lineage']=None
    reject(lambda:p.verify(bad),'prior_cost_erasure_refused')
    before=p.sha(p.REGISTRY)
    with p.leased(c):capacity=p.capacity(c,True)
    check(p.sha(p.REGISTRY)==before,'registry_byte_identical_no_enrollment')
    check(not j.exists() and not target.parent.exists(),'new_job_and_artifact_absent')
    for name in ('dualteacher_recovery.py','resumable_dualteacher.py','public_operator.py','admission.py'):
        ast.parse((p.HERE/name).read_bytes());check(True,'parse_'+name)
    proofdoc=p.read(p.read(p.HERE/'template.json')['numerical_operator_proof']['path'])
    check(proofdoc['new_native_iterations']==proofdoc['new_effective_optimizer_updates']==proofdoc['new_actual_test_evaluations']==0,
        'synthetic_cuda_proof_not_counted_as_native_qualification')
    check(proofdoc['source_sha256']['dualteacher_numerics.py']==p.sha(p.HERE/'dualteacher_numerics.py'),'tested_numeric_source_exact')
    result=dict(status='PASS_NEW_NUMERIC_BINDING_STATE_LINEAGE_CPU_ONLY',checks=len(checks),check_names=checks,
        source_sha256={f.name:p.sha(f) for f in p.HERE.glob('*.py')},capacity_projection=capacity,
        new_jobs=0,new_reservations=0,new_native_model_constructions=0,new_cuda_loads=0,new_patient_loads=0,
        new_effective_optimizer_updates=0,new_actual_test_evaluations=0,prior_engineering_updates=20,
        full_native_gpu_qualification=False,at=time.time())
    out=p.HERE.parent/'dualteacher_numeric_binding_cpu_20261027_01.json';p.exclusive(out,result)
    print(json.dumps(dict(output=str(out),sha256=p.sha(out),checks=len(checks))))

if __name__=='__main__':main()
