"""Only the new declared-tolerance/engineering-context migration; no model IO."""
import argparse,copy,json,math,sys
from pathlib import Path
from unittest.mock import patch


def main():
    p=argparse.ArgumentParser();p.add_argument('--native-dir',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();sys.path.insert(0,str(Path(__file__).resolve().parent));sys.path.insert(1,a.native_dir)
    import qualification_migration as m
    from relif_state import counts,plan,FORMAT,require
    checks=[]
    def ok(n,v):require(v,n);checks.append(n)
    def reject(n,f):
        try:f()
        except (ValueError,KeyError):checks.append(n);return
        raise AssertionError('Not rejected: '+n)
    ok('declared_additive_tolerance_accepts_recorded_distillation_difference',m.numeric_close(.10526208579540253,.10525038838386536))
    ok('old_math_isclose_max_rule_reproduces_rejection',not math.isclose(.10526208579540253,.10525038838386536,rel_tol=1e-4,abs_tol=1e-5))
    ok('new_rule_rejects_beyond_declared_sum',not m.numeric_close(.1053,.10525038838386536))
    ok('new_rule_rejects_nan_inf',not m.numeric_close(float('nan'),1.) and not m.numeric_close(float('inf'),1.))
    recipe=plan(5);pairs=[dict(i=i) for i in range(recipe['original_clock_iterations'])]
    c=dict(recipe=recipe,qualification_origin=dict(contract=dict(sha256='a'*64)))
    for role,start in [('engineering_warmup',0),('engineering_larc',recipe['original_larc_start_index'])]:
        state=dict(format=FORMAT,contract_sha256='a'*64,execution_role=role,recipe=recipe,
            core_state=dict(student={'synthetic':1},projection={'synthetic':2},optimizer={'momentum':3},step=start+2,
                total=recipe['original_clock_iterations'],shape=[128]*3,implementation='synthetic'),
            position=start+2,actual_rows=pairs[start:start+2],counts=counts(2),cycles_completed=0,
            module_modes={k:{'':k!='prior'} for k in ('student','projection','prior')},
            requires_grad={k:{'weight':k!='prior'} for k in ('student','projection','prior')},
            foundation=dict(variant='vit_b_ori',model_family='SAM-Med3D_original'),
            frozen_prior_signature='c'*64,rng=[1,2,3,4],initial_signatures={'synthetic':5})
        ref=dict(origin_contract_sha256='a'*64,execution_role=role)
        with patch.object(m,'contract_sha',return_value='b'*64):
            migrated=m.migrate_engineering_identity(c,state,ref,pairs)
            ok(role+'_only_contract_context_changes',state['contract_sha256']=='a'*64 and migrated['contract_sha256']=='b'*64
               and {k:v for k,v in migrated.items() if k!='contract_sha256'}=={k:v for k,v in state.items() if k!='contract_sha256'})
            reject(role+'_wrong_origin_rejected',lambda:m.migrate_engineering_identity(c,state,dict(ref,origin_contract_sha256='d'*64),pairs))
            reject(role+'_formal_role_rejected',lambda:m.migrate_engineering_identity(c,state,dict(ref,execution_role='formal'),pairs))
            reject(role+'_wrong_counts_rejected',lambda:m.migrate_engineering_identity(c,dict(state,counts=counts(1)),ref,pairs))
    ok('no_model_cuda_numpy_imports','torch' not in sys.modules and 'numpy' not in sys.modules)
    result=dict(passed=True,checks=len(checks),check_names=checks,
        scope='bounded predicate correction and only engineering contract-context migration; synthetic metadata, no tensor/model/GPU/real resource proof',
        new_model_loads=0,new_optimization_updates=0,new_test_evaluations=0,
        sources={p.name:m.sha(p) for p in Path(__file__).resolve().parent.glob('*.py')})
    m.exclusive(Path(a.out),result);print(json.dumps(dict(passed=True,checks=len(checks))))


if __name__=='__main__':main()
