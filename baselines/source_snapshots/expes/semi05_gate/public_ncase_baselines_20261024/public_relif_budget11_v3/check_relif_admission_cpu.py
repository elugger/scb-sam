"""Targeted admission and pinned-diagnostic checks; synthetic proofs, no GPU."""
import argparse,contextlib,copy,json,os,sys,tempfile
from pathlib import Path
from unittest.mock import patch


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--native-dir',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();here=Path(__file__).resolve().parent;sys.path.insert(0,str(here));sys.path.insert(1,a.native_dir)
    import admission as ad
    import relif_protocol as protocol
    from relif_state import plan,counts,require
    checks=[]
    def ok(name,v):require(v,name);checks.append(name)
    def reject(name,fn):
        try:fn()
        except (ValueError,AssertionError,KeyError,FileNotFoundError):checks.append(name);return
        raise AssertionError('Not rejected: '+name)
    records=protocol.known_warning_records()
    ok('two_installed_warning_sources_sha_and_ast_verified',len(records)==2)
    for raw in (b'',records[0],records[1],b''.join(records)):
        ok('exact_native_warning_sequence_accepted',bool(protocol.check_logs(raw)))
    ok('live_exact_incomplete_warning_prefix_accepted',bool(protocol.check_logs(records[0][:25],True)))
    reject('terminal_partial_warning_rejected',lambda:protocol.check_logs(records[0][:25]))
    reject('unknown_diagnostic_rejected',lambda:protocol.check_logs(b'unknown'))
    reject('known_warning_cannot_hide_appended_traceback',lambda:protocol.check_logs(records[0]+b'Traceback'))
    reject('duplicate_warning_rejected',lambda:protocol.check_logs(records[0]*2))
    with tempfile.TemporaryDirectory(prefix='relif_admission_cpu_') as td:
        j=Path(td);c=dict(output=str(j/'formal'),recipe=plan(5));(j/'contract.json').write_text(json.dumps(c))
        ch=protocol.sha(j/'contract.json');rows=[dict(path=str(j/('slot%d.pth'%n)),bound=ad.RECOVERY_BOUND) for n in range(6)]
        q=dict(status='OWN_WINDOWS_QUALIFIED_FRESH_PROBES_PENDING',contract_sha256=ch,actual_extra_engineering_updates=8,
            engineering_labeled_inputs=8,engineering_unlabeled_inputs=8,new_formal_updates=0,new_test_evaluations=0,
            actual_disk_roundtrips=True,references={},pid=100)
        def write(p,v):p.write_text(json.dumps(v))
        for window,slot,pid in [('warmup',3,101),('larc',4,102)]:
            role='engineering_'+window;d=j/('qualification_'+role);d.mkdir();cp=Path(rows[slot]['path']);cp.write_bytes(b'opaque')
            cp=dict(path=str(cp),bytes=6,sha256=protocol.sha(cp));start=0 if window=='warmup' else plan(5)['original_larc_start_index']
            ref=dict(execution_role=role,contract_sha256=ch,checkpoint=cp,expected_counts=counts(4),
                steps=[dict(completed_primary_pairs=start+3),dict(completed_primary_pairs=start+4)],active_larc_actual_gradient_verified=window=='larc')
            write(d/'reference.json',ref);q['references'][role]=dict(path=str(d/'reference.json'),sha256=protocol.sha(d/'reference.json'))
            probe=dict(status='PASS',contract_sha256=ch,role=role,checkpoint_sha256=cp['sha256'],fresh_process=True,
                state_restoration_bitwise_equal=True,indices_equal=True,losses_close=True,rng_sequence_equal=True,
                numeric_tolerance=dict(rtol=1e-4,atol=1e-5),actual_extra_engineering_updates=2,engineering_labeled_inputs=2,
                engineering_unlabeled_inputs=2,new_formal_updates=0,new_test_evaluations=0,pid=pid,
                continuation_state_bitwise_equal=False,max_abs_loss_difference=1e-7)
            write(j/('resume_qualification_'+window+'.json'),probe)
        write(j/'qualification.json',q)
        for name in ('qualify.log','probe_warmup.log','probe_larc.log'):(j/name).write_bytes(b''.join(records))
        registry=j/'registry.json';write(registry,rows)
        with contextlib.ExitStack() as stack:
            for name,value in dict(verify=lambda c:None,rows=lambda c:rows,REGISTRY=registry,
                    leased=lambda c:contextlib.nullcontext(),capacity=lambda *args:dict(mock=True)).items():
                stack.enter_context(patch.object(ad,name,value))
            ready=ad.inspect_ready(c,j)
            ok('admission_binds_both_roles_and_all_12_engineering_updates',ready['total_engineering_updates']==12
               and ready['engineering_labeled_inputs']==ready['engineering_unlabeled_inputs']==12 and ready['new_formal_updates']==0)
            def tamper(name,path,field,value):
                orig=protocol.read(path);v=copy.deepcopy(orig);v[field]=value;write(path,v)
                reject(name,lambda:ad.inspect_ready(c,j));write(path,orig)
            p=j/'resume_qualification_larc.json'
            for field,value in [('role','formal'),('state_restoration_bitwise_equal',False),('indices_equal',False),
                ('rng_sequence_equal',False),('numeric_tolerance',dict(rtol=1,atol=1)),('pid',100),('new_test_evaluations',1)]:
                tamper('admission_rejects_'+field,p,field,value)
            tamper('fresh_probes_require_distinct_pids',p,'pid',101)
            (j/'probe_larc.log').write_bytes(b'unknown')
            reject('unknown_qualification_log_blocks_start',lambda:ad.inspect_ready(c,j));(j/'probe_larc.log').write_bytes(b'')
            write(j/'launch_intent.json',{})
            reject('existing_launch_cannot_be_admitted_again',lambda:ad.inspect_ready(c,j));(j/'launch_intent.json').unlink()
            (j/'formal').mkdir();reject('existing_formal_directory_blocks_fresh_start',lambda:ad.inspect_ready(c,j))
    ok('no_model_or_cuda_imports','torch' not in sys.modules and 'numpy' not in sys.modules)
    result=dict(passed=True,checks=len(checks),check_names=checks,
        scope='new admission and exact pinned native warnings only; synthetic qualification metadata; mocked capacity/lease; no actual model or GPU proof',
        new_model_loads=0,new_cuda_loads=0,new_patient_loads=0,new_optimization_updates=0,new_test_evaluations=0,
        formal_jobs_created=0,reservations_registered=0,dispatches=0,
        sources={p.name:protocol.sha(p) for p in here.glob('*.py')})
    protocol.exclusive(Path(a.out),result);print(json.dumps(dict(passed=True,checks=len(checks))))


if __name__=='__main__':main()
