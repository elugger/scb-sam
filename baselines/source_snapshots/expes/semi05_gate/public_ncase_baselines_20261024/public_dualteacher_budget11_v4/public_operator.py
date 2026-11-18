"""One-shot DualTeacher own engineering qualification and explicit formal dispatch."""
import argparse
import contextlib
import gc
import math
import time
import traceback

from public_dualteacher_protocol import *


def backend(c,output,role='formal'):
    return load_native_backend(c)(c,output,role)


def prepare(a):
    j,target=paths(a.labels,a.tag)
    require(not j.exists() and not target.parent.exists(),'Job/artifact already exists')
    template=read(HERE/'template.json');origin=read(template['origin_contract']['path'])
    b=binding(dict(labeled_cases=a.labels));s=b.splits
    sources={}
    for row in origin['sources']+origin['environment']['files']:
        if row['path'] in sources:require(sources[row['path']]['sha256']==row['sha256'],'Conflicting original source')
        sources[row['path']]=dict(row)
    for p in [*sorted(HERE.glob('*.py')),*[HERE/n for n in ('template.json','budget11.json')]]:
        require(str(p) not in sources,'Duplicate new source');sources[str(p)]=dict(path=str(p),sha256=sha(p))
    c=dict(method='dualteacher_sam',dataset='Public433',gpu=1,fold=2,seed=2027,tag=a.tag,
        labeled_cases=a.labels,ratio=a.labels,labels_are_case_counts_not_percent=True,
        target=str(target),output=str(j/'formal'),splits=s,case_files=bind_case_files('Public433',s),
        recipe=b.recipe,primary_pairs=b.recipe['primary_inputs_per_stream'],foundation=origin['foundation'],
        official_code_root=origin['official_code_root'],native_validate_sha256=origin['native_validate_sha256'],
        sources=list(sources.values()),inputs=[dict(path=str(p),sha256=sha(p)) for p in sorted(SPLITS.glob('*.txt'))],
        recovery_interval=100,recovery_bound=RECOVERY_BOUND,checkpoint_bound=FINAL_BOUND,log_bound=LOG_BOUND,
        schedule_policy='retain_original_36_cycle_lr_ramp_clocks; cutoff_only',
        selection='terminal_after_fixed_exposure',test_evaluations=1,early_stop=False,
        continuation_policy='full_saved_state_exact; abs(actual-reference)<=1e-5+1e-4*abs(reference); not_bitwise_guaranteed',
        qualification_windows={'engineering_warmup':4,'engineering_sam':4},
        numerical_runtime_policy=__import__('dualteacher_numerics').POLICY,
        repair_lineage=template['repair_lineage'] if a.labels==5 else None,
        standing_human_authorization='Run remaining Public5/10 baselines with frozen 11-cycle input budget; engineering repairs and qualified recovery authorized; CPAC held')
    c['dualteacher_recipe']=public_recipe(c)
    verify(c);capacity(c,True);j.mkdir(parents=True,exist_ok=False)
    try:
        cap=enroll(c);exclusive(j/'contract.json',c)
        exclusive(j/'preparation.json',dict(status='BOUND_NOT_QUALIFIED',contract_sha256=contract_sha(c),capacity=cap))
    except BaseException as error:
        exclusive(j/'prepare_failure.json',dict(error=repr(error),traceback=traceback.format_exc()));raise
    return dict(status='BOUND_NOT_QUALIFIED',job=str(j),contract_sha256=contract_sha(c),recipe=c['recipe'])


def take_step(b):
    step=b.train_iteration(b.position)
    b.append_step(dict(step=step,cumulative=copy_counts(b.progress)))
    return step


def copy_counts(value):
    return dict(value)


def release_backend(b,torch):
    b.core.close()
    b.data=None;b.core=None
    gc.collect();torch.cuda.empty_cache()


def qualify(c,j):
    require(not (j/'qualification_intent.json').exists() and not Path(c['output']).exists(),'Qualification repeated')
    with leased(c):
        os.environ['CUDA_VISIBLE_DEVICES']=UUID
        import torch
        from recovery_state import tree_hash
        exclusive(j/'qualification_intent.json',dict(contract_sha256=contract_sha(c),at=time.time()))
        refs={};b=None
        try:
            for role,slot in (('engineering_warmup',3),('engineering_sam',4)):
                out=j/('qualification_'+role);out.mkdir(exist_ok=False)
                b=backend(c,out,role);prepared=b.prepare_initial()
                initial=tree_hash(b.capture_recovery())
                first=[take_step(b) for _ in range(2)]
                state=b.capture_recovery()
                artifact=write_tensor(c,rows(c)[slot]['path'],state,RECOVERY_BOUND,torch)
                loaded=load_tensor(c,artifact,torch)
                require(tree_hash(loaded)==tree_hash(state),'Serialized complete state differs')
                measured=[take_step(b) for _ in range(2)];end=b.capture_recovery()
                require(end['counts']==counts(4),'Engineering count differs')
                active=role=='engineering_sam'
                if active:
                    require(all(x['native_receipt']['clock']['sam_enabled'] for x in first+measured)
                        and any(x['native_receipt']['sam_valid_cases']>0 and x['native_receipt']['sam_loss_requires_grad'] and x['losses']['sam_loss']>0
                                for x in first+measured),'Own active SAM window lacks actual valid distillation gradient')
                ref=dict(execution_role=role,contract_sha256=contract_sha(c),checkpoint=artifact,
                    checkpoint_state_sha256=tree_hash(state),initial_state_sha256=initial,
                    expected_state_sha256=tree_hash(end),expected_rng_sha256=tree_hash(end['rng']),
                    expected_counts=end['counts'],steps=measured,preparation=prepared,
                    active_sam_actual_gradient_verified=active,pid=os.getpid())
                exclusive(out/'reference.json',ref)
                refs[role]=dict(path=str(out/'reference.json'),sha256=sha(out/'reference.json'))
                release_backend(b,torch);b=None
            result=dict(status='OWN_WINDOWS_QUALIFIED_FRESH_PROBES_PENDING',contract_sha256=contract_sha(c),
                actual_extra_engineering_iterations=8,actual_extra_engineering_updates=16,engineering_labeled_inputs=16,engineering_unlabeled_inputs=16,
                new_formal_updates=0,new_test_evaluations=0,actual_disk_roundtrips=True,references=refs,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(),pid=os.getpid(),at=time.time())
            exclusive(j/'qualification.json',result)
            return result
        except BaseException as error:
            exclusive(j/'qualification_failure.json',dict(error=repr(error),traceback=traceback.format_exc(),
                actual_progress=getattr(b,'progress',None),native_partial_events=partial_events(b),failed_partial_compute_not_assumed_zero=True));raise


def probe(c,j,window):
    role='engineering_'+window
    require(window in ('warmup','sam') and (j/'qualification.json').exists() and not Path(c['output']).exists(),
            'Invalid probe stage')
    intent=j/('resume_probe_'+window+'_intent.json')
    require(not intent.exists(),'Probe already attempted')
    with leased(c):
        os.environ['CUDA_VISIBLE_DEVICES']=UUID
        import torch
        from recovery_state import tree_hash
        exclusive(intent,dict(contract_sha256=contract_sha(c),role=role,at=time.time()))
        b=None
        try:
            ref=read(j/('qualification_'+role)/'reference.json')
            out=j/('probe_'+role);out.mkdir(exist_ok=False)
            b=backend(c,out,role);b.prepare_initial(for_restore=True)
            state=load_tensor(c,ref['checkpoint'],torch)
            require(tree_hash(state)==ref['checkpoint_state_sha256'],'Probe checkpoint differs')
            restored_expected_hash=tree_hash(state)
            b.restore(state)
            require(tree_hash(b.capture_recovery())==restored_expected_hash,'Fresh process state not exact')
            measured=[take_step(b) for _ in range(2)];end=b.capture_recovery()
            indices=all(x['actual_pairs']==y['actual_pairs'] and x['native_receipt']['clock']==y['native_receipt']['clock']
                        for x,y in zip(measured,ref['steps']))
            differences=[abs(x['losses'][k]-y['losses'][k]) for x,y in zip(measured,ref['steps']) for k in x['losses']]
            numeric_close=lambda actual,reference:abs(actual-reference)<=1e-5+1e-4*abs(reference)
            losses=all(set(x['losses'])==set(y['losses']) and all(numeric_close(x['losses'][k],y['losses'][k])
                       for k in x['losses']) for x,y in zip(measured,ref['steps']))
            rng_equal=tree_hash(end['rng'])==ref['expected_rng_sha256']
            comparable=dict(end)
            exclusive(out/'continuation_measurement.json',dict(indices_equal=indices,losses_close=losses,
                rng_sequence_equal=rng_equal,max_abs_loss_difference=max(differences),counts=end['counts'],
                expected_counts=ref['expected_counts'],rng_sha256=tree_hash(end['rng']),state_sha256=tree_hash(end),
                numeric_rule='abs(actual-reference) <= atol + rtol * abs(reference)',numeric_tolerance=dict(rtol=1e-4,atol=1e-5)))
            require(len(measured)==len(ref['steps'])==2 and indices and losses and rng_equal
                    and end['counts']==ref['expected_counts'],'Fresh native continuation differs')
            result=dict(status='PASS',role=role,contract_sha256=contract_sha(c),fresh_process=True,
                checkpoint_sha256=ref['checkpoint']['sha256'],state_restoration_bitwise_equal=True,
                indices_equal=indices,losses_close=losses,rng_sequence_equal=rng_equal,
                continuation_state_bitwise_equal=tree_hash(comparable)==ref['expected_state_sha256'],
                numeric_tolerance=dict(rtol=1e-4,atol=1e-5),max_abs_loss_difference=max(differences),
                actual_extra_engineering_iterations=2,actual_extra_engineering_updates=4,engineering_labeled_inputs=4,engineering_unlabeled_inputs=4,
                new_formal_updates=0,new_test_evaluations=0,pid=os.getpid(),at=time.time())
            exclusive(j/('resume_qualification_'+window+'.json'),result)
            return result
        except BaseException as error:
            exclusive(j/('resume_probe_'+window+'_failure.json'),dict(error=repr(error),traceback=traceback.format_exc(),
                actual_progress=getattr(b,'progress',None),native_partial_events=partial_events(b),failed_partial_compute_not_assumed_zero=True));raise


def launch(c,j,a,resume=False):
    require(a.confirm==('RESUME_VALIDATED_PUBLIC_DUALTEACHER' if resume else 'START_PUBLIC_NCASE'),'Explicit start token missing')
    from admission import inspect_ready
    old=read(j/'admission.json')
    require(old['status']=='READY_INDEPENDENTLY_VERIFIED' and old['contract_sha256']==contract_sha(c), 'No independent admission')
    current=inspect_ready(c,j,resume=resume)
    with leased(c) as fd:
        cap=capacity(c,True)
        if resume:
            from resume_review import reconcile
            event=reconcile(c,j);suffix='_resume_'+event['event']
        else:
            require(not Path(c['output']).exists() and not (j/'launch_intent.json').exists(),'Formal launch already attempted')
            suffix=''
        exclusive(j/('launch_intent'+suffix+'.json'),dict(contract_sha256=contract_sha(c),admission=current,capacity=cap,at=time.time()))
        command=[sys.executable,str(HERE/'public_operator.py'),'--action','worker','--job',str(j),
                 '--contract-sha',contract_sha(c),'--fd',str(fd)]
        if resume:command.append('--resume-worker')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUID,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONWARNINGS='')
        log=j/('boot'+suffix+'.log')
        with log.open('xb') as stream:
            process=subprocess.Popen(command,cwd=str(ROOT),env=env,pass_fds=(fd,),start_new_session=True,
                                     stdout=stream,stderr=subprocess.STDOUT)
        if resume:
            # The consumed review is bound to this new PID before the worker
            # proceeds; worker waits only for this bounded parent handoff.
            from resume_review import bind_worker_pid
            bind_worker_pid(c,process.pid)
        exclusive(j/('dispatch'+suffix+'.json'),dict(pid=process.pid,command=command,boot=str(log),
                  contract_sha256=contract_sha(c),at=time.time()))
    return dict(status='DISPATCHED',pid=process.pid,job=str(j),contract_sha256=contract_sha(c))


def worker(c,j,a):
    require(a.fd is not None,'Missing inherited lease FD')
    x,y=os.fstat(a.fd),LEASE.stat()
    require((x.st_dev,x.st_ino)==(y.st_dev,y.st_ino),'Wrong inherited lease')
    assert_worker_lease(c)
    from dualteacher_controller import FixedCutoffRun
    try:
        if a.resume_worker:
            from resume_review import wait_worker_binding
            wait_worker_binding(c)
        b=backend(c,c['output'])
        result=FixedCutoffRun(c['recipe'],c['output'],c['splits']['test'],resume=a.resume_worker).run(b)
        exclusive(j/('exit_%d.json'%os.getpid()),dict(exit_code=0,status=result['status'],at=time.time()))
    except BaseException as error:
        exclusive(j/('worker_error_%d.json'%os.getpid()),dict(error=repr(error),traceback=traceback.format_exc(),at=time.time()));raise


def partial_events(b):
    core=getattr(b,'core',None)
    return {name:list(getattr(getattr(core,name,None),'events',[])) for name in ('audit_student','audit_op')}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--action',required=True,choices=('prepare','qualify','resume-probe','admit','start','worker','review-resume','resume'))
    parser.add_argument('--labels',type=int,choices=(5,10));parser.add_argument('--tag',default='r1')
    parser.add_argument('--job');parser.add_argument('--contract-sha');parser.add_argument('--confirm')
    parser.add_argument('--window',choices=('warmup','sam'));parser.add_argument('--fd',type=int)
    parser.add_argument('--checkpoint-sha');parser.add_argument('--reason');parser.add_argument('--resume-worker',action='store_true')
    a=parser.parse_args()
    if a.action=='prepare':result=prepare(a)
    else:
        j=Path(a.job);require(sha(j/'contract.json')==a.contract_sha,'Contract identity differs')
        c=read(j/'contract.json');verify(c)
        if a.action in ('qualify','resume-probe'):
            name='qualify.log' if a.action=='qualify' else 'probe_'+str(a.window)+'.log'
            with (j/name).open('x') as capture,contextlib.redirect_stdout(capture),contextlib.redirect_stderr(capture):
                result=qualify(c,j) if a.action=='qualify' else probe(c,j,a.window)
        elif a.action=='admit':
            from admission import inspect_ready
            result=inspect_ready(c,j);exclusive(j/'admission.json',result)
        elif a.action=='review-resume':
            from resume_review import create_review
            result=create_review(c,j,a.checkpoint_sha,a.reason)
        elif a.action in ('start','resume'):result=launch(c,j,a,a.action=='resume')
        else:worker(c,j,a);return
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=='__main__':main()
