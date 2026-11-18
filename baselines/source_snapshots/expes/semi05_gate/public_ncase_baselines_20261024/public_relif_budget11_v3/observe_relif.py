"""Model-free ReLiF live identity audit and separate fixed-terminal acceptance."""
import argparse
import fcntl
import time
from relif_protocol import *
from relif_ledger import scan,project
from relif_controller import check_metrics


def ledger_snapshot(c):
    # Writers use this same original lock. Parsed bytes, inode and current
    # generations are one coherent snapshot, with no checkpoint deserialization.
    with resource(c).original_lock():
        pin=read_pin(Path(c['output'])/'steps.jsonl',job(c),True);raw=pin.pop('raw')
        review=scan(raw,binding(c))
        require(review['partial_bytes']==0,'Partial durable training ledger requires independent review')
        commits=store(c)._commits();latest={r['path']:r for r in commits};valid=[]
        for path,cp in latest.items():
            artifact=read_pin(path,STORE)
            require(all(artifact[k]==cp[k] for k in ('bytes','sha256')),'Recovery generation bytes changed')
            projection=project(raw,pin,cp,binding(c),contract_sha(c))
            valid.append(cp)
        valid.sort(key=lambda x:x['generation'],reverse=True)
        if valid:require(0<=review['counts']['iterations']-valid[0]['counts']['iterations']<=100,
                         'Latest committed boundary outside declared interval')
        require(len(valid)==min(2,len(commits)),'Missing valid recovery generation')
    return pin,raw,review,valid


def check_previous(c,old,dp,pin,raw):
    require(old['contract_sha256']==contract_sha(c) and old['job']==str(job(c)),'Previous audit identity differs')
    prior=old['ledger']
    if old['dispatch_sha256']==sha(dp):
        require(prior['inode'] is None or (prior['inode'],prior['device'])==(pin['inode'],pin['device']),
                'Ledger inode changed without recovery history')
        require(len(raw)>=prior['bytes'] and hashlib.sha256(raw[:prior['bytes']]).hexdigest()==prior['sha256'],
                'Previously parsed exact byte prefix changed')
        return None
    require(dp.stem.startswith('dispatch_resume_'),'Unknown new worker lineage')
    event=dp.stem[len('dispatch_resume_'):];d=job(c)/'recovery_history'/event
    r=read(d/'review.json');rc=read(d/'reconciliation.json');consumed=read(d/'consumed.json')
    require(r['contract_sha256']==contract_sha(c) and rc['checkpoint']==r['checkpoint']
            and consumed['consumed_by_pid']==read(dp)['pid'],'Recovery binding differs')
    require(old['dispatch_sha256'] in [v['sha256'] for k,v in r['evidence'].items()
            if Path(k).name.startswith('dispatch')],'Old dispatch missing from recovery review')
    original=read_pin(d/'original_steps.jsonl',job(c),True);preserved=original.pop('raw')
    require(original['sha256']==rc['original_ledger_sha256']==r['projection']['original_ledger_pin']['sha256']
        and len(preserved)>=prior['bytes'] and hashlib.sha256(preserved[:prior['bytes']]).hexdigest()==prior['sha256'],
        'Previously observed prefix missing from preserved original ledger')
    cp=r['checkpoint'];n=cp['ledger']['bytes']
    require(len(raw)>=n and hashlib.sha256(raw[:n]).hexdigest()==cp['ledger']['sha256']
            and raw[:n]==preserved[:n], 'Restored committed prefix differs')
    require(sha(d/'uncommitted_steps.jsonl')==r['projection']['entire_uncommitted_tail']['sha256'],
            'Uncommitted tail not preserved')
    loaded=read(job(c)/('restoration_verified_%d.json'%read(dp)['pid']))
    require(loaded['loaded_state_exact'] and loaded['contract_sha256']==contract_sha(c)
            and loaded['checkpoint_sha256']==cp['sha256'] and loaded['ledger']==cp['ledger'],
            'Complete state restoration proof missing')
    return dict(event=event,projection=r['projection'])


def inspect(j,expected,previous=None,final=False):
    require(sha(j/'contract.json')==expected,'Wrong contract SHA');c=read(j/'contract.json');verify(c)
    dp=max(j.glob('dispatch*.json'),key=lambda p:read(p)['at']);d=read(dp);pid=d['pid'];proc=Path('/proc')/str(pid)
    alive=proc.exists();f=Path(c['output'])
    require(d['contract_sha256']==expected,'Dispatch contract differs')
    pin,raw,review,cps=ledger_snapshot(c)
    recovery=check_previous(c,read(previous),dp,pin,raw) if previous else None
    for p in [*j.glob('boot*.log'),j/'qualify.log',j/'probe_warmup.log',j/'probe_larc.log']:
        require(p.is_file(),'Missing bound log: '+str(p));check_logs(p.read_bytes(),allow_partial=alive and p==Path(d['boot']))
    require(not list(j.glob('worker_error*.json')) and not (f/'failure.json').exists(),'Worker failed; preserve evidence')
    gpu=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid,pci.bus_id,memory.used,utilization.gpu',
                                 '--format=csv,noheader'],universal_newlines=True).strip()
    require(gpu.split(',')[0].strip()==UUID and gpu.split(',')[1].strip().endswith('B3:00.0'),'Wrong physical GPU1')
    if alive:
        require((proc/'cmdline').read_bytes().decode().rstrip('\0').split('\0')==d['command'],'PID command differs')
        env=(proc/'environ').read_bytes().split(b'\0')
        require(all(v in env for v in (b'CUDA_VISIBLE_DEVICES='+UUID.encode(),b'PYTHONNOUSERSITE=1',
                                       b'PYTHONDONTWRITEBYTECODE=1')),'Worker environment differs')
        fd=d['command'][d['command'].index('--fd')+1];a,b=(proc/'fd'/fd).stat(),LEASE.stat()
        require((a.st_dev,a.st_ino)==(b.st_dev,b.st_ino),'Missing inherited original lease')
        competitor=lease_open()
        try:
            try:fcntl.flock(competitor,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:pass
            else:raise ValueError('Lease unexpectedly free while worker alive')
        finally:os.close(competitor)
        status='EVALUATING_FIXED_TERMINAL' if (f/'native_test_started.json').exists() else 'RUNNING_SSL'
    else:
        m=read(f/'final_metrics.json')
        require(m['status']=='COMPLETE_NEEDS_INDEPENDENT_ACCEPTANCE' and m['counts']==review['counts']
                ==counts(c['recipe']['ssl_iterations']),'Incomplete native terminal')
        require(read(j/('exit_%d.json'%pid))['exit_code']==0,'Missing normal worker exit')
        gpu_idle();fd=lease_open()
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        finally:os.close(fd)
        status='COMPLETE_NEEDS_INDEPENDENT_ACCEPTANCE'
    try:cap=capacity(c)
    except (ValueError,AssertionError,RuntimeError) as error:
        require(str(error).startswith(('Insufficient filesystem capacity:','Insufficient RAM including cgroup ancestors')),
                'Unknown resource-gate error: '+str(error))
        wait=read(j/'resource_status.json')
        require(alive and wait['state']=='WAIT_CAPACITY' and time.time()-wait['at']<180,'Unconfirmed resource wait')
        cap=dict(wait=wait,error=str(error));status='WAIT_CAPACITY'
    out=dict(status=status,at=time.time(),job=str(j),contract_sha256=expected,pid=pid,pid_alive=alive,
        dispatch_sha256=sha(dp),counts=review['counts'],actual_training_cases=review['actual_train_cases'],
        complete_reference_cycles=review['complete_reference_cycles'],ledger=pin,recovery_generations=cps,
        recovery_transition=recovery,gpu1=gpu,capacity=cap,source_inputs_verified=True,
        lease_verified=alive,lease_released=not alive,test_started=(f/'final_test_intent.json').exists(),
        new_model_loads=0,new_optimization_updates=0,new_test_evaluations=0)
    if final:
        require(not alive and status=='COMPLETE_NEEDS_INDEPENDENT_ACCEPTANCE','Final acceptance requires normal termination')
        cov=read(f/'coverage_verified.json');artifact=m['artifact'];reload=read(f/'relif_reload_verified.json')
        intent=read(f/'final_test_intent.json');started=read(f/'native_test_started.json')
        require(artifact['path']==c['target'] and artifact['sha256']==sha(c['target'])
                and Path(c['target']).stat().st_size==artifact['bytes']<=FINAL_BOUND,'Fixed terminal artifact differs')
        require(reload==m['strict_reload'] and reload['sha256']==artifact['sha256'] and all(reload[k] for k in
            ('strict','native_student_identity_verified','native_projection_identity_verified')),'Strict reload proof differs')
        require(intent==dict(selection='terminal_after_fixed_exposure',final_sha256=artifact['sha256'],
                test_case_ids=c['splits']['test'],test_evaluation=1)
                and started==dict(final_sha256=artifact['sha256'],contract_sha256=expected,test_evaluation=1),
                'Unique test intent/entry identity differs')
        require(cov['actual_case_coverage_verified'] and cov['actual_training_cases']==review['actual_train_cases']==346
                and cov['complete_reference_cycles']==review['complete_reference_cycles']==11
                and cov['primary_labeled_inputs']==cov['primary_unlabeled_inputs']==c['recipe']['primary_inputs_per_stream']
                and cov['successful_optimizer_updates']==c['recipe']['ssl_iterations'],'Budget/coverage differs')
        require(m['test_evaluations']==1 and m['recipe']==c['recipe'],'Terminal selection differs')
        require(len(cps)==2 and cps[0]['counts']==review['counts'] and cps[0]['ledger']==pin,'Terminal recovery mismatch')
        check_metrics(f,m['final_metrics'],c['splits']['test'],artifact)
        out.update(status='ACCEPTED_FIXED11_RELIF_TERMINAL',metrics={k:m['final_metrics'][k] for k in
            ('dice','jaccard','hd95','asd')},distance_unit='voxel',test_cases=87,test_evaluations=1,
            final_artifact=artifact,strict_reload=reload,evidence={str(p):read_pin(p,j) for p in
            [f/'final_metrics.json',f/'coverage_verified.json',f/'relif_reload_verified.json',f/'final_test_intent.json',
             f/'native_test_started.json',f/'final_test_cases.jsonl',j/('exit_%d.json'%pid)]})
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--contract-sha',required=True)
    p.add_argument('--previous');p.add_argument('--out',type=Path,required=True);p.add_argument('--final',action='store_true')
    a=p.parse_args();r=inspect(a.job,a.contract_sha,a.previous,a.final);exclusive(a.out,r)
    print(json.dumps({k:r[k] for k in ('status','counts','actual_training_cases','pid')}))
