"""Read-only own live ledger observer. No models, inference or dispatch."""
import argparse,datetime,fcntl,hashlib,json,os,subprocess,sys
from pathlib import Path
from supervisor_verify_dualteacher import ROOT,BASE,DEPLOY,LOCK_ID,need,sha,read_json,regular,source_inputs,verify_ledger

def observe(job,expected,previous=None,previous_sha=None):
    need(sha(job/'contract.json')==expected,'External contract SHA differs')
    c=read_json(job/'contract.json');budget,pairs=source_inputs(c,DEPLOY,job)
    dispatch=read_json(job/'dispatch.json');ready_sha=sha(job/'ready.json');pid=dispatch['pid']
    wanted=['__RECORDED_PYTHON__',str(DEPLOY/'run_dualteacher.py'),'--action','worker','--method','dualteacher_sam','--dataset','BY','--ratio',str(c['ratio']),'--tag',c['tag'],'--gpu','1','--contract-sha',expected,'--ready-sha',ready_sha,'--lease-fd']
    need(len(dispatch['command'])==20 and dispatch['command'][:-1]==wanted and dispatch['command'][-1].isdigit()
         and dispatch['contract_sha256']==expected and dispatch['ready_sha256']==ready_sha,'Dispatched command/contract chain differs')
    boot=(job/'boot.log').read_bytes();need(boot==b'','Unexpected worker stdout/stderr; preserve for explicit review')
    failures=[str(p.relative_to(job)) for p in job.rglob('*') if 'fail' in p.name.lower()]
    need(not failures,'Failure evidence preserved, do not retry')
    ledger=job/'formal/steps.jsonl'
    raw=ledger.read_bytes() if ledger.exists() else b'';end=raw.rfind(b'\n')+1;complete=raw[:end]
    rows=[json.loads(x) for x in complete.splitlines()]
    counts,cycles=verify_ledger(c,budget,pairs,rows,False)
    inode=(ledger.stat().st_dev,ledger.stat().st_ino) if ledger.exists() else None
    if previous is not None:
        need(previous_sha and sha(previous)==previous_sha,'Previous snapshot SHA differs')
        old=read_json(previous);need(old['contract_sha256']==expected and old['steps']<=len(rows),'Snapshot identity/progress regressed')
        cursor=old['ledger_cursor']
        if cursor is not None:
            need(list(inode)==cursor['identity'] and len(complete)>=cursor['bytes'] and hashlib.sha256(complete[:cursor['bytes']]).hexdigest()==cursor['sha256'],'Old ledger inode/pinned prefix changed')
    total=budget.primary_pairs//2;final=Path(c['target']);partial=Path(str(final)+'.partial')
    test_marker=job/'formal/dualteacher_final_test_started.json'
    need(len(rows)==total or not any(p.exists() for p in (final,partial,test_marker)),'Terminal save/test before full native budget')
    gpu=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True).strip()
    gp=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines()
    uuid='GPU_CONFIGURED_AT_RUNTIME';need(gpu.split(',')[1].strip()==uuid,'Physical GPU1 UUID differs')
    gpu_pids=[int(r.split(',')[1]) for r in gp if r.split(',')[0].strip()==uuid]
    proc=Path('/proc')/str(pid);alive=proc.exists();lease_proof=None;actual_command=None
    if alive:
        need(proc.stat().st_uid==os.getuid(),'Own process owner differs')
        actual_command=(proc/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        need(actual_command==dispatch['command'],'Actual PID full command differs')
        env=(proc/'environ').read_bytes().split(b'\0');need(b'CUDA_VISIBLE_DEVICES=1' in env and b'PYTHONNOUSERSITE=1' in env,'Actual GPU1/process environment isolation differs')
        need(gpu_pids==[pid] or (not rows and not gpu_pids),'GPU1 PID attribution differs')
        lease=Path(c['native_gpu1_lease_path']);need(lease==BASE/'native_gpu1_r4.lock' and lease.read_bytes()==b'NATIVE_MAIN_TABLE_R4_GPU1_EXCLUSIVE_OPERATION_V1\n','Original native lease path/marker differs');s=regular(lease);fdnum=int(dispatch['command'][-1]);fdpath=proc/'fd'/str(fdnum)
        stat=fdpath.stat();need((stat.st_dev,stat.st_ino)==(s.st_dev,s.st_ino) and dispatch['native_lease']==dict(device=s.st_dev,inode=s.st_ino),'Actual inherited lease FD identity differs')
        competing=os.open(lease,os.O_RDONLY|os.O_NOFOLLOW)
        try:
            try:fcntl.flock(competing,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:lease_proof=dict(device=s.st_dev,inode=s.st_ino,competing_exclusive_lock_rejected=True)
            else:need(False,'Own native lease not held')
        finally:os.close(competing)
        status='RUNNING_TERMINAL_SAVE_OR_TEST' if len(rows)==total else 'RUNNING_DUALTEACHER_LEDGER_VERIFIED'
    else:
        need((job/'exited.json').exists() and read_json(job/'exited.json')['exit_code']==0 and len(rows)==total,'Worker disappeared without normal full completion')
        need(not gpu_pids,'Designated GPU1 not idle for final acceptance')
        status='FINAL_ACCEPTANCE_REQUIRED'
    sys.path.insert(0,str(DEPLOY));from runtime_resources import resource_backend,RAM_REQUIRED
    guard=resource_backend(final,c['checkpoint_bound'])
    with guard.original_lock():capacity=guard.capacity_snapshot(RAM_REQUIRED)
    need(all(capacity['save_lock'][k]==v for k,v in LOCK_ID.items()),'Original save lock identity differs')
    return dict(at=datetime.datetime.now().astimezone().isoformat(),status=status,contract_sha256=expected,actual_pid=pid,pid_alive=alive,actual_command=actual_command,
        gpu1=gpu,gpu1_compute_pids=gpu_pids,native_lease=lease_proof,steps=len(rows),total_steps=total,percent=100*len(rows)/total,primary_inputs_per_stream=2*len(rows),
        complete_cycles=len(cycles),counts=counts,source_input_pins_verified=True,whole_native_ledger_verified=True,models_loaded=0,tests_repeated=0,boot_bytes=0,failures=[],
        ledger_cursor=dict(identity=list(inode),bytes=len(complete),sha256=hashlib.sha256(complete).hexdigest()) if inode else None,
        trailing_incomplete_ledger_bytes=len(raw)-end,previous_snapshot_sha256=previous_sha,final_exists=final.exists(),own_terminal_partial_exists=partial.exists(),test_started=test_marker.exists(),capacity=capacity)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=Path,required=True);p.add_argument('--contract-sha',required=True);p.add_argument('--previous',type=Path);p.add_argument('--previous-sha');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    need(not a.output.exists() and ROOT in a.output.parents and not any(p in a.output.parents for p in (a.job,DEPLOY,ROOT/'expes/artifacts')),'New external observation output required')
    result=observe(a.job,a.contract_sha,a.previous,a.previous_sha)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(status=result['status'],steps=result['steps'],total=result['total_steps'],sha256=sha(a.output))))

if __name__=='__main__':main()
