"""Prepare or dispatch the one authorized non-bitwise sup100 recovery branch."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import recovery_support_20261019 as support
import train_public_sup_recovery_20261019 as runner
import public_recovery_guard_20261019 as guard
from expes.semi05_gate.public_sup_20261018 import train_public_basic_sup_20261018 as old

ROOT,BASE,STORE,ARM=support.ROOT,support.BASE,support.STORE,support.ARM
HERE=ROOT/'expes/semi05_gate/public_sup_20261018'
PYTHON='__RECORDED_PYTHON__'
SOURCES=['train_public_sup_recovery_20261019.py','recovery_support_20261019.py',
    'public_recovery_guard_20261019.py','launch_public_sup_recovery_20261019.py',
    'test_public_sup_recovery_20261019.py','diagnose_sup100_replay_20261018.py']
require=support.require
def new_json(p,data):
    with p.open('x') as f:
        json.dump(data,f,indent=2,sort_keys=True,allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
def absent():
    require(not list((BASE/'control').glob(ARM+'.*')), 'Existing intent/boot/dispatch; no retry')
    for p in [BASE/ARM,STORE/ARM]:
        guard.no_symlink(p)
        require(not os.path.lexists(p), 'Existing recovery output/artifact; inspect first')
    require(not list((STORE/'sup100').glob('*.partial')), 'Old partial requires investigation')
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',required=True,choices=['prepare','launch'])
    p.add_argument('--preparation-sha')
    args=p.parse_args()
    require(Path(sys.executable).resolve()==Path(PYTHON).resolve(),'Wrong environment')
    absent()
    oldcfg=json.loads((HERE/'sup100.json').read_text())
    require(oldcfg==json.loads((BASE/'sup100/config.json').read_text()),'Original config changed')
    old.verify_manifests(oldcfg)
    require(runner.sha256_file(support.CKPT)==support.CKPT_SHA,'Original checkpoint changed')
    require(runner.sha256_file(BASE/'sup100/training_steps.jsonl')==support.LEDGER_SHA,'Original ledger changed')
    failure=json.loads((BASE/'sup100/failure.json').read_text())
    require(failure['successful_updates']==14930 and failure['completed_epochs']==43,'Failure changed')
    require(not (BASE/'sup100/test_evaluation_started.json').exists(),'Old test started')
    cp=HERE/'sup100_recovery_r1.json'; prep=HERE/'recovery_preparation_20261019.json'
    if args.action=='prepare':
        require(not cp.exists() and not prep.exists(),'Preparation exists; do not repeat')
        cfg=dict(oldcfg)
        cfg.update(output_dir=str(BASE/ARM),artifact_dir=str(STORE/ARM),
            resource_guard_module='public_recovery_guard_20261019',source_sha256=dict(oldcfg['source_sha256']),
            recovery={'boundary_updates':14878,'boundary_epochs':43,'discarded_original_suffix_updates':52,
                'continuation_replay_verified':False,'authorized_non_bitwise':True,'amp_fixed_scale':1024.0,
                'checkpoint_sha256':support.CKPT_SHA,'original_ledger_sha256':support.LEDGER_SHA})
        for name in SOURCES:
            cfg['source_sha256'][str((HERE/name).relative_to(ROOT))]=runner.sha256_file(HERE/name)
        runner.validate_config(cfg); runner.verify_manifests(cfg)
        with guard.original_lock():
            capacity=guard.capacity_snapshot(14*guard.GIB+guard.MARGIN)
        check=subprocess.run([PYTHON,str(HERE/'test_public_sup_recovery_20261019.py')],capture_output=True,text=True)
        require(check.returncode==0,'Recovery CPU tests failed: '+check.stdout+check.stderr)
        new_json(cp,cfg)
        new_json(prep,{'status':'READY_NOT_LAUNCHED','time':runner.now_iso(),'config_sha256':runner.sha256_file(cp),
            'source_sha256':cfg['source_sha256'],'capacity':capacity,'cpu_tests':check.stdout+check.stderr,
            'authorization':'2026-10-19 human selected new branch from 43 completed epochs; disclose non-bitwise',
            'no_test_before_final':True,'no_auto_retry':True})
        print(json.dumps({'status':'READY_NOT_LAUNCHED','preparation_sha256':runner.sha256_file(prep),
            'config_sha256':runner.sha256_file(cp),'capacity':capacity})); return
    require(args.preparation_sha==runner.sha256_file(prep),'Preparation SHA mismatch')
    receipt=json.loads(prep.read_text()); cfg=json.loads(cp.read_text())
    require(receipt['status']=='READY_NOT_LAUNCHED' and receipt['config_sha256']==runner.sha256_file(cp), 'Config SHA mismatch')
    require(receipt['source_sha256']==cfg['source_sha256'],'Source receipt mismatch')
    runner.validate_config(cfg); runner.verify_manifests(cfg)
    control=BASE/'control'
    fd=os.open(control/'gpu1.lock',os.O_RDWR|os.O_NOFOLLOW)
    s=os.fstat(fd)
    runner.validate_gpu_claim(cfg,1,fd,s.st_dev,s.st_ino)
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total',
        '--format=csv,noheader,nounits'],text=True)
    row=next(x.split(', ') for x in rows.splitlines() if x.startswith('1,'))
    require(int(row[2])<256 and int(row[3])>=23000,'GPU1 not idle/large enough')
    procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    require(row[1] not in procs,'GPU1 compute process present')
    with guard.original_lock():
        capacity=guard.capacity_snapshot(14*guard.GIB+guard.MARGIN)
    absent()
    command=[PYTHON,'-u',str(HERE/'train_public_sup_recovery_20261019.py'),'--config',str(cp),
        '--config-sha256',runner.sha256_file(cp),'--gpu','1','--gpu-claim-fd',str(fd),
        '--gpu-claim-device',str(s.st_dev),'--gpu-claim-inode',str(s.st_ino)]
    intent={'time':runner.now_iso(),'arm':ARM,'gpu':1,'command':command,'preparation_sha256':args.preparation_sha,
        'config_sha256':runner.sha256_file(cp),'capacity':capacity,'automatic_retry':False}
    new_json(control/(ARM+'.intent.json'),intent)
    try:
        env=dict(os.environ,PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES='1',PYTHONUNBUFFERED='1')
        with (control/(ARM+'.boot.log')).open('xb') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,pass_fds=(fd,))
        new_json(control/(ARM+'.dispatch.json'),dict(intent,pid=child.pid,status='DISPATCHED_NOT_STEP_VERIFIED'))
        print(json.dumps({'pid':child.pid,'arm':ARM,'gpu':1,'status':'DISPATCHED_NOT_STEP_VERIFIED'}))
    except BaseException as e:
        new_json(control/(ARM+'.dispatch_failure.json'),{'error':str(e),'no_auto_retry':True}); raise
    finally:
        os.close(fd)
if __name__=='__main__':
    main()
