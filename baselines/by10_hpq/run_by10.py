"""Explicit one-shot BY10 supervisor -> HPQ initialization -> bridge -> formal queue."""
import argparse,fcntl,hashlib,importlib,json,os,subprocess,sys,traceback
from pathlib import Path
from by_protocol import *
from prepare_by5 import dump_new,stamp,make_bound_config
from by_resources import original_lock,capacity_snapshot,enroll_locked

BASE,STORE,SRC,CONTROL=map(Path,(BASE,STORE,SRC,CONTROL))

def status(state,**kw):
    from train_by5 import atomic_json
    atomic_json(CONTROL/'status.json',dict(at=stamp(),state=state,pid=os.getpid(),**kw))

def prepare(bundle):
    verify_bundle(bundle)
    require(not CONTROL.exists() and not STORE.exists(),'Existing run; no overwrite')
    cfg=make_bound_config(bundle)
    from by_inputs import bind,identity
    from train_by5 import read_case_ids
    ids={n:read_case_ids(SPLIT/(n+'.txt')) for n in ['train_lab','train_unlab','test']}
    old=ROOT/'expes/splits/by_semi05_cv5/fold2'
    require(set(read_case_ids(old/'train_lab.txt'))<=set(ids['train_lab']),'Non-nested labeled split')
    require(read_case_ids(old/'test.txt')==ids['test'],'Different evaluation set')
    require(len(set(sum(ids.values(),[])))==125,'Overlapping split')
    metadata=bind(cfg,ids['train_lab']+ids['train_unlab'],ids['test'])
    inputs={v['path']:dict(identity=v,sha256=sha(v['path'])) for cases in metadata.values() for pair in cases.values() for v in pair.values()}
    for p,v in inputs.items():require(identity(p,Path(ROOT)/'data')==v['identity'],'Changed input')
    ancestry_path=Path(ROOT)/'expes/semi05_gate/by5_four_arm_initialization_check_20261030/ancestry.json'
    ancestor=json.loads(ancestry_path.read_text())
    require(hashlib.sha256(ancestor['config_text'].encode()).hexdigest()==ancestor['config_sha256'],'Ancestor config corrupt')
    for p,h in ancestor['source_hashes'].items():require(sha(p)==h,'Historical scientific source changed: '+p)
    for name in ['expes.semi05_gate.paired_fcd_matrix_20261006','expes.semi05_gate.fcd_deterministic_losses_r2_20261005','expes.common.seg_metrics','expes.common.semi_data','utils.cascade_train_utils']:
        importlib.import_module(name)
    sources=dict(cfg['source_sha256'])
    for m in list(sys.modules.values()):
        p=Path(getattr(m,'__file__','') or '/nonexistent').resolve()
        if Path(ROOT) in p.parents and p.is_file():sources[str(p)]=sha(p)
    sources[str(ancestry_path)]=sha(ancestry_path)
    from by_runtime import ensure_gpu0_idle
    gpu=ensure_gpu0_idle()
    with original_lock():
        capacity=capacity_snapshot(config=cfg)
        enrollment=enroll_locked()
        capacity=capacity_snapshot(config=cfg)
        CONTROL.mkdir(parents=True)
        dump_new(CONTROL/'supervised_config.json',cfg)
        dump_new(CONTROL/'preparation.json',dict(status='PREPARED',at=stamp(),bundle_sha256=bundle,
            prefix_config_sha256=sha(CONTROL/'supervised_config.json'),source_hashes=sources,
            inputs=inputs,case_file_metadata=metadata,case_ids=ids,ancestor=ancestor,
            supervised_epochs=60,supervised_updates=600,semi_epochs=38,semi_successful_updates=684,semi_main_pairs=3420,
            budget_note='Supervision: 60 exposures per labeled case. Semi: equal total main pairs and actual updates to BY5; 90U gives 38 epochs. Original epoch-based curricula retained.',
            gpu=gpu,capacity=capacity,enrollment=enrollment))
    print(json.dumps(dict(status='PREPARED',bundle_sha=bundle,preparation_sha=sha(CONTROL/'preparation.json'),source_count=len(sources),input_files=len(inputs))),flush=True)

def verify(bundle,prep_sha):
    verify_bundle(bundle)
    require(sha(CONTROL/'preparation.json')==prep_sha,'Preparation changed')
    prep=json.loads((CONTROL/'preparation.json').read_text())
    require(prep['bundle_sha256']==bundle,'Bundle mismatch')
    require(sha(CONTROL/'supervised_config.json')==prep['prefix_config_sha256'],'Config changed')
    for p,h in prep['source_hashes'].items():require(sha(p)==h,'Source changed: '+p)
    return prep

def launch(a):
    verify(a.bundle,a.preparation)
    require(not (CONTROL/'dispatch.json').exists(),'Already dispatched')
    from by_runtime import ensure_gpu0_idle
    ensure_gpu0_idle()
    fd=os.open(CONTROL/'gpu0.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    oldfd=os.open(Path(ROOT)/'expes/work_dir/semi05_gate/BY5_FOUR_ARM_RERUN_20261030/control/gpu0.lock',os.O_RDWR|os.O_NOFOLLOW)
    fcntl.flock(oldfd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stat=os.fstat(fd)
    cmd=[sys.executable,'-u','-B',str(SRC/'run_by10.py'),'--action','supervise','--bundle',a.bundle,'--preparation',a.preparation,'--fd',str(fd),'--dev',str(stat.st_dev),'--ino',str(stat.st_ino)]
    env=dict(os.environ,PYTHONPATH=str(SRC)+':'+str(ROOT),CUDA_VISIBLE_DEVICES=GPU_UUID,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with original_lock():
        capacity_snapshot()
        with (CONTROL/'queue.log').open('xb') as log:
            p=subprocess.Popen(cmd,cwd=str(ROOT),env=env,pass_fds=(fd,oldfd),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        receipt=dict(at=stamp(),pid=p.pid,command=cmd,gpu=0,gpu_uuid=GPU_UUID,phases=['supervised','initial','bridge','formal'],automatic_retry=False)
        dump_new(CONTROL/'dispatch.json',receipt)
    os.close(fd);os.close(oldfd)
    print(json.dumps(receipt),flush=True)

def supervise(a):
    prep=verify(a.bundle,a.preparation)
    try:
        for phase in ['supervised','accept','initial','bridge','formal']:
            verify(a.bundle,a.preparation)
            cmd=[sys.executable,'-u','-B',str(SRC/'run_by10.py'),'--action','worker','--phase',phase,'--bundle',a.bundle,'--preparation',a.preparation,'--fd',str(a.fd),'--dev',str(a.dev),'--ino',str(a.ino)]
            with (CONTROL/(phase+'.log')).open('xb') as log:
                p=subprocess.Popen(cmd,cwd=str(ROOT),pass_fds=(a.fd,),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                status('RUNNING',phase=phase,child_pid=p.pid)
                rc=p.wait()
            require(rc==0,'Phase failed: '+phase+'; see '+str(CONTROL/(phase+'.log')))
        result=json.loads((BASE/'formal/HPQ/result.json').read_text())
        require(result['successful_updates']==684 and result['completed_epochs']==38,'Formal budget incomplete')
        status('COMPLETE',metrics=result['metrics'],checkpoint=result['checkpoint'])
    except BaseException:
        status('STOPPED_NEEDS_ATTENTION',error=traceback.format_exc())
        raise

def worker(a):
    prep=verify(a.bundle,a.preparation)
    cfg=json.loads((CONTROL/'supervised_config.json').read_text())
    from train_by5 import validate_gpu_claim
    claim=validate_gpu_claim(cfg,0,a.fd,a.dev,a.ino)
    if a.phase=='supervised':
        from train_by5 import run
        run(cfg,prep['prefix_config_sha256'],0,gpu_claim=claim)
    elif a.phase=='accept':
        from train_by5 import load_trusted_checkpoint,state_dict_sha256
        result=json.loads((BASE/'supervised/final_metrics.json').read_text())
        reload=json.loads((BASE/'supervised/final_reload_verified.json').read_text())
        require(result['status']=='COMPLETE' and result['successful_updates']==600 and result['completed_epochs']==60,'Prefix incomplete')
        p=STORE/'supervised/final.pth'
        with original_lock():
            capacity_snapshot(config=cfg)
            require(sha(p)==result['final_sha256']==reload['sha256'] and reload['strict_load'],'Prefix readback failed')
            ck=load_trusted_checkpoint(p,map_location='cpu')
            require(ck['successful_updates']==600 and ck['completed_epochs']==60,'Checkpoint budget mismatch')
            acceptance=dict(successful_updates=600,strict_reload=True,initializer=dict(sha256=sha(p),state_sha256=state_dict_sha256(ck['student'])))
        dump_new(CONTROL/'initializer_acceptance.json',acceptance)
        prep['acceptance_sha256']=sha(CONTROL/'initializer_acceptance.json')
        dump_new(CONTROL/'hpq_preparation.json',prep)
    else:
        from types import SimpleNamespace
        from four_training import run
        from four_contract import validate_bridge
        hp=sha(CONTROL/'hpq_preparation.json')
        bridge=validate_bridge(a.bundle,hp)['sha256'] if a.phase=='formal' else None
        run(SimpleNamespace(bundle_sha=a.bundle,prepare_sha=hp,phase=a.phase,group='HPQ',fd=a.fd,dev=a.dev,ino=a.ino,bridge_sha=bridge))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--action',choices=['prepare','launch','supervise','worker'],required=True)
    p.add_argument('--bundle',required=True);p.add_argument('--preparation');p.add_argument('--phase')
    for n in ['fd','dev','ino']:p.add_argument('--'+n,type=int)
    a=p.parse_args();os.chdir(str(ROOT))
    if a.action=='prepare':prepare(a.bundle)
    elif a.action=='launch':launch(a)
    elif a.action=='supervise':supervise(a)
    else:worker(a)
