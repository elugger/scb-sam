"""Explicit prepare -> real qualification -> one launch; no automatic queue."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

from budget_protocol import BUDGETS
from fixed_endpoint import Endpoint, FixedFinalRun, exclusive_json
from runtime_data import ROOT, bound_splits, bind_case_files
from runtime_resources import enroll_target, resource_backend, RAM_REQUIRED, sha

HERE=Path(__file__).resolve().parent
BASE=ROOT/'expes/work_dir/semi05_gate/MAIN_TABLE_FIXED_FINAL_20261019'
STORE=ROOT/'expes/artifacts/FCD_A_MATRIX_20261006/MAIN_TABLE_FIXED_FINAL_20261019'
OWN_FILES=('budget_protocol.py','historical_kernels.py','fixed_endpoint.py','runtime_data.py',
           'runtime_resources.py','mt_runtime.py','run_main_baseline.py','source_inventory.json','dataset_pins.json')


def job_paths(method,dataset,ratio,tag):
    if method not in ('mt_semisam','uamt_semisam') or dataset not in ('BY','Public433') or ratio not in (5,10):
        raise ValueError('Unsupported closed main-table arm')
    if not re.fullmatch(r'r[1-9][0-9]*',tag):
        raise ValueError('Invalid engineering tag')
    rel=Path(method)/dataset/('semi'+str(ratio).zfill(2))/'fold2'/tag
    return BASE/rel,STORE/rel/'final.pth'


def verify_contract(c):
    for row in c['sources']:
        p=Path(row['path'])
        if p.is_symlink() or sha(p)!=row['sha256']:
            raise ValueError('Bound source changed: '+str(p))
    if bound_splits(c['dataset'],c['ratio'])!=c['splits']:
        raise ValueError('Actual split changed')
    if bind_case_files(c['dataset'],c['splits'])!=c['case_files']:
        raise ValueError('Actual named case mapping changed')
    if sha(ROOT/'ckpt/sam_med3d_turbo.pth')!=c['sam_sha256']:
        raise ValueError('Foundation initializer changed')
    expected=next(b for b in BUDGETS if b.dataset==c['dataset'] and b.ratio==c['ratio'])
    if c['primary_pairs']!=expected.primary_pairs or c['seed']!=2027 or c['fold']!=2:
        raise ValueError('Frozen training contract differs')
    if Path(c['target'])!=job_paths(c['method'],c['dataset'],c['ratio'],c['tag'])[1]:
        raise ValueError('Wrong exact artifact target')


def prepare(method,dataset,ratio,tag,gpu):
    job,target=job_paths(method,dataset,ratio,tag)
    if job.exists():
        raise ValueError('Existing preparation/control; inspect rather than repeat')
    job.mkdir(parents=True)
    sources=[]
    for name in OWN_FILES:
        p=HERE/name;sources.append(dict(path=str(p),sha256=sha(p)))
    inventory=json.loads((HERE/'source_inventory.json').read_text())
    for row in inventory:
        p=ROOT/row['remote_path']
        if sha(p)!=row['remote_sha256']:
            raise ValueError('Original historical source changed')
        sources.append(dict(path=str(p),sha256=row['remote_sha256']))
    dependency_paths=list((ROOT/'segment_anything').rglob('*.py'))
    dependency_paths+=list((ROOT/'SemiSAM-main/code_semisam+/networks').glob('*.py'))
    dependency_paths += [ROOT/'SemiSAM-main/code_semisam+/utils/click_method.py',
                         ROOT/'utils/cascade_train_utils.py']
    for p in dependency_paths:
        if p.is_symlink():
            raise ValueError('Source dependency symlink')
        sources.append(dict(path=str(p),sha256=sha(p)))
    sources=list({r['path']:r for r in sources}.values())
    splits=bound_splits(dataset,ratio)
    budget=next(b for b in BUDGETS if b.dataset==dataset and b.ratio==ratio)
    recipe=dict(semi=dict(consistency=0.1,consistency_rampup=200.,ramp_iter_div=150,
                         sam_consistency_scale=0.1,prompt='unc',sam_num_clicks=10,
                         noise_std=0.1,noise_clamp=0.2,uamt_mc_samples=8),ema=dict(alpha=0.99))
    c=dict(method=method,dataset=dataset,ratio=ratio,tag=tag,gpu=gpu,fold=2,seed=2027,
           target=str(target),splits=splits,case_files=bind_case_files(dataset,splits),primary_pairs=budget.primary_pairs,
           sam_sha256=json.loads((HERE/'dataset_pins.json').read_text())['SAM_SHA'],
           sources=sources,recipe=recipe,output=str(job/'formal'),
           comparison_scope='primary 3D SSL exposure; architecture/init/auxiliary compute disclosed',
           original_reference_iterations=30000,selection='terminal_after_fixed_exposure',
           test_evaluations=1,early_stop=False,exact_resume_supported=False)
    verify_contract(c)
    capacity=enroll_target(target)
    exclusive_json(job/'contract.json',c)
    exclusive_json(job/'preparation_bound.json',dict(contract_sha256=sha(job/'contract.json'),
                   source_files=len(sources),capacity=capacity,actual_gpu_qualified=False))
    return dict(status='BOUND_NEEDS_ACTUAL_QUALIFICATION',job=str(job),contract_sha256=sha(job/'contract.json'))


def gpu_idle(gpu):
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    uuid=next(line.split(',')[1].strip() for line in rows.splitlines() if int(line.split(',')[0])==gpu)
    if any(line.split(',')[0].strip()==uuid for line in processes.splitlines() if line.strip()):
        raise ValueError('Designated GPU already has a compute process')


def endpoint(c):
    return Endpoint(c['method'],c['primary_pairs'],c['primary_pairs'],1,tuple(c['splits']['test']))


def qualify(job,c):
    gpu_idle(c['gpu'])
    exclusive_json(job/'qualification_intent.json',dict(contract_sha256=sha(job/'contract.json'),gpu=c['gpu']))
    os.environ['CUDA_VISIBLE_DEVICES']=str(c['gpu'])
    from mt_runtime import SemiSAMBackend
    try:
        result=SemiSAMBackend(c,job/'qualification_backend',verify_contract).qualify(endpoint(c))
        result.update(contract_sha256=sha(job/'contract.json'),gpu=c['gpu'])
        exclusive_json(job/'qualification.json',result)
        ready=dict(status='READY',contract_sha256=sha(job/'contract.json'),
                   qualification_sha256=sha(job/'qualification.json'),gpu=c['gpu'])
        exclusive_json(job/'ready.json',ready)
        return dict(status='READY',ready_sha256=sha(job/'ready.json'),job=str(job))
    except BaseException as error:
        exclusive_json(job/'qualification_failure.json',dict(error=repr(error),traceback=traceback.format_exc(),
                       formal_updates=0,formal_test_evaluations=0))
        raise


def launch(job,c,ready_sha):
    if sha(job/'ready.json')!=ready_sha:
        raise ValueError('Ready receipt identity differs')
    ready=json.loads((job/'ready.json').read_text())
    if (ready['contract_sha256']!=sha(job/'contract.json') or
            ready['qualification_sha256']!=sha(job/'qualification.json')):
        raise ValueError('Actual qualification binding differs')
    verify_contract(c)
    guard=resource_backend(c['target'])
    with guard.original_lock():
        guard.capacity_snapshot(RAM_REQUIRED)
        gpu_idle(c['gpu'])
        for p in [job/'formal',job/'started.json',job/'dispatch.json',job/'boot.log',
                  Path(c['target']),Path(c['target']+'.partial'),job/'worker_failure.json']:
            if p.exists():
                raise ValueError('Existing formal/control/artifact residual: '+str(p))
        exclusive_json(job/'launch_intent.json',dict(ready_sha256=ready_sha,contract_sha256=sha(job/'contract.json')))
        command=[sys.executable,str(HERE/'run_main_baseline.py'),'--action','worker',
                 '--method',c['method'],'--dataset',c['dataset'],'--ratio',str(c['ratio']),
                 '--tag',c['tag'],'--gpu',str(c['gpu']),'--ready-sha',ready_sha]
        env=dict(os.environ,PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES=str(c['gpu']),PYTHONDONTWRITEBYTECODE='1')
        with (job/'boot.log').open('xb') as stream:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=stream,start_new_session=True)
        exclusive_json(job/'dispatch.json',dict(pid=proc.pid,command=command,ready_sha256=ready_sha))
    return dict(status='DISPATCHED_ONCE',pid=proc.pid,job=str(job))


def worker(job,c,ready_sha):
    if sha(job/'ready.json')!=ready_sha:
        raise ValueError('Worker ready identity differs')
    exclusive_json(job/'started.json',dict(pid=os.getpid(),command=[sys.executable]+sys.argv))
    from mt_runtime import SemiSAMBackend
    try:
        backend=SemiSAMBackend(c,job/'formal',verify_contract)
        result=FixedFinalRun(endpoint(c),job/'formal').run(backend)
        exclusive_json(job/'exited.json',dict(pid=os.getpid(),exit_code=0,status=result['status']))
    except BaseException as error:
        exclusive_json(job/'worker_failure.json',dict(pid=os.getpid(),error=repr(error),traceback=traceback.format_exc()))
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--action',choices=('prepare','qualify','launch','worker'),required=True)
    parser.add_argument('--method',choices=('mt_semisam','uamt_semisam'),required=True)
    parser.add_argument('--dataset',choices=('BY','Public433'),required=True)
    parser.add_argument('--ratio',type=int,choices=(5,10),required=True)
    parser.add_argument('--gpu',type=int,choices=(1,),default=1)
    parser.add_argument('--tag',default='r1')
    parser.add_argument('--ready-sha')
    args=parser.parse_args()
    job,_=job_paths(args.method,args.dataset,args.ratio,args.tag)
    if args.action=='prepare':
        result=prepare(args.method,args.dataset,args.ratio,args.tag,args.gpu)
    else:
        c=json.loads((job/'contract.json').read_text())
        if c['gpu']!=args.gpu:
            raise ValueError('Actual GPU differs')
        result=(qualify(job,c) if args.action=='qualify' else
                launch(job,c,args.ready_sha) if args.action=='launch' else worker(job,c,args.ready_sha))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
