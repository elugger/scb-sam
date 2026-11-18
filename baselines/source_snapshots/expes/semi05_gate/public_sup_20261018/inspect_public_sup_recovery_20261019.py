"""Read-only recovery monitor: never imports Torch, launches, saves weights or evaluates."""
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
from datetime import datetime,timezone
import recovery_support_20261019 as support
import train_public_sup_recovery_20261019 as api
import public_recovery_guard_20261019 as guard

def inspect():
    b=support.BASE/support.ARM
    cfg=json.loads((b/'config.json').read_text())
    support.validate(cfg)
    for group in ['source_sha256','input_sha256']:
        for name,digest in cfg[group].items():
            support.require(api.sha256_file(api.resolve_path(cfg,name))==digest,'Bound file changed: '+name)
    started=json.loads((b/'started.json').read_text())
    restored=json.loads((b/'recovery_restored.json').read_text())
    support.require(restored['checkpoint_sha256']==support.CKPT_SHA and restored['inherited_updates']==14878,
        'Restoration identity mismatch')
    def complete_rows(path):
        raw=path.read_text()
        if raw and not raw.endswith('\n'):
            raw=raw[:raw.rfind('\n')+1]
        return [json.loads(x) for x in raw.splitlines()]
    rows=complete_rows(b/'training_steps.jsonl')
    epochs=complete_rows(b/'training_epochs.jsonl')
    ids=api.read_case_ids(api.resolve_path(cfg,cfg['train_list']))
    new=rows[14878:]
    for i,r in enumerate(new,14879):
        support.require(r['successful_update']==i and r['epoch']==(i-1)//346 and r['epoch_position']==(i-1)%346,
            'New ledger discontinuity')
        support.require(r['optimizer_steps']==1 and not r['skipped'] and r['gradient_finite']
            and r['amp_scale']==1024.0 and math.isfinite(r['loss']) and r['case_id'] in ids,
            'New numerical/case gate failed')
    for e in range(43,len(epochs)):
        actual=rows[e*346:(e+1)*346]
        support.require(len(actual)==346 and set(r['case_id'] for r in actual)==set(ids)
            and epochs[e]['actual_case_ids']==[r['case_id'] for r in actual], 'New complete epoch coverage failed')
    procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory',
        '--format=csv,noheader'],text=True).strip()
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used',
        '--format=csv,noheader'],text=True).strip()
    gpu1=next(x.split(', ')[1] for x in gpu.splitlines() if x.startswith('1,'))
    gpu_pid_match=any(x.split(', ')[0]==str(started['pid']) and gpu1 in x for x in procs.splitlines())
    process=Path('/proc')/str(started['pid'])
    live=False
    if (process/'stat').exists():
        ticks=(process/'stat').read_text().rsplit(')',1)[1].split()[19]
        live=ticks==started['proc_start_ticks']
    failure=json.loads((b/'failure.json').read_text()) if (b/'failure.json').exists() else None
    complete=(b/'final_metrics.json').exists()
    status='FAILED_PRESERVED' if failure else 'COMPLETE_NEEDS_ACCEPTANCE' if complete else 'RUNNING' if live else 'NOT_RUNNING_NEEDS_INSPECTION'
    with guard.original_lock():
        capacity=guard.capacity_snapshot()
    elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(restored['time'])).total_seconds()
    if status=='RUNNING':
        support.require(gpu_pid_match,'Live PID does not match GPU1 compute process')
    result={'time':api.now_iso(),'arm':support.ARM,'status':status,'pid':started['pid'],'pid_identity_live':live,
        'gpu_pid_match':gpu_pid_match,
        'successful_updates':len(rows),'inherited_updates':14878,'new_updates':len(new),'total_updates':20760,
        'percent':len(rows)/20760*100,'completed_epochs':len(epochs),'new_full_epochs':len(epochs)-43,
        'new_step_checks':'PASS','new_epoch_actual_case_checks':'PASS','source_input_checks':'PASS',
        'test_started':(b/'test_evaluation_started.json').exists(),'failure':failure,
        'last20_compute_mean_seconds':statistics.mean(r['elapsed_seconds'] for r in new[-20:]) if new else None,
        'observed_branch_wall_seconds':elapsed,'training_and_save_remaining_hours_estimate':
            (20760-len(rows))*elapsed/len(new)/3600 if len(new)>50 and not failure else None,
        'eta_excludes_final_test_and_acceptance':True,'gpu_processes':procs,'gpus':gpu,'capacity':capacity,
        'continuation_replay_verified':False,'discarded_original_suffix_updates':52,
        'partial_files':[str(p) for p in (support.STORE/support.ARM).glob('*.partial')]}
    return result
if __name__=='__main__':
    print(json.dumps(inspect(),indent=2,allow_nan=False))
