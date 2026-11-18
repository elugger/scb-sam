"""Independent, model-free admission for DualTeacher's own two engineering windows."""
import time
from public_dualteacher_protocol import *


def engineering_history(c):
    if c['labeled_cases']!=5:return dict(prior_updates=0,partial_inputs_per_stream=0,prior_failed_probe_updates=0)
    template=read(HERE/'template.json');rows=template.get('additional_failed_attempts',[])
    require(c['repair_lineage']['engineering_effective_updates']==20,'Missing original actual engineering cost')
    for row in rows:
        require(type(row['successful_optimizer_updates']) is int and row['successful_optimizer_updates']>=0 and row['successful_optimizer_updates']%2==0,'Invalid historical complete update count')
        require(type(row['partial_inputs_loaded_per_stream']) is int and row['partial_inputs_loaded_per_stream']>=0 and row['partial_forward_cost_not_fully_measured'] is True,'Missing partial failure accounting')
    return dict(prior_updates=20+sum(row['successful_optimizer_updates'] for row in rows),
                partial_inputs_per_stream=sum(row['partial_inputs_loaded_per_stream'] for row in rows),prior_failed_probe_updates=4)


def inspect_ready(c,j,resume=False):
    verify(c);identity=contract_sha(c)
    require(not list(j.glob('qualification_failure*')) and not list(j.glob('resume_probe_*_failure*')),
            'Failed qualification cannot be admitted')
    q=read(j/'qualification.json')
    require(q['status']=='OWN_WINDOWS_QUALIFIED_FRESH_PROBES_PENDING' and q['contract_sha256']==identity,
            'Missing own qualification')
    require(q['actual_extra_engineering_updates']==q['engineering_labeled_inputs']==q['engineering_unlabeled_inputs']==16 and q['actual_extra_engineering_iterations']==8
            and q['new_formal_updates']==q['new_test_evaluations']==0 and q['actual_disk_roundtrips'],
            'Engineering accounting differs')
    probes={}
    for window,slot in (('warmup',3),('sam',4)):
        role='engineering_'+window
        source=q['references'][role]
        require(source['path']==str(j/('qualification_'+role)/'reference.json') and sha(source['path'])==source['sha256'],
                'Engineering reference changed')
        ref=read(source['path']);p=j/('resume_qualification_'+window+'.json');r=read(p)
        require(ref['execution_role']==r['role']==role and ref['contract_sha256']==r['contract_sha256']==identity,
                'Engineering role/contract mismatch')
        cp=ref['checkpoint']
        require(cp['path']==rows(c)[slot]['path'] and sha(cp['path'])==cp['sha256']==r['checkpoint_sha256']
                and Path(cp['path']).stat().st_size==cp['bytes']<=RECOVERY_BOUND, 'Engineering checkpoint differs')
        require(r['status']=='PASS' and all(r[k] for k in ('fresh_process','state_restoration_bitwise_equal',
                'indices_equal','losses_close','rng_sequence_equal')), 'Fresh-process continuation not proven')
        require(r['numeric_tolerance']==dict(rtol=1e-4,atol=1e-5) and ref['expected_counts']==counts(4),
                'Tolerance/counts changed')
        require(r['actual_extra_engineering_updates']==r['engineering_labeled_inputs']==r['engineering_unlabeled_inputs']==4 and r['actual_extra_engineering_iterations']==2
                and r['new_formal_updates']==r['new_test_evaluations']==0, 'Probe accounting changed')
        require(r['pid']!=q['pid'] and r['pid']>0,'Probe not a separate process')
        start=0 if window=='warmup' else c['recipe']['original_sam_start_index']
        require(len(ref['steps'])==2 and [x['completed_primary_pairs'] for x in ref['steps']]==[2*(start+3),2*(start+4)],
                'Wrong engineering clock window')
        if window=='sam':require(ref['active_sam_actual_gradient_verified'] is True,'SAM branch not exercised')
        probes[window]=dict(path=str(p),sha256=sha(p),max_abs_loss_difference=r['max_abs_loss_difference'],
                           continuation_state_bitwise_equal=r['continuation_state_bitwise_equal'])
    for name in ('qualify.log','probe_warmup.log','probe_sam.log'):
        require((j/name).is_file(),'Missing engineering log: '+name);check_logs((j/name).read_bytes())
    require(read(j/'resume_qualification_warmup.json')['pid']!=read(j/'resume_qualification_sam.json')['pid'],
            'Both probes require their own fresh process')
    if resume:
        from resume_review import validate_review
        validate_review(c,j)
    else:
        require(not Path(c['output']).exists() and not list(j.glob('dispatch*.json'))
                and not list(j.glob('launch_intent*.json')) and not list(j.glob('worker_error*.json')),
                'Formal lineage already exists')
    registered=read(REGISTRY)
    for row in rows(c):require([x for x in registered if x['path']==row['path']]==[row],'Missing reserved growth')
    with leased(c):cap=capacity(c,True)
    history=engineering_history(c);prior=history['prior_updates'];partial=history['partial_inputs_per_stream']
    return dict(status='READY_INDEPENDENTLY_VERIFIED',contract_sha256=identity,
        qualification_sha256=sha(j/'qualification.json'),probes=probes,total_engineering_iterations=12+prior//2,total_engineering_updates=24+prior,
        successful_qualification_and_probe_updates=24,preserved_failed_engineering_updates=4 if prior else 0,
        preserved_prior_engineering_updates=prior,numerical_runtime_policy=c['numerical_runtime_policy'],
        engineering_labeled_inputs=24+prior+partial,engineering_unlabeled_inputs=24+prior+partial,
        successful_engineering_inputs_per_stream=24+prior,preserved_partial_inputs_per_stream=partial,
        failed_partial_forward_compute_not_fully_measured=bool(partial),new_formal_updates=0,
        new_test_evaluations=0,new_model_loads=0,role_separation_verified=True,capacity=cap,at=time.time())
