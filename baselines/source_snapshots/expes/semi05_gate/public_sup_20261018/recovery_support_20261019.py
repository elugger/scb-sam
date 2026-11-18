"""Explicitly authorized non-bitwise sup100 continuation; no automatic retry."""
import json
from pathlib import Path
import random
ROOT = Path('__RECORDED_PROJECT_ROOT__')
BASE = ROOT/'expes/work_dir/semi05_gate/PUBLIC_BASIC_SUP_FOLD2_20261018'
STORE = ROOT/'expes/artifacts/FCD_A_MATRIX_20261006/PUBLIC_BASIC_SUP_FOLD2_20261018'
ARM = 'sup100_recovery_r1'
CKPT = STORE/'sup100/latest.pth'
CKPT_SHA = 'e96bb38fec82c83b1a5e17ce722d21e0bedea084dc4f0218d71919cc94ad51be'
LEDGER_SHA = 'dc4b8b5a249f374da82118b2120de588eb916c5d69133cc717ffdf0d8e740bbf'
def require(ok, message):
    if not ok:
        raise RuntimeError(message)
def validate(cfg):
    from expes.semi05_gate.public_sup_20261018 import train_public_basic_sup_20261018 as old
    original = json.loads((BASE/'sup100/config.json').read_text())
    old.validate_config(original)
    permitted = {'output_dir','artifact_dir','resource_guard_module','source_sha256','recovery'}
    require(set(cfg) == set(original) | {'recovery'}, 'Unexpected recovery config keys')
    for k in set(original)-permitted:
        require(cfg[k] == original[k], 'Scientific/input configuration changed: '+k)
    require(cfg['label_percent'] == 100, 'Only sup100 authorized')
    require(cfg['output_dir'] == str(BASE/ARM) and cfg['artifact_dir'] == str(STORE/ARM), 'Wrong new branch')
    require(cfg['resource_guard_module'] == 'public_recovery_guard_20261019', 'Wrong joint guard')
    require(all(cfg['source_sha256'].get(k) == v for k,v in original['source_sha256'].items()), 'Old source bindings changed')
    require(cfg['recovery'] == {'boundary_updates':14878,'boundary_epochs':43,'discarded_original_suffix_updates':52,
        'continuation_replay_verified':False,'authorized_non_bitwise':True,'amp_fixed_scale':1024.0,
        'checkpoint_sha256':CKPT_SHA,'original_ledger_sha256':LEDGER_SHA}, 'Recovery policy changed')
    return original
def validate_boundary(cp, original, checked, config_sha):
    require(cp['successful_updates']==14878 and cp['completed_epochs']==43 and cp['next_epoch']==43
            and cp['test_evaluations']==0, 'Wrong checkpoint boundary')
    require(cp['config']==original and cp['config_sha256']==config_sha and cp['source_manifest']==checked,
            'Original checkpoint provenance mismatch')
def restore(cfg, student, optimizer, scaler, generator, torch, np, guard, output, api):
    from expes.semi05_gate.public_sup_20261018 import train_public_basic_sup_20261018 as old
    from expes.semi05_gate.public_sup_20261018.diagnose_sup100_replay_20261018 import exact_tree_hash
    original=validate(cfg)
    checked=old.verify_manifests(original)
    require(api.sha256_file(CKPT)==CKPT_SHA, 'Original checkpoint changed')
    source=BASE/'sup100'
    require(api.sha256_file(source/'training_steps.jsonl')==LEDGER_SHA, 'Original ledger changed')
    require(json.loads((source/'failure.json').read_text())['successful_updates']==14930, 'Failure boundary changed')
    require(not (source/'test_evaluation_started.json').exists() and not (source/'final_metrics.json').exists(),
            'Original test already started')
    with guard.original_lock():
        capacity=guard.capacity_snapshot(required_ram=14*guard.GIB+guard.MARGIN)
    cp=api.load_trusted_checkpoint(CKPT, map_location='cpu')
    validate_boundary(cp, original, checked, api.sha256_file(ROOT/'expes/semi05_gate/public_sup_20261018/sup100.json'))
    model_hash=api.state_dict_sha256(cp['student'])
    optimizer_hash=exact_tree_hash(cp['optimizer'])
    student.load_state_dict(cp['student'], strict=True)
    optimizer.load_state_dict(cp['optimizer'])
    require(api.state_dict_sha256(student.state_dict())==model_hash, 'Restored model mismatch')
    require(exact_tree_hash(optimizer.state_dict())==optimizer_hash, 'Restored optimizer mismatch')
    scaler.load_state_dict(cp['scaler'])
    old_scaler=scaler.state_dict()
    new_scaler=dict(old_scaler, scale=1024.0, growth_interval=2147483647, _growth_tracker=0)
    scaler.load_state_dict(new_scaler)
    require(scaler.state_dict()==new_scaler, 'Numerical policy load mismatch')
    steps=(source/'training_steps.jsonl').read_text().splitlines()
    epochs=(source/'training_epochs.jsonl').read_text().splitlines()
    require(len(steps)==14930 and len(epochs)==43, 'Unexpected inherited record lengths')
    ids=set(api.read_case_ids(api.resolve_path(cfg,cfg['train_list'])))
    for i,line in enumerate(steps[:14878]):
        row=json.loads(line)
        require(row['successful_update']==i+1 and row['epoch']==i//346 and row['epoch_position']==i%346
                and row['optimizer_steps']==1 and not row['skipped'] and row['gradient_finite'], 'Inherited step invalid')
    for e in range(43):
        rows=[json.loads(x) for x in steps[e*346:(e+1)*346]]
        er=json.loads(epochs[e])
        require(len(rows)==346 and set(r['case_id'] for r in rows)==ids
                and er['actual_case_ids']==[r['case_id'] for r in rows] and er['successful_updates']==(e+1)*346,
                'Inherited case coverage invalid')
    for name,lines in [('training_steps.jsonl',steps[:14878]),('training_epochs.jsonl',epochs)]:
        with (output/name).open('x') as f:
            f.write('\n'.join(lines)+'\n')
    provenance={'time':api.now_iso(),'status':'RESTORED_NOT_BITWISE_REPLAY','checkpoint_sha256':CKPT_SHA,
        'model_state_sha256':model_hash,'optimizer_state_sha256':optimizer_hash,'original_ledger_sha256':LEDGER_SHA,
        'inherited_updates':14878,'inherited_epochs':43,'discarded_suffix_updates':52,'remaining_updates':5882,
        'original_scaler':old_scaler,'recovery_scaler':new_scaler,'capacity':capacity,
        'test_evaluations':0,'continuation_replay_verified':False,'user_authorized_non_bitwise':True}
    api.atomic_json(output/'recovery_restored.json',provenance)
    random.setstate(cp['rng']['python'])
    np.random.set_state(cp['rng']['numpy'])
    torch.set_rng_state(cp['rng']['torch_cpu'])
    torch.cuda.set_rng_state_all(cp['rng']['torch_cuda_all'])
    generator.set_state(cp['rng']['data_generator'])
    require(exact_tree_hash(api.capture_rng(torch,np,generator))==exact_tree_hash(cp['rng']), 'RNG restore mismatch')
    del cp
    return 14878,43
