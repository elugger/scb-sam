"""Freeze one fresh basic-supervised protocol after CPU-only checks; never launch."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path('__RECORDED_PROJECT_ROOT__')
HERE = ROOT/'expes/semi05_gate/public_sup_20261018'
BASE = ROOT/'expes/work_dir/semi05_gate/PUBLIC_BASIC_SUP_FOLD2_20261018'
STORE = ROOT/'expes/artifacts/FCD_A_MATRIX_20261006/PUBLIC_BASIC_SUP_FOLD2_20261018'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def write_new(p, d):
    with p.open('x') as f:
        json.dump(d, f, indent=2, sort_keys=True)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def main():
    assert Path(sys.executable).resolve() == Path('__RECORDED_PYTHON__').resolve()
    assert not (HERE/'preparation_ready.json').exists() and not BASE.exists() and not STORE.exists(), 'Existing preparation or run; inspect first'
    import train_public_basic_sup_20261018 as runner
    import public_basic_sup_guard_20261018 as guard
    test = subprocess.run([sys.executable, str(HERE/'test_public_basic_sup_20261018.py')],
                           cwd=HERE, capture_output=True, text=True)
    assert test.returncode == 0, test.stdout+test.stderr
    for name in ['segment_anything.auto_sam_seg', 'segment_anything.build_sam3D_cascade',
                 'segment_anything.modeling.sam3D_cascade', 'utils.auto_sam_loss',
                 'utils.cascade_train_utils', 'expes.common.semi_data', 'expes.common.seg_metrics']:
        importlib.import_module(name)
    scientific_paths = set()
    for name, module in list(sys.modules.items()):
        file = getattr(module, '__file__', None)
        if file and (name.startswith('segment_anything') or name.startswith('utils.') or name.startswith('expes.common.')):
            p = Path(file).resolve()
            if ROOT in p.parents:
                scientific_paths.add(p)
    code_names = ['train_public_basic_sup_20261018.py', 'public_basic_sup_guard_20261018.py',
                  'launch_public_basic_sup_20261018.py', 'prepare_public_basic_sup_20261018.py',
                  'test_public_basic_sup_20261018.py']
    sources = {str(p.relative_to(ROOT)):sha(p) for p in sorted(scientific_paths | {HERE/n for n in code_names})}
    split05=ROOT/'expes/splits/public/public_semi05_cv5/fold2'
    split10=ROOT/'expes/splits/public/public_semi10_cv5/fold2'
    train_paths=[split05/'train_lab.txt', split10/'train_lab.txt', split05/'train_whole.txt']
    test_path=split05/'test.txt'
    ids=[runner.read_case_ids(p) for p in train_paths]
    tests=runner.read_case_ids(test_path)
    assert list(map(len,ids)) == [17,35,346] and len(tests)==87
    assert set(ids[0]) < set(ids[1]) < set(ids[2]) and not set(ids[2]) & set(tests)
    assert tests == runner.read_case_ids(split10/'test.txt')
    assert set(ids[2]) == set(runner.read_case_ids(split10/'train_whole.txt'))
    data=ROOT/'data/Public_433_Folds/fold_2'
    for split, cases in [('train',ids[2]),('test',tests)]:
        for case in cases:
            for sub in ('imagesTr','labelsTr'):
                assert (data/split/sub/(case+'.nii.gz')).is_file(), 'Missing image/label pair'
    inputs={str(p.relative_to(ROOT)):sha(p) for p in train_paths+[test_path, split10/'test.txt',split10/'train_whole.txt']}
    checkpoint=ROOT/'ckpt/sam_med3d_turbo.pth'
    checkpoint_sha=sha(checkpoint)
    assert checkpoint_sha=='899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c'
    assert len(os.sched_getaffinity(0))>=8, 'Insufficient CPU affinity for two four-thread tasks'
    with guard.original_lock():
        resource=guard.capacity_snapshot(14*guard.GIB+guard.MARGIN)
    configs={}
    for arm,percent,count,train_path in zip(['sup05','sup10','sup100'],[5,10,100],[17,35,346],train_paths):
        cfg={'project_root':str(ROOT), 'output_dir':str(BASE/arm), 'artifact_dir':str(STORE/arm),
             'fold_cv_dir':str(data), 'train_list':str(train_path.relative_to(ROOT)),
             'test_list':str(test_path.relative_to(ROOT)), 'sam_checkpoint':str(checkpoint.relative_to(ROOT)),
             'sam_checkpoint_sha256':checkpoint_sha, 'source_sha256':sources, 'input_sha256':inputs,
             'resource_guard_module':'public_basic_sup_guard_20261018', 'train_count':count, 'test_count':87,
             'label_percent':percent, 'fold':2, 'seed':2027, 'epochs':60, 'batch_size':1, 'num_workers':0,
             'test_policy':'fixed_last_epoch_once', 'mode':'sup', 'use_hpq':False, 'use_unlabeled':False,
             'resume':False, 'amp_init_scale':1024.0,
             'fold_selection':'development_fold_chosen_from_seen_historical_test_centrality',
             'equal_budget_definition':'equal_sixty_exposures_per_labeled_case_not_equal_total_updates',
             'data_source_identity_formally_verified':False, 'patient_alias_boundary_verified':False,
             'gt_roi_used':True}
        runner.validate_config(cfg)
        runner.verify_manifests(cfg)
        path=HERE/(arm+'.json'); write_new(path,cfg); configs[path.name]=sha(path)
    (BASE/'control').mkdir(parents=True, exist_ok=False)
    receipt={'created_at':datetime.now(timezone.utc).isoformat(), 'status':'CPU_READY_NOT_LAUNCHED',
             'source_sha256':sources, 'config_sha256':configs, 'input_sha256':inputs,
             'cpu_test_returncode':test.returncode,'cpu_test_output':test.stdout+test.stderr,
             'resource_snapshot':resource, 'sam_checkpoint_sha256':checkpoint_sha,
             'training_or_model_forward_performed':False,'test_evaluations':0,
             'cleanup_status':'BLOCKED_NO_DELETIONS_UNINSPECTABLE_PROCESS_HANDLES',
             'cleanup_inventory_sha256':'4474f8d80158f6d65e55b4ca1fd2f626daffe9828ba9b2a04364d0c93641d88e',
             'runtime':{'python':sys.version, 'torch':importlib.import_module('torch').__version__,
                        'cpu_affinity':len(os.sched_getaffinity(0))}}
    write_new(HERE/'preparation_ready.json',receipt)
    print(json.dumps({'status':receipt['status'],'sha256':sha(HERE/'preparation_ready.json'),
                      'source_count':len(sources),'cpu_test_summary':(test.stdout+test.stderr)[-160:]}))


if __name__=='__main__':
    main()
