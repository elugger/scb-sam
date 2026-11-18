"""BY10 HPQ contract: same scientific classes, isolated initialization and evidence."""
import copy,json
from pathlib import Path
from by_protocol import ROOT as ROOT_P, SRC as SRC_P, BASE as BASE_P, STORE as STORE_P, require,sha,verify_bundle
from prepare_by5 import dump_new,stamp
from by_resources import original_lock,capacity_snapshot,infra
from step_gates import check_step
ROOT,SRC,BASE,STORE=map(Path,(ROOT_P,SRC_P,BASE_P,STORE_P))
CONTROL=BASE/'control'
GPU='__RECORDED_GPU_UUID__'
GROUPS={'HPQ':('A1B','B')}
ACCEPT=CONTROL/'initializer_acceptance.json'
INIT_SHA=json.loads(ACCEPT.read_text())['initializer']['sha256'] if ACCEPT.exists() else None

def group_config(ancestor,group,phase):
    require(group=='HPQ' and phase in ('initial','bridge','formal'),'Unexpected phase')
    cfg=copy.deepcopy(ancestor)
    cfg['experiment'].update(output_dir=str(BASE/phase/group),gpu=GPU,seed=2027)
    cfg['dataset']['split_root']='expes/splits/by_semi10_cv5'
    cfg['semi'].update(paired_checkpoint_dir=str(STORE/group if phase=='formal' else BASE/phase/group/'no_weights'),paired_workers=0,matrix_group='A1B',matrix_c='B',ucp_epochs=38)
    require(cfg['semi']['ucp_accumulation']==5,'Accumulation changed')
    return cfg

def verify_prepared(bundle_sha,prepare_sha):
    verify_bundle(bundle_sha)
    require(sha(CONTROL/'hpq_preparation.json')==prepare_sha,'Preparation changed')
    prep=json.loads((CONTROL/'hpq_preparation.json').read_text())
    require(prep['bundle_sha256']==bundle_sha,'Bundle mismatch')
    for p,h in prep['source_hashes'].items(): require(sha(p)==h,'Source changed: '+p)
    from by_inputs import identity
    for p,row in prep['inputs'].items():
        require(identity(p,ROOT/'data')==row['identity'] and sha(p)==row['sha256'],'Input changed: '+p)
    cfg=json.loads((CONTROL/'supervised_config.json').read_text())
    require(sha(CONTROL/'supervised_config.json')==prep['prefix_config_sha256'],'Config changed')
    accepted=json.loads(ACCEPT.read_text())
    require(sha(ACCEPT)==prep['acceptance_sha256'] and sha(STORE/'supervised/final.pth')==INIT_SHA,'Initializer changed')
    require(accepted['successful_updates']==600 and accepted['strict_reload'],'Initializer incomplete')
    initial_path=BASE/'initial/HPQ/result.json'
    initial={'results':{'HPQ':json.loads(initial_path.read_text())}} if initial_path.exists() else {'results':{}}
    return cfg,prep['ancestor'],accepted,initial,prep

def validate_bridge(bundle_sha,prepare_sha):
    verify_prepared(bundle_sha,prepare_sha)
    d=BASE/'bridge/HPQ';row=json.loads((d/'result.json').read_text())
    require(row['status']=='PASS_FINITE_BRIDGE' and row['successful_updates']==row['optimizer_successful_calls']==2 and row['attempted_updates']==3,'Bridge incomplete')
    require(row['initializer_sha256']==INIT_SHA and row['overflow_model_optimizer_unchanged'] and row['strict_memory_roundtrip'],'Bridge invariant failed')
    initial=json.loads((BASE/'initial/HPQ/result.json').read_text())
    require(row['initial_roots']==initial['root_sha256'],'Bridge initial values differ')
    for n,h in row['evidence_sha256'].items():require(sha(d/n)==h,'Bridge evidence changed')
    steps=[json.loads(s) for s in (d/'steps.jsonl').read_text().splitlines()]
    require(len(steps)==3,'Bridge step count')
    for i,s in enumerate(steps):check_step(s,'HPQ',overflow=i==2)
    return {'sha256':sha(d/'result.json')}
