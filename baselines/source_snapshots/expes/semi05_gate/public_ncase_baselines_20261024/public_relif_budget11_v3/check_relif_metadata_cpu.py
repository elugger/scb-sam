"""Bounded stdlib-only actual-index/clock and synthetic recovery metadata checks."""
import ast
import copy
import dataclasses
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from relif_state import FORMAT, counts, plan, role_window, validate, accept, validate_payload

ROOT=Path('__RECORDED_PROJECT_ROOT__')
BASE=ROOT/'expes/semi05_gate/public_ncase_baselines_20261024'
OLD=ROOT/'expes/semi05_gate/main_table_20261019_r4'
CP=ROOT/'expes/work_dir/semi05_gate/MAIN_TABLE_FIXED_FINAL_20261019/relif3d/BY/semi05/fold2/r1/contract.json'
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
stable=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main(outpath):
    c=json.loads(CP.read_text())
    assert c['method']=='relif3d' and c['dataset']=='BY' and c['seed']==2027 and c['fold']==2
    names=('budget_protocol.py','runtime_data.py','relif_runtime.py','relif_components.py',
           'relif_native.py','relif_backend.py','relif_official_pin.json')
    pins=[]
    for name in names:
        p=OLD/name; row=next(x for x in c['sources'] if x['path']==str(p))
        assert sha(p)==row['sha256']; pins.append(row)
    official=c['relif_recipe']['official_code_root']
    pin=json.loads((OLD/'relif_official_pin.json').read_text())
    assert pin['commit']=='bbda984aa337520c009091d4a02d2e104bb176d6'
    for name,row in pin['files'].items():
        p=Path(official)/name
        assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
    budget_file=BASE/'public_sfr_budget11_v1/budget11.json'
    assert sha(budget_file)=='9c79216dafe33283c4e49bd64c40e4dbe9c9ff92cff8f0a3b3af02c6c844677b'
    budget=json.loads(budget_file.read_text())
    env=dict(__name__='__main__',dataclass=dataclasses.dataclass,random=random,math=math,NATIVE_ITERATIONS=30000)
    for path,kind,names_ in [(OLD/'budget_protocol.py',ast.ClassDef,{'Budget'}),
                             (OLD/'runtime_data.py',ast.FunctionDef,{'schedule'}),
                             (OLD/'relif_runtime.py',ast.FunctionDef,{'clock'})]:
        tree=ast.parse(path.read_text()); nodes=[n for n in tree.body if isinstance(n,kind) and n.name in names_]
        assert len(nodes)==len(names_)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    tree=ast.parse((OLD/'relif_runtime.py').read_text())
    core=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ReLiFCore')
    update=next(n for n in core.body if isinstance(n,ast.FunctionDef) and n.name=='update')
    calls=[n for n in ast.walk(update) if isinstance(n,ast.Call)]
    assert any(ast.unparse(n.func)=='clock' and [ast.unparse(a) for a in n.args]==['self.step_count','self.total'] for n in calls)
    sobf=[n for n in calls if ast.unparse(n.func)=="self.native['sobf_augment']"]
    assert len(sobf)==1 and [ast.unparse(a) for a in sobf[0].args]==['images','self.step_count','self.total']
    assert [k.arg for k in sobf[0].keywords]==['k_mul','k_add','a_mul_max','a_add_max']
    sp=ROOT/'expes/semi05_gate/public_ncase_20261019'
    sys.path.insert(0,str(sp))
    from ncase_protocol import validate_splits
    splits=validate_splits(sp/'splits'); records=[]; checks=[]
    def ok(name, action):
        action(); checks.append(name)
    def reject(name, action):
        try: action()
        except (ValueError,KeyError,TypeError): checks.append(name);return
        raise AssertionError('Expected rejection: '+name)
    def require(v):
        assert v
    def mutate(v,key,value):
        x=copy.deepcopy(v);cur=x
        for k in key[:-1]:cur=cur[k]
        cur[key[-1]]=value
        return x
    for n in (5,10):
        s=splits[n];split=dict(train_lab=s['lab'],train_unlab=s['unlab'],test=s['test'])
        old=env['Budget']('Public433',n,n,346-n,cycles=36)
        cutoff=env['Budget']('Public433',n,n,346-n,cycles=11)
        r=plan(n); pairs=env['schedule'](old,split); short=env['schedule'](cutoff,split)
        mapping=next(x for row in budget['rows'] if row['labeled_cases']==n for x in row['method_mapping'] if x['method']=='relif3d')
        assert r['ssl_iterations']==cutoff.primary_pairs==mapping['native_ssl_iterations']
        assert mapping['actual_primary_inputs_per_stream']==cutoff.primary_pairs and mapping['overshoot_per_stream']==0
        assert pairs[:cutoff.primary_pairs]==short and r['original_clock_iterations']==old.primary_pairs
        for i in range(11):
            cycle=short[i*cutoff.primary_pairs_per_cycle:(i+1)*cutoff.primary_pairs_per_cycle]
            assert set(p['labeled'] for p in cycle)==set(s['lab']) and set(p['unlabeled'] for p in cycle)==set(s['unlab'])
            assert sum(p['unlabeled_replay'] for p in cycle)==cutoff.replay_per_cycle
        assert not set(s['test']) & {p[k] for p in short for k in ('labeled','unlabeled')}
        clocks=[env['clock'](i,old.primary_pairs) for i in range(cutoff.primary_pairs)]
        compressed=[env['clock'](i,cutoff.primary_pairs) for i in range(cutoff.primary_pairs)]
        assert clocks!=compressed
        active=next(i for i,v in enumerate(clocks) if v['larc_enabled'])
        assert active==r['original_larc_start_index']
        assert clocks[active-1]['larc_enabled'] is False and clocks[active]['larc_enabled'] is True
        assert clocks[r['original_larc_full_weight_index']]['larc_weight']==.08
        records.append(dict(labeled_cases=n,unlabeled_cases=346-n,ssl_iterations=cutoff.primary_pairs,
             primary_inputs_per_stream=cutoff.primary_pairs,successful_optimizer_updates=cutoff.primary_pairs,
             prefix_updates=0,primary_pairs_per_cycle=cutoff.primary_pairs_per_cycle,cycles=11,actual_train_cases=346,
             original_clock_iterations=old.primary_pairs,original_schedule_prefix_exact=True,pair_prefix_sha256=stable(short),
             native_clock_sha256=stable(clocks),first_clock=clocks[0],last_clock=clocks[-1],
             original_larc_start_index=active,compressed_larc_start_index=next(i for i,v in enumerate(compressed) if v['larc_enabled']),
             original_larc_full_weight_index=r['original_larc_full_weight_index'],compressed_clock_rejected=True,
             original_sobf_total_argument_retained=True,all_11_actual_index_cycles_checked=True,
             teacher_case_forwards=0,student_case_forwards=4*cutoff.primary_pairs,foundation_case_forwards=2*cutoff.primary_pairs,
             sobf_views=2*cutoff.primary_pairs,extra_donor_inputs=0))
        for role in ('formal','engineering_warmup','engineering_larc'):
            start,stop=role_window(r,role); i=start;pair=pairs[i];clock=env['clock'](i,old.primary_pairs)
            native=dict(step=i+1,clock=clock,primary_labeled=1,primary_unlabeled=1,optimizer_updates=1,
                student_case_forwards=4,foundation_case_forwards=2,sobf_views=2,teacher_case_forwards=0,
                extra_donor_inputs=0,test_evaluations=0,unlab_gt_in_loss=False,case_ids=dict(labeled=pair['labeled'],unlabeled=pair['unlabeled']))
            row=dict(actual_pair=pair,completed_primary_pairs=i+1,labeled_inputs=1,unlabeled_inputs=1,
                successful_optimizer_updates=1,finite_loss=True,finite_nonzero_gradients=True,actual_case_ids_verified=True,
                teacher_no_grad=True,skip_count=0,unlabeled_gt_in_loss=False,native_receipt=native,gradient_norm=1.,
                losses={k:1. for k in ('loss','supervised_ce','supervised_dice','distillation','cgcr','larc')})
            tag=str(n)+'_'+role
            ok(tag+'_one_actual_update',lambda:require(accept(counts(0),row,r,role,pair,clock)==counts(1)))
            reject(tag+'_extra_update_rejected',lambda:accept(counts(0),mutate(row,['successful_optimizer_updates'],2),r,role,pair,clock))
            reject(tag+'_wrong_index_rejected',lambda:accept(counts(0),mutate(row,['actual_pair','unlabeled'],'other'),r,role,pair,clock))
            reject(tag+'_compressed_clock_rejected',lambda:accept(counts(0),mutate(row,['native_receipt','clock'],compressed[i]),r,role,pair,clock))
            reject(tag+'_wrong_input_rejected',lambda:accept(counts(0),mutate(row,['unlabeled_inputs'],2),r,role,pair,clock))
            state=dict(format=FORMAT,contract_sha256='a'*64,recipe=r,execution_role=role,
                counts=counts(1),position=i+1,core_state=dict(student={'synthetic':1},projection={'synthetic':2},optimizer={'synthetic':3},
                step=i+1,total=old.primary_pairs,shape=(128,128,128),implementation=c['relif_recipe']['implementation']),
                actual_rows=pairs[i:i+1],cycles_completed=0,
                module_modes=dict(student={'':True},projection={'':True},prior={'':False}),
                requires_grad=dict(student={'p':True},projection={'p':True},prior={'p':False}),
                foundation=dict(variant='vit_b_ori',model_family='SAM-Med3D_original'),frozen_prior_signature='b'*64,rng=[0,0,0,0])
            ok(tag+'_synthetic_payload_metadata',lambda:validate_payload(state,'a'*64,r,role,pairs))
            for name,key,value in [('cursor_reset',['position'],0),('clock_shortened',['core_state','total'],cutoff.primary_pairs),
                ('row_substitution',['actual_rows'],[]),('missing_optimizer',['core_state'],{}),
                ('prior_train',['module_modes','prior',''],True),('prior_grad',['requires_grad','prior','p'],True),
                ('rng_missing',['rng'],[0,0,0]),('contract',['contract_sha256'],'c'*64)]:
                reject(tag+'_'+name,lambda key=key,value=value:validate_payload(mutate(state,key,value),'a'*64,r,role,pairs))
            other='formal' if role!='formal' else 'engineering_warmup'
            reject(tag+'_role_rejected',lambda:validate_payload(state,'a'*64,r,other,pairs))
        reject(str(n)+'_boolean_count',lambda:validate(dict(iterations=True,labeled_inputs=1,unlabeled_inputs=1,successful_optimizer_updates=1),r))
        reject(str(n)+'_overrun',lambda:validate(counts(cutoff.primary_pairs+1),r))
    for bad in (True,17,35,0):reject('labels_'+str(bad),lambda bad=bad:plan(bad))
    assert 'torch' not in sys.modules and not any(n.startswith('torch.') for n in sys.modules)
    source={p.name:dict(path=str(p),sha256=sha(p)) for p in HERE.glob('*') if p.is_file()}
    result=dict(status='RELIF_INDEX_CLOCK_AND_SYNTHETIC_RECOVERY_METADATA_PASS_NOT_GPU_QUALIFIED',passed=True,at=time.time(),
       native_sources=pins,origin_contract=dict(path=str(CP),sha256=sha(CP)),official_commit=pin['commit'],
       budget=dict(path=str(budget_file),sha256=sha(budget_file)),
       split_protocol=dict(path=str(sp/'ncase_protocol.py'),sha256=sha(sp/'ncase_protocol.py')),
       split_files={str(p):sha(p) for p in sorted((sp/'splits').glob('*.txt'))},records=records,
       synthetic_metadata_check_count=len(checks),synthetic_metadata_checks=checks,source=source,
       snapshot_tensor_roundtrip_tested=False,new_model_loads=0,new_cuda_loads=0,new_patient_inputs=0,
       new_optimizer_updates=0,new_test_evaluations=0,new_job_created=False,new_targets_enrolled=False,
       remaining=['native backend and strict cutoff controller integration','bounded storage and reviewed resume',
                  'explicit operator and independent admission/observer','own warmup and active LARC GPU qualification and fresh process recovery probes'])
    with Path(outpath).open('x') as f:json.dump(result,f,indent=2,sort_keys=True)
    print(json.dumps(dict(status=result['status'],checks=len(checks),records=records,sha256=sha(outpath))))


if __name__=='__main__':
    main(sys.argv[1])
