"""Read-only stdlib AST/index/clock preparation, never GPU qualification."""
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
from dualteacher_budget import plan, counts, validate, validate_clock

ROOT = Path('__RECORDED_PROJECT_ROOT__')
BASE = ROOT/'expes/semi05_gate/public_ncase_baselines_20261024'
OLD = ROOT/'expes/semi05_gate/dualteacher_fixed_final_20261021_v4_conda'
CP = ROOT/'expes/work_dir/semi05_gate/MAIN_TABLE_FIXED_FINAL_20261019/dualteacher_sam/BY/semi05/fold2/r3/contract.json'
SPLIT_ANCHOR = ROOT/'expes/work_dir/semi05_gate/MAIN_TABLE_FIXED_FINAL_20261019/public_ncase_baselines_20261024/relif3d/Public433/lab005/fold2/r2/contract.json'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
stable = lambda v: hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main(outpath):
    assert sha(CP) == 'b82d2a57777af9f97925768590eebca718587e44ae40ba073b8751cc41a1a04f'
    origin = json.loads(CP.read_bytes())
    assert origin['method']=='dualteacher_sam' and origin['dataset']=='BY' and origin['seed']==2027
    pinned = []
    for row in origin['sources']:
        if str(OLD)+'/' in row['path']:
            p=Path(row['path']); assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
            pinned.append(row)
    assert sha(SPLIT_ANCHOR)=='9d7262de5d2fbf5437526a31c2e3d36697237edc3eb5e244b5fcc92710ac5b12'
    anchor=json.loads(SPLIT_ANCHOR.read_bytes()); split_files={}; lists={}
    for row in anchor['inputs']:
        p=Path(row['path']); assert sha(p)==row['sha256']; ids=p.read_text().split()
        assert len(ids)==len(set(ids)); lists[p.stem]=ids; split_files[p.name]=row
    train=set(lists['train_whole']); test=set(lists['test'])
    assert len(train)==346 and len(test)==87 and not train & test
    assert set(lists['n005_lab']) < set(lists['n010_lab']) <= set(lists['original_lab17']) <= train
    budget_file=BASE/'public_relif_budget11_v3/budget11.json'
    assert sha(budget_file)=='9c79216dafe33283c4e49bd64c40e4dbe9c9ff92cff8f0a3b3af02c6c844677b'
    frozen=json.loads(budget_file.read_bytes())
    env=dict(__name__=__name__,dataclass=dataclasses.dataclass,random=random,math=math,
             NATIVE_EPOCHS=200,NATIVE_ITERS_PER_EPOCH=200,need=lambda ok,msg:None if ok else (_ for _ in ()).throw(ValueError(msg)))
    for file,kind,name in [('budget_protocol.py',ast.ClassDef,'Budget'),('runtime_data.py',ast.FunctionDef,'schedule'),('dualteacher_core.py',ast.FunctionDef,'clock')]:
        tree=ast.parse((OLD/file).read_text()); nodes=[n for n in tree.body if isinstance(n,kind) and n.name==name]
        assert len(nodes)==1; exec(compile(ast.Module(body=nodes,type_ignores=[]),str(OLD/file),'exec'),env)
    coretree=ast.parse((OLD/'dualteacher_core.py').read_text())
    core=next(n for n in coretree.body if isinstance(n,ast.ClassDef) and n.name=='DualteacherCore')
    update=next(n for n in core.body if isinstance(n,ast.FunctionDef) and n.name=='update')
    native_clock=[n for n in ast.walk(update) if isinstance(n,ast.Call) and ast.unparse(n.func)=='clock']
    assert len(native_clock)==1 and [ast.unparse(a) for a in native_clock[0].args]==['self.step_count','self.total']
    calls=[n for n in ast.walk(update) if isinstance(n,ast.Call) and ast.unparse(n.func)=='self.step_function']
    assert len(calls)==1 and ast.unparse(calls[0].args[13])=="c['native_epoch']"
    checks=[]; records=[]
    def reject(name, action):
        try: action()
        except (ValueError,KeyError,TypeError): checks.append(name); return
        raise AssertionError('Expected rejection: '+name)
    for n in (5,10):
        lab=lists[f'n{n:03d}_lab']; unlab=lists[f'n{n:03d}_unlab']
        assert len(lab)==n and len(unlab)==346-n and not set(lab)&set(unlab) and set(lab)|set(unlab)==train
        splits=dict(train_lab=lab,train_unlab=unlab,test=lists['test'])
        old=env['Budget']('Public433',n,n,346-n,cycles=36); short=env['Budget']('Public433',n,n,346-n,cycles=11)
        recipe=plan(n); pairs=env['schedule'](old,splits); target=env['schedule'](short,splits)
        mapping=next(m for r in frozen['rows'] if r['labeled_cases']==n for m in r['method_mapping'] if m['method']=='dualteacher_sam')
        assert recipe['ssl_iterations']==mapping['native_ssl_iterations'] and recipe['primary_inputs_per_stream']==mapping['actual_primary_inputs_per_stream']
        assert recipe['overshoot_per_stream']==mapping['overshoot_per_stream'] and recipe['original_clock_iterations']==old.primary_pairs//2
        assert pairs[:len(target)]==target
        actual=pairs[:recipe['primary_inputs_per_stream']]
        assert len(actual)==2*recipe['ssl_iterations'] and not test & {p[k] for p in actual for k in ('labeled','unlabeled')}
        for cycle in range(11):
            chunk=actual[cycle*short.primary_pairs_per_cycle:(cycle+1)*short.primary_pairs_per_cycle]
            assert set(p['labeled'] for p in chunk)==set(lab) and set(p['unlabeled'] for p in chunk)==set(unlab)
            assert sum(p['unlabeled_replay'] for p in chunk)==short.replay_per_cycle
        clocks=[env['clock'](i,old.primary_pairs//2) for i in range(recipe['ssl_iterations'])]
        for i,c in enumerate(clocks): validate_clock(c,i,recipe)
        active=next(i for i,c in enumerate(clocks) if c['sam_enabled'])
        assert active==recipe['original_sam_start_index'] and not clocks[active-1]['sam_enabled']
        assert [len(actual[i:i+2]) for i in range(0,len(actual),2)]==[2]*recipe['ssl_iterations']
        for i in (0,active-1,active,recipe['ssl_iterations']-1):
            reject(f'{n}_compressed_clock_{i}',lambda i=i:validate_clock(env['clock'](i,recipe['ssl_iterations']),i,recipe))
        validate(counts(recipe['ssl_iterations']),recipe); checks.append(f'{n}_all_original_indices_clocks_and_terminal_counts')
        for key in counts(1):
            bad=counts(1); bad[key]=True
            reject(f'{n}_boolean_{key}',lambda bad=bad:validate(bad,recipe))
        bad=counts(1);bad['successful_optimizer_updates']=3
        reject(f'{n}_step_calls_are_not_effective_updates',lambda:validate(bad,recipe))
        reject(f'{n}_overrun',lambda:validate(counts(recipe['ssl_iterations']+1),recipe))
        altered=copy.deepcopy(recipe);altered['original_clock_iterations']=recipe['ssl_iterations']
        reject(f'{n}_recipe_clock_compression',lambda:validate(counts(0),altered))
        records.append(dict(recipe=recipe,actual_train_cases=len({p[k] for p in actual for k in ('labeled','unlabeled')}),
            actual_primary_pair_sha256=stable(actual),native_clock_sha256=stable(clocks),first_clock=clocks[0],last_clock=clocks[-1],
            full_reference_cycles_verified=11,extra_pairs=actual[len(target):],native_sam_start=active,
            compressed_sam_start=math.ceil(6*recipe['ssl_iterations']/200),no_test_indices=True,native_batch_unchanged=True))
    for n in (True,0,17,35):reject('invalid_labels_'+str(n),lambda n=n:plan(n))
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    result=dict(status='DUALTEACHER_PUBLIC_BUDGET_INDEX_CLOCK_CPU_PASS_NOT_GPU_QUALIFIED',at=time.time(),passed=True,
        origin_contract=dict(path=str(CP),sha256=sha(CP)),native_source_bindings=pinned,
        split_anchor=dict(path=str(SPLIT_ANCHOR),sha256=sha(SPLIT_ANCHOR),usage='case-list pins only; not borrowed qualification'),
        split_files=split_files,budget=dict(path=str(budget_file),sha256=sha(budget_file)),records=records,
        checks=checks,check_count=len(checks),source={p.name:dict(path=str(p),sha256=sha(p)) for p in Path(__file__).parent.glob('*.py')},
        new_model_loads=0,new_cuda_loads=0,new_patient_inputs=0,new_optimizer_updates=0,new_test_evaluations=0,
        new_job_created=False,new_targets_enrolled=False,
        remaining=['isolated constructor BY-only guard adaptation; preserve native update AST',
        'complete student EMA OP criterion plus dual optimizer EMA-step thresholds modes flags gradients RNG sampling recovery',
        'frozen Turbo full signature and original SHA strict reload',
        'bounded dual storage and history/tail review; operator admission live/final observer',
        'own actual warmup/SAM-active GPU windows and fresh-process continuation probes after ReLiF acceptance'])
    with Path(outpath).open('x') as f: json.dump(result,f,sort_keys=True,indent=2)
    print(json.dumps(dict(status=result['status'],checks=len(checks),records=records,sha256=sha(outpath))))


if __name__=='__main__': main(sys.argv[1])
