"""Own binding plus mock backend sequencing. No native model/tensor execution.

Reuses fixture builders only from the previous check module, not its test run.
Does not prove real storage, tensor restoration, initialization or inference.
"""
import argparse,copy,hashlib,json,sys,tempfile,time
from pathlib import Path
from types import SimpleNamespace as NS
from dualteacher_binding import DualTeacherBinding,pinned_bytes
from dualteacher_budget import counts
from dualteacher_state import role_window
from resumable_dualteacher import make_backend
from check_dualteacher_recovery_cpu import receipt

checks=[]
def yes(name,condition):
    if not condition:raise AssertionError(name)
    checks.append(name)
def reject(name,fn):
    try:fn()
    except (ValueError,TypeError,KeyError,RuntimeError):checks.append(name);return
    raise AssertionError('Accepted invalid boundary: '+name)

class Model:
    def __init__(self,frozen=False):
        self.p=NS(requires_grad=not frozen,grad=None);self.training=True
    def parameters(self):return iter([self.p])
class Core:
    def __init__(self,total,start):
        self.model=Model();self.op=Model();self.bc=Model();self.sam=Model(True);self.sam.training=False
        self.ema=NS(model=Model(True),step=0);self.total=total;self.step_count=start
        self.optimizer=NS(param_groups=[dict(params=list(self.model.parameters()))])
        self.optimizer_op=NS(param_groups=[dict(params=list(self.op.parameters()))])

class Base:
    def __init__(self,c,out,verify):
        self.contract=c;self.output=Path(out);self.position=0;self.actual_pairs=[];self.native_rows=[]
        self.core=None;self.test_started=False;self.reloaded_sha=None
    def consume_batch(self,primary):
        self.events.append('native');self.native_calls+=1
        if self.inject_failure:raise RuntimeError('Synthetic partial native failure')
        index=primary//2;row=receipt(index,self.recipe,self.pairs)
        self.core.step_count+=1;self.core.ema.step+=1
        if self.inject_bad_receipt:row['native_receipt']['effective_op_updates']=0
        return row

class Protocol:
    def __init__(self,b):self.b=b;self.events=[];self.capacity_errors=[];self.records=[];self.append_errors=[]
    def verify(self,c):pass
    def binding(self,c):return self.b
    def contract_sha(self,c):return 'a'*64
    def output_check(self,*args):pass
    def capacity(self,c):
        self.events.append('capacity')
        if self.capacity_errors:raise ValueError(self.capacity_errors.pop(0))
        return dict(mock=True)
    def resource_status(self,c,v):self.records.append(copy.deepcopy(v))
    def append_step(self,c,out,row,role):
        self.events.append('append')
        if self.append_errors:raise ValueError(self.append_errors.pop(0))
        self.records.append(copy.deepcopy(row))
    def verify_actual_ledger(self,c,pairs,progress):self.events.append('ledger')

def backend(binding,role='formal'):
    p=Protocol(binding);C=make_backend(NS(DualteacherBackend=Base,signature=lambda m:'mock'),p)
    b=C({'target':'unused-mock'},'unused-mock',role);b.initialized=True
    b.core=Core(b.recipe['original_clock_iterations'],b.role_start);b.position=b.role_start
    b.events=p.events;b.native_calls=0;b.inject_failure=False;b.inject_bad_receipt=False
    return b,p

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--native-dir',required=True);ap.add_argument('--by-contract',required=True)
    ap.add_argument('--budget-file',required=True);ap.add_argument('--split-dir',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    args=dict(native_dir=a.native_dir,by_contract=a.by_contract,budget_file=a.budget_file,split_dir=a.split_dir)
    bindings=[DualTeacherBinding(n,**args) for n in (5,10)]
    for b in bindings:
        n=b.recipe['labeled_cases'];r=b.recipe
        yes(f'binding_{n}_own_native_closure',len(b.evidence()['native_source_bindings'])==6)
        yes(f'binding_{n}_no_admission_claim',not b.evidence()['gpu_qualified'] and not b.evidence()['patient_files_bound'])
        yes(f'binding_{n}_original_pair_sequence',len(b.pairs)==2*r['original_clock_iterations'])
        changed=b.pairs;changed[0]['labeled']='changed';yes(f'binding_{n}_defensive_pairs',b.pairs[0]['labeled']!='changed')
        changed=b.recipe;changed['ssl_iterations']=1;yes(f'binding_{n}_defensive_recipe',b.recipe==r)
        reject(f'binding_{n}_short_clock',lambda:b.require_core_total(r['ssl_iterations']))
        for role in ('formal','engineering_warmup','engineering_sam'):
            start,stop=role_window(r,role)
            yes(f'binding_{n}_{role}_actual_window',b.executed_prefix(start+2,role)==b.pairs[2*start:2*(start+2)])
            reject(f'binding_{n}_{role}_past_stop',lambda:b.pair_batch(stop,role))
            reject(f'binding_{n}_{role}_before_start',lambda:b.executed_prefix(start-1,role))
        x,p=backend(b,'engineering_sam');start=x.role_start;x.check_boundary()
        yes(f'backend_{n}_engineering_arming',x.core.step_count==start and x.core.ema.step==0 and x.progress==counts(0))
        row=x.train_iteration(start)
        yes(f'backend_{n}_gate_before_update',p.events==['capacity','native'])
        yes(f'backend_{n}_native_counters',x.progress==counts(1) and x.core.ema.step==1 and x.native_calls==1)
        reject(f'backend_{n}_nondurable_boundary',x.check_boundary)
        reject(f'backend_{n}_nondurable_next_step',lambda:x.train_iteration(start+1))
        reject(f'backend_{n}_replaced_receipt',lambda:x.append_step({}))
        x.append_step(copy.deepcopy(x.pending_row));x.check_boundary()
        yes(f'backend_{n}_append_then_boundary',p.events[-1]=='append' and x.pending_row is None and x.native_calls==1)
        reject(f'backend_{n}_old_prepare',x.prepare);reject(f'backend_{n}_old_qualify',x.qualify)
        reject(f'backend_{n}_engineering_final',lambda:x.verify_training_coverage())
        reject(f'backend_{n}_skipped_native_cursor',lambda:x.train_iteration(start+2))
    b=bindings[0]
    for failure in ('native','receipt'):
        x,p=backend(b);setattr(x,'inject_failure' if failure=='native' else 'inject_bad_receipt',True)
        reject('backend_failure_'+failure,lambda:x.train_iteration(0))
        yes('backend_failed_flag_'+failure,x.failed_update is True and x.native_calls==1)
        reject('backend_no_retry_'+failure,lambda:x.train_iteration(0));reject('backend_no_checkpoint_boundary_'+failure,x.check_boundary)
    x,p=backend(b);p.capacity_errors=['Unknown resource failure']
    reject('unknown_gate_failure',lambda:x.train_iteration(0));yes('no_native_after_gate_failure',x.native_calls==0 and x.progress==counts(0))
    x,p=backend(b);p.capacity_errors=['Insufficient filesystem capacity: mock'];reject('qualification_does_not_wait',lambda:x.train_iteration(0))
    yes('qualification_gate_no_update',x.native_calls==0)
    old_sleep=time.sleep;waits=[];time.sleep=lambda seconds:waits.append(seconds)
    try:
        x,p=backend(b);x.allow_wait=True;p.capacity_errors=['Insufficient RAM including cgroup ancestors mock']
        x.train_iteration(0);yes('formal_gate_wait_then_one_update',waits==[60] and x.native_calls==1)
        yes('formal_wait_status',p.records[0]['state']=='WAIT_CAPACITY' and p.records[1]['state']=='PASS')
        p.append_errors=['Insufficient filesystem capacity: mock'];x.append_step(copy.deepcopy(x.pending_row))
        yes('retry_append_without_duplicate_update',x.native_calls==1 and p.events.count('append')==2 and x.pending_row is None)
        p.capacity_errors=['unknown'];reject('formal_unknown_gate_not_waited',lambda:x.train_iteration(1))
        yes('formal_unknown_no_extra_update',x.native_calls==1 and waits==[60])
    finally:time.sleep=old_sleep
    # Terminal audit after strict reload must work for unchanged native test.
    x,p=backend(b);n=x.recipe['ssl_iterations'];x.position=x.core.step_count=x.core.ema.step=n;x.progress=counts(n)
    x.actual_pairs=b.pairs[:2*n];x.native_rows=[receipt(i,x.recipe,x.pairs)['native_receipt'] for i in range(n)]
    x.reloaded_sha='b'*64;v=x.verify_training_coverage()
    yes('post_strict_reload_readonly_coverage',v['complete_reference_cycles']==11 and v['primary_labeled_inputs']==3796 and v['successful_optimizer_updates']==3796)
    reject('post_reload_training_boundary_still_closed',x.check_boundary)
    x.test_started=True;reject('already_tested_terminal_closed',x.verify_training_coverage);x.test_started=False
    x.actual_pairs[-1]=copy.deepcopy(x.actual_pairs[-2]);reject('terminal_actual_pair_changed',x.verify_training_coverage)
    with tempfile.TemporaryDirectory(prefix='dual_binding_cpu_') as tmp:
        path=Path(tmp)/'payload';raw=b'bounded small synthetic source';path.write_bytes(raw);digest=hashlib.sha256(raw).hexdigest()
        yes('same_fd_source_read',pinned_bytes(path,digest,len(raw))==raw)
        reject('source_wrong_sha',lambda:pinned_bytes(path,'0'*64));reject('source_wrong_size',lambda:pinned_bytes(path,digest,1))
        if sys.platform!='win32':
            link=Path(tmp)/'link';link.symlink_to(path);reject('source_symlink',lambda:pinned_bytes(link,digest))
    yes('no_heavy_runtime_modules',all(n not in sys.modules for n in ('torch','numpy','SimpleITK')))
    source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    result=dict(status='PASS_OWN_BINDING_AND_MOCK_BACKEND_ONLY',checks=len(checks),check_names=checks,
        at=time.time(),source_sha256=source,binding_evidence=[b.evidence() for b in bindings],
        scope='Real pinned small source/split files and mock backend control; no real tensors, models, optimizer, shared storage or GPU proof',
        model_loads=0,cuda_loads=0,patient_loads=0,actual_optimization_updates=0,actual_test_evaluations=0,new_jobs=0,new_reservations=0,
        previous_34_and_91_tests_reused_not_rerun=True)
    with Path(a.out).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(status=result['status'],checks=len(checks))))

if __name__=='__main__':main()
