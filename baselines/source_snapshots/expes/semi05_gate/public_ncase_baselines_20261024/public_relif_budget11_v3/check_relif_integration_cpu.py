"""New integration checks only: synthetic backend/receipts and mocked resource gate.

Real isolated filesystem preservation and fsync; no torch/model/patient/CUDA,
optimization, evaluation, formal job, registered reservation or subprocess launch.
"""
import argparse
import ast
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import patch


def main():
    p=argparse.ArgumentParser()
    for name in ('native-dir','by-contract','budget-file','split-dir','out'):p.add_argument('--'+name,required=True)
    a=p.parse_args();here=Path(__file__).resolve().parent
    sys.path.insert(0,str(here));sys.path.insert(1,a.native_dir)
    from relif_binding import ReLiFBinding
    from relif_state import counts,require
    from relif_storage import read_pin
    import relif_protocol as protocol
    import resume_review as resume
    import admission
    import resumable_relif
    from relif_recovery import RecoveryMixin
    import observe_relif as observer
    binding=ReLiFBinding(5,native_dir=a.native_dir,by_contract=a.by_contract,budget_file=a.budget_file,split_dir=a.split_dir)
    checks=[]
    def ok(name,v):require(v,name);checks.append(name)
    def reject(name,fn):
        try:fn()
        except (ValueError,AssertionError,KeyError,FileExistsError):checks.append(name);return
        raise AssertionError('Not rejected: '+name)
    def dump(path,v):path.write_text(json.dumps(v,allow_nan=False))
    def receipt(i):
        pair=binding.pair(i);clock=binding.original_clock(i)
        return dict(actual_pair=pair,completed_primary_pairs=i+1,labeled_inputs=1,unlabeled_inputs=1,
            successful_optimizer_updates=1,finite_loss=True,finite_nonzero_gradients=True,actual_case_ids_verified=True,
            teacher_no_grad=True,skip_count=0,unlabeled_gt_in_loss=False,gradient_norm=1.,
            losses={k:1. for k in ('loss','supervised_ce','supervised_dice','distillation','cgcr','larc')},
            native_receipt=dict(step=i+1,clock=clock,primary_labeled=1,primary_unlabeled=1,optimizer_updates=1,
                student_case_forwards=4,foundation_case_forwards=2,sobf_views=2,teacher_case_forwards=0,
                extra_donor_inputs=0,test_evaluations=0,unlab_gt_in_loss=False,
                case_ids=dict(labeled=pair['labeled'],unlabeled=pair['unlabeled'])))
    for f in here.glob('*.py'):ast.parse(f.read_text(),filename=str(f))
    ok('all_new_sources_parse',True)
    import relif_backend as native
    real=resumable_relif.make_backend(native,protocol)
    ok('native_transform_and_consume_are_inherited_unchanged',real.transform is native.ReLiFBackend.transform
       and real.consume_pair is native.ReLiFBackend.consume_pair)
    ok('core_scientific_sources_remain_pinned',bool(binding.evidence()['native_sources']))
    calls=[]
    class Native:
        def __init__(self,c,output,verify):self.position=0;self.actual_rows=[];self.core=None;self.output=Path(output)
        def consume_pair(self,i):calls.append(i);self.core.step_count+=1;return receipt(i)
    fake=types.SimpleNamespace(ReLiFBackend=Native,schedule=lambda *args:copy.deepcopy(binding._pairs),signature=lambda m:'x')
    gate=types.SimpleNamespace(verify=lambda c:None,binding=lambda c:binding,clock_budget=lambda n:None,
        contract_sha=lambda c:'a'*64,capacity=lambda c:{},append_step=lambda *args:None)
    # The core is a counter only. Complete tensor restoration is deliberately
    # left to the separately required own GPU qualification and fresh probes.
    with patch.object(RecoveryMixin,'check_boundary',lambda self:None):
        cls=resumable_relif.make_backend(fake,gate);b=cls({'labeled_cases':5},'synthetic')
        b.core=types.SimpleNamespace(step_count=0,features=[])
        row=b.train_iteration(0)
        reject('pending_receipt_rejects_next_update',lambda:b.train_iteration(1))
        reject('pending_receipt_rejects_checkpoint_boundary',b.check_boundary)
        reject('wrong_durable_receipt_rejected',lambda:b.append_step({}))
        b.append_step(dict(step=row,cumulative=counts(1)))
        ok('native_update_and_durable_receipt_count_once',b.progress==counts(1) and calls==[0])
        before=list(calls)
        with patch.object(gate,'capacity',side_effect=ValueError('Insufficient filesystem capacity: synthetic')):
            reject('capacity_gate_precedes_any_native_update',lambda:b.train_iteration(1))
        ok('capacity_rejection_performed_zero_updates',calls==before)
        with patch.object(b,'consume_pair',side_effect=ValueError('unknown native failure')):
            reject('unknown_native_failure_is_preserved_and_poisoned',lambda:b.train_iteration(1))
        reject('failed_native_update_cannot_be_checkpointed',b.check_boundary)
    with tempfile.TemporaryDirectory(prefix='relif_integration_cpu_') as temp:
        root=Path(temp);j=root/'job';out=j/'formal';out.mkdir(parents=True);art=root/'artifacts';art.mkdir()
        c=dict(output=str(out),target=str(art/'final.pth'),recipe=binding.recipe)
        dump(j/'contract.json',c);identity=protocol.sha(j/'contract.json')
        ledger=out/'steps.jsonl';lines=[(json.dumps(dict(step=receipt(i),cumulative=counts(i+1)))+'\n').encode() for i in range(4)]
        ledger.write_bytes(b''.join(lines[:2]));cp_pin=read_pin(ledger,j)
        cp=dict(path=str(art/'recovery.pth'),bytes=6,sha256=hashlib.sha256(b'opaque').hexdigest(),
            contract_sha256=identity,execution_role='formal',generation=1,counts=counts(2),ledger=cp_pin)
        Path(cp['path']).write_bytes(b'opaque');ledger.write_bytes(b''.join(lines)+b'{"partial":')
        # write_bytes truncates in place on the same inode; inspect that fact.
        ok('synthetic_original_prefix_inode_retained',read_pin(ledger,j)['inode']==cp_pin['inode'])
        dump(j/'dispatch.json',dict(pid=2000000000,at=0));(j/'boot.log').write_bytes(b'')
        dump(out/'failure.json',dict(error='synthetic failed boundary'))
        dummy_store=types.SimpleNamespace(available=lambda:[cp],_commits=lambda:[cp])
        guard=types.SimpleNamespace(original_lock=contextlib.nullcontext)
        with contextlib.ExitStack() as stack:
            for module in (protocol,resume):
                for name,value in dict(STORE=art,verify=lambda c:None,binding=lambda c:binding,rows=lambda c:[],
                    store=lambda c:dummy_store,resource=lambda c:guard,capacity_locked=lambda c:{},
                    capacity=lambda *args:{'mock':True},leased=lambda c:contextlib.nullcontext()).items():
                    stack.enter_context(patch.object(module,name,value))
            for test_name in protocol.TEST_NAMES:
                (out/test_name).write_bytes(b'{}')
                reject('resume_rejects_'+test_name,lambda:resume.create_review(c,j,cp['sha256'],'synthetic explicit technical review'))
                (out/test_name).unlink()
            (j/'boot.log').write_bytes(b'unknown diagnostic')
            reject('reason_string_cannot_whitelist_unknown_boot',lambda:resume.create_review(c,j,cp['sha256'],'synthetic explicit technical review'))
            (j/'boot.log').write_bytes(b'')
            r=resume.create_review(c,j,cp['sha256'],'synthetic explicit technical review')
            before=ledger.read_bytes();inode=ledger.stat().st_ino
            ok('review_is_read_only_and_counts_partial_cost_separately',ledger.read_bytes()==before and r['projection']['complete_tail']['iterations']==2
               and r['projection']['partial_tail']['bytes']>0 and not r['projection']['failed_or_inflight_compute_fully_measured'])
            (j/'boot.log').write_bytes(b'changed')
            reject('review_rejects_changed_old_evidence',lambda:resume.validate_review(c,j))
            (j/'boot.log').write_bytes(b'')
            result=resume.reconcile(c,j);history=j/'recovery_history'/r['event']
            ok('rollback_preserves_full_original_and_exact_tail_first',(history/'original_steps.jsonl').read_bytes()==before
               and (history/'uncommitted_steps.jsonl').read_bytes()==before[cp_pin['bytes']:])
            ok('rollback_retains_committed_bytes_and_original_inode',ledger.read_bytes()==b''.join(lines[:2]) and ledger.stat().st_ino==inode)
            ok('failure_evidence_moved_not_discarded',not (out/'failure.json').exists() and len(result['preserved_failures'])==1)
            reject('review_cannot_be_reconciled_twice',lambda:resume.reconcile(c,j))
            resume.bind_worker_pid(c,os.getpid());consumed=resume.require_consumed_review(c)
            ok('review_consumed_by_exact_worker_pid',consumed['consumed_by_pid']==os.getpid())
            with patch.object(resume.os,'getpid',return_value=os.getpid()+1):
                reject('other_worker_cannot_consume_review',lambda:resume.require_consumed_review(c))
            reject('test_intent_in_preserved_history_blocks_future_testing',lambda:(
                (history/'old_final_test_intent.json').write_bytes(b'{}'),protocol.guard_test_lineage(c)))
        # Exact previous byte-prefix checks are exercised without any GPU call.
        dp=j/'dispatch.json';pin=read_pin(ledger,j);raw=ledger.read_bytes()
        old=dict(contract_sha256=identity,job=str(j),dispatch_sha256=protocol.sha(dp),ledger=pin)
        ok('live_previous_exact_prefix_accepts_same_file',observer.check_previous(c,old,dp,pin,raw) is None)
        reject('live_previous_same_length_byte_tamper_rejected',lambda:observer.check_previous(c,old,dp,pin,b'x'+raw[1:]))
        reject('live_previous_inode_replacement_rejected',lambda:observer.check_previous(c,old,dp,dict(pin,inode=pin['inode']+1),raw))
    ok('no_tensor_or_cuda_modules_imported','torch' not in sys.modules and 'numpy' not in sys.modules)
    result=dict(passed=True,checks=len(checks),check_names=checks,scope='new integration only; synthetic native counter and small evidence; mock shared capacity and lease; real filesystem preservation/fsync; not model/GPU/resource admission',
        new_model_loads=0,new_cuda_loads=0,new_patient_loads=0,new_optimization_updates=0,new_test_evaluations=0,
        formal_jobs_created=0,reservations_registered=0,dispatches=0,sources={f.name:protocol.sha(f) for f in here.glob('*.py')})
    path=Path(a.out);require(not path.exists(),'CPU receipt exists');protocol.exclusive(path,result)
    print(json.dumps(dict(passed=True,checks=len(checks),out=str(path))))


if __name__=='__main__':main()
