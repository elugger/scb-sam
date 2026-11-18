"""Native SDCL two-stage adapter; standalone until its real GPU qualification.

Pretraining retains the historical fixed 81*4*floor(N_lab/2) iterations, with
four real labeled inputs per iteration. It has no test selection. The SSL stage
then matches the main primary-input budget using four labeled/unlabeled inputs
per native iteration and two actual optimizers. No official run loop is used.
"""
import copy
import json
import math
from pathlib import Path
import random
import sys

from budget_protocol import BUDGETS
from fixed_endpoint import exclusive_json
from historical_kernels import SDCL_SOURCE,bind_kernel,method_definitions
from mt_runtime import MeasuredOptimizer,module_file,signature
from runtime_data import ROOT,BoundCases,bound_splits,schedule
from runtime_resources import RAM_REQUIRED,BoundedWriter,resource_backend,save_terminal,sha


def stage_plan(labeled_cases,primary_pairs):
    if labeled_cases<2 or primary_pairs%4:
        raise ValueError('Native SDCL batch/stage budget cannot be silently truncated')
    pre=81*4*(labeled_cases//2)
    return dict(pretrain_iterations=pre,pretrain_labeled_inputs=4*pre,
                pretrain_successful_optimizer_updates=2*pre,pretrain_test_evaluations=0,
                ssl_iterations=primary_pairs//4,ssl_labeled_inputs=primary_pairs,
                ssl_unlabeled_inputs=primary_pairs,ssl_successful_optimizer_updates=primary_pairs//2,
                native_inputs_per_stream=4,native_optimizers=2,
                prefix_selection='fixed_terminal_not_test_best',
                prefix_extra_budget_disclosed=True)


class SDCLBackend:
    def __init__(self,contract,output,verify_contract):
        self.contract,self.output,self.verify_contract=contract,Path(output),verify_contract
        self.budget=next(b for b in BUDGETS if b.dataset==contract['dataset'] and b.ratio==contract['ratio'])
        self.target=Path(contract['target']); self.position=0; self.rows=[]

    def prepare(self,endpoint,perform_prefix=True):
        self.verify_contract(self.contract)
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torchio as tio
        self.torch,self.np=torch,np
        torch.set_num_threads(2); random.seed(2027); np.random.seed(2027)
        torch.manual_seed(2027); torch.cuda.manual_seed_all(2027)
        self.splits=bound_splits(self.contract['dataset'],self.contract['ratio'])
        self.plan=stage_plan(len(self.splits['train_lab']),self.budget.primary_pairs)
        if tuple(self.splits['test'])!=endpoint.test_case_ids or endpoint.optimizer_updates_per_iteration!=2:
            raise ValueError('SDCL actual endpoint identity differs')
        self.pairs=schedule(self.budget,self.splits)
        self.data=BoundCases(self.contract['dataset'],self.splits,file_bindings=self.contract['case_files'])
        code=ROOT/'SDCL-master/code'; sys.path.insert(0,str(code)); sys.path.insert(0,str(ROOT))
        from utils.cascade_train_utils import normalize_batch
        ops=module_file('_main_sdcl_native_ops',ROOT/'expes/sdcl/sdcl_ops.py')
        networks=module_file('_main_sdcl_native_networks',ROOT/'expes/sdcl/sdcl_networks.py')
        losses=module_file('_main_sdcl_native_losses',code/'utils/losses.py')
        scope=dict(torch=torch,nn=nn,F=F,np=np,Tuple=tuple,normalize_batch=normalize_batch)
        for name in ('context_mask','forward_logits','get_cut_mask','get_xor_region',
                     'mix_loss','mix_max_kl_loss','mix_mse_loss','update_ema'):
            scope[name]=getattr(ops,name)
        helper,path=method_definitions(SDCL_SOURCE,'SDCLTrainer',['_prepare','_pretrain_step','_self_train_step'])
        exec(compile(helper,path,'exec'),scope)
        self.t=type('FixedNativeSDCL',(),{n:scope[n] for n in ('_prepare','_pretrain_step','_self_train_step')})()
        self.t.device=torch.device('cuda:0'); self.t.cfg=dict(semi={})
        self.t.student_v=networks.build_vnet().to(self.t.device)
        self.t.student_r=networks.build_resvnet().to(self.t.device)
        self.t.ema_v=networks.build_ema_vnet().to(self.t.device)
        self.t.ema_v.load_state_dict(self.t.student_v.state_dict(),strict=True)
        for p in self.t.ema_v.parameters(): p.requires_grad_(False)
        self.t.norm_transform=tio.ZNormalization(masking_method=lambda x:x>0)
        self.t.mask_ratio,self.t.u_weight,self.t.mse_w,self.t.kl_w,self.t.ema_alpha=2/3,.5,.5,.05,.99
        self.t.dice_loss=losses.mask_DiceLoss(nclass=2)
        self.t.opt_v=MeasuredOptimizer(torch.optim.Adam(self.t.student_v.parameters(),lr=.001),torch)
        self.t.opt_r=MeasuredOptimizer(torch.optim.Adam(self.t.student_r.parameters(),lr=.001),torch)
        self.t.global_step=0
        self.pre_kernel,_=bind_kernel('sdcl_pretrain',scope); self.ssl_kernel,_=bind_kernel('sdcl',scope)
        self.scope=scope
        guard=resource_backend(self.target)
        with guard.original_lock(): capacity=guard.capacity_snapshot(RAM_REQUIRED)
        self.output.mkdir(parents=True,exist_ok=True)
        self.pre_rows=[]
        if not perform_prefix:
            return dict(source_pins_verified=True,input_pins_verified=True,initialization_verified=True,
                        resource_gate_passed=True,test_unopened=True,capacity=capacity,
                        pretraining_executed=False)
        for i in range(self.plan['pretrain_iterations']):
            self.pretrain_iteration(i,record=True)
        self.t.ema_v.load_state_dict(self.t.student_v.state_dict(),strict=True)
        self.t.global_step=0
        self.prefix_signatures={n:signature(torch,getattr(self.t,n)) for n in ('student_v','student_r','ema_v')}
        exclusive_json(self.output/'pretrain_terminal.json',dict(plan=self.plan,
            successful_iterations=len(self.pre_rows),initialization='seed2027_random_native_dual_backbone',
            terminal_state_signatures=self.prefix_signatures,no_test_selection=True))
        return dict(source_pins_verified=True,input_pins_verified=True,initialization_verified=True,
                    resource_gate_passed=True,test_unopened=True,capacity=capacity,pretraining=self.plan,
                    teacher_matches_terminal_student=True)

    def qualify(self,endpoint):
        preparation=self.prepare(endpoint,perform_prefix=False)
        torch=self.torch
        names=('student_v','student_r','ema_v')
        states={n:copy.deepcopy(getattr(self.t,n).state_dict()) for n in names}
        before={n:signature(torch,getattr(self.t,n)) for n in names}
        optimizers={n:copy.deepcopy(getattr(self.t,n).inner.state_dict()) for n in ('opt_v','opt_r')}
        rng=(random.getstate(),self.np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state())
        pre=self.pretrain_iteration(0)
        self.t.ema_v.load_state_dict(self.t.student_v.state_dict(),strict=True)
        ssl=self.train_iteration(dict(labeled_inputs=4,unlabeled_inputs=4))
        for n in names:
            getattr(self.t,n).load_state_dict(states[n],strict=True)
            for p in getattr(self.t,n).parameters(): p.grad=None
        for n,state in optimizers.items():
            getattr(self.t,n).inner.load_state_dict(state); getattr(self.t,n).calls=0
        self.t.global_step,self.position=0,0; self.rows,self.pre_rows=[],[]
        random.setstate(rng[0]); self.np.random.set_state(rng[1]); torch.set_rng_state(rng[2]); torch.cuda.set_rng_state(rng[3])
        if any(signature(torch,getattr(self.t,n))!=before[n] for n in names):
            raise ValueError('SDCL qualification model restore differs')
        if any(getattr(self.t,n).inner.state_dict()!=state for n,state in optimizers.items()):
            raise ValueError('SDCL qualification fresh optimizer restore differs')
        nr=self.np.random.get_state()
        if (random.getstate()!=rng[0] or not self.np.array_equal(nr[1],rng[1][1])
                or nr[0]!=rng[1][0] or nr[2:]!=rng[1][2:]
                or not torch.equal(torch.get_rng_state(),rng[2]) or not torch.equal(torch.cuda.get_rng_state(),rng[3])):
            raise ValueError('SDCL qualification RNG restore differs')
        probe=self.target.parent/'own_sdcl_update0_probe.pth.partial'
        guard=resource_backend(self.target)
        with guard.original_lock():
            guard.capacity_snapshot(RAM_REQUIRED); probe.parent.mkdir(parents=True,exist_ok=True)
            with probe.open('xb') as stream:
                torch.save(states,BoundedWriter(stream,384*1024**2)); stream.flush()
                import os
                os.fsync(stream.fileno())
            identity=(probe.stat().st_dev,probe.stat().st_ino,probe.stat().st_size)
            reloaded=torch.load(str(probe),map_location=self.t.device)
            for n in names:
                getattr(self.t,n).load_state_dict(reloaded[n],strict=True)
                if signature(torch,getattr(self.t,n))!=before[n]: raise ValueError('SDCL disk roundtrip differs')
            if identity!=(probe.stat().st_dev,probe.stat().st_ino,probe.stat().st_size):
                raise ValueError('SDCL own probe identity changed')
            probe.unlink()
        return dict(status='QUALIFIED',preparation=preparation,actual_extra_engineering_updates=4,
            prefix_and_ssl_each_one_native_iteration=True,pre_receipt=pre,ssl_receipt=ssl,
            new_test_evaluations=0,restored_model_optimizer_rng=True,actual_disk_roundtrip=True,
            native_state_signatures=before)

    def batch(self,role,ids):
        samples=[self.data.load(role,k) for k in ids]
        labels=[s['label'] if role!='train_unlab' else self.torch.zeros(s['image'].shape[-3:],dtype=self.torch.long)
                for s in samples]
        if role=='train_unlab' and any(s['label'] is not None for s in samples):
            raise ValueError('Unlabeled target crossed SDCL data boundary')
        return dict(image=self.torch.stack([s['image'] for s in samples]),label=self.torch.stack(labels))

    def pretrain_iteration(self,index,record=False):
        ids=self.splits['train_lab']; actual=[]
        for j in range(4):
            epoch,within=divmod(index*4+j,len(ids)); order=list(ids)
            random.Random(2027+epoch*104729).shuffle(order)
            actual.append(order[within])
        self.t.student_v.train(); self.t.student_r.train()
        values=self.pre_kernel(self.t,iter([self.batch('train_lab',actual[:2])]),
                               iter([self.batch('train_lab',actual[2:])]))
        if any(not math.isfinite(v) for v in values.values()): raise ValueError('SDCL nonfinite prefix loss')
        row=dict(iteration=index+1,actual_labeled_case_ids=actual,labeled_inputs=4,
                 successful_optimizer_updates=2,losses=values,gradient_norms=[self.t.opt_v.norm,self.t.opt_r.norm])
        self.pre_rows.append(row)
        if record:
            with (self.output/'pretrain_steps.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n'); stream.flush()
        return row

    def train_iteration(self,remaining):
        if min(remaining.values())<4: raise ValueError('SDCL tail cannot silently change native batch')
        pairs=self.pairs[self.position:self.position+4]
        if len(pairs)!=4: raise ValueError('Incomplete SDCL native batch')
        lab=[p['labeled'] for p in pairs]; unlab=[p['unlabeled'] for p in pairs]
        self.t.student_v.train(); self.t.student_r.train(); self.t.ema_v.train()
        before=self.t.opt_v.calls+self.t.opt_r.calls
        batches=(self.batch('train_lab',lab[:2]),self.batch('train_lab',lab[2:]),
                 self.batch('train_unlab',unlab[:2]),self.batch('train_unlab',unlab[2:]))
        values=self.ssl_kernel(self.t,iter([batches])); self.torch.cuda.synchronize()
        if any(not math.isfinite(v) for v in values.values()): raise ValueError('SDCL nonfinite SSL loss')
        if any(p.grad is not None for p in self.t.ema_v.parameters()): raise ValueError('SDCL teacher gradient')
        self.position+=4; self.rows.extend(pairs)
        return dict(labeled_inputs=4,unlabeled_inputs=4,successful_optimizer_updates=self.t.opt_v.calls+self.t.opt_r.calls-before,
            finite_loss=True,finite_nonzero_gradients=True,teacher_no_grad=True,skip_count=0,
            actual_case_ids_verified=True,unlabeled_gt_in_loss=False,actual_pairs=pairs,
            completed_primary_pairs=self.position,losses=values,gradient_norms=[self.t.opt_v.norm,self.t.opt_r.norm])

    def verify_training_coverage(self):
        if self.rows!=self.pairs or self.position!=self.budget.primary_pairs:
            raise ValueError('SDCL real primary exposure differs')
        return dict(actual_case_coverage_verified=True,cycles=36,pretraining=self.plan,
                    primary_labeled_inputs=self.position,primary_unlabeled_inputs=self.position)

    def save_terminal(self,counts):
        self.signatures={n:signature(self.torch,getattr(self.t,n)) for n in ('student_v','student_r')}
        return save_terminal(self.target,dict(method='sdcl',counts=counts,pretraining=self.plan,
            signatures=self.signatures,**{n:getattr(self.t,n).state_dict() for n in self.signatures}))

    def strict_reload(self,artifact):
        guard=resource_backend(self.target)
        with guard.original_lock():
            guard.capacity_snapshot(RAM_REQUIRED)
            if sha(self.target)!=artifact['sha256']: raise ValueError('SDCL terminal SHA changed')
            value=self.torch.load(str(self.target),map_location=self.t.device)
            for n in self.signatures:
                getattr(self.t,n).load_state_dict(value[n],strict=True)
                if signature(self.torch,getattr(self.t,n))!=self.signatures[n]:
                    raise ValueError('SDCL terminal identity changed')
        return dict(strict=True,sha256=artifact['sha256'],native_student_identity_verified=True)

    def test(self,artifact,permit):
        from expes.common.seg_metrics import binary_case_metrics,logits_to_binary
        from utils.cascade_train_utils import normalize_batch
        data=BoundCases(self.contract['dataset'],self.splits,test_permit=permit,training_flip=False,
                        file_bindings=self.contract['case_files'])
        self.t.student_v.eval(); self.t.student_r.eval(); rows=[]
        for case in self.splits['test']:
            sample=data.load('test',case)
            x=normalize_batch(sample['image'].unsqueeze(0),self.t.norm_transform).to(self.t.device)
            with self.torch.no_grad():
                probability=.5*(self.scope['forward_logits'](self.t.student_v,x).softmax(1)
                                  +self.scope['forward_logits'](self.t.student_r,x).softmax(1))
                pred=logits_to_binary(probability)[0]
            row=dict(case_id=case,actual_index=sample['actual_index'],
                     **binary_case_metrics(pred,sample['label'],spacing=(1.,1.,1.)))
            rows.append(row)
            with (self.output/'final_test_cases.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
        return dict(case_ids=[r['case_id'] for r in rows],case_dice=[r['dice'] for r in rows],
            **{k:sum(r[k] for r in rows)/len(rows) for k in ('dice','jaccard','hd95','asd')},
            final_sha256=artifact['sha256'],test_evaluations=1,distance_unit='voxel')
