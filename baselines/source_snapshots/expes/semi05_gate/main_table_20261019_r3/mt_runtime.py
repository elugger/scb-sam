"""Real-data MT/UA-MT SemiSAM+ backend for the fixed terminal controller."""
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import time
import types

from budget_protocol import BUDGETS
from fixed_endpoint import exclusive_json
from historical_kernels import (MT_SOURCE, UA_SOURCE, bind_kernel, class_method,
                                load_pinned, method_definitions)
from runtime_data import ROOT, BoundCases, bound_splits, schedule
from runtime_resources import (FINAL_BOUND, RAM_REQUIRED, BoundedWriter,
                               resource_backend, save_terminal, sha)


def module_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def signature(torch, model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class MeasuredOptimizer:
    def __init__(self, optimizer, torch):
        self.inner, self.torch, self.calls, self.norm = optimizer, torch, 0, None
    @property
    def param_groups(self):
        return self.inner.param_groups
    def zero_grad(self, **kwargs):
        return self.inner.zero_grad(**kwargs)
    def step(self):
        total = 0.
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if not bool(self.torch.isfinite(p.grad).all()):
                        raise ValueError('Nonfinite actual optimizer gradient')
                    total += float(p.grad.detach().double().square().sum())
        self.norm = math.sqrt(total)
        if not math.isfinite(self.norm) or self.norm <= 0:
            raise ValueError('Zero/nonfinite actual joint gradient')
        self.inner.step()
        self.calls += 1


class SemiSAMBackend:
    def __init__(self, contract, output, verify_contract):
        self.contract, self.output, self.verify_contract = contract, Path(output), verify_contract
        self.method = contract['method']
        self.budget = next(b for b in BUDGETS if b.dataset == contract['dataset'] and b.ratio == contract['ratio'])
        self.target = Path(contract['target'])
        self.position, self.actual_rows, self.cycle_rows = 0, [], []

    def prepare(self, endpoint):
        self.verify_contract(self.contract)
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import numpy as np
        import torchio as tio
        self.torch, self.np = torch, np
        torch.set_num_threads(2)
        random.seed(2027)
        np.random.seed(2027)
        torch.manual_seed(2027)
        torch.cuda.manual_seed_all(2027)
        self.splits = bound_splits(self.contract['dataset'], self.contract['ratio'])
        if tuple(self.splits['test']) != endpoint.test_case_ids:
            raise ValueError('Endpoint actual test identity differs')
        self.pairs = schedule(self.budget, self.splits)
        self.data = BoundCases(self.contract['dataset'], self.splits, file_bindings=self.contract['case_files'])
        sys.path.insert(0, str(ROOT))
        from utils.cascade_train_utils import normalize_batch
        ramps = module_file('_main_ramps', ROOT/'expes/common/ramps.py')
        scope = dict(torch=torch, nn=nn, F=F, np=np, math=math, Dict=dict, Any=object,
                     normalize_batch=normalize_batch, sigmoid_rampup=ramps.sigmoid_rampup)
        tree, source = load_pinned(MT_SOURCE)
        dice_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DiceLoss')
        exec(compile(ast.fix_missing_locations(ast.Module(body=[dice_class], type_ignores=[])),str(source),'exec'),scope)
        names = ['_prepare_batch', '_update_teacher', '_add_noise', '_consistency_weight',
                 '_sam_consistency_weight', '_poly_lr', '_specialist_consistency_loss', '_sam_consistency_loss']
        helpers, path = method_definitions(MT_SOURCE, 'MTSemiSAMTrainer', names)
        exec(compile(helpers,path,'exec'),scope)
        methods = {name:scope[name] for name in names}
        if self.method == 'uamt_semisam':
            names_ua = ['_teacher_uncertainty','_specialist_consistency_loss']
            helper, path = method_definitions(UA_SOURCE,'UAMTSemiSAMTrainer',names_ua)
            exec(compile(helper,path,'exec'),scope)
            methods.update({name:scope[name] for name in names_ua})
        # Preserve the reference ramp as a function of fractional budget progress.
        reference_consistency = methods['_consistency_weight']
        reference_sam = methods['_sam_consistency_weight']
        total = self.budget.primary_pairs
        methods['_consistency_weight'] = lambda t, i: reference_consistency(t, i*30000/total)
        methods['_sam_consistency_weight'] = lambda t, i, maximum: reference_sam(t,i*30000/total,30000)
        self.t = types.SimpleNamespace()
        self.t = type('FixedSemiSAMState', (), methods)()
        self.t.cfg = copy.deepcopy(self.contract['recipe'])
        self.t.device = torch.device('cuda:0')
        self.t.norm_transform = tio.ZNormalization(masking_method=lambda x:x>0)
        self.t.global_step, self.t.base_lr = 0, 0.01
        semicode = ROOT/'SemiSAM-main/code_semisam+'
        sys.path.insert(0,str(semicode))
        from networks.unet_3D import unet_3D
        self.t.student = unet_3D(n_classes=2,in_channels=1).to(self.t.device)
        self.t.teacher = unet_3D(n_classes=2,in_channels=1).to(self.t.device)
        self.t.teacher.load_state_dict(self.t.student.state_dict(),strict=True)
        for p in self.t.teacher.parameters():
            p.requires_grad_(False)
        self.t.ce_loss, self.t.dice_loss = nn.CrossEntropyLoss(),scope['DiceLoss'](2)
        self.t.optimizer = MeasuredOptimizer(torch.optim.SGD(self.t.student.parameters(),lr=0.01,
                                             momentum=0.9,weight_decay=0.0001),torch)
        gpath=ROOT/'expes/artifacts/mt-semisam-plus/code/expes/mt_semisam_plus/semisam_generalist.py'
        generalist=module_file('_main_generalist',gpath)
        generalist.ROOT, generalist.SEMISAM_CODE = ROOT, semicode
        # Load the exact click helper directly; project utils is a different package.
        clicks=module_file('_main_clicks',semicode/'utils/click_method.py')
        generalist.get_next_click3D_torch_ritm=clicks.get_next_click3D_torch_ritm
        generalist.get_next_click3D_torch_2=clicks.get_next_click3D_torch_2
        generalist._CLICK_READY=True
        guard=resource_backend(self.target)
        with guard.original_lock():
            capacity=guard.capacity_snapshot(RAM_REQUIRED)
            checkpoint=ROOT/'ckpt/sam_med3d_turbo.pth'
            if sha(checkpoint)!=self.contract['sam_sha256']:
                raise ValueError('Actual SAM initializer changed')
            self.t.generalist=generalist.SemiSAMGeneralist(checkpoint,self.t.device,prompt='unc',num_clicks=10)
        if signature(torch,self.t.student)!=signature(torch,self.t.teacher):
            raise ValueError('Actual student/teacher initialization differs')
        if any(p.requires_grad for p in self.t.generalist.sam.parameters()):
            raise ValueError('Foundation generalist unexpectedly trainable')
        self.kernel,self.kernel_audit=bind_kernel(self.method,scope)
        self.initial_signature=signature(torch,self.t.student)
        return dict(source_pins_verified=True,input_pins_verified=True,initialization_verified=True,
                    resource_gate_passed=True,test_unopened=True,initial_student_sha=self.initial_signature,
                    generalist_state_tensors=len(self.t.generalist.sam.state_dict()),
                    generalist_checkpoint_sha256=self.contract['sam_sha256'],capacity=capacity,
                    initialization_kind='seed2027_random_UNet_plus_frozen_generic_SAM_Turbo',
                    historical_adaptation=True,ramp_mapping='reference_30000_fractional_progress')

    def batch(self, sample):
        torch=self.torch
        # Unlabeled GT was used only by the disclosed spatial ROI transform.
        label=sample['label']
        if sample['role']=='train_unlab':
            if label is not None:
                raise ValueError('Unlabeled target crossed dataset boundary')
            label=torch.zeros(sample['image'].shape[-3:],dtype=torch.long)
        return dict(image=sample['image'].unsqueeze(0),label=label.unsqueeze(0))

    def train_iteration(self, remaining):
        if remaining['labeled_inputs']<=0 or remaining['unlabeled_inputs']<=0:
            raise ValueError('Invalid remaining native batch')
        pair=self.pairs[self.position]
        a=self.data.load('train_lab',pair['labeled'])
        b=self.data.load('train_unlab',pair['unlabeled'])
        if a['case_id']!=pair['labeled'] or b['case_id']!=pair['unlabeled']:
            raise ValueError('Actual case substituted')
        self.t.student.train()
        self.t.teacher.train()
        before=self.t.optimizer.calls
        values=self.kernel(self.t,iter([self.batch(a)]),iter([self.batch(b)]),self.budget.primary_pairs,
                           self.t.cfg['semi'],0.99,1)
        self.torch.cuda.synchronize()
        if any(not math.isfinite(v) for v in values.values()):
            raise ValueError('Actual nonfinite loss')
        if any(p.grad is not None for p in self.t.teacher.parameters()):
            raise ValueError('Actual teacher gradient')
        if any(p.grad is not None for p in self.t.generalist.sam.parameters()):
            raise ValueError('Actual SAM generalist gradient')
        self.position+=1
        self.actual_rows.append(pair)
        receipt=dict(labeled_inputs=1,unlabeled_inputs=1,successful_optimizer_updates=self.t.optimizer.calls-before,
                    finite_loss=True,finite_nonzero_gradients=True,teacher_no_grad=True,skip_count=0,
                    actual_case_ids_verified=True,unlabeled_gt_in_loss=False,gradient_norm=self.t.optimizer.norm,
                    actual_pair=pair,losses=values,completed_primary_pairs=self.position)
        if self.position % self.budget.primary_pairs_per_cycle==0:
            rows=self.actual_rows[-self.budget.primary_pairs_per_cycle:]
            if (set(x['unlabeled'] for x in rows)!=set(self.splits['train_unlab'])
                    or set(x['labeled'] for x in rows)!=set(self.splits['train_lab'])
                    or sum(x['unlabeled_replay'] for x in rows)!=self.budget.replay_per_cycle):
                raise ValueError('Actual complete cycle coverage differs')
            self.cycle_rows.append(dict(epoch=pair['epoch'],actual_case_coverage_verified=True,
                                       unlabeled_replay=self.budget.replay_per_cycle))
        return receipt

    def verify_training_coverage(self):
        if self.position!=self.budget.primary_pairs or len(self.cycle_rows)!=36:
            raise ValueError('Incomplete actual primary exposure')
        return dict(actual_case_coverage_verified=True,cycles=self.cycle_rows,
                    primary_labeled_inputs=self.position,primary_unlabeled_inputs=self.position)

    def save_terminal(self, counts):
        if signature(self.torch,self.t.student)==self.initial_signature:
            raise ValueError('Terminal student unchanged')
        self.final_signature=signature(self.torch,self.t.student)
        return save_terminal(self.target,dict(format='main_table_terminal_v1',method=self.method,
                             counts=counts,student=self.t.student.state_dict(),student_signature=self.final_signature,
                             sam_checkpoint_sha256=self.contract['sam_sha256'],selection='fixed_exposure_terminal'))

    def strict_reload(self, artifact):
        guard=resource_backend(self.target)
        with guard.original_lock():
            guard.capacity_snapshot(RAM_REQUIRED)
            if sha(self.target)!=artifact['sha256']:
                raise ValueError('Final changed before reload')
            loaded=self.torch.load(str(self.target),map_location=self.t.device)
            self.t.student.load_state_dict(loaded['student'],strict=True)
            if signature(self.torch,self.t.student)!=self.final_signature:
                raise ValueError('Native terminal identity changed')
        return dict(strict=True,sha256=artifact['sha256'],native_student_identity_verified=True)

    def test(self, artifact, permit):
        from expes.common.seg_metrics import binary_case_metrics,logits_to_binary
        from utils.cascade_train_utils import normalize_batch
        data=BoundCases(self.contract['dataset'],self.splits,test_permit=permit,training_flip=False,
                        file_bindings=self.contract['case_files'])
        rows=[]
        self.t.student.eval()
        for case in self.splits['test']:
            sample=data.load('test',case)
            image=normalize_batch(sample['image'].unsqueeze(0),self.t.norm_transform).to(self.t.device)
            with self.torch.no_grad():
                pred=logits_to_binary(self.t.student(image))[0]
            metrics=binary_case_metrics(pred,sample['label'],spacing=(1.,1.,1.))
            row=dict(case_id=case,actual_index=sample['actual_index'],**metrics)
            with (self.output/'final_test_cases.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            rows.append(row)
        means={k:sum(r[k] for r in rows)/len(rows) for k in ('dice','jaccard','hd95','asd')}
        return dict(case_ids=[r['case_id'] for r in rows],case_dice=[r['dice'] for r in rows],
                    final_sha256=artifact['sha256'],**means,distance_unit='voxel',test_evaluations=1)

    def qualify(self, endpoint):
        preparation=self.prepare(endpoint)
        torch=self.torch
        state={name:copy.deepcopy(getattr(self.t,name).state_dict()) for name in ('student','teacher')}
        optimizer=copy.deepcopy(self.t.optimizer.inner.state_dict())
        rng=(random.getstate(),self.np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state())
        before={k:signature(torch,getattr(self.t,k)) for k in state}
        receipts=[self.train_iteration(dict(labeled_inputs=2,unlabeled_inputs=2)) for _ in range(2)]
        for name,value in state.items():
            getattr(self.t,name).load_state_dict(value,strict=True)
            for p in getattr(self.t,name).parameters():
                p.grad=None
        self.t.optimizer.inner.load_state_dict(optimizer)
        self.t.optimizer.calls,self.t.global_step,self.position=0,0,0
        self.actual_rows,self.cycle_rows=[],[]
        random.setstate(rng[0]);self.np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state(rng[3])
        if any(signature(torch,getattr(self.t,k))!=before[k] for k in before):
            raise ValueError('Qualification did not restore model state')
        if self.t.optimizer.inner.state_dict()!=optimizer:
            raise ValueError('Qualification did not restore fresh optimizer')
        nr=self.np.random.get_state()
        if (random.getstate()!=rng[0] or nr[0]!=rng[1][0] or not self.np.array_equal(nr[1],rng[1][1])
                or nr[2:]!=rng[1][2:] or not torch.equal(torch.get_rng_state(),rng[2])
                or not torch.equal(torch.cuda.get_rng_state(),rng[3])):
            raise ValueError('Qualification did not restore actual RNG states')
        # Exercise actual filesystem serialization/reload in the shared atomic slot.
        probe=self.target.parent/'own_update0_probe.pth.partial'
        guard=resource_backend(self.target)
        with guard.original_lock():
            guard.capacity_snapshot(RAM_REQUIRED)
            probe.parent.mkdir(parents=True,exist_ok=True)
            with probe.open('xb') as stream:
                torch.save(state,BoundedWriter(stream,96*1024**2));stream.flush()
                import os
                os.fsync(stream.fileno())
            identity=(probe.stat().st_dev,probe.stat().st_ino,probe.stat().st_size)
            loaded=torch.load(str(probe),map_location=self.t.device)
            for name,value in loaded.items():
                getattr(self.t,name).load_state_dict(value,strict=True)
                if signature(torch,getattr(self.t,name))!=before[name]:
                    raise ValueError('Own actual disk roundtrip differs')
            if identity!=(probe.stat().st_dev,probe.stat().st_ino,probe.stat().st_size):
                raise ValueError('Own probe identity changed')
            probe.unlink()  # Only the newly created, verified own probe.
        return dict(status='QUALIFIED',preparation=preparation,actual_extra_engineering_updates=2,
                    new_test_evaluations=0,restored_model_optimizer_rng=True,actual_disk_roundtrip=True,
                    receipts=receipts,native_student_signature=before['student'])
