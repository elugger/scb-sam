"""Unscheduled native Dualteacher core; no loader, writer, test or dispatch.

The official inner training body is extracted by AST and source-bound. Native
optimizer membership, teacher train mode, augmentations and all loss branches
are retained. Caller must provide only two labeled targets, never unlabeled GT.
This candidate is not a qualified training backend.
"""
import ast
import copy
import hashlib
import importlib
import json
import math
import inspect
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

OFFICIAL_COMMIT='3710e3216db8bdb444296f0460eab9532b253848'
TRAIN_SHA='5aa647cf1a46dec0c67e360cdb261ac3cdbd16cf38ed2d7350b62235facf3d78'
NATIVE_EPOCHS=200
NATIVE_ITERS_PER_EPOCH=200

def need(ok,message):
    if not ok:raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def clock(index,total):
    need(type(index) is int and type(total) is int and 0<=index<total and total>0,'Invalid native iteration clock')
    epoch=min(NATIVE_EPOCHS-1,(NATIVE_EPOCHS*index)//total)
    return dict(iteration=index,total_iterations=total,native_epoch=epoch,native_reference_iteration=NATIVE_EPOCHS*NATIVE_ITERS_PER_EPOCH*index/total,
        primary_progress=index/total,learning_rate=max(1e-6,.0025*(1-epoch/NATIVE_EPOCHS)**.9),sam_enabled=epoch>5)

def extract_native_step(code,namespace):
    p=Path(code)/'train.py';need(sha(p)==TRAIN_SHA,'Official train source differs')
    raw=p.read_text();tree=ast.parse(raw)
    outer=[n for n in tree.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='index']
    need(len(outer)==1,'Official epoch loop differs')
    inner=[n for n in outer[0].body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='i']
    need(len(inner)==1,'Official native iteration loop differs')
    nodes=inner[0].body
    need(len(nodes)>20,'Native update unexpectedly empty')
    # Exactly four native loader/read-to-device statements are replaced with
    # explicit caller tensors; the original unlabeled target is never exposed.
    expected=['train_l_data, train_l_label, _','train_l_data, train_l_label','train_u_data, train_u_label, name','train_u_data, train_u_label']
    for n,wanted in zip(nodes[:4],expected):need(isinstance(n,ast.Assign) and ast.unparse(n.targets[0])==f'({wanted})','Native input statements differ')
    end=next((i for i,n in enumerate(nodes) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Subscript) and isinstance(n.targets[0].value,ast.Name) and n.targets[0].value.id=='cost'),None)
    need(end is not None and end>4,'Native logging boundary differs')
    body=copy.deepcopy(nodes[4:end])
    need(ast.unparse(body[-1])=='model.train()','Native optimizer body endpoint differs')
    # Paper equation (5)/(10) requires student gradients. The public source
    # accidentally computes student probability and MSE inside no_grad. Keep
    # external-teacher generation frozen, re-enable autograd only for these
    # same four student/selection/MSE statements, without changing values.
    sam_if=[n for n in body if isinstance(n,ast.If) and ast.unparse(n.test)=='index > 5']
    need(len(sam_if)==1 and len(sam_if[0].body)==1 and isinstance(sam_if[0].body[0],ast.With),'SAM gradient scope differs')
    frozen=sam_if[0].body[0]
    valid=[n for n in frozen.body if isinstance(n,ast.If) and ast.unparse(n.test)=='len(valid_batch_indices) > 0']
    need(len(valid)==1,'SAM valid-case branch differs')
    tail=valid[0].body[-4:]
    need([ast.unparse(n.targets[0]) for n in tail if isinstance(n,ast.Assign)]==['student_fg_prob','student_valid','sam_valid','loss_sam_aux'],'SAM autodiff repair scope differs')
    valid[0].body[-4:]=[ast.With(items=[ast.withitem(context_expr=ast.parse('torch.enable_grad()',mode='eval').body,optional_vars=None)],body=tail,type_comment=None)]
    selected_names={'resize_volume','get_points_from_mask','compute_unsupervised_loss_with_uncertainty'}
    helpers=[copy.deepcopy(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in selected_names]
    need({n.name for n in helpers}==selected_names,'Native helper closure differs')
    args='train_l_data, train_l_label, train_u_data, model, op_module, ema, sam_model, config, data_loader, bc_criterion, optimizer, optimizer_op, device, index, dynamic_threshold_bg, dynamic_threshold_fg'
    fn=ast.parse(f'def official_step({args}):\n    pass').body[0]
    result="dict(loss=loss, sup_loss=sup_loss, unsup_loss=unsup_loss, bc_loss=loss_bc, sam_loss=loss_sam_aux, op_loss=op_loss, dynamic_threshold_fg=dynamic_threshold_fg, sam_valid_cases=len(locals().get('valid_batch_indices', [])))"
    fn.body=body+[ast.Return(value=ast.parse(result,mode='eval').body)]
    ns=dict(namespace,torch=torch,F=F,np=np,alpha_ema=.99,lambda_bc=.1,lambda_sam=.1)
    compiled=ast.fix_missing_locations(ast.Module(body=helpers+[fn],type_ignores=[]))
    exec(compile(compiled,str(p),'exec'),ns)
    return ns['official_step'],dict(original_train_sha256=TRAIN_SHA,
        extracted_algorithm_ast_sha256=hashlib.sha256(ast.dump(ast.Module(body=body,type_ignores=[]),include_attributes=False).encode()).hexdigest(),
        removed_input_statements=4,removed_logging_only=True,native_algorithm_statements=len(body),
        dynamic_threshold_config_repair='explicit true from author config_50; config_25 other algorithm parameters retained',
        native_untransformed_uncertainty_mask_preserved=True,native_criterion_optimizer_membership_preserved=True,
        sam_autodiff_repair='only four existing student/MSE statements re-enabled under torch.enable_grad; SAM/EMA frozen; paper equations5,10')

class AuditedOptimizer:
    def __init__(self,optimizer,label):self.optimizer,self.label=optimizer,label;self.events=[]
    def reset(self):self.events=[]
    def zero_grad(self,*args,**kwargs):return self.optimizer.zero_grad(*args,**kwargs)
    def step(self,*args,**kwargs):
        gradients=[p.grad for g in self.optimizer.param_groups for p in g['params'] if p.grad is not None]
        need(all(bool(torch.isfinite(x).all()) for x in gradients),'Nonfinite '+self.label+' gradient')
        l1=sum(float(x.detach().abs().sum()) for x in gradients)
        need(math.isfinite(l1),'Nonfinite gradient total')
        # OP's first step has zero gradient buffers on later iterations, yet
        # native SGD momentum/weight decay still changes its weights. Measure
        # the small OP module rather than calling this extra update a no-op.
        parameters=[p for g in self.optimizer.param_groups for p in g['params']]
        before=[p.detach().clone() for p in parameters] if self.label=='op' else None
        result=self.optimizer.step(*args,**kwargs)
        delta=sum(float((p.detach()-b).abs().sum()) for p,b in zip(parameters,before)) if before is not None else None
        event=dict(parameters_with_gradient=len(gradients),gradient_l1=l1,effective_update=(delta>0 if delta is not None else bool(gradients and l1>0)),parameter_delta_l1=delta)
        self.events.append(event)
        return result

def load_native_modules(code):
    code=Path(code).resolve();need((code/'Model/Unet.py').is_file(),'Official component root missing')
    sys.path.insert(0,str(code))
    modules={n:importlib.import_module(n) for n in ('Model.Unet','module_list','dataset','utilsm.loss','boundary_contrastive')}
    for n,m in modules.items():need(code in Path(m.__file__).resolve().parents,'Native import escaped exact source root: '+n)
    return modules

class DualteacherCore:
    def __init__(self,code,foundation,total,device):
        need(type(total) is int and total in (1710,1620),'Only authorized BY native iteration budgets')
        self.code=Path(code);self.total,self.step_count=total,0;self.device=torch.device(device)
        need(os.environ.get('PYTHONNOUSERSITE')=='1' and torch.__version__=='2.0.1+cu117','Isolated coherent existing Conda Torch2 runtime required')
        need(inspect.signature(torch.optim.SGD.zero_grad).parameters['set_to_none'].default is True,'Native Conda optimizer default changed')
        self.native=load_native_modules(self.code)
        import medim
        import random
        random.seed(2027);np.random.seed(2027);torch.manual_seed(2027)
        if self.device.type=='cuda':torch.cuda.manual_seed_all(2027)
        cfg=json.loads((self.code/'configs/config_25.json').read_text());cfg50=json.loads((self.code/'configs/config_50.json').read_text())
        need('use_dynamic_threshold' not in cfg and cfg50['use_dynamic_threshold'] is True,'Author config switch evidence differs')
        self.config=dict(cfg,use_dynamic_threshold=True)
        need({k:v for k,v in self.config.items() if k!='use_dynamic_threshold'}==cfg,'Silent algorithm hyperparameter change')
        self.model=self.native['Model.Unet'].UNet_kMaX(in_channels=1,is_batchnorm=True,n_classes=2,mun_pro=4,num_queries=16,query_dim=64).to(self.device)
        self.op=self.native['Model.Unet'].opmoudle().to(self.device)
        p=Path(foundation['path']);need(p.is_file() and p.stat().st_size==foundation['bytes']==402163626 and sha(p)==foundation['sha256'],'Bound Turbo changed')
        import medim
        self.sam=medim.create_model('SAM-Med3D',pretrained=False)
        state=torch.load(str(p),map_location='cpu');need(set(state)=={'model_state_dict'},'Turbo wrapper differs')
        loaded=self.sam.load_state_dict(state['model_state_dict'],strict=True);need(not loaded.missing_keys and not loaded.unexpected_keys,'Turbo strict mismatch')
        del state;self.sam=self.sam.to(self.device).eval()
        for p in self.sam.parameters():p.requires_grad_(False)
        self.bc=self.native['boundary_contrastive'].BoundaryContrastiveLoss(temperature=.07,lambda_soft=1.,lambda_global=.5,lambda_hard=.5,feature_dim=4,query_dim=64).to(self.device)
        self.optimizer=torch.optim.SGD(self.model.parameters(),lr=.0025,weight_decay=.0005,momentum=.9,nesterov=True)
        self.optimizer_op=torch.optim.SGD(self.op.parameters(),lr=.0025,weight_decay=.0005,momentum=.9,nesterov=True)
        # Author EMA updates parameters while teacher forwards stay train-mode.
        self.ema=self.native['module_list'].EMA(self.model,.99)
        for p in self.ema.model.parameters():p.requires_grad_(False)
        self.audit_student=AuditedOptimizer(self.optimizer,'student');self.audit_op=AuditedOptimizer(self.optimizer_op,'op')
        self.data_proxy=SimpleNamespace(crop_size=[96,96,96],scale_size=(.5,1.5))
        ns=dict(dice_loss=self.native['utilsm.loss'].dice_loss,batch_transform=self.native['dataset'].batch_transform,generate_unsup_data=self.native['module_list'].generate_unsup_data)
        self.step_function,self.extraction=extract_native_step(self.code,ns)
        self.dynamic_threshold_bg=self.dynamic_threshold_fg=.5
        self.forward_cases={};self.hooks=[]
        for name,model in [('student',self.model),('ema',self.ema.model),('op',self.op),('sam_encoder',self.sam.image_encoder)]:
            def hook(module,args,out,key=name):self.forward_cases[key]=self.forward_cases.get(key,0)+int(args[0].shape[0])
            self.hooks.append(model.register_forward_hook(hook))

    def update(self,labeled,target,unlabeled,case_ids):
        need(labeled.shape==unlabeled.shape==(2,1,96,96,96) and target.shape==(2,96,96,96) and target.dtype==torch.long,'Native batch/crop/target shape differs')
        need(set(case_ids)=={'labeled','unlabeled'} and all(len(v)==2 and all(isinstance(x,str) and x for x in v) for v in case_ids.values()),'Native case identity missing')
        need(not set(case_ids['labeled'])&set(case_ids['unlabeled']),'Labeled/unlabeled case overlap')
        need(all(x.device==self.device for x in (labeled,target,unlabeled)) and bool(torch.isfinite(labeled).all() and torch.isfinite(unlabeled).all()),'Native tensor device/value mismatch')
        need(bool(((target==0)|(target==1)).all()),'Nonbinary labeled target')
        c=clock(self.step_count,self.total)
        for opt in (self.optimizer,self.optimizer_op):
            for g in opt.param_groups:g['lr']=c['learning_rate']
        self.model.train();self.op.train();self.ema.model.train();self.sam.eval()
        self.forward_cases={};self.audit_student.reset();self.audit_op.reset()
        first_actual_update=self.ema.step==0
        r=self.step_function(labeled,target,unlabeled,self.model,self.op,self.ema,self.sam,self.config,self.data_proxy,self.bc,self.audit_student,self.audit_op,self.device,c['native_epoch'],self.dynamic_threshold_bg,self.dynamic_threshold_fg)
        need(not r['sam_valid_cases'] or r['sam_loss'].requires_grad,'Valid SAM loss detached from student')
        losses={k:float(r[k].detach()) for k in ('loss','sup_loss','unsup_loss','bc_loss','sam_loss','op_loss')}
        need(all(math.isfinite(v) for v in losses.values()),'Nonfinite native loss')
        need(len(self.audit_student.events)==1 and self.audit_student.events[0]['effective_update'],'Missing native student update')
        need(len(self.audit_op.events)==2 and self.audit_op.events[0]['gradient_l1']==0 and self.audit_op.events[1]['effective_update'],'Native OP gradient/step calls differ')
        need(self.audit_op.events[0]['effective_update'] is False,'Native Torch2 OP first call must have no gradient or effective update')
        need(not any(p.grad is not None for m in (self.sam,self.ema.model) for p in m.parameters()),'Teacher/foundation gradient escaped no_grad')
        need(self.forward_cases['student']==6 and self.forward_cases['op']==4 and self.forward_cases['ema']==(4 if c['sam_enabled'] else 2),'Native student/teacher/fusion compute count differs')
        need(self.forward_cases.get('sam_encoder',0)==(2 if c['sam_enabled'] and r['sam_valid_cases'] else 0),'SAM native batch/branch count differs')
        self.dynamic_threshold_fg=float(r['dynamic_threshold_fg']);self.step_count+=1
        return dict(step=self.step_count,clock=c,case_ids=case_ids,primary_labeled=2,primary_unlabeled=2,
            optimizer_updates=2,primary_optimizer_updates=1,student_optimizer_calls=1,op_optimizer_calls=2,
            effective_student_updates=1,effective_op_updates=1,
            op_step_events=copy.deepcopy(self.audit_op.events),
            student_gradient_l1=self.audit_student.events[0]['gradient_l1'],op_gradient_l1=self.audit_op.events[1]['gradient_l1'],
            student_case_forwards=self.forward_cases['student'],teacher_case_forwards=self.forward_cases['ema'],op_case_forwards=self.forward_cases['op'],
            foundation_case_forwards=self.forward_cases.get('sam_encoder',0),sam_valid_cases=r['sam_valid_cases'],dynamic_threshold_fg=self.dynamic_threshold_fg,
            unlab_gt_in_loss=False,test_evaluations=0,criterion_projection_optimizer_owned=False,sam_loss_requires_grad=r['sam_loss'].requires_grad,**losses)

    def state_dict(self):
        return copy.deepcopy(dict(student=self.model.state_dict(),ema=self.ema.model.state_dict(),op=self.op.state_dict(),criterion=self.bc.state_dict(),
            optimizer=self.optimizer.state_dict(),optimizer_op=self.optimizer_op.state_dict(),ema_step=self.ema.step,
            position=self.step_count,dynamic_threshold_fg=self.dynamic_threshold_fg,dynamic_threshold_bg=self.dynamic_threshold_bg))

    def load_state_dict(self,state):
        need(set(state)=={'student','ema','op','criterion','optimizer','optimizer_op','ema_step','position','dynamic_threshold_fg','dynamic_threshold_bg'},'Core state closure differs')
        for name,model in [('student',self.model),('ema',self.ema.model),('op',self.op),('criterion',self.bc)]:
            model.load_state_dict(state[name],strict=True)
            for p in model.parameters():p.grad=None
        self.optimizer.load_state_dict(state['optimizer']);self.optimizer_op.load_state_dict(state['optimizer_op'])
        self.ema.step=state['ema_step'];self.step_count=state['position']
        self.dynamic_threshold_fg=state['dynamic_threshold_fg'];self.dynamic_threshold_bg=state['dynamic_threshold_bg']

    def close(self):
        for h in self.hooks:h.remove()
        self.hooks=[]
