"""Prospective BY-only recipe and source-derived native terminal inference.

Candidate metadata, never permission to prepare or dispatch. Source pinning,
resource checks, complete backend/GPU qualification and independent acceptance
are still required.
"""
import ast
import hashlib
import json
import math
from pathlib import Path

METHOD='dualteacher_sam'
COMMIT='3710e3216db8bdb444296f0460eab9532b253848'
IMPLEMENTATION='author-source adaptation with disclosed config/device and paper-equation5 SAM autodiff repair'
FINAL_BOUND=256*1024**2
TRAIN_SHA='5aa647cf1a46dec0c67e360cdb261ac3cdbd16cf38ed2d7350b62235facf3d78'

def need(ok,message):
    if not ok:raise ValueError(message)

def binding(p):
    p=Path(p).absolute();need(p.is_file() and not p.is_symlink(),'Missing/symlink source')
    raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

def recipe(code,foundation,ratio,validate_sha):
    code=Path(code).absolute()
    need(ratio in (5,10),'Only authorized BY5/BY10')
    need(binding(code/'train.py')['sha256']==TRAIN_SHA,'Official train changed')
    need(binding(code/'validate.py')['sha256']==validate_sha,'Official validation changed')
    base=json.loads((code/'configs/config_25.json').read_text());cfg50=json.loads((code/'configs/config_50.json').read_text())
    need('use_dynamic_threshold' not in base and cfg50['use_dynamic_threshold'] is True,'Author config evidence changed')
    pairs=3420 if ratio==5 else 3240;iterations=pairs//2
    need(foundation['bytes']==402163626 and foundation['sha256']=='899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c','Own Turbo binding differs')
    keys=('lr','weight_decay','apply_aug','strong_threshold','weak_threshold','xi','num_negatives','num_queries','temp','mun_pro')
    return dict(method=METHOD,dataset='BY',ratio=ratio,fold=2,seed=2027,official_commit=COMMIT,implementation=IMPLEMENTATION,
        official_root=str(code),source_evidence={n:binding(code/n) for n in ('train.py','dataset.py','module_list.py','boundary_contrastive.py','Model/Unet.py','validate.py','configs/config_25.json','configs/config_50.json')},
        foundation=dict(foundation),initialization='seed2027_random_native_UNet_kMaX_and_op_plus_frozen_existing_Turbo',
        actual_native_batches=dict(labeled=2,unlabeled=2),primary_pairs=pairs,native_iterations=iterations,
        native_reference=dict(epochs=200,iterations_per_epoch=200),clock_policy='floor(200*native_iteration/fixed_native_iterations); native epoch schedules retained',
        native_algorithm_config={k:base[k] for k in keys},explicit_config_repair=dict(use_dynamic_threshold=True,source='author config_50.json'),
        student=dict(feature_scale=4,in_channels=1,n_classes=2,mun_pro=4,num_queries=16,query_dim=64),
        optimizer=dict(student='SGD',op='SGD',learning_rate=.0025,momentum=.9,weight_decay=.0005,nesterov=True,
            primary_student_updates=iterations,op_step_calls=2*iterations,op_effective_updates_expected=iterations,
            all_native_effective_optimizer_updates_expected=2*iterations,
            caveat='Isolated existing Conda Torch2.0.1 zero_grad defaults set_to_none=True; OP first call has no gradients and no parameter update in every iteration. Both source step calls remain; actual deltas and environment are pinned.'),
        prototype=dict(boundary_temperature=.07,local_weight=1.,global_weight=.5,hard_weight=.5,boundary_loss_weight=.1,
            criterion_query_projection='native initialized orthogonal projection remains outside native optimizers; save its state; model query/prototype parameters remain trainable'),
        consistency=dict(weight=.5,uncertainty_mask='preserve native raw-view mask behavior',ema_alpha=.99,teacher_mode='train with no_grad; native parameter-only EMA update'),
        sam=dict(weight=.1,start_native_epoch_exclusive=5,points_per_case=10,positive_points=5,negative_points=5,
            input_resize=[128,128,128],native_output_resize=[96,96,96],native_batch=2,loss='MSE student foreground vs continuous sigmoid SAM probability on native valid cases',
            autodiff_repair='Native student/MSE four statements re-enabled under torch.enable_grad; external teacher target stays frozen',paper_equations=[5,10]),
        preprocessing=dict(shared_roi=[128,128,128],native_h5_numeric_equivalence='unknown_not_claimed',gt_roi_disclosed=True,patient_alias_identity_unresolved=True,
            train=['shared ToCanonical+GT CropOrPad128 without added random flip','native RandomRotFlip','native minmax Normalise','native RandomBrightness(.75,1.25)','native RandomCrop96','native ToTensor'],
            unlabeled_targets='shape-only zero placeholder for image/label joint geometric transforms, never GT returned or used in loss',
            test=['shared ToCanonical+GT CropOrPad128','native minmax Normalise','native EMA+op sliding window96 stride16'],
            native_minmax_constant_input_policy='require finite; preserve failed sample and repair separately if encountered'),
        fixed_endpoint=dict(selection='fixed_terminal_EMA_plus_OP',native_inference_model='author validation uses EMA model and op module',
            native_patch=[96,96,96],native_stride=[16,16,16],native_windows_per_shared_roi=27,post_processing=False,
            test_case_count=25,test_evaluations=1,intermediate_validation=False,best_checkpoint_selection=False,early_stop=False,
            shared_case_metrics='same pinned common binary_case_metrics with spacing(1,1,1)',repeated_test=False),
        prefix_training_updates=0,terminal_contains=['student','ema','op','criterion'],checkpoint_bound=FINAL_BOUND,
        primary_budget_only=True,total_compute_matched=False,exact_resume_supported=False,
        process_environment=dict(PYTHONNOUSERSITE='1',torch='2.0.1+cu117',torchvision='0.15.2+cu117',native_zero_grad_set_to_none_default=True))

def load_native_inference(code,expected_sha,torch,np):
    p=Path(code)/'validate.py';need(binding(p)['sha256']==expected_sha,'Native inference source changed')
    tree=ast.parse(p.read_text());fns=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='test_single_case']
    need(len(fns)==1,'Native inference function closure differs')
    fn=fns[0];changes=[]
    class DevicePortability(ast.NodeTransformer):
        def visit_Call(self,n):
            self.generic_visit(n)
            if isinstance(n.func,ast.Attribute) and n.func.attr=='cuda':
                need(not n.args and len(n.keywords)==1 and n.keywords[0].arg=='non_blocking' and isinstance(n.keywords[0].value,ast.Constant) and n.keywords[0].value.value is True,'Unexpected native CUDA call')
                n.func.attr='to';n.keywords.insert(0,ast.keyword(arg='device',value=ast.parse('next(net.parameters()).device',mode='eval').body));changes.append(n.lineno)
            return n
    fn=DevicePortability().visit(fn);need(len(changes)==1,'Native inference device transfer changed')
    def no_postprocess(*args,**kwargs):raise ValueError('Unfrozen postprocessing rejected')
    ns=dict(torch=torch,np=np,math=math,F=torch.nn.functional,cct=no_postprocess)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(p),'exec'),ns)
    native=ns['test_single_case']
    def infer(ema,op,image):
        need(not ema.training and not op.training,'Terminal models must be eval')
        need(tuple(image.shape)==(128,128,128),'Shared terminal ROI shape changed')
        # Original author function retained, including window aggregation and
        # op fusion; only explicit tensor-to-model-device transfer is portable.
        with torch.no_grad():return native(ema,op,image,stride_xy=16,stride_z=16,patch_size=(96,96,96),num_classes=2,post_process=False)
    return infer,dict(source=binding(p),device_transfer_only_lines=changes,intermediate_test_enabled=False,
        frozen_selection='terminal EMA+OP',patch=[96,96,96],stride=[16,16,16],post_processing=False)
