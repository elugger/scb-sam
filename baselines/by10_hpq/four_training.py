"""Execution-only matched adapter; frozen scientific methods remain inherited."""
import copy,hashlib,io,json,math,os,sys,time,traceback
from pathlib import Path
from four_contract import *


def build_trainer(prefix_cfg,ancestor,accepted,initial,prep,group,phase):
    import torch,yaml
    from train_by5 import make_strict_dataset,read_case_ids,load_trusted_checkpoint,state_dict_sha256
    from by_inputs import identity
    from expes.common.semi_data import build_eval_transform
    from expes.semi05_gate.paired_fcd_matrix_20261006 import MatrixHalf,MatrixStyle,MatrixMixin
    from expes.semi05_gate.paired_zea_cross_scale_half_20260911 import PairedCrossScaleHalf
    from expes.semi05_gate.paired_l_cross_style_complement_20261001 import PairedCrossStyleComplement
    data_cfg=copy.deepcopy(prefix_cfg);data_cfg['case_file_metadata']=prep['case_file_metadata']
    ids=prep['case_ids'];cfg=group_config(yaml.safe_load(ancestor['config_text']),group,phase)
    Base=MatrixStyle if group in ('HP','HPQ') else MatrixHalf

    class NoUnlabeledTarget(torch.utils.data.Dataset):
        def __init__(self,original):self.original=original
        def __len__(self):return len(self.original)
        def __getitem__(self,index):
            value=self.original[index]
            return {k:v for k,v in value.items() if k not in ('label','foreground_voxels')}

    class MatchedBY(Base):
        def _build_dataloaders(self):
            self.lab_set=make_strict_dataset(data_cfg,ids['train_lab'],'train')
            u=make_strict_dataset(data_cfg,ids['train_unlab'],'train')
            u.transform=build_eval_transform(128)
            self.unlab_set=NoUnlabeledTarget(u)
            # Development data cannot be built through the inherited trainer.
        def _load(self,path):
            require(Path(path)==STORE/'supervised/final.pth','Wrong shared initialization')
            with original_lock():
                capacity_snapshot(config=prefix_cfg)
                require(sha(path)==INIT_SHA,'Initializer changed before load')
                ck=load_trusted_checkpoint(path,map_location='cpu')
                require(ck['successful_updates']==600 and ck['completed_epochs']==60,'Initializer budget differs')
                self.student.load_state_dict(ck['student'],strict=True)
                require(state_dict_sha256(self.student.state_dict())==accepted['initializer']['state_sha256'],'Initializer values differ')
                for n,m in (('head',self.student_fclr_head),('adapter',self.student_zea_adapter)):
                    m.load_state_dict(self.cold[n],strict=True)
                for s,t in (('student','teacher'),('head','teacher_head'),('adapter','teacher_adapter')):
                    self.roots()[t].load_state_dict(self.roots()[s].state_dict(),strict=True)
                self._fclr_loaded_head=False;self._matrix_initial_head_state=self.cold['head']
                del ck
        def _donor_amplitude(self,name):
            require(name in ids['train_unlab'],'Donor outside 90U')
            path=ROOT/'data/BY_125_Folds/fold_2/train/imagesTr'/(name+'.nii.gz')
            require(identity(path,ROOT/'data')==prep['inputs'][str(path)]['identity'],'Donor input identity changed')
            return super()._donor_amplitude(name)
        def _save_ucp(self,*args,**kwargs):raise RuntimeError('Only guarded fixed-final save is permitted')
        def _evaluate(self,*args,**kwargs):raise RuntimeError('Only fixed-final explicit evaluation is permitted')

    require(MatchedBY.step_matrix is MatrixMixin.step_matrix and MatchedBY.contextual_gradients is MatrixMixin.contextual_gradients,'Scientific step/CCBS changed')
    require(MatchedBY._losses is (PairedCrossStyleComplement._losses if group in ('HP','HPQ') else PairedCrossScaleHalf._losses),'CCL path differs')
    trainer=MatchedBY(cfg)
    trainer.cold={n:{k:v.detach().cpu().clone() for k,v in trainer.roots()[n].state_dict().items()} for n in ('head','adapter')}
    trainer.initialize_matrix(str(STORE/'supervised/final.pth'))
    roots={n:state_dict_sha256(m.state_dict()) for n,m in trainer.roots().items()}
    reference=initial['results'].get(group)
    if phase=='initial':
        require(not trainer.optimizer.state and trainer.matrix_updates==0,'Initial state is not cold')
        require([g['lr'] for g in trainer.optimizer.param_groups]==[5e-8,2e-7,2e-7,2e-7,1e-4,1e-4],'Learning rates changed')
        for student,teacher in [('student','teacher'),('head','teacher_head'),('adapter','teacher_adapter')]:
            require(roots[student]==roots[teacher],'Initial teacher differs')
        return trainer,cfg,data_cfg,roots
    require(roots==reference['root_sha256'],'Training initializer differs from qualified six-root state')
    require([g['lr'] for g in trainer.optimizer.param_groups]==reference['optimizer_lrs'],'Learning rates changed')
    require(trainer.scaler.state_dict()==reference['scaler'] and not trainer.optimizer.state and trainer.matrix_updates==0,'Optimizer/scaler not cold')
    require(trainer.pair_seed==2027 and not trainer.refresh and trainer.workers==0,'Paired execution changed')
    return trainer,cfg,data_cfg,roots


def run(args):
    os.chdir(ROOT)
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==GPU and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','GPU/determinism environment differs')
    prefix_cfg,ancestor,accepted,initial,prep=verify_prepared(args.bundle_sha,args.prepare_sha)
    from train_by5 import validate_gpu_claim,state_dict_sha256,load_trusted_checkpoint,install_optimizer_step_counter,append_jsonl
    from by_runtime import ensure_gpu0_idle,own_gpu0_fds
    claim=validate_gpu_claim(prefix_cfg,0,args.fd,args.dev,args.ino)
    ensure_gpu0_idle()
    with original_lock():capacity=capacity_snapshot(config=prefix_cfg)
    group,phase=args.group,args.phase;output=BASE/phase/group
    require(not output.exists(),'Existing arm output; no replay')
    if phase=='formal':
        bridge=validate_bridge(args.bundle_sha,args.prepare_sha)
        require(bridge['sha256']==args.bridge_sha,'Formal bridge binding differs')
        require(not (STORE/group).exists(),'Existing formal model store')
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    from expes.semi05_gate.fcd_deterministic_losses_r2_20261005 import deterministic_training
    from expes.semi05_gate.paired_rng_data_20260908 import stream_seed,tensor_digest
    from utils.cascade_train_utils import normalize_batch
    started=time.monotonic();trainer=None;counter={'count':0};attempts=0;completed_epochs=0
    input_pairs=[];exposures={c:0 for c in prep['case_ids']['train_lab']};unlab_exposures={c:0 for c in prep['case_ids']['train_unlab']}
    append=lambda name,row:append_jsonl(output/name,row)
    def signature(value):
        h=hashlib.sha256()
        def visit(x):
            if torch.is_tensor(x):
                h.update(str((x.dtype,tuple(x.shape))).encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
            elif isinstance(x,dict):
                for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
            elif isinstance(x,(tuple,list)):
                for v in x:visit(v)
            else:h.update(repr(x).encode())
        visit(value);return h.hexdigest()
    try:
        with deterministic_training():
            trainer,cfg,data_cfg,initial_roots=build_trainer(prefix_cfg,ancestor,accepted,initial,prep,group,phase)
            require(torch.are_deterministic_algorithms_enabled() and torch.backends.cudnn.deterministic
                and not torch.backends.cudnn.benchmark,'Constructor changed deterministic execution')
            install_optimizer_step_counter(trainer.optimizer,counter)
            gpu_fd=own_gpu0_fds()
            dump_new(output/'started.json',dict(at=stamp(),phase=phase,group=group,pid=os.getpid(),claim=claim,
                initializer_sha256=INIT_SHA,initial_roots=initial_roots,own_gpu_fds=gpu_fd,
                bundle_sha256=args.bundle_sha,preparation_sha256=args.prepare_sha,
                development_evaluations=0,capacity=capacity,actual_input_guard=True))
            def raw_groups(epoch):
                lab,unlab=trainer._loaders(epoch);batch=[]
                for step,bu in enumerate(unlab):
                    bl=next(lab)
                    lc,uc=bl['case_id'][0],bu['case_id'][0]
                    require('label' not in bu and uc==prep['case_ids']['train_unlab'][step],'Unlabeled target/order boundary failed')
                    require(lc in exposures and int(bl['requested_index'])==prep['case_ids']['train_lab'].index(lc),'Labeled requested/actual differs')
                    xl,yl=trainer._prepare_batch(bl);xu=normalize_batch(bu['image'],trainer.norm_transform).to(trainer.device)
                    evidence=dict(epoch=epoch,step=step,labeled_case=lc,unlabeled_case=uc,
                        label_seed=int(bl['augmentation_seed']),unlab_seed=int(bu['augmentation_seed']),
                        student_seed=stream_seed(2027,'student-step',epoch,step),
                        label_image_sha=tensor_digest(bl['image']),unlab_image_sha=tensor_digest(bu['image']),
                        no_unlabeled_target=True)
                    batch.append(dict(epoch=epoch,step=step,image_l=xl,label_l=yl,image_u=xu))
                    input_pairs.append(evidence);append('pairs.jsonl',evidence)
                    exposures[lc]+=1;unlab_exposures[uc]+=1
                    if len(batch)==5:yield batch;batch=[]
                require(not batch and step==89,'Incomplete epoch actual pair coverage')
            def step(items,overflow=False):
                nonlocal attempts
                attempts+=1
                append('attempts.jsonl',dict(at=stamp(),attempt=attempts,successful_updates_before=counter['count'],
                    epoch=items[0]['epoch'],steps=[i['step'] for i in items],artificial_overflow=overflow))
                calls=[];styles=[];original_ema=trainer._update_teacher;original_loss=trainer._losses
                def ema(a):calls.append(a);return original_ema(a)
                def loss(*a,**kw):
                    r=original_loss(*a,**kw)
                    if group in ('HP','HPQ'):styles.append(copy.deepcopy(trainer._b_last))
                    return r
                trainer._update_teacher=ema;trainer._losses=loss
                try:row=trainer.step_matrix(items,compute_context=True,inject_overflow=overflow)
                finally:trainer._update_teacher=original_ema;trainer._losses=original_loss
                row.update(at=stamp(),attempt=attempts,ema_calls=len(calls),artificial_overflow=overflow,
                    optimizer_successful_calls=counter['count'],actual_pairs=input_pairs[-5:])
                if styles:row['styles']=styles
                append('steps.jsonl',row)
                require(calls==([] if row['skipped'] else [.999]),'EMA momentum/calls changed')
                require(counter['count']==trainer.matrix_updates,'Optimizer actual calls differ')
                check_step(row,group,overflow)
                if group in ('HP','HPQ'):
                    require(len(styles)==5,'Missing CCL diagnostics')
                    for s in styles:
                        require(s['donor1']!=s['donor2'] and not s['donor_labels_used'] and s['native_labeled_supervision'] and s['native_teacher'] and not s['old_fvc_active'] and s['complementary_sum']==2.,'CCL boundary differs')
                return row
            def payload(epoch):
                return dict(format='BY5_FOUR_ARM_FIXED_FINAL_V1',group=group,seed=2027,fold=2,labeled_cases=10,
                    roots={n:m.state_dict() for n,m in trainer.roots().items()},epoch=epoch,
                    successful_updates=counter['count'],cfg=cfg,initializer_sha256=INIT_SHA,
                    exact_resume_supported=False,bundle_sha256=args.bundle_sha,preparation_sha256=args.prepare_sha)
            def roundtrip(source):
                require(all(bool(torch.isfinite(v).all()) for m in trainer.roots().values()
                    for v in m.state_dict().values() if v.is_floating_point()),'Nonfinite checkpoint tensor')
                before={n:state_dict_sha256(m.state_dict()) for n,m in trainer.roots().items()}
                ck=load_trusted_checkpoint(source,map_location='cpu')
                require(ck['group']==group and ck['successful_updates']==counter['count'] and ck['cfg']==cfg and ck['initializer_sha256']==INIT_SHA,'Serialized identity differs')
                require(set(ck['roots'])==set(trainer.roots()),'Missing serialized root')
                for n,m in trainer.roots().items():
                    m.load_state_dict(ck['roots'][n],strict=True)
                    require(state_dict_sha256(m.state_dict())==before[n],'Strict readback differs: '+n)
                del ck
                return before
            if phase=='initial':
                dump_new(output/'result.json',dict(status='PASS_COLD_INITIALIZATION',root_sha256=initial_roots,optimizer_lrs=[g['lr'] for g in trainer.optimizer.param_groups],scaler=trainer.scaler.state_dict(),successful_updates=0,development_evaluations=0,initializer_sha256=INIT_SHA))
            elif phase=='bridge':
                batches=raw_groups(0)
                step(next(batches));step(next(batches))
                third=next(batches)
                before=signature([{n:m.state_dict() for n,m in trainer.roots().items()},trainer.optimizer.state_dict()])
                scale=float(trainer.scaler.get_scale());step(third,overflow=True)
                require(signature([{n:m.state_dict() for n,m in trainer.roots().items()},trainer.optimizer.state_dict()])==before,'Overflow changed weights/optimizer')
                require(float(trainer.scaler.get_scale())==scale/2 and counter['count']==2,'Overflow scaler/budget differs')
                del third,batches
                with original_lock():
                    capacity_snapshot(config=prefix_cfg)
                    with io.BytesIO() as stream:
                        torch.save(payload(-1),stream)
                        size=stream.getbuffer().nbytes;require(size<=2*1024**3,'Memory checkpoint exceeds reserved final bound')
                        stream.seek(0);roundtrip(stream)
                result=dict(status='PASS_FINITE_BRIDGE',at=stamp(),group=group,successful_updates=2,
                    optimizer_successful_calls=counter['count'],attempted_updates=attempts,artificial_overflow_attempts=1,
                    development_evaluations=0,initializer_sha256=INIT_SHA,initial_roots=initial_roots,
                    overflow_model_optimizer_unchanged=True,strict_memory_roundtrip=True,serialized_bytes=size,
                    temporary_labeled_exposures=exposures,temporary_unlabeled_exposures=unlab_exposures,
                    elapsed_seconds=time.monotonic()-started,own_gpu_fds=own_gpu0_fds(),
                    evidence_sha256={n:sha(output/n) for n in ('steps.jsonl','pairs.jsonl','attempts.jsonl','started.json')})
                dump_new(output/'result.json',result)
            else:
                for epoch in range(38):
                    n=0
                    for items in raw_groups(epoch):step(items);n+=1
                    require(n==18 and counter['count']==(epoch+1)*18,'Fixed successful-update budget violated')
                    completed_epochs=epoch+1
                    require(all(v==completed_epochs*9 for v in exposures.values()) and all(v==completed_epochs for v in unlab_exposures.values()),'Exposure imbalance')
                    append('epochs.jsonl',dict(at=stamp(),completed_epochs=completed_epochs,successful_updates=counter['count'],
                        labeled_exposures=exposures,unlabeled_exposures=unlab_exposures,development_evaluations=0))
                    own_gpu0_fds()
                require(counter['count']==684 and attempts==684 and len(input_pairs)==3420,'Final budget incomplete')
                target=STORE/group/'final.pth';partial=Path(str(target)+'.partial');h=infra()
                with original_lock():
                    capacity=capacity_snapshot(config=prefix_cfg);h.no_symlink(target);h.no_symlink(partial)
                    require(not os.path.lexists(target) and not os.path.lexists(partial),'Existing final/partial; no overwrite')
                    with partial.open('xb') as f:
                        torch.save(payload(37),h.BoundedWriter(f,2*1024**3));f.flush();os.fsync(f.fileno())
                    digest=sha(partial);size=partial.stat().st_size
                    h.publish_no_replace(partial,target);h.fsync_directory(target.parent)
                    readback=roundtrip(target);require(sha(target)==digest,'Checkpoint changed during strict readback')
                dump_new(output/'final_readback.json',dict(at=stamp(),path=str(target),sha256=digest,bytes=size,roots=readback,strict=True,capacity=capacity))
                # First and only development entry, after fixed budget and save/readback.
                dump_new(output/'evaluation_intent.json',dict(at=stamp(),checkpoint_sha256=digest,successful_updates=684,epoch=37,count=25))
                from train_by5 import make_strict_dataset
                from expes.common.seg_metrics import aggregate_metrics,binary_case_metrics,logits_to_binary
                ds=make_strict_dataset(data_cfg,prep['case_ids']['test'],'test')
                loader=torch.utils.data.DataLoader(ds,batch_size=1,shuffle=False,num_workers=0)
                for m in trainer.roots().values():m.eval()
                metrics=[]
                with torch.no_grad():
                    for i,batch in enumerate(loader):
                        cid=batch['case_id'][0];require(cid==prep['case_ids']['test'][i],'Development actual case differs')
                        image,label=trainer._prepare_batch(batch)
                        pred=logits_to_binary(trainer._forward_main_logits(trainer.student,image))[0]
                        gt=(label>0).squeeze(1).cpu().numpy().astype(bool)[0]
                        value=binary_case_metrics(pred,gt,spacing=(1.,1.,1.))
                        require(all(math.isfinite(float(x)) for x in value.values()),'Nonfinite development metric')
                        append('final_cases.jsonl',dict(at=stamp(),case_id=cid,metrics=value,checkpoint_sha256=digest,
                            foreground_voxels=int(gt.sum()),predicted_foreground_voxels=int(pred.sum())))
                        metrics.append(value)
                require(len(metrics)==25,'Incomplete development coverage')
                require(sha(target)==digest and all(state_dict_sha256(m.state_dict())==readback[n]
                    for n,m in trainer.roots().items()),'Development evaluation changed checkpoint/model state')
                result=dict(status='COMPLETE_FIXED_FINAL',at=stamp(),group=group,successful_updates=684,attempted_updates=684,
                    completed_epochs=38,labeled_exposures=exposures,unlabeled_exposures=unlab_exposures,development_evaluations=1,
                    metrics=aggregate_metrics(metrics),initializer_sha256=INIT_SHA,checkpoint_sha256=digest,
                    checkpoint=str(target),selection='native_student_epoch37_fixed_final',seed=2027,
                    gt_roi_used=True,independent_test=False,distance_unit='voxel_not_mm',
                    elapsed_seconds=time.monotonic()-started,evidence_sha256={n:sha(output/n) for n in
                        ('started.json','attempts.jsonl','pairs.jsonl','steps.jsonl','epochs.jsonl','final_readback.json','evaluation_intent.json','final_cases.jsonl')})
                dump_new(output/'result.json',result)
    except BaseException:
        output.mkdir(parents=True,exist_ok=True)
        dump_new(output/'failure.json',dict(at=stamp(),group=group,phase=phase,traceback=traceback.format_exc(),
            optimizer_successful_calls=counter['count'],matrix_updates=getattr(trainer,'matrix_updates',None),
            attempted_updates=attempts,completed_epochs=completed_epochs,labeled_exposures=exposures,
            unlabeled_exposures=unlab_exposures,automatic_retry=False,automatic_resume=False,
            partial_development=(output/'evaluation_intent.json').exists()))
        raise
