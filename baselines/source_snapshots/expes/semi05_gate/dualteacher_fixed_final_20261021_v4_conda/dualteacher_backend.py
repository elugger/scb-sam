"""Unscheduled BY adapter draft. Qualifier implementation not yet executed; controller/verifier pending.

No CLI and no dispatch. No admission until own actual GPU checks, source review
and a complete independent controller/verifier have passed.
"""
import contextlib,copy,io,json,math,os,random,traceback,warnings
from pathlib import Path
from budget_protocol import BUDGETS
from fixed_endpoint import exclusive_json
from runtime_data import ROOT,BoundCases,bound_splits,bind_case_files,schedule
from runtime_resources import RAM_REQUIRED,REGISTRY,BoundedWriter,resource_backend,save_terminal,sha
from dualteacher_util import signature,same_state,load_file
from dualteacher_protocol import METHOD,FINAL_BOUND,IMPLEMENTATION,recipe,load_native_inference,need

class DualteacherBackend:
    def __init__(self,contract,output,verify_contract):
        self.contract,self.output,self.verify_contract=contract,Path(output),verify_contract
        self.target=Path(contract['target']);self.position=0;self.actual_pairs=[];self.cycles=[]
        self.native_rows=[];self.core=None;self.test_started=False;self.reloaded_sha=None

    def signatures(self):
        return {n:signature(m) for n,m in [('student',self.core.model),('ema',self.core.ema.model),('op',self.core.op),('criterion',self.core.bc)]}

    def prepare(self,endpoint):
        output,error=io.StringIO(),io.StringIO()
        def record(caught,failed):
            value=dict(stdout=output.getvalue(),stderr=error.getvalue(),failed=failed,
                warnings=[dict(message=str(w.message),filename=w.filename,lineno=w.lineno,category=w.category.__name__) for w in caught])
            exclusive_json(self.output/'native_initialization_diagnostics.json',value)
            return value
        try:
            with warnings.catch_warnings(record=True) as caught,contextlib.redirect_stdout(output),contextlib.redirect_stderr(error):
                warnings.simplefilter('always')
                prepared=self._prepare_native(endpoint)
        except BaseException:
            record(caught,True)
            exclusive_json(self.output/'native_initialization_failure_trace.json',dict(traceback=traceback.format_exc()))
            raise
        value=record(caught,False)
        need(value['stderr']=='','Unexpected initialization stderr, preserve all diagnostics')
        need(value['stdout']=='creating model SAM-Med3D\n','Unexpected factory stdout, preserve')
        expected=dict(message='Importing from timm.models.layers is deprecated, please import via timm.layers',
            category='FutureWarning',filename='__RECORDED_HOME__/anaconda3/envs/sam3d/lib/python3.10/site-packages/timm/models/layers/__init__.py',lineno=48)
        need(len(value['warnings'])<=1 and all(w==expected for w in value['warnings']),'Unreviewed initialization warning, preserve')
        need(any(x['path']==expected['filename'] for x in self.contract['environment']['files']),'Warning source not bound')
        prepared['initialization_diagnostics_sha256']=sha(self.output/'native_initialization_diagnostics.json')
        prepared['narrow_startup_diagnostics_verified']=True
        return prepared

    def _prepare_native(self,endpoint):
        c=self.contract;self.verify_contract(c)
        need(c['method']==METHOD and c['dataset']=='BY' and c['ratio'] in (5,10) and c['gpu']==1 and c['seed']==2027 and c['fold']==2,'Closed BY identity differs')
        self.budget=next(b for b in BUDGETS if b.dataset=='BY' and b.ratio==c['ratio'])
        need(endpoint.method==METHOD and endpoint.labeled_inputs==endpoint.unlabeled_inputs==self.budget.primary_pairs and endpoint.optimizer_updates_per_iteration==1,'Primary endpoint differs')
        self.splits=bound_splits('BY',c['ratio']);need(self.splits==c['splits'] and bind_case_files('BY',self.splits)==c['case_files'],'Split/case identity differs')
        self.pairs=schedule(self.budget,self.splits)
        self.code=Path(c['official_code_root']);self.plan=c['dualteacher_recipe']
        need(self.plan==recipe(self.code,c['foundation'],c['ratio'],c['native_validate_sha256']),'Prospective recipe changed')
        rel=Path(METHOD)/'BY'/('semi%02d'%c['ratio'])/'fold2'/c['tag']
        job=ROOT/'expes/work_dir/semi05_gate/MAIN_TABLE_FIXED_FINAL_20261019'/rel
        need(self.target==ROOT/'expes/artifacts/FCD_A_MATRIX_20261006/MAIN_TABLE_FIXED_FINAL_20261019'/rel/'final.pth','Wrong own terminal path')
        need(self.output in (job/'formal',job/'qualification_backend') and Path(c['output'])==job/'formal','Wrong output')
        need([r['bound'] for r in json.loads(REGISTRY.read_text()) if r['path']==str(self.target)]==[FINAL_BOUND],'Exact terminal target not registered')
        need(os.environ.get('CUDA_VISIBLE_DEVICES')=='1','GPU1 visibility required')
        guard=resource_backend(self.target,FINAL_BOUND)
        with guard.original_lock():
            capacity=guard.capacity_snapshot(RAM_REQUIRED)
            os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'
            import torch,numpy as np
            from dualteacher_core import DualteacherCore
            torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            torch.backends.cudnn.enabled=True;self.torch,self.np=torch,np;self.device=torch.device('cuda:0')
            self.core=DualteacherCore(self.code,c['foundation'],self.budget.primary_pairs//2,self.device)
            self.initial=self.signatures()
            tensor_bytes=sum(t.numel()*t.element_size() for m in (self.core.model,self.core.ema.model,self.core.op,self.core.bc) for t in m.state_dict().values())
            need(tensor_bytes<FINAL_BOUND-1024**2,'Terminal exceeds reservation')
        self.data=BoundCases('BY',self.splits,training_flip=False,file_bindings=c['case_files'])
        d=self.core.native['dataset']
        self.transforms=[d.RandomRotFlip(),d.Normalise(),d.RandomBrightness((.75,1.25)),d.RandomCrop((96,96,96)),d.ToTensor()]
        self.normalise=d.Normalise()
        self.infer,self.inference_evidence=load_native_inference(self.code,c['native_validate_sha256'],torch,np)
        self.metrics=load_file('_dualteacher_shared_metrics',ROOT/'expes/common/seg_metrics.py')
        return dict(source_pins_verified=True,input_pins_verified=True,initialization_verified=True,resource_gate_passed=True,
            test_unopened=True,capacity=capacity,initial_state_signatures=self.initial,native_terminal_tensor_bytes=tensor_bytes,
            initialization=self.plan['initialization'],prefix_training_updates=0,native_inputs_per_stream=2,
            primary_optimizer_updates_per_iteration=1,auxiliary_optimizer_accounting=self.plan['optimizer'],implementation=IMPLEMENTATION,
            native_inference_evidence=self.inference_evidence,own_GPU_qualification_claimed=False)

    def transform(self,sample,training):
        t=self.torch;need(sample['image'].shape==(1,128,128,128),'Shared ROI shape changed')
        need(sample['case_replaced'] is False and sample['unlab_gt_returned'] is False,'Case/GT boundary violated')
        target=sample['label']
        if sample['role']=='train_unlab':need(target is None,'Unlabeled GT crossed loader')
        else:need(target is not None,'Missing labeled/test target')
        placeholder=target if target is not None else t.zeros((128,128,128),dtype=t.long)
        value=dict(image=sample['image'][0].numpy(),label=placeholder.numpy())
        for transform in (self.transforms if training else [self.normalise]):value=transform(value)
        image=value['image'];label=value['label'] if target is not None else None
        if training:
            need(image.shape==(1,96,96,96) and bool(t.isfinite(image).all()),'Nonfinite native training crop')
        else:need(image.shape==(128,128,128) and bool(self.np.isfinite(image).all()),'Nonfinite native test normalization')
        return image,label

    def consume_batch(self,index):
        need(index%2==0 and self.core.step_count==index//2,'Native/primary input clocks differ')
        pairs=self.pairs[index:index+2];need(len(pairs)==2,'Native batch cannot be shortened')
        labeled=[];targets=[];unlabeled=[]
        for p in pairs:
            samples=[self.data.load(role,p[key]) for role,key in [('train_lab','labeled'),('train_unlab','unlabeled')]]
            for sample,role,key in zip(samples,('train_lab','train_unlab'),('labeled','unlabeled')):
                need(sample['case_id']==p[key] and sample['role']==role and sample['actual_index']==self.splits[role].index(p[key]),'Actual case substituted')
            a,y=self.transform(samples[0],True);b,no_target=self.transform(samples[1],True);need(no_target is None,'Unlabeled target reached loss')
            labeled.append(a);targets.append(y);unlabeled.append(b)
        native=self.core.update(self.torch.stack(labeled).to(self.device),self.torch.stack(targets).to(self.device),self.torch.stack(unlabeled).to(self.device),
            {key:[p[key] for p in pairs] for key in ('labeled','unlabeled')})
        self.torch.cuda.synchronize()
        need(native['primary_labeled']==native['primary_unlabeled']==2 and native['primary_optimizer_updates']==1,'Native batch/update receipt differs')
        losses={k:native[k] for k in ('loss','sup_loss','unsup_loss','bc_loss','sam_loss','op_loss')}
        gradient_norm=math.sqrt(sum(float(p.grad.detach().double().square().sum()) for m in (self.core.model,self.core.op) for p in m.parameters() if p.grad is not None))
        need(math.isfinite(gradient_norm) and gradient_norm>0,'Nonfinite native gradient norm')
        return dict(labeled_inputs=2,unlabeled_inputs=2,successful_optimizer_updates=1,finite_loss=True,finite_nonzero_gradients=True,
            teacher_no_grad=True,skip_count=0,actual_case_ids_verified=True,unlabeled_gt_in_loss=False,actual_pairs=pairs,
            completed_primary_pairs=index+2,losses=losses,gradient_norm=gradient_norm,native_receipt=native)

    def train_iteration(self,remaining):
        need(remaining==dict(labeled_inputs=self.budget.primary_pairs-self.position,unlabeled_inputs=self.budget.primary_pairs-self.position),'Remaining exposure differs')
        row=self.consume_batch(self.position);old=self.position;self.position+=2;self.actual_pairs.extend(row['actual_pairs']);self.native_rows.append(row['native_receipt'])
        size=self.budget.primary_pairs_per_cycle
        for epoch in range(old//size,self.position//size):
            chunk=self.actual_pairs[epoch*size:(epoch+1)*size]
            need(set(x['labeled'] for x in chunk)==set(self.splits['train_lab']) and set(x['unlabeled'] for x in chunk)==set(self.splits['train_unlab']) and sum(x['unlabeled_replay'] for x in chunk)==self.budget.replay_per_cycle,'Actual cycle coverage differs')
            self.cycles.append(dict(epoch=epoch,actual_case_coverage_verified=True,unlabeled_replay=self.budget.replay_per_cycle))
        return row

    def verify_training_coverage(self):
        n=self.budget.primary_pairs//2
        need(self.position==self.budget.primary_pairs and self.actual_pairs==self.pairs and len(self.cycles)==36 and len(self.native_rows)==self.core.step_count==n,'Incomplete fixed native exposure')
        need(sum(x['effective_student_updates'] for x in self.native_rows)==n and sum(x['effective_op_updates'] for x in self.native_rows)==n,'Native effective optimizer cost differs')
        return dict(actual_case_coverage_verified=True,cycles=self.cycles,primary_labeled_inputs=self.position,primary_unlabeled_inputs=self.position,
            primary_student_optimizer_updates=n,auxiliary_op_optimizer_updates=n,total_native_optimizer_updates=2*n,
            student_case_forwards=sum(x['student_case_forwards'] for x in self.native_rows),teacher_case_forwards=sum(x['teacher_case_forwards'] for x in self.native_rows),
            foundation_case_forwards=sum(x['foundation_case_forwards'] for x in self.native_rows),op_case_forwards=sum(x['op_case_forwards'] for x in self.native_rows),prefix_training_updates=0,extra_donor_inputs=0)

    def save_terminal(self,counts):
        self.verify_training_coverage();n=self.core.total
        need(counts==dict(labeled_inputs=2*n,unlabeled_inputs=2*n,iterations=n,successful_optimizer_updates=n),'Primary terminal counts differ')
        self.final_signatures=self.signatures();need(self.final_signatures['student']!=self.initial['student'],'Terminal student unchanged')
        payload=dict(format='dualteacher_fixed_terminal_v1',method=METHOD,selection='fixed_terminal_EMA_plus_OP',counts=counts,
            signatures=self.final_signatures,foundation=self.contract['foundation'],recipe=self.plan,
            student=self.core.model.state_dict(),ema=self.core.ema.model.state_dict(),op=self.core.op.state_dict(),criterion=self.core.bc.state_dict())
        return save_terminal(self.target,payload)

    def strict_reload(self,artifact):
        need(self.reloaded_sha is None and Path(artifact['path'])==self.target,'Reload target/repetition differs')
        with resource_backend(self.target,FINAL_BOUND).original_lock():
            resource_backend(self.target,FINAL_BOUND).capacity_snapshot(RAM_REQUIRED);need(sha(self.target)==artifact['sha256'],'Terminal changed before reload')
            state=self.torch.load(str(self.target),map_location=self.device)
            need(state['method']==METHOD and state['selection']=='fixed_terminal_EMA_plus_OP' and state['signatures']==self.final_signatures and state['foundation']==self.contract['foundation'] and state['recipe']==self.plan,'Reload metadata differs')
            for name,m in [('student',self.core.model),('ema',self.core.ema.model),('op',self.core.op),('criterion',self.core.bc)]:m.load_state_dict(state[name],strict=True)
            need(self.signatures()==self.final_signatures,'Strict composite terminal state differs')
        proof=dict(strict=True,sha256=artifact['sha256'],native_student_identity_verified=True,native_ema_op_identity_verified=True,signatures=self.final_signatures)
        exclusive_json(self.output/'dualteacher_reload_verified.json',proof);self.reloaded_sha=artifact['sha256'];return proof

    def test(self,artifact,permit):
        need(not self.test_started and self.reloaded_sha==artifact['sha256'],'Single terminal test/reload required')
        self.verify_training_coverage();p=json.loads(Path(permit).read_text())
        need(p['final_sha256']==artifact['sha256'] and p['test_case_ids']==self.splits['test'] and p['test_evaluation']==1,'Terminal permit differs')
        self.test_started=True;exclusive_json(self.output/'dualteacher_final_test_started.json',dict(final_sha256=artifact['sha256'],permit_sha256=sha(permit),test_evaluations=1,selection='fixed_terminal_EMA_plus_OP'))
        self.data.test_permit=permit;self.core.ema.model.eval();self.core.op.eval();rows=[]
        with (self.output/'final_test_cases.jsonl').open('x') as f:
            for index,case in enumerate(self.splits['test']):
                sample=self.data.load('test',case);need(sample['actual_index']==index,'Terminal case order differs')
                image,target=self.transform(sample,False);before=dict(self.core.forward_cases)
                prediction,_=self.infer(self.core.ema.model,self.core.op,image)
                need(self.core.forward_cases.get('ema',0)-before.get('ema',0)==27 and self.core.forward_cases.get('op',0)-before.get('op',0)==27,'Native terminal windows differ')
                row=dict(case_id=case,actual_index=index,native_windows=27,selection='EMA+OP',**self.metrics.binary_case_metrics(self.torch.from_numpy(prediction),self.torch.from_numpy(target),spacing=(1.,1.,1.)))
                need(all(math.isfinite(row[k]) and row[k]>=0 for k in ('dice','jaccard','hd95','asd')) and row['dice']<=1 and row['jaccard']<=1,'Invalid case metrics')
                f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno());rows.append(row)
        need(sha(self.target)==artifact['sha256'],'Terminal changed during test')
        return dict(case_ids=[r['case_id'] for r in rows],case_dice=[r['dice'] for r in rows],
            **{k:sum(r[k] for r in rows)/len(rows) for k in ('dice','jaccard','hd95','asd')},final_sha256=artifact['sha256'],test_evaluations=1,native_patch_forwards=27*len(rows),selection='fixed_terminal_EMA_plus_OP')

    def qualify(self,endpoint):
        preparation=self.prepare(endpoint)
        t,np=self.torch,self.np
        state=copy.deepcopy(self.core.state_dict())
        rng=(random.getstate(),np.random.get_state(),t.get_rng_state(),t.cuda.get_rng_state())
        frozen_signature=signature(self.core.sam)
        need(not any(p.requires_grad or p.grad is not None for p in self.core.sam.parameters()),'Frozen Turbo grad boundary differs')
        active=math.ceil(6*self.core.total/200)
        receipts=[];self.measured_extra_engineering_optimizer_updates=0
        self.measured_extra_native_optimizer_updates=0
        self.qualification_partial_native_cost=[]
        for index,phase in ((0,'native_warmup'),(active,'native_SAM_start')):
            # These are explicit engineering clock probes, not formal resume
            # or exposure. Every formal worker is freshly initialized.
            self.core.step_count=index
            try:
                row=self.consume_batch(2*index)
            finally:
                student=copy.deepcopy(self.core.audit_student.events)
                op=copy.deepcopy(self.core.audit_op.events)
                self.measured_extra_engineering_optimizer_updates+=sum(x['effective_update'] for x in student)
                self.measured_extra_native_optimizer_updates+=sum(x['effective_update'] for x in student+op)
                self.qualification_partial_native_cost.append(dict(phase=phase,native_index=index,student_events=student,op_events=op))
                exclusive_json(self.output/('qualification_cost_'+phase+'.json'),self.qualification_partial_native_cost[-1])
            row.update(qualification_phase=phase,qualification_native_index=index,formal_budget_inputs=0)
            receipts.append(row)
        nr=receipts[1]['native_receipt']
        need(nr['clock']['sam_enabled'] and nr['foundation_case_forwards']==2 and nr['sam_valid_cases']>0
             and nr['sam_loss_requires_grad'] and nr['sam_loss']>0,'Own actual native SAM distillation branch not witnessed; preserve failure')
        need(self.measured_extra_engineering_optimizer_updates==2 and self.measured_extra_native_optimizer_updates==4,'Native qualification optimizer cost differs')
        need(signature(self.core.sam)==frozen_signature and not any(p.grad is not None for p in self.core.sam.parameters()),'Frozen foundation changed')
        self.core.load_state_dict(state)
        random.setstate(rng[0]);np.random.set_state(rng[1]);t.set_rng_state(rng[2]);t.cuda.set_rng_state(rng[3])
        need(same_state(self.core.state_dict(),state,t) and self.signatures()==self.initial,'Complete native state restore differs')
        now=np.random.get_state()
        need(random.getstate()==rng[0] and now[0]==rng[1][0] and np.array_equal(now[1],rng[1][1]) and now[2:]==rng[1][2:]
             and t.equal(t.get_rng_state(),rng[2]) and t.equal(t.cuda.get_rng_state(),rng[3]),'Native RNG restore differs')
        need(self.position==0 and not self.actual_pairs and not self.cycles and not self.native_rows,'Qualification contaminated formal exposure ledger')
        probe=self.target.parent/'own_dualteacher_update0_probe.pth.partial'
        guard=resource_backend(self.target,FINAL_BOUND)
        with guard.original_lock():
            probe_capacity=guard.capacity_snapshot(RAM_REQUIRED)
            need(not probe.exists() and not any(x.is_symlink() for x in (probe,*probe.parents)),'Existing/symlink own probe')
            probe.parent.mkdir(parents=True,exist_ok=True)
            with probe.open('xb') as f:
                t.save(state,BoundedWriter(f,512*1024**2));f.flush();os.fsync(f.fileno())
            identity=(probe.stat().st_dev,probe.stat().st_ino,probe.stat().st_size,probe.stat().st_nlink)
            probe_sha=sha(probe);value=t.load(str(probe),map_location=self.device)
            self.core.load_state_dict(value)
            need(same_state(self.core.state_dict(),state,t) and self.signatures()==self.initial,'Own composite disk roundtrip differs')
            need(identity==(probe.stat().st_dev,probe.stat().st_ino,probe.stat().st_size,probe.stat().st_nlink) and identity[3]==1
                 and sha(probe)==probe_sha,'Own probe identity changed')
            probe.unlink() # Only this newly created, identity/SHA-verified own temporary file.
        return dict(status='QUALIFIED',preparation=preparation,actual_extra_engineering_updates=2,
            actual_extra_engineering_primary_inputs_per_stream=4,actual_extra_native_optimizer_updates=4,
            qualification_clock_probe_indices=[0,active],new_test_evaluations=0,formal_training_updates=0,
            restored_model_optimizer_rng=True,actual_disk_roundtrip=True,receipts=receipts,
            foundation_unchanged=True,foundation_signature=frozen_signature,native_state_signatures=self.initial,
            implementation=IMPLEMENTATION,sam_autodiff_repair_disclosed=True,
            probe_roundtrip=dict(path=str(probe),bytes=identity[2],sha256=probe_sha,bound=512*1024**2,
                deleted_exact_own_probe=True,original_lock_verified=True,capacity_before=probe_capacity))

    def close(self):
        if self.core is not None:self.core.close()
