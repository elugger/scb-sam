"""Isolated native backend draft; explicit protocol/admission still required.

Native transform, consume_batch, signatures, inference/test are inherited from
the exact BY backend. No module-level native/model imports or implicit launch.
"""
import contextlib,copy,io,os,sys,time,traceback,warnings
from pathlib import Path
from dualteacher_budget import counts,require
from dualteacher_state import accept,role_window,validate_payload
from dualteacher_recovery import RecoveryMixin
from dualteacher_native import load_public_core


def make_backend(native,protocol):
    class Backend(RecoveryMixin,native.DualteacherBackend):
        def __init__(self,contract,output,execution_role='formal'):
            super().__init__(contract,output,protocol.verify)
            self.c=contract;self.execution_role=execution_role
            self.binding=protocol.binding(contract);self.recipe=self.binding.recipe;self.splits=self.binding.splits
            self.budget=self.binding.clock_budget;self.pairs=self.binding.pairs
            self.contract_identity=protocol.contract_sha(contract);self.signature=native.signature
            self.initialized=False;self.allow_wait=False;self.pending_row=None;self.failed_update=False;self.progress=counts(0)
            self.role_start,self.role_stop=role_window(self.recipe,execution_role)
            protocol.output_check(contract,output,execution_role)

        def gate(self):
            while True:
                try:
                    result=protocol.capacity(self.c)
                    if self.allow_wait:protocol.resource_status(self.c,dict(state='PASS',at=time.time(),counts=self.progress))
                    return result
                except (ValueError,RuntimeError,AssertionError) as error:
                    if not self.allow_wait or not str(error).startswith(('Insufficient filesystem capacity:','Insufficient RAM including cgroup ancestors')):raise
                    protocol.resource_status(self.c,dict(state='WAIT_CAPACITY',at=time.time(),counts=self.progress,error=str(error)))
                    time.sleep(60)

        def io_with_wait(self,operation):
            while True:
                try:return operation()
                except (ValueError,RuntimeError,AssertionError) as error:
                    if not self.allow_wait or not str(error).startswith(('Insufficient filesystem capacity:','Insufficient RAM including cgroup ancestors')):raise
                    self.gate()

        def prepare(self,*args,**kwargs):raise ValueError('Old BY prepare is not a Public entry point')
        def qualify(self,*args,**kwargs):raise ValueError('Own complete-state Public qualification required')

        def prepare_initial(self,for_restore=False):
            require(not self.initialized and self.core is None,'Single-use native preparation')
            protocol.verify(self.c);protocol.assert_worker_lease(self.c)
            require(os.environ.get('CUDA_VISIBLE_DEVICES')==protocol.UUID,'Bound physical GPU1 UUID required')
            capacity=self.gate();protocol.verify_test_prerequisites(self.c)
            stdout,stderr=io.StringIO(),io.StringIO();caught=[]
            def diagnostics(failed):
                value=dict(stdout=stdout.getvalue(),stderr=stderr.getvalue(),failed=failed,
                    warnings=[dict(message=str(w.message),filename=w.filename,lineno=w.lineno,category=w.category.__name__) for w in caught])
                self.record('native_initialization_diagnostics.json',value)
                return value
            try:
                with warnings.catch_warnings(record=True) as caught,contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
                    warnings.simplefilter('always')
                    result=self._initialize_public(capacity,for_restore)
            except BaseException:
                diagnostics(True);self.record('native_initialization_failure_trace.json',dict(traceback=traceback.format_exc()))
                self.failed_update=True
                raise
            value=diagnostics(False);protocol.check_initialization_diagnostics(self.c,value)
            return dict(result,narrow_startup_diagnostics_verified=True)

        def _initialize_public(self,capacity,for_restore):
            # Same native deterministic flags/seed construction and transforms.
            os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8';sys.path.insert(0,str(protocol.ROOT))
            import torch,numpy as np
            self.torch,self.np=torch,np
            from dualteacher_numerics import install
            install(torch)
            require(torch.cuda.is_available() and torch.cuda.device_count()==1,'GPU mapping differs')
            torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.enabled=True
            self.device=torch.device('cuda:0');self.code=Path(self.c['official_code_root'])
            Core,adaptation=load_public_core(self.binding.native_path('dualteacher_core.py'))
            with protocol.resource(self.c).original_lock():
                protocol.capacity_locked(self.c)
                self.core=Core(self.code,self.c['foundation'],self.recipe['original_clock_iterations'],self.device)
            self.binding.require_core_total(self.core.total)
            require(self.core.step_count==self.core.ema.step==0,'Native core not fresh')
            self.initial=self.signatures();self.frozen_sam_signature=self.signature(self.core.sam)
            self.data=native.BoundCases('Public433',self.splits,training_flip=False,file_bindings=self.c['case_files'])
            d=self.core.native['dataset']
            self.transforms=[d.RandomRotFlip(),d.Normalise(),d.RandomBrightness((.75,1.25)),d.RandomCrop((96,96,96)),d.ToTensor()]
            self.normalise=d.Normalise()
            self.infer,self.inference_evidence=native.load_native_inference(self.code,self.c['native_validate_sha256'],torch,np)
            self.metrics=native.load_file('_public_dualteacher_shared_metrics',protocol.ROOT/'expes/common/seg_metrics.py')
            tensor_bytes=sum(t.numel()*t.element_size() for m in (self.core.model,self.core.ema.model,self.core.op,self.core.bc) for t in m.state_dict().values())
            require(tensor_bytes<self.c['checkpoint_bound']-1024**2,'Terminal exceeds reservation')
            self.initialized=True
            if not for_restore:
                self.position=self.role_start;self.core.step_count=self.role_start;self.check_boundary()
            return dict(source_pins_verified=True,input_pins_verified=True,initialization_verified=True,resource_gate_passed=True,
                capacity=capacity,test_unopened=True,native_terminal_tensor_bytes=tensor_bytes,initial_state_signatures=self.initial,
                frozen_sam_signature=self.frozen_sam_signature,execution_role=self.execution_role,original_clock_iterations=self.core.total,
                cutoff_recipe=self.recipe,actual_prefix_optimization_updates=0,skipped_engineering_clock_updates_counted=False,
                native_constructor_adaptation=adaptation,native_inference_evidence=self.inference_evidence,
                numerical_runtime=__import__('dualteacher_numerics').snapshot(torch))

        def check_boundary(self):
            require(not self.failed_update and self.pending_row is None,'Failed or nondurable update is not a recovery boundary')
            super().check_boundary()
            require(self.progress==counts(self.position-self.role_start),'Backend counters differ')

        def original_clock(self,index):return self.binding.original_clock(index,self.execution_role)

        def train_iteration(self,index):
            self.check_boundary();require(type(index) is int and index==self.position<self.role_stop,'Wrong native iteration')
            self.gate()
            try:
                row=self.consume_batch(2*index)
                proposed=accept(self.progress,row,self.recipe,self.execution_role,self.binding.pair_batch(index,self.execution_role),self.original_clock(index))
                require(self.core.step_count==index+1 and self.core.ema.step==self.progress['iterations']+1,'Incomplete native/EMA boundary')
                self.actual_pairs.extend(copy.deepcopy(row['actual_pairs']));self.native_rows.append(copy.deepcopy(row['native_receipt']))
                self.position+=1;self.progress=proposed;self.pending_row=dict(step=copy.deepcopy(row),cumulative=copy.deepcopy(proposed))
                return row
            except BaseException:
                self.failed_update=True
                raise

        def append_step(self,row):
            require(self.pending_row is not None and row==self.pending_row,'Receipt differs from actual native update')
            self.io_with_wait(lambda:protocol.append_step(self.c,self.output,row,self.execution_role))
            self.pending_row=None;self.check_boundary()

        def restore(self,state):
            require(self.initialized and self.position==self.core.step_count==self.core.ema.step==0 and not self.pending_row and not self.failed_update,'Fresh separate backend required for restoration')
            validate_payload(state,self.contract_identity,self.recipe,self.execution_role,self.pairs)
            self.progress=copy.deepcopy(state['counts'])
            try:self.restore_recovery(state)
            except BaseException:
                self.failed_update=True
                raise
            self.check_boundary()

        def checkpoint(self,expected):
            require(self.execution_role=='formal' and expected==self.progress,'Wrong checkpoint role/counts')
            state=self.capture_recovery();state['ledger']=protocol.ledger_pin(self.c)
            return self.io_with_wait(lambda:protocol.save_checkpoint(self.c,state,self.torch))

        def prepare_resume(self):
            require(self.execution_role=='formal','Engineering cannot resume formal state')
            prepared=self.prepare_initial(for_restore=True)
            commit,state,consumed=protocol.load_reviewed_checkpoint(self.c,self.torch)
            require(consumed.get('independent_resume_review_consumed') is True,'Independent resume review not consumed')
            self.restore(state);protocol.record_restoration(self.c,commit,state,self.progress)
            return dict(prepared,loaded_state_exact=True,ledger_prefix_verified=True,independent_resume_review_consumed=True,counts=copy.deepcopy(self.progress))

        def verify_training_coverage(self,expected=None):
            if expected is None:expected=self.progress
            # Native test rechecks coverage after strict terminal reload. Keep
            # recovery/training barriers strict; allow this read-only audit.
            require(self.initialized and not self.failed_update and self.pending_row is None and not self.test_started,'Incomplete or already tested terminal boundary')
            require(self.core.step_count==self.position and self.core.ema.step==self.position
                and self.core.total==self.recipe['original_clock_iterations'] and self.progress==counts(self.position),'Terminal cursor/counts differ')
            require(self.execution_role=='formal' and expected==self.progress and self.position==self.recipe['ssl_iterations']
                and self.actual_pairs==self.pairs[:2*self.position],'Incomplete fixed-cutoff training')
            cycle=self.recipe['primary_pairs_per_cycle']
            for epoch in range(11):
                rows=self.actual_pairs[epoch*cycle:(epoch+1)*cycle]
                require(len(rows)==cycle and all(r['epoch']==epoch for r in rows) and {r['labeled'] for r in rows}==set(self.splits['train_lab'])
                    and {r['unlabeled'] for r in rows}==set(self.splits['train_unlab'])
                    and sum(r['unlabeled_replay'] for r in rows)==self.budget.replay_per_cycle,'Actual reference coverage differs')
            require(sum(r['effective_student_updates'] for r in self.native_rows)==self.position
                and sum(r['effective_op_updates'] for r in self.native_rows)==self.position,'Effective native update ledger differs')
            protocol.verify_actual_ledger(self.c,self.actual_pairs,self.progress)
            return dict(actual_case_coverage_verified=True,complete_reference_cycles=11,actual_training_cases=346,
                primary_labeled_inputs=2*self.position,primary_unlabeled_inputs=2*self.position,successful_optimizer_updates=2*self.position,
                primary_student_optimizer_updates=self.position,auxiliary_op_optimizer_updates=self.position,
                **{k:sum(r[k] for r in self.native_rows) for k in ('student_case_forwards','teacher_case_forwards','foundation_case_forwards','op_case_forwards')},
                prefix_training_updates=0,extra_donor_inputs=0,rounding_overshoot_per_stream=self.recipe['overshoot_per_stream'])

        def save_terminal(self,expected):
            self.verify_training_coverage(expected);self.final_signatures=self.signatures()
            require(self.final_signatures['student']!=self.initial['student'],'Terminal student unchanged')
            from recovery_state import cpu
            models=dict(student=self.core.model,ema=self.core.ema.model,op=self.core.op,criterion=self.core.bc)
            payload=dict(format='public_dualteacher_fixed11_terminal_v1',method='dualteacher_sam',execution_role='formal',
                selection='fixed_terminal_EMA_plus_OP',contract_sha256=self.contract_identity,recipe=self.recipe,counts=expected,
                signatures=self.final_signatures,foundation=self.c['foundation'],native_recipe=self.c['dualteacher_recipe'],
                **{k:cpu(m.state_dict()) for k,m in models.items()})
            return self.io_with_wait(lambda:protocol.save_terminal(self.c,payload,self.torch))

        def strict_reload(self,artifact):
            require(self.reloaded_sha is None and self.execution_role=='formal','Wrong or repeated reload')
            state=self.io_with_wait(lambda:protocol.load_terminal(self.c,artifact,self.torch))
            require(state['format']=='public_dualteacher_fixed11_terminal_v1' and state['method']=='dualteacher_sam'
                and state['execution_role']=='formal' and state['selection']=='fixed_terminal_EMA_plus_OP'
                and state['contract_sha256']==self.contract_identity and state['recipe']==self.recipe and state['counts']==self.progress
                and state['signatures']==self.final_signatures and state['foundation']==self.c['foundation']
                and state['native_recipe']==self.c['dualteacher_recipe'],'Terminal metadata differs')
            for name,m in [('student',self.core.model),('ema',self.core.ema.model),('op',self.core.op),('criterion',self.core.bc)]:m.load_state_dict(state[name],strict=True)
            require(self.signatures()==self.final_signatures,'Strict composite tensor signature mismatch')
            proof=dict(strict=True,sha256=artifact['sha256'],native_student_identity_verified=True,native_ema_op_identity_verified=True,signatures=self.final_signatures)
            self.record('dualteacher_reload_verified.json',proof);self.reloaded_sha=artifact['sha256'];return proof

        def verify_test_prerequisites(self):
            require(self.execution_role=='formal','Engineering cannot open test');protocol.verify_test_prerequisites(self.c)
            require(callable(self.infer) and callable(self.metrics.binary_case_metrics),'Native test prerequisites missing')

        def test(self,artifact,permit):
            self.verify_test_prerequisites();protocol.guard_native_test_lineage(self.c);self.gate()
            self.record('native_test_started.json',dict(final_sha256=artifact['sha256'],contract_sha256=self.contract_identity,test_evaluation=1))
            result=super().test(artifact,permit)
            return dict(result,distance_unit='voxel')

        def guard_run_lineage(self,resume):return protocol.guard_run_lineage(self.c,resume)
        def guard_test_lineage(self):return protocol.guard_test_lineage(self.c)
        def resume_intent_name(self):return protocol.resume_intent_name(self.c)
        def record(self,name,value,consistent=False):
            return self.io_with_wait(lambda:protocol.record(self.c,self.output,name,value,self.execution_role,consistent=consistent))
    return Backend
