"""Unintegrated full-state mixin; qualification/admission/storage still required."""
import copy
from dualteacher_budget import counts,require
from dualteacher_state import FORMAT,role_window,validate_payload


class RecoveryMixin:
    def recovery_models(self):
        return dict(student=self.core.model,ema=self.core.ema.model,op=self.core.op,criterion=self.core.bc,sam=self.core.sam)

    def immutable_native_config(self):
        scalars={k:v for k,v in self.core.bc.__dict__.items() if not k.startswith('_') and k!='training' and type(v) in (bool,int,float,str,type(None))}
        return copy.deepcopy(dict(config=self.core.config,ema_alpha=self.core.ema.alpha,criterion_scalars=scalars,
                                  data_proxy=vars(self.core.data_proxy),extraction=self.core.extraction))

    def check_boundary(self):
        from dualteacher_numerics import snapshot
        snapshot(self.torch)
        require(self.initialized and not self.test_started and not self.reloaded_sha,'Not a training boundary')
        start,stop=role_window(self.recipe,self.execution_role)
        require(start<=self.position<=stop and self.core.step_count==self.position and self.core.total==self.recipe['original_clock_iterations'],'Native cursor/clock differs')
        require(self.actual_pairs==self.pairs[2*start:2*self.position] and len(self.native_rows)==self.position-start,'Actual input/native rows differ')
        require(self.core.ema.step==self.position-start,'EMA step differs from actual executed updates')
        for optimizer,model in ((self.core.optimizer,self.core.model),(self.core.optimizer_op,self.core.op)):
            actual=[id(p) for g in optimizer.param_groups for p in g['params']]
            require(actual==[id(p) for p in model.parameters()],'Native optimizer membership/order changed')
        for model in (self.core.sam,self.core.ema.model):
            require(all(not p.requires_grad and p.grad is None for p in model.parameters()),'Frozen teacher/foundation gradient')
        require(not self.core.sam.training,'SAM left eval')

    def capture_recovery(self):
        from recovery_state import cpu,rng
        from dualteacher_numerics import snapshot,state_layout
        self.check_boundary();models=self.recovery_models();start,stop=role_window(self.recipe,self.execution_role)
        require(self.signature(self.core.sam)==self.frozen_sam_signature,'Frozen Turbo tensor signature changed')
        state=dict(format=FORMAT,contract_sha256=self.contract_identity,execution_role=self.execution_role,
            recipe=copy.deepcopy(self.recipe),core_state=cpu(self.core.state_dict()),position=self.position,primary_position=2*self.position,
            original_clock_iterations=self.core.total,counts=counts(self.position-start),actual_pairs=copy.deepcopy(self.actual_pairs),native_rows=copy.deepcopy(self.native_rows),
            cycles_completed=2*self.position//self.recipe['primary_pairs_per_cycle'] if self.execution_role=='formal' else 0,
            foundation=copy.deepcopy(self.contract['foundation']),frozen_sam_signature=self.frozen_sam_signature,initial_signatures=copy.deepcopy(self.initial),
            module_modes={n:{k:m.training for k,m in v.named_modules()} for n,v in models.items()},
            requires_grad={n:{k:p.requires_grad for k,p in v.named_parameters()} for n,v in models.items()},
            parameter_gradients={n:{k:cpu(p.grad) for k,p in v.named_parameters()} for n,v in models.items()},
            instrumentation=copy.deepcopy(dict(forward_cases=self.core.forward_cases,student_events=self.core.audit_student.events,op_events=self.core.audit_op.events)),
            immutable_native_config=self.immutable_native_config(),rng=rng(),
            numerical_runtime=snapshot(self.torch),tensor_layout=state_layout(self.core,models,self.torch))
        validate_payload(state,self.contract_identity,self.recipe,self.execution_role,self.pairs)
        return state

    def restore_recovery(self,state):
        from recovery_state import restore_rng,tree_hash
        from dualteacher_numerics import snapshot
        validate_payload(state,self.contract_identity,self.recipe,self.execution_role,self.pairs)
        require(self.initialized and self.position==self.core.step_count==self.core.ema.step==0 and not self.actual_pairs and not self.native_rows and not self.test_started and not self.reloaded_sha,'Restore only into fresh native backend')
        require(state['initial_signatures']==self.initial and state['foundation']==self.contract['foundation'],'Native initialization/foundation differs')
        require(state['immutable_native_config']==self.immutable_native_config(),'Native immutable config changed')
        require(self.signature(self.core.sam)==state['frozen_sam_signature']==self.frozen_sam_signature,'Turbo full signature differs')
        require(state['numerical_runtime']==snapshot(self.torch),'Saved numerical policy differs')
        self.core.load_state_dict(state['core_state'])
        for name,model in self.recovery_models().items():
            require({k:p.requires_grad for k,p in model.named_parameters()}==state['requires_grad'][name],'Native flags differ')
            modules=dict(model.named_modules());require(set(modules)==set(state['module_modes'][name]),'Mode closure differs')
            for key,module in modules.items():module.training=state['module_modes'][name][key]
            for key,p in model.named_parameters():
                value=state['parameter_gradients'][name][key]
                if value is None:p.grad=None
                else:
                    require(value.shape==p.shape and value.dtype==p.dtype and bool(self.torch.isfinite(value).all()),'Gradient buffer mismatch')
                    p.grad=value.to(device=p.device).clone()
        self.position=state['position'];self.actual_pairs=copy.deepcopy(state['actual_pairs']);self.native_rows=copy.deepcopy(state['native_rows'])
        self.core.forward_cases=copy.deepcopy(state['instrumentation']['forward_cases'])
        self.core.audit_student.events=copy.deepcopy(state['instrumentation']['student_events'])
        self.core.audit_op.events=copy.deepcopy(state['instrumentation']['op_events'])
        restore_rng(state['rng']);self.check_boundary()
        expected={k:v for k,v in state.items() if k!='ledger'}
        require(tree_hash(self.capture_recovery())==tree_hash(expected),'Complete recovered value differs')
