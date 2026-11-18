"""Unintegrated ReLiF snapshot mixin; no dispatcher or implicit model load.

The original frozen SAM is reloaded from its original SHA and its full tensor
signature is checked. Student/projection/optimizer tensors are serialized.
prepare, lease/storage, source binding and qualification still need integration.
"""
import copy
from relif_state import FORMAT, counts, plan, role_window, validate_payload, require


class RecoveryMixin:
    def recovery_models(self):
        return dict(student=self.core.model, projection=self.core.projection, prior=self.core.prior.model)

    def check_boundary(self):
        require(self.initialized and not self.test_started and not self.reloaded_sha,
                'Recovery is only valid at a training boundary')
        require(not self.core.features, 'Live feature-hook buffer at boundary')
        start, stop = role_window(self.recipe, self.execution_role)
        require(self.core.total == self.recipe['original_clock_iterations'] and self.core.step_count == self.position,
                'Native/core cursor or clock changed')
        require(start <= self.position <= stop and len(self.actual_rows) == self.position-start,
                'Executed rows and native cursor differ')
        require(self.actual_rows == self.pairs[start:self.position], 'Input schedule changed')
        require(all(not p.requires_grad and p.grad is None for p in self.core.prior.model.parameters())
                and not self.core.prior.model.training, 'Frozen prior received gradients or left eval')
        from relif_components import verify_projection_optimizer
        verify_projection_optimizer(self.core.optimizer, self.core.projection)

    def capture_recovery(self):
        from recovery_state import cpu, rng
        self.check_boundary()
        models = self.recovery_models()
        require(self.signature(models['prior']) == self.frozen_prior_signature, 'Frozen original SAM tensors changed')
        start, stop = role_window(self.recipe, self.execution_role)
        state = dict(format=FORMAT, contract_sha256=self.contract_identity, execution_role=self.execution_role,
                     recipe=copy.deepcopy(self.recipe), core_state=cpu(self.core.state_dict()),
                     foundation=copy.deepcopy(self.core.prior.identity), frozen_prior_signature=self.frozen_prior_signature,
                     initial_signatures=copy.deepcopy(self.initial),
                     module_modes={n:{k:m.training for k,m in v.named_modules()} for n,v in models.items()},
                     requires_grad={n:{k:p.requires_grad for k,p in v.named_parameters()} for n,v in models.items()},
                     rng=rng(), position=self.position, actual_rows=copy.deepcopy(self.actual_rows),
                     counts=counts(self.position-start),
                     cycles_completed=self.position//self.recipe['primary_pairs_per_cycle'] if self.execution_role=='formal' else 0)
        validate_payload(state,self.contract_identity,self.recipe,self.execution_role,self.pairs)
        return state

    def restore_recovery(self, state):
        from recovery_state import restore_rng, tree_hash
        validate_payload(state,self.contract_identity,self.recipe,self.execution_role,self.pairs)
        # A freshly built core only: no rewind of a used model or native hook.
        require(self.initialized and self.position == 0 and self.core.step_count == 0
                and not self.actual_rows and not self.test_started and not self.reloaded_sha,
                'Restore requires fresh initial backend')
        require(state['initial_signatures'] == self.initial and state['foundation'] == self.core.prior.identity,
                'Recreated native initialization/foundation identity differs')
        require(self.signature(self.core.prior.model) == state['frozen_prior_signature'] == self.frozen_prior_signature,
                'Original frozen SAM was not reconstructed exactly')
        self.core.load_state_dict(state['core_state'])
        for name, model in self.recovery_models().items():
            require({k:p.requires_grad for k,p in model.named_parameters()} == state['requires_grad'][name],
                    'Native gradient flags differ')
            modules = dict(model.named_modules())
            require(set(modules) == set(state['module_modes'][name]), 'Native module mode keys differ')
            for k,m in modules.items():
                m.training = state['module_modes'][name][k]
            for p in model.parameters():
                p.grad = None
        self.position = state['position']
        self.actual_rows = copy.deepcopy(state['actual_rows'])
        self.core.features.clear()
        restore_rng(state['rng'])
        self.check_boundary()
        expected = {k:v for k,v in state.items() if k != 'ledger'}
        require(tree_hash(self.capture_recovery()) == tree_hash(expected), 'Full recovery value roundtrip differs')
