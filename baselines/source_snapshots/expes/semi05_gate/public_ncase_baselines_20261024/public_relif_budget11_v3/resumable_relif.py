"""ReLiF native-backend integration draft. No launcher or implicit model loading.

The protocol must independently bind the complete contract, lease, patient
files, original native sources and shared resource gate before prepare_initial.
This module deliberately cannot supply or bypass those protocol proofs.
"""
import copy
import os
import random
import sys
import time
from pathlib import Path

from relif_recovery import RecoveryMixin
from relif_state import accept, counts, require, role_window, validate_payload


def make_backend(native, protocol):
    """Use the original transform/consume_pair/test implementations unchanged."""
    class Backend(RecoveryMixin, native.ReLiFBackend):
        def __init__(self, contract, output, execution_role='formal'):
            super().__init__(contract, output, protocol.verify)
            self.c = contract
            self.execution_role = execution_role
            self.binding = protocol.binding(contract)
            self.recipe = self.binding.recipe
            self.splits = self.binding.splits
            self.budget = protocol.clock_budget(contract['labeled_cases'])
            self.pairs = native.schedule(self.budget, self.splits)
            require(len(self.pairs) == self.recipe['original_clock_iterations'],
                    'Native full pair sequence was shortened')
            self.contract_identity = protocol.contract_sha(contract)
            self.signature = native.signature
            self.initialized = False
            self.allow_wait = False
            self.pending_row = None
            self.failed_update = False
            self.progress = counts(0)
            self.role_start, self.role_stop = role_window(self.recipe, execution_role)

        def gate(self):
            while True:
                try:
                    result = protocol.capacity(self.c)
                    if self.allow_wait:
                        protocol.resource_status(self.c, dict(state='PASS', at=time.time(), counts=self.progress))
                    return result
                except (ValueError, RuntimeError, AssertionError) as error:
                    if not self.allow_wait or not str(error).startswith((
                        'Insufficient filesystem capacity:', 'Insufficient RAM including cgroup ancestors')):
                        raise
                    protocol.resource_status(self.c, dict(state='WAIT_CAPACITY', at=time.time(),
                                                         counts=self.progress, error=str(error)))
                    time.sleep(60)

        def io_with_wait(self, operation):
            while True:
                try:
                    return operation()
                except (ValueError, RuntimeError, AssertionError) as error:
                    if not self.allow_wait or not str(error).startswith((
                        'Insufficient filesystem capacity:', 'Insufficient RAM including cgroup ancestors')):
                        raise
                    self.gate()

        def prepare_initial(self, for_restore=False):
            require(not self.initialized and self.core is None, 'Preparation is single-use')
            protocol.verify(self.c)
            protocol.assert_worker_lease(self.c)
            require(os.environ.get('CUDA_VISIBLE_DEVICES') == protocol.UUID,
                    'Only the bound physical GPU1 UUID is allowed')
            capacity = self.gate()
            # Resolve project imports before model construction and well before
            # a terminal test intent. Original files remain SHA-bound by verify.
            sys.path.insert(0, str(protocol.ROOT))
            r = self.c['relif_recipe']
            norm = r['preprocessing']['source_evidence']['shared_normalization']['path']
            self.normalize_batch = native.load_file('_public_relif_normalization', norm).normalize_batch
            self.metrics = native.load_file('_public_relif_metrics', protocol.ROOT/'expes/common/seg_metrics.py')
            import numpy as np
            import torch
            import torchio as tio
            from relif_runtime import build_core
            self.torch, self.np = torch, np
            require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'GPU mapping differs')
            torch.set_num_threads(2)
            random.seed(2027)
            np.random.seed(2027)
            torch.manual_seed(2027)
            torch.cuda.manual_seed_all(2027)
            self.device = torch.device('cuda:0')
            self.data = native.BoundCases('Public433', self.splits, training_flip=False,
                                         file_bindings=self.c['case_files'])
            with protocol.resource(self.c).original_lock():
                protocol.capacity_locked(self.c)
                self.core = build_core(r['official_code_root'], r['original_sam'],
                                       self.recipe['original_clock_iterations'], self.device)
            self.binding.require_core_total(self.core.total)
            require(self.core.step_count == 0 and not self.core.features, 'Native core is not fresh')
            self.initial = dict(student=self.signature(self.core.model),
                                projection=self.signature(self.core.projection))
            self.frozen_prior_signature = self.signature(self.core.prior.model)
            self.norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
            self.geometry = native.load_geometry(r['preprocessing'], torch, np)
            tensor_bytes = sum(t.numel()*t.element_size() for model in (self.core.model, self.core.projection)
                               for t in model.state_dict().values())
            require(tensor_bytes < self.c['checkpoint_bound']-1024**2, 'Terminal reservation too small')
            self.initialized = True
            # Only the explicitly labelled engineering LARC window skips the
            # clock. Its skipped warmup is never counted as executed work.
            if not for_restore:
                self.position = self.role_start
                self.core.step_count = self.role_start
                self.check_boundary()
            return dict(source_pins_verified=True, input_pins_verified=True, initialization_verified=True,
                resource_gate_passed=True, test_unopened=True, capacity=capacity,
                native_terminal_tensor_bytes=tensor_bytes, initial_state_signatures=self.initial,
                frozen_prior_signature=self.frozen_prior_signature, execution_role=self.execution_role,
                original_clock_iterations=self.core.total, cutoff_recipe=self.recipe,
                actual_prefix_optimization_updates=0, skipped_engineering_clock_updates_counted=False)

        def check_boundary(self):
            require(not self.failed_update and self.pending_row is None,
                    'Failed update or non-durable receipt cannot become a recovery boundary')
            super().check_boundary()
            require(self.progress == counts(self.position-self.role_start), 'Backend counter differs')

        def train_iteration(self, index):
            self.check_boundary()
            require(type(index) is int and index == self.position < self.role_stop, 'Wrong native iteration')
            self.gate()  # Exact capacity failures occur before any native update.
            try:
                row = self.consume_pair(index)  # Unchanged original scientific implementation.
                proposed = accept(self.progress, row, self.recipe, self.execution_role,
                                  self.binding.pair(index, self.execution_role), self.original_clock(index))
                require(self.core.step_count == index+1 and not self.core.features,
                        'Native update did not reach a complete boundary')
                self.actual_rows.append(copy.deepcopy(row['actual_pair']))
                self.position += 1
                self.progress = proposed
                self.pending_row = dict(step=copy.deepcopy(row), cumulative=copy.deepcopy(proposed))
                return row
            except BaseException:
                self.failed_update = True
                raise

        def append_step(self, row):
            require(self.pending_row is not None and row == self.pending_row,
                    'Durable receipt does not match the completed native update')
            # Engineering paths are separate; the protocol must reject any
            # engineering receipt or payload addressed to formal files.
            self.io_with_wait(lambda: protocol.append_step(self.c, self.output, row, self.execution_role))
            self.pending_row = None
            self.check_boundary()

        def original_clock(self, index):
            return self.binding.original_clock(index, self.execution_role)

        def restore(self, state):
            require(self.initialized and self.position == 0 and self.core.step_count == 0,
                    'Restore requires a separately prepared fresh native core')
            require(not self.pending_row and not self.failed_update, 'Unresolved update before restore')
            # RecoveryMixin recaptures state during its exact-value check.
            # Its counter invariant must already refer to the restored payload.
            validate_payload(state, self.contract_identity, self.recipe, self.execution_role, self.pairs)
            self.progress = copy.deepcopy(state['counts'])
            try:
                self.restore_recovery(state)
            except BaseException:
                self.failed_update = True
                raise
            self.check_boundary()

        def checkpoint(self, expected):
            require(self.execution_role == 'formal' and expected == self.progress, 'Wrong checkpoint role/counts')
            state = self.capture_recovery()
            state['ledger'] = protocol.ledger_pin(self.c)
            return self.io_with_wait(lambda: protocol.save_checkpoint(self.c, state, self.torch))

        def prepare_resume(self):
            require(self.execution_role == 'formal', 'Formal resume cannot load engineering state')
            prepared = self.prepare_initial(for_restore=True)
            commit, state, consumed = protocol.load_reviewed_checkpoint(self.c, self.torch)
            require(consumed.get('independent_resume_review_consumed') is True, 'Resume review not consumed')
            self.restore(state)
            protocol.record_restoration(self.c, commit, state, self.progress)
            return dict(prepared, loaded_state_exact=True, ledger_prefix_verified=True,
                        independent_resume_review_consumed=True, counts=copy.deepcopy(self.progress))

        def verify_training_coverage(self, expected):
            self.check_boundary()
            require(self.execution_role == 'formal' and expected == self.progress
                    and self.position == self.recipe['ssl_iterations']
                    and self.actual_rows == self.pairs[:self.position], 'Incomplete fixed-cutoff training')
            cycle = self.recipe['primary_pairs_per_cycle']
            for epoch in range(11):
                rows = self.actual_rows[epoch*cycle:(epoch+1)*cycle]
                require(len(rows) == cycle and all(r['epoch'] == epoch for r in rows)
                        and {r['labeled'] for r in rows} == set(self.splits['train_lab'])
                        and {r['unlabeled'] for r in rows} == set(self.splits['train_unlab'])
                        and sum(r['unlabeled_replay'] for r in rows) == self.budget.replay_per_cycle,
                        'Actual reference-cycle coverage differs')
            protocol.verify_actual_ledger(self.c, self.actual_rows, self.progress)
            return dict(actual_case_coverage_verified=True, complete_reference_cycles=11,
                actual_training_cases=len(set(self.splits['train_lab']+self.splits['train_unlab'])),
                primary_labeled_inputs=self.position, primary_unlabeled_inputs=self.position,
                successful_optimizer_updates=self.position, student_case_forwards=4*self.position,
                foundation_case_forwards=2*self.position, sobf_views=2*self.position,
                teacher_case_forwards=0, prefix_training_updates=0, extra_donor_inputs=0)

        def save_terminal(self, expected):
            self.verify_training_coverage(expected)
            self.final_signatures = dict(student=self.signature(self.core.model),
                                         projection=self.signature(self.core.projection))
            require(self.final_signatures['student'] != self.initial['student'], 'Student did not change')
            from recovery_state import cpu
            payload = dict(format='public_relif_fixed11_terminal_v1', method='relif3d',
                execution_role='formal', contract_sha256=self.contract_identity, recipe=self.recipe,
                counts=expected, signatures=self.final_signatures,
                student=cpu(self.core.model.state_dict()), projection=cpu(self.core.projection.state_dict()),
                original_sam=self.c['relif_recipe']['original_sam'],
                preprocessing=self.c['relif_recipe']['preprocessing'], implementation=native.IMPLEMENTATION)
            return self.io_with_wait(lambda: protocol.save_terminal(self.c, payload, self.torch))

        def strict_reload(self, artifact):
            require(self.reloaded_sha is None and self.execution_role == 'formal', 'Wrong/repeated terminal reload')
            value = self.io_with_wait(lambda: protocol.load_terminal(self.c, artifact, self.torch))
            require(value['contract_sha256'] == self.contract_identity and value['recipe'] == self.recipe
                    and value['counts'] == self.progress and value['execution_role'] == 'formal'
                    and value['signatures'] == self.final_signatures
                    and value['implementation'] == native.IMPLEMENTATION
                    and value['original_sam'] == self.c['relif_recipe']['original_sam']
                    and value['preprocessing'] == self.c['relif_recipe']['preprocessing'], 'Terminal metadata changed')
            self.core.model.load_state_dict(value['student'], strict=True)
            self.core.projection.load_state_dict(value['projection'], strict=True)
            require(self.signature(self.core.model) == self.final_signatures['student']
                    and self.signature(self.core.projection) == self.final_signatures['projection'], 'Reload tensors differ')
            result = dict(strict=True, sha256=artifact['sha256'], native_student_identity_verified=True,
                          native_projection_identity_verified=True, signatures=self.final_signatures)
            self.record('relif_reload_verified.json', result)
            self.reloaded_sha = artifact['sha256']
            return result

        def verify_test_prerequisites(self):
            require(self.execution_role == 'formal', 'Engineering may not open test')
            protocol.verify_test_prerequisites(self.c)
            require(callable(self.metrics.logits_to_binary) and callable(self.metrics.binary_case_metrics),
                    'Native metrics not resolved')

        def test(self, artifact, permit):
            self.verify_test_prerequisites()
            protocol.guard_native_test_lineage(self.c)
            self.gate()
            self.record('native_test_started.json', dict(final_sha256=artifact['sha256'],
                        contract_sha256=self.contract_identity, test_evaluation=1))
            return super().test(artifact, permit)  # Unchanged original case inference and metrics.

        def guard_run_lineage(self, resume):
            return protocol.guard_run_lineage(self.c, resume)

        def guard_test_lineage(self):
            return protocol.guard_test_lineage(self.c)

        def resume_intent_name(self):
            return protocol.resume_intent_name(self.c)

        def record(self, name, value, consistent=False):
            return self.io_with_wait(lambda: protocol.record(self.c, self.output, name, value,
                                                            self.execution_role, consistent=consistent))

    return Backend
