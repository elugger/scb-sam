"""Unqualified SFR adapter draft; no CLI, dispatcher or implicit model load.

The protocol service has not yet been integrated. Native update methods are
called unchanged; full-state storage and separate own GPU proofs remain required.
"""
import copy
import os
from pathlib import Path
import random
import time
from recovery_state import cpu, rng, restore_rng, tree_hash
from sfr_state import accept, initial, plan, prefix_inputs, ready_for_test, transition, validate


def make_backend(native, protocol):
    class Backend(native.SFRBackend):
        def __init__(self, contract, output, engineering=False):
            protocol.verify(contract)
            protocol.configure_native(native, contract)
            self.c = self.contract = contract
            self.output = Path(output)
            self.recipe = plan(contract['labeled_cases'])
            self.execution_role = 'engineering_two_phase_probe' if engineering else 'formal'
            if engineering:
                assert self.output == protocol.job(contract)/'qualification_backend'
                assert contract['engineering_probe_prefix_iterations'] == 4
                self.recipe.update(prefix_iterations=4, prefix_updates=4,
                                   prefix_labeled_inputs=prefix_inputs(4, contract['labeled_cases']))
            else:
                assert self.output == Path(contract['output'])
            self.budget = protocol.clock_budget(contract['labeled_cases'])
            self.plan = native.stage_plan('Public433', contract['labeled_cases'])
            assert self.plan['ssl_iterations'] == self.recipe['original_clock_ssl_iterations']
            if engineering:
                self.plan.update(prefix_iterations=4, prefix_optimizer_updates=4,
                                 prefix_labeled_inputs=self.recipe['prefix_labeled_inputs'])
            self.targets = contract['sfr_targets']
            self.target = Path(self.targets['final']['path'])
            self.splits = contract['splits']
            self.pairs = native.schedule(self.budget, self.splits)
            self.progress = initial()
            self.initialized = self.allow_wait = self.reload_verified = self.test_started = False
            self.prefix_artifact = self.final_artifact = None
            self.initial_student_signature = None
            self.initial_trainable = None
            self.cycle_receipts = []

        def _verify(self):
            protocol.verify(self.c)

        def gate(self):
            while True:
                try:
                    value = protocol.capacity(self.c)
                    if self.allow_wait:
                        protocol.resource_status(self.c, dict(state='PASS', at=time.time(), progress=self.progress))
                    return value
                except (ValueError, RuntimeError, AssertionError) as e:
                    if not self.allow_wait or not str(e).startswith(('Insufficient filesystem capacity:', 'Insufficient RAM including cgroup ancestors')):
                        raise
                    protocol.resource_status(self.c, dict(state='WAIT_CAPACITY', at=time.time(), progress=self.progress, error=str(e)))
                    time.sleep(60)

        def io_with_wait(self, action):
            while True:
                try:
                    return action()
                except (ValueError, RuntimeError, AssertionError) as e:
                    if not self.allow_wait or not str(e).startswith(('Insufficient filesystem capacity:', 'Insufficient RAM including cgroup ancestors')):
                        raise
                    # Only a capacity gate that fails before writing may wait.
                    # Unknown or partial-write failures always escape and preserve evidence.
                    self.gate()

        def prepare_initial(self):
            assert not self.initialized
            self._verify(); capacity = self.gate()
            protocol.assert_worker_lease(self.c)
            import numpy as np
            import torch
            self.np, self.torch = np, torch
            assert os.environ.get('CUDA_VISIBLE_DEVICES') == protocol.UUID
            assert torch.cuda.is_available() and torch.cuda.device_count() == 1
            self.device = torch.device('cuda:0')
            torch.set_num_threads(2)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            random.seed(2027); np.random.seed(2027); torch.manual_seed(2027); torch.cuda.manual_seed_all(2027)
            bound = native.BoundCases('Public433', self.splits, training_flip=False, file_bindings=self.c['case_files'])
            with protocol.resource(self.c).original_lock():
                protocol.capacity_locked(self.c)
                init = self.c['foundation']
                sam, self.foundation_receipt = native.build_bound_sam(self.c['official_code_root'], init['path'], init['sha256'], self.device)
            self.core = native.SFRTrainingCore(self.c['official_code_root'], sam, self.foundation_receipt,
                         'Public433', self.c['labeled_cases'], self.splits['train_lab'], self.pairs, self.device)
            assert self.core.plan['ssl_iterations'] == self.recipe['original_clock_ssl_iterations']
            if self.execution_role != 'formal':
                self.core.plan = copy.deepcopy(self.plan)
                self.core.prefix_schedule = self.core.prefix_schedule[:4]
            assert self.core.plan == self.plan
            self.core.prefix_optimizer = native.CountedOptimizer(self.core.prefix_optimizer)
            self.data = native.NativeBoundTrainingData(bound, self.core.ops)
            self.initial_sam_signature = native.tensor_state_signature(sam.state_dict())
            self.initial_trainable = {k: p.requires_grad for k, p in sam.named_parameters()}
            self.initialized = True
            self.check_invariants()
            return dict(source_pins_verified=True, input_pins_verified=True, initialization_verified=True,
                        resource_gate_passed=True, test_unopened=True, capacity=capacity,
                        foundation=self.foundation_receipt, native_clock_plan=self.plan, cutoff_recipe=self.recipe,
                        initial_sam_signature=self.initial_sam_signature, execution_role=self.execution_role)

        def check_invariants(self):
            validate(self.progress, self.recipe)
            core = self.core; pre = self.progress['prefix']['iterations']; ssl = self.progress['ssl']['iterations']
            assert core.plan == self.plan and core.plan['ssl_iterations'] == self.recipe['original_clock_ssl_iterations']
            assert len(core.prefix_rows) == pre and len(core.ssl_rows) == ssl and core.ssl_position == ssl*2
            assert [tuple(r['actual_labeled_case_ids']) for r in core.prefix_rows] == core.prefix_schedule[:pre]
            assert [p for r in core.ssl_rows for p in r['actual_pairs']] == self.pairs[:2*ssl]
            assert sum(r['labeled_inputs'] for r in core.prefix_rows) == self.progress['prefix']['labeled_inputs']
            if self.progress['phase'] == 'prefix':
                assert core.stage == 'sam_finetuning' and core.prefix_optimizer.calls == pre
                assert not core.prefix_complete and self.prefix_artifact is None
                assert {k: p.requires_grad for k, p in core.sam.named_parameters()} == self.initial_trainable
            else:
                assert core.stage == 'vnet_retraining' and core.prefix_complete and core.prefix_optimizer is None
                assert self.prefix_artifact is not None and self.initial_student_signature is not None
                assert core.optimizer.calls == ssl
                assert not core.sam.training
                assert all(not p.requires_grad and p.grad is None for m in (core.sam, core.teacher) for p in m.parameters())

        def prefix_iteration(self, index):
            self.check_invariants(); assert self.progress['phase'] == 'prefix'
            assert index == self.progress['prefix']['iterations'] < self.recipe['prefix_iterations']
            self.gate()
            row = self.core.finetune_iteration(self.data.batch('train_lab', self.core.prefix_schedule[index]))
            self.torch.cuda.synchronize()
            assert self.core.prefix_optimizer.calls == index+1
            result = dict(row, phase='prefix', unlabeled_inputs=0, finite_loss=True,
                          finite_nonzero_gradients=True, actual_case_ids_verified=True,
                          unlabeled_gt_in_loss=False, skip_count=0)
            self.progress = accept(self.progress, result, self.recipe)
            self.check_invariants()
            return result

        def enter_ssl(self, progress):
            self.check_invariants(); assert progress == self.progress
            proposed = transition(progress, self.recipe)
            self.gate()
            payload = self.core.prefix_terminal_payload()
            payload.update(execution_role=self.execution_role, contract_sha256=protocol.contract_sha(self.c))
            self.prefix_signature = native.tensor_state_signature(payload['native_lora_state'])
            # The protocol must be idempotent across a crash after artifact commit:
            # bind this exact state/ledger/contract, never overwrite a different file.
            artifact, loaded = self.io_with_wait(lambda: protocol.persist_prefix_and_reload(self.c, payload, self.torch, self.execution_role))
            assert tree_hash(payload) == tree_hash(loaded)
            self.core.begin_retraining(loaded)
            self.core.optimizer = native.CountedOptimizer(self.core.optimizer)
            self.prefix_artifact = artifact
            self.initial_student_signature = native.tensor_state_signature(self.core.student.state_dict())
            self.progress = proposed
            self.check_invariants()
            return dict(strict_prefix_disk_reload=True, prefix_artifact=artifact,
                        prefix_native_lora_signature=self.prefix_signature, frozen_sam=True,
                        fresh_stage_seed=2027, native_teacher_initialization='independent')

        def ssl_iteration(self, index):
            self.check_invariants(); assert self.progress['phase'] == 'ssl'
            assert index == self.progress['ssl']['iterations'] < self.recipe['ssl_iterations']
            self.gate()
            pairs = self.pairs[2*index:2*index+2]
            row = self.core.retrain_iteration(self.data.batch('train_lab', [p['labeled'] for p in pairs]),
                                             self.data.batch('train_unlab', [p['unlabeled'] for p in pairs]))
            self.torch.cuda.synchronize()
            assert self.core.optimizer.calls == index+1
            result = dict(row, phase='ssl', finite_loss=True, finite_nonzero_gradients=True, actual_case_ids_verified=True)
            self.progress = accept(self.progress, result, self.recipe)
            self.check_invariants()
            return result

        def capture(self):
            self.check_invariants()
            core = self.core; ssl = self.progress['phase'] == 'ssl'
            names = ('sam', 'student', 'teacher') if ssl else ('sam',)
            optimizer = core.optimizer if ssl else core.prefix_optimizer
            return dict(format='PUBLIC_SFR_COMPLETE_PHASE_RECOVERY_V1', contract_sha256=protocol.contract_sha(self.c),
                        execution_role=self.execution_role, recipe=copy.deepcopy(self.recipe), native_clock_plan=copy.deepcopy(self.plan),
                        foundation=copy.deepcopy(self.foundation_receipt), initial_sam_signature=self.initial_sam_signature,
                        initial_trainable=copy.deepcopy(self.initial_trainable), initial_student_signature=self.initial_student_signature,
                        model_states={n: cpu(getattr(core, n).state_dict()) for n in names},
                        module_modes={n: {k: m.training for k, m in getattr(core, n).named_modules()} for n in names},
                        requires_grad={n: {k: p.requires_grad for k, p in getattr(core, n).named_parameters()} for n in names},
                        optimizer_state=cpu(optimizer.state_dict()), optimizer_calls=optimizer.calls,
                        rng=rng(), progress=copy.deepcopy(self.progress), stage=core.stage,
                        prefix_complete=core.prefix_complete, ssl_position=core.ssl_position,
                        prefix_rows=copy.deepcopy(core.prefix_rows), ssl_rows=copy.deepcopy(core.ssl_rows),
                        prefix_artifact=copy.deepcopy(self.prefix_artifact))

        def restore(self, state):
            assert state['format'] == 'PUBLIC_SFR_COMPLETE_PHASE_RECOVERY_V1'
            assert state['contract_sha256'] == protocol.contract_sha(self.c)
            assert state['execution_role'] == self.execution_role and state['recipe'] == self.recipe
            assert state['native_clock_plan'] == self.plan and state['foundation'] == self.foundation_receipt
            assert state['initial_sam_signature'] == self.initial_sam_signature and state['initial_trainable'] == self.initial_trainable
            validate(state['progress'], self.recipe)
            core = self.core; ssl = state['progress']['phase'] == 'ssl'
            names = ('sam', 'student', 'teacher') if ssl else ('sam',)
            assert set(state['model_states']) == set(state['module_modes']) == set(state['requires_grad']) == set(names)
            assert self.progress == initial() and core.stage == 'sam_finetuning'
            core.sam.load_state_dict(state['model_states']['sam'], strict=True)
            core.prefix_rows = copy.deepcopy(state['prefix_rows'])
            if ssl:
                artifact = state['prefix_artifact']
                loaded = protocol.load_bound_prefix(self.c, artifact, self.torch, self.execution_role)
                assert state['prefix_complete'] is True
                core.prefix_complete = True
                core.begin_retraining(loaded)  # Constructs native architecture; saved RNG is restored last.
                core.optimizer = native.CountedOptimizer(core.optimizer)
                assert native.tensor_state_signature(core.student.state_dict()) == state['initial_student_signature']
                self.prefix_artifact = copy.deepcopy(artifact)
                self.prefix_signature = native.tensor_state_signature(loaded['native_lora_state'])
                self.initial_student_signature = state['initial_student_signature']
            else:
                assert state['prefix_complete'] is False and state['prefix_artifact'] is None
                assert state['initial_student_signature'] is None
            for name in names:
                model = getattr(core, name)
                assert {k: p.requires_grad for k, p in model.named_parameters()} == state['requires_grad'][name]
                model.load_state_dict(state['model_states'][name], strict=True)
                modules = dict(model.named_modules()); assert set(modules) == set(state['module_modes'][name])
                for k, m in modules.items(): m.training = state['module_modes'][name][k]
                for p in model.parameters(): p.grad = None
            optimizer = core.optimizer if ssl else core.prefix_optimizer
            optimizer.load_state_dict(state['optimizer_state']); optimizer.calls = state['optimizer_calls']
            core.ssl_rows = copy.deepcopy(state['ssl_rows']); core.ssl_position = state['ssl_position']
            assert core.stage == state['stage'] and core.prefix_complete == state['prefix_complete']
            self.progress = copy.deepcopy(state['progress'])
            restore_rng(state['rng'])
            self.check_invariants()
            assert tree_hash(self.capture()) == tree_hash({k: v for k, v in state.items() if k != 'ledgers'})

        def checkpoint(self, progress):
            assert progress == self.progress
            self.gate(); state = self.capture(); state['ledgers'] = protocol.ledgers(self.c)
            return self.io_with_wait(lambda: protocol.save_checkpoint(self.c, state, self.torch, self.execution_role))

        def prepare_resume(self):
            result = self.prepare_initial()
            commit, state = protocol.load_reviewed_checkpoint(self.c, self.torch)
            assert state['ledgers'] == commit['ledgers'] == protocol.ledgers(self.c)
            self.restore(state)
            protocol.record_restoration(self.c, commit, tree_hash(self.capture()), self.progress)
            return dict(result, progress=copy.deepcopy(self.progress), loaded_state_exact=True, both_ledger_prefixes_verified=True)

        def verify_training_coverage(self, progress):
            self.check_invariants(); assert progress == self.progress and ready_for_test(progress, self.recipe)
            actual = [p for r in self.core.ssl_rows for p in r['actual_pairs']]
            assert actual == self.pairs[:self.recipe['primary_inputs']]
            protocol.verify_actual_ledgers(self.c, self.core.prefix_rows, self.core.ssl_rows)
            cycle = self.budget.primary_pairs_per_cycle
            complete = [self._verify_cycle(i, actual[i*cycle:(i+1)*cycle]) for i in range(11)]
            return dict(actual_case_coverage_verified=True, prefix=progress['prefix'], ssl=progress['ssl'],
                        reference_inputs=self.recipe['reference_inputs'], actual_inputs=len(actual),
                        rounding_extra_inputs=self.recipe['rounding_extra_inputs'], original_clock_inputs=self.budget.primary_pairs,
                        complete_cycles=complete, prefix_artifact=self.prefix_artifact,
                        ssl_teacher_input_forwards=sum(r['teacher_input_forwards'] for r in self.core.ssl_rows),
                        ssl_sam_stitched_board_forwards=sum(r['sam_stitched_board_forwards'] for r in self.core.ssl_rows))

        def save_terminal(self, progress):
            assert self.execution_role == 'formal'
            self.verify_training_coverage(progress)
            self.final_signature = native.tensor_state_signature(self.core.student.state_dict())
            assert self.final_signature != self.initial_student_signature
            payload = dict(method='sfr', stage='fixed_ssl_terminal', plan=self.plan, cutoff_recipe=self.recipe,
                           counts=progress, foundation=self.foundation_receipt, student_state=cpu(self.core.student.state_dict()),
                           native_student_signature=self.final_signature, prefix_artifact=self.prefix_artifact,
                           selection='fixed_terminal_no_test_selection', test_evaluations=0,
                           execution_role='formal', contract_sha256=protocol.contract_sha(self.c))
            self.gate()
            self.final_artifact = self.io_with_wait(lambda: protocol.save_terminal(self.c, payload, self.torch))
            return self.final_artifact

        def _reload(self, role, artifact, verifier):
            self.gate()
            return self.io_with_wait(lambda: protocol.reload_artifact(self.c, role, artifact, self.torch, verifier))

        def guard_test_lineage(self):
            assert self.execution_role == 'formal'
            protocol.guard_test_lineage(self.c)

    return Backend
