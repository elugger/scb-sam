"""Draft full-state wrapper. No standalone GPU or formal-launch entry.

make_backend receives the SHA-verified immutable r3 SDCL module plus a separate
protocol service. That service must bind all source/input hashes, two ledgers,
two bounded recovery generations, current resource gates and the GPU1 lease.
The launch/admission layer must prove both prefix and SSL fresh-process resume.
"""
import copy
import math
from pathlib import Path
import time
from recovery_state import cpu, rng, restore_rng, tree_hash
from stage_state import counts, initial, plan, transition, validate


def make_backend(native, protocol):
    class Backend(native.SDCLBackend):
        models = ('student_v', 'student_r', 'ema_v')
        optimizers = ('opt_v', 'opt_r')

        def __init__(self, contract, output, engineering=False):
            # Change only the bound dataset/budget/resource adapters in this
            # isolated process. The original methods and kernels stay intact.
            native.BUDGETS = tuple(protocol.clock_budget(n) for n in (5, 10))
            native.bound_splits = protocol.bound_splits
            native.resource_backend = lambda target: protocol.resource(contract)
            super().__init__(contract, output, protocol.verify)
            self.c = contract
            self.recipe = plan(contract['labeled_cases'])
            self.execution_role = 'engineering_two_phase_probe' if engineering else 'formal'
            if engineering:
                assert Path(output).parent == protocol.job(contract) and Path(output).name == 'qualification_backend'
                assert contract['engineering_probe_prefix_iterations'] == 4
                self.recipe.update(prefix_iterations=4, prefix_labeled_inputs=16, prefix_updates=8)
            else:
                assert Path(output) == Path(contract['output'])
            self.progress = initial()
            self.allow_wait = False

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

        def prepare_initial(self):
            self.gate()
            result = super().prepare(protocol.endpoint(self.c), perform_prefix=False)
            assert self.budget.primary_pairs == self.recipe['original_clock_inputs']
            assert len(self.pairs) == self.recipe['original_clock_inputs']
            # The clock and full original sampling sequence remain 36 cycles;
            # only the required fixed endpoint plan is shortened.
            self.plan = native.stage_plan(self.c['labeled_cases'], self.recipe['primary_inputs'])
            if self.execution_role != 'formal':
                self.plan.update(pretrain_iterations=4, pretrain_labeled_inputs=16,
                                 pretrain_successful_optimizer_updates=8,
                                 engineering_probe_only_not_formal_prefix=True)
            self.initial_signatures = {n: native.signature(self.torch, getattr(self.t, n)) for n in self.models}
            self.prefix_signatures = None
            self.initial_trainable = {n: {k: p.requires_grad for k, p in getattr(self.t, n).named_parameters()} for n in self.models}
            self.check_invariants()
            return dict(result, native_initial_signatures=self.initial_signatures,
                        fixed_prefix_and_ssl_plan=self.plan, original_clock_inputs=self.budget.primary_pairs)

        def check_invariants(self):
            validate(self.progress, self.recipe)
            pre = self.progress['prefix']['iterations']; ssl = self.progress['ssl']['iterations']
            assert self.position == 4*ssl and len(self.rows) == self.position
            assert self.rows == self.pairs[:self.position] and len(self.pre_rows) == pre
            assert self.t.global_step == (pre if self.progress['phase'] == 'prefix' else ssl)
            assert all(getattr(self.t, n).calls == pre+ssl for n in self.optimizers)
            assert all(p.grad is None for p in self.t.ema_v.parameters())
            for n in self.models:
                assert {k: p.requires_grad for k, p in getattr(self.t, n).named_parameters()} == self.initial_trainable[n]
            if self.progress['phase'] == 'ssl':
                assert self.prefix_signatures is not None

        def prefix_iteration(self, index):
            self.check_invariants()
            assert self.progress['phase'] == 'prefix' and index == len(self.pre_rows) < self.recipe['prefix_iterations']
            self.gate()
            before = sum(getattr(self.t, n).calls for n in self.optimizers)
            row = super().pretrain_iteration(index, record=False)
            self.torch.cuda.synchronize()
            assert sum(getattr(self.t, n).calls for n in self.optimizers)-before == 2
            assert all(math.isfinite(v) and v > 0 for v in row['gradient_norms'])
            self.progress['prefix'] = counts(index+1, 4*(index+1), 0, 2*(index+1))
            self.check_invariants()
            return dict(row, phase='prefix', unlabeled_inputs=0, finite_loss=True,
                        finite_nonzero_gradients=True, actual_case_ids_verified=True,
                        teacher_no_grad=True, skip_count=0, unlabeled_gt_in_loss=False)

        def enter_ssl(self, progress):
            self.check_invariants()
            assert progress == self.progress
            new_progress = transition(progress, self.recipe)
            before_rng = tree_hash(rng())
            before_optim = tree_hash({n: getattr(self.t, n).inner.state_dict() for n in self.optimizers})
            self.t.ema_v.load_state_dict(self.t.student_v.state_dict(), strict=True)
            self.t.global_step = 0
            self.prefix_signatures = {n: native.signature(self.torch, getattr(self.t, n)) for n in self.models}
            self.progress = new_progress
            self.check_invariants()
            assert tree_hash(rng()) == before_rng
            assert tree_hash({n: getattr(self.t, n).inner.state_dict() for n in self.optimizers}) == before_optim
            assert self.prefix_signatures['ema_v'] == self.prefix_signatures['student_v']
            return dict(teacher_matches_terminal_student=True, optimizer_state_preserved=True,
                        rng_preserved=True, terminal_state_signatures=self.prefix_signatures)

        def ssl_iteration(self, index):
            self.check_invariants()
            assert self.progress['phase'] == 'ssl' and index == self.progress['ssl']['iterations'] < self.recipe['ssl_iterations']
            self.gate()
            remaining = self.recipe['primary_inputs']-self.position
            row = super().train_iteration(dict(labeled_inputs=remaining, unlabeled_inputs=remaining))
            self.progress['ssl'] = counts(index+1, 4*(index+1), 4*(index+1), 2*(index+1))
            self.check_invariants()
            return dict(row, phase='ssl', iteration=index+1)

        def capture(self):
            self.check_invariants()
            return dict(format='PUBLIC_SDCL_BOTH_PHASE_RECOVERY_V1', contract_sha256=protocol.contract_sha(self.c),
                        execution_role=self.execution_role, recipe=copy.deepcopy(self.recipe),
                        model_states={n: cpu(getattr(self.t, n).state_dict()) for n in self.models},
                        module_modes={n: {k: m.training for k, m in getattr(self.t, n).named_modules()} for n in self.models},
                        optimizer_states={n: cpu(getattr(self.t, n).inner.state_dict()) for n in self.optimizers},
                        optimizer_calls={n: getattr(self.t, n).calls for n in self.optimizers},
                        optimizer_norms={n: getattr(self.t, n).norm for n in self.optimizers},
                        rng=rng(), progress=copy.deepcopy(self.progress), global_step=self.t.global_step,
                        position=self.position, rows=copy.deepcopy(self.rows), pre_rows=copy.deepcopy(self.pre_rows),
                        initial_signatures=self.initial_signatures, initial_trainable=self.initial_trainable,
                        prefix_signatures=self.prefix_signatures)

        def restore(self, state):
            assert state['format'] == 'PUBLIC_SDCL_BOTH_PHASE_RECOVERY_V1'
            assert state['contract_sha256'] == protocol.contract_sha(self.c)
            assert state['execution_role'] == self.execution_role and state['recipe'] == self.recipe
            validate(state['progress'], self.recipe)
            assert state['initial_signatures'] == self.initial_signatures and state['initial_trainable'] == self.initial_trainable
            for n in self.models:
                module = getattr(self.t, n)
                module.load_state_dict(state['model_states'][n], strict=True)
                for k, m in module.named_modules():
                    m.training = state['module_modes'][n][k]
                for p in module.parameters():
                    p.grad = None
            for n in self.optimizers:
                optimizer = getattr(self.t, n)
                optimizer.inner.load_state_dict(state['optimizer_states'][n])
                optimizer.calls = state['optimizer_calls'][n]
                optimizer.norm = state['optimizer_norms'][n]
            for n in ('position', 'rows', 'pre_rows', 'prefix_signatures', 'progress'):
                setattr(self, n, copy.deepcopy(state[n]))
            self.t.global_step = state['global_step']
            restore_rng(state['rng'])
            self.check_invariants()
            assert tree_hash(self.capture()) == tree_hash({k: v for k, v in state.items() if k != 'ledgers'})

        def checkpoint(self, progress):
            assert progress == self.progress
            self.gate()
            state = self.capture()
            state['ledgers'] = protocol.ledgers(self.c)
            while True:
                try:
                    return protocol.save_checkpoint(self.c, state)
                except (ValueError, RuntimeError, AssertionError) as e:
                    if not str(e).startswith(('Insufficient filesystem capacity:', 'Insufficient RAM including cgroup ancestors')):
                        raise
                    self.gate()

        def prepare_resume(self):
            result = self.prepare_initial()
            commit, state = protocol.load_reviewed_checkpoint(self.c, self.torch)
            assert state['ledgers'] == commit['ledgers'] == protocol.ledgers(self.c)
            self.restore(state)
            protocol.record_restoration(self.c, commit, tree_hash(self.capture()), self.progress)
            return dict(result, progress=copy.deepcopy(self.progress), loaded_state_exact=True, both_ledger_prefixes_verified=True)

        def verify_training_coverage(self, progress):
            self.check_invariants()
            assert progress == self.progress and progress['phase'] == 'ssl'
            assert self.position == self.recipe['primary_inputs'] and len(self.pre_rows) == self.recipe['prefix_iterations']
            protocol.verify_actual_ledgers(self.c, self.pre_rows, self.rows)
            return dict(actual_case_coverage_verified=True, prefix=progress['prefix'], ssl=progress['ssl'],
                        reference_inputs=self.recipe['reference_inputs'], actual_inputs=self.position,
                        rounding_extra_inputs=self.recipe['rounding_extra_inputs'], original_clock_inputs=self.budget.primary_pairs)

        def save_terminal(self, progress):
            self.verify_training_coverage(progress)
            self.signatures = {n: native.signature(self.torch, getattr(self.t, n)) for n in ('student_v', 'student_r')}
            payload = dict(method='sdcl', counts=progress, pretraining=self.plan, signatures=self.signatures,
                           **{n: getattr(self.t, n).state_dict() for n in self.signatures})
            while True:
                self.gate()
                try:
                    return protocol.save_terminal(self.c, payload, self.torch, tree_hash)
                except (ValueError, RuntimeError, AssertionError) as e:
                    if not str(e).startswith(('Insufficient filesystem capacity:', 'Insufficient RAM including cgroup ancestors')):
                        raise

        def strict_reload(self, artifact):
            while True:
                self.gate()
                try:
                    return super().strict_reload(artifact)
                except (ValueError, RuntimeError, AssertionError) as e:
                    if not str(e).startswith(('Insufficient filesystem capacity:', 'Insufficient RAM including cgroup ancestors')):
                        raise

        def guard_test_lineage(self):
            protocol.guard_test_lineage(self.c)

    return Backend
