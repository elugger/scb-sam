"""DualTeacher fixed-cutoff lifecycle; no launcher or implicit model construction.

An independently verified protocol/backend must supply storage, source checks,
the original native clock and native updates. This module cannot authorize a run.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import time

from dualteacher_budget import counts, plan, require, validate
from dualteacher_state import accept


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check_metrics(output, result, test_ids, artifact):
    """Check all four aggregates against the one native per-case ledger."""
    require(result.get('case_ids') == test_ids and result.get('test_evaluations') == 1,
            'Terminal test identity/count differs')
    require(result.get('final_sha256') == artifact['sha256'] and result.get('distance_unit') == 'voxel',
            'Terminal artifact or distance unit differs')
    path = Path(output) / 'final_test_cases.jsonl'
    require(path.is_file() and not path.is_symlink(), 'Missing native case ledger')
    raw = path.read_bytes()
    require(raw.endswith(b'\n'), 'Partial native test case row')
    rows = [json.loads(line) for line in raw.splitlines()]
    require(len(rows) == 87 and [r.get('case_id') for r in rows] == test_ids
            and [r.get('actual_index') for r in rows] == list(range(87)),
            'Test cases missing, duplicated, reordered or substituted')
    require(all(r.get('native_windows') == 27 and r.get('selection') == 'EMA+OP' for r in rows)
            and result.get('selection') == 'fixed_terminal_EMA_plus_OP'
            and result.get('native_patch_forwards') == 27*87, 'Native EMA+OP/window selection differs')
    for key in ('dice', 'jaccard', 'hd95', 'asd'):
        values = [r.get(key) for r in rows]
        require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0
                    and (key not in ('dice', 'jaccard') or v <= 1) for v in values),
                'Invalid case metric: ' + key)
        aggregate = result.get(key)
        require(type(aggregate) in (int, float) and math.isfinite(aggregate)
                and math.isclose(sum(values) / 87, aggregate, rel_tol=0, abs_tol=1e-12),
                'Native case aggregate differs: ' + key)
    require(result.get('case_dice') == [r['dice'] for r in rows], 'Case Dice list differs')
    require(file_sha(artifact['path']) == artifact['sha256'], 'Terminal changed during test')


class FixedCutoffRun:
    """Formal run only. Engineering windows have a separate future operator."""
    def __init__(self, recipe, output, test_ids, resume=False):
        require(recipe == plan(recipe['labeled_cases']), 'Noncanonical cutoff recipe')
        require(len(test_ids) == len(set(test_ids)) == 87, 'Expected 87 unique test IDs')
        self.recipe, self.output = copy.deepcopy(recipe), Path(output)
        self.test_ids, self.resume = list(test_ids), resume
        self.progress = counts(0)

    def run(self, backend):
        require(backend.execution_role == 'formal' and backend.recipe == self.recipe,
                'Engineering state or different recipe cannot run formally')
        require(Path(backend.output) == self.output, 'Backend output differs')
        # This also inspects preserved history. A reason string cannot override
        # a prior test, failure or unreviewed ledger rewind.
        backend.guard_run_lineage(self.resume)
        backend.record('run_intent.json' if not self.resume else backend.resume_intent_name(),
                       dict(recipe=self.recipe, test_ids=self.test_ids, resume=self.resume,
                            selection='terminal_after_fixed_exposure'))
        started = time.monotonic()
        try:
            prepared = backend.prepare_resume() if self.resume else backend.prepare_initial()
            for key in ('source_pins_verified', 'input_pins_verified', 'initialization_verified',
                        'resource_gate_passed', 'test_unopened'):
                require(prepared.get(key) is True, 'Missing preparation proof: ' + key)
            if self.resume:
                require(prepared.get('loaded_state_exact') is True
                        and prepared.get('ledger_prefix_verified') is True
                        and prepared.get('independent_resume_review_consumed') is True,
                        'Resume lacks exact state, original prefix or independent review')
                self.progress = copy.deepcopy(validate(prepared['counts'], self.recipe))
            else:
                backend.record('preparation_verified.json', prepared)
            require(self.progress == backend.progress, 'Restored backend/controller counts differ')
            backend.allow_wait = True
            backend.checkpoint(copy.deepcopy(self.progress))
            while self.progress['iterations'] < self.recipe['ssl_iterations']:
                i = self.progress['iterations']
                # Resource checks must precede model updates; the backend must
                # preserve unknown failures and only wait on exact capacity errors.
                step = backend.train_iteration(i)
                proposed = accept(self.progress, step, self.recipe, 'formal',
                                  backend.binding.pair_batch(i), backend.original_clock(i))
                require(proposed == backend.progress, 'Native/backend/controller counts differ')
                backend.append_step(dict(step=step, cumulative=proposed))
                self.progress = proposed
                if proposed['iterations'] % 100 == 0:
                    backend.checkpoint(copy.deepcopy(proposed))
            backend.checkpoint(copy.deepcopy(self.progress))
            coverage = backend.verify_training_coverage(copy.deepcopy(self.progress))
            require(coverage.get('actual_case_coverage_verified') is True
                    and coverage.get('complete_reference_cycles') == 11
                    and coverage.get('primary_labeled_inputs') == self.recipe['primary_inputs_per_stream']
                    and coverage.get('primary_unlabeled_inputs') == self.recipe['primary_inputs_per_stream']
                    and coverage.get('successful_optimizer_updates') == self.recipe['effective_optimizer_updates'],
                    'Incomplete actual 11-cycle coverage')
            backend.record('coverage_verified.json', coverage, consistent=True)
            artifact = backend.save_terminal(copy.deepcopy(self.progress))
            require(artifact.get('resource_lock_verified') is True
                    and file_sha(artifact['path']) == artifact['sha256'], 'Unbound final artifact')
            reloaded = backend.strict_reload(artifact)
            for key in ('strict', 'native_student_identity_verified', 'native_ema_op_identity_verified'):
                require(reloaded.get(key) is True, 'Incomplete strict reload: ' + key)
            require(reloaded.get('sha256') == artifact['sha256'], 'Strict reload SHA differs')
            # Resolve pinned prerequisite modules before opening a test intent.
            backend.verify_test_prerequisites()
            backend.guard_test_lineage()
            permit = self.output / 'final_test_intent.json'
            backend.record(permit.name, dict(selection='terminal_after_fixed_exposure',
                           final_sha256=artifact['sha256'], test_case_ids=self.test_ids, test_evaluation=1))
            result = backend.test(artifact, permit)
            check_metrics(self.output, result, self.test_ids, artifact)
            record = dict(status='COMPLETE_NEEDS_INDEPENDENT_ACCEPTANCE', counts=self.progress,
                          recipe=self.recipe, artifact=artifact, strict_reload=reloaded,
                          final_metrics=result, test_evaluations=1,
                          wall_seconds=time.monotonic() - started,
                          disclosure='SSL main inputs only; native 36-cycle clock retained; engineering and replay separate')
            backend.record('final_metrics.json', record)
            return record
        except BaseException as error:
            backend.record('failure.json', dict(status='FAILED_PRESERVED', counts=self.progress,
                           error_type=type(error).__name__, error=str(error),
                           failed_partial_compute_not_assumed_zero=True))
            raise
