"""Draft recoverable SDCL lifecycle. Formal launch requires separate own admission.

The backend owns original native kernels, full-state saves, pinned inputs and
resource waits. This controller accounts the mandatory prefix separately from
SSL and opens the single terminal evaluation only after both are complete.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time
from stage_state import accept, initial, ready_for_test, transition, validate


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def exclusive(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, allow_nan=False, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())


def consistent_record(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != json.loads(json.dumps(value)):
            raise ValueError('Existing phase/terminal evidence differs: '+str(path))
    else:
        exclusive(path, value)


class TwoStageRun:
    def __init__(self, recipe, output, test_ids, interval=100, resume=False):
        self.recipe, self.output = recipe, Path(output)
        self.test_ids = list(test_ids)
        self.interval, self.resume = interval, resume
        if interval != 100 or len(self.test_ids) != 87 or len(set(self.test_ids)) != 87:
            raise ValueError('Recovery interval or test identity differs')
        self.progress = initial()

    def run(self, backend):
        self.output.mkdir(parents=True, exist_ok=self.resume)
        if any((self.output/n).exists() for n in ('failure.json', 'final_test_intent.json', 'final_metrics.json')):
            raise ValueError('Failure or terminal intent requires independent review, never replay automatically')
        if not self.resume:
            exclusive(self.output/'run_intent.json', dict(recipe=self.recipe, test_ids=self.test_ids,
                      terminal_selection='fixed_primary_input_endpoint', prefix_selection='fixed_native_terminal'))
        started = time.monotonic()
        try:
            preparation = backend.prepare_resume() if self.resume else backend.prepare_initial()
            for name in ('source_pins_verified', 'input_pins_verified', 'initialization_verified', 'resource_gate_passed', 'test_unopened'):
                if preparation.get(name) is not True:
                    raise ValueError('Missing preparation proof: '+name)
            if self.resume:
                self.progress = copy.deepcopy(validate(preparation['progress'], self.recipe))
                if preparation.get('loaded_state_exact') is not True or preparation.get('both_ledger_prefixes_verified') is not True:
                    raise ValueError('Missing full-state/two-ledger resume proof')
            else:
                exclusive(self.output/'preparation_verified.json', preparation)
            backend.allow_wait = True
            backend.checkpoint(copy.deepcopy(self.progress))
            if self.progress['phase'] == 'prefix':
                while self.progress['prefix']['iterations'] < self.recipe['prefix_iterations']:
                    self.step(backend)
                # Commit the last pretraining state before the deterministic EMA handoff.
                backend.checkpoint(copy.deepcopy(self.progress))
                boundary = backend.enter_ssl(copy.deepcopy(self.progress))
                if boundary.get('teacher_matches_terminal_student') is not True or boundary.get('optimizer_state_preserved') is not True or boundary.get('rng_preserved') is not True:
                    raise ValueError('Native prefix/SSL handoff differs')
                consistent_record(self.output/'pretrain_terminal.json', dict(prefix=self.progress['prefix'], boundary=boundary))
                self.progress = transition(self.progress, self.recipe)
                backend.checkpoint(copy.deepcopy(self.progress))
            while self.progress['ssl']['iterations'] < self.recipe['ssl_iterations']:
                self.step(backend)
            backend.checkpoint(copy.deepcopy(self.progress))
            if not ready_for_test(self.progress, self.recipe):
                raise ValueError('Terminal before both phase endpoints')
            coverage = backend.verify_training_coverage(copy.deepcopy(self.progress))
            if coverage.get('actual_case_coverage_verified') is not True:
                raise ValueError('Actual training inputs not proven')
            consistent_record(self.output/'coverage_verified.json', coverage)
            artifact = backend.save_terminal(copy.deepcopy(self.progress))
            if artifact.get('resource_lock_verified') is not True or sha(artifact['path']) != artifact['sha256']:
                raise ValueError('Terminal identity/resource gate differs')
            reload = backend.strict_reload(artifact)
            if reload.get('strict') is not True or reload.get('native_student_identity_verified') is not True or reload.get('sha256') != artifact['sha256']:
                raise ValueError('Both final students must strictly reload')
            backend.guard_test_lineage()
            permit = self.output/'final_test_intent.json'
            exclusive(permit, dict(selection='terminal_after_fixed_exposure', final_sha256=artifact['sha256'],
                      test_case_ids=self.test_ids, test_evaluation=1))
            result = backend.test(artifact, permit)
            if result.get('case_ids') != self.test_ids or result.get('final_sha256') != artifact['sha256'] or result.get('test_evaluations') != 1:
                raise ValueError('Wrong terminal/test identities')
            values = result.get('case_dice')
            if not isinstance(values, list) or len(values) != 87 or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in values):
                raise ValueError('Invalid individual test metrics')
            if not math.isclose(sum(values)/87, result['dice'], rel_tol=0, abs_tol=1e-12) or sha(artifact['path']) != artifact['sha256']:
                raise ValueError('Terminal/test aggregate changed')
            record = dict(status='COMPLETE_NEEDS_INDEPENDENT_ACCEPTANCE', counts=copy.deepcopy(self.progress),
                          recipe=self.recipe, artifact=artifact, strict_reload=reload, final_metrics=result,
                          test_evaluations=1, wall_seconds=time.monotonic()-started,
                          disclosure='Native mandatory prefix and SSL/recovery/engineering compute are separately counted')
            exclusive(self.output/'final_metrics.json', record)
            return record
        except BaseException as e:
            exclusive(self.output/'failure.json', dict(status='FAILED_PRESERVED', counts=self.progress,
                      error_type=type(e).__name__, error=str(e), failed_partial_compute_not_assumed_zero=True))
            raise

    def step(self, backend):
        phase = self.progress['phase']
        step = backend.prefix_iteration(self.progress['prefix']['iterations']) if phase == 'prefix' else backend.ssl_iteration(self.progress['ssl']['iterations'])
        proposed = accept(self.progress, step, self.recipe)
        row = json.dumps(dict(step=step, cumulative=proposed[phase]), allow_nan=False)+'\n'
        if len(row.encode('utf-8')) > 8192:
            raise ValueError('Phase ledger row exceeds reserved bound')
        path = self.output/('pretrain_steps.jsonl' if phase == 'prefix' else 'steps.jsonl')
        with path.open('a', encoding='utf-8', newline='\n') as f:
            f.write(row)
            f.flush()
            os.fsync(f.fileno())
        self.progress = proposed
        if self.progress[phase]['iterations'] % self.interval == 0:
            backend.checkpoint(copy.deepcopy(self.progress))
