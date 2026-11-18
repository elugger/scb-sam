"""Synthetic controller failure/recovery checks; never load training data/models."""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from stage_state import accept, counts, initial, plan, ready_for_test, transition, validate
from two_stage_controller import TwoStageRun, sha


class FakeBackend:
    def __init__(self, output, recipe, failure=None):
        self.output, self.recipe = Path(output), recipe
        self.progress = initial(); self.saved = None; self.failure = failure
        self.checkpoints = []; self.test_calls = 0; self.transition_calls = 0
        self.extra_updates = 0

    def prepare_initial(self):
        return {k: True for k in ('source_pins_verified', 'input_pins_verified', 'initialization_verified', 'resource_gate_passed', 'test_unopened')}

    def prepare_resume(self):
        assert self.saved is not None
        self.progress = copy.deepcopy(self.saved['progress'])
        return dict(self.prepare_initial(), progress=copy.deepcopy(self.progress), loaded_state_exact=True, both_ledger_prefixes_verified=True)

    def checkpoint(self, progress):
        assert progress == self.progress
        if self.failure == 'transition_save' and progress['phase'] == 'ssl' and progress['ssl']['iterations'] == 0:
            self.failure = None
            raise RuntimeError('synthetic checkpoint interruption after EMA handoff')
        self.saved = dict(progress=copy.deepcopy(progress), files={})
        for name in ('steps.jsonl', 'pretrain_steps.jsonl'):
            p = self.output/name
            self.saved['files'][name] = p.read_bytes() if p.exists() else b''
        self.checkpoints.append(copy.deepcopy(progress))

    def update(self, phase, index):
        if self.failure == (phase, index):
            self.failure = None
            raise RuntimeError('synthetic unknown forward failure')
        step = dict(phase=phase, iteration=index+1, labeled_inputs=4, unlabeled_inputs=0 if phase=='prefix' else 4,
                    successful_optimizer_updates=2, finite_loss=True, finite_nonzero_gradients=True,
                    actual_case_ids_verified=True, teacher_no_grad=True, skip_count=0, unlabeled_gt_in_loss=False,
                    losses=dict(loss_v=0.25, loss_r=0.5), gradient_norms=[1.0, 2.0])
        self.progress = accept(self.progress, step, self.recipe)
        return step

    def prefix_iteration(self, index):
        return self.update('prefix', index)

    def ssl_iteration(self, index):
        return self.update('ssl', index)

    def enter_ssl(self, progress):
        assert progress == self.progress
        if self.failure == 'before_transition':
            self.failure = None
            raise RuntimeError('synthetic interruption before EMA handoff')
        self.progress = transition(progress, self.recipe)
        self.transition_calls += 1
        return dict(teacher_matches_terminal_student=True, optimizer_state_preserved=True, rng_preserved=True,
                    terminal_state_signatures=dict(student_v='synthetic_v', student_r='synthetic_r', ema_v='synthetic_v'))

    def verify_training_coverage(self, progress):
        assert ready_for_test(progress, self.recipe) and progress == self.progress
        return dict(actual_case_coverage_verified=True, counts=progress)

    def save_terminal(self, progress):
        assert ready_for_test(progress, self.recipe)
        p = self.output/'synthetic_terminal'
        if not p.exists():
            p.write_bytes(b'CPU fake scalar state, not a model')
        return dict(path=str(p), sha256=sha(p), resource_lock_verified=True)

    def strict_reload(self, artifact):
        if self.failure == 'strict_reload':
            self.failure = None
            raise ValueError('synthetic strict reload rejection')
        return dict(strict=True, native_student_identity_verified=True, sha256=artifact['sha256'])

    def guard_test_lineage(self):
        assert not (self.output/'final_test_intent.json').exists()

    def test(self, artifact, permit):
        self.test_calls += 1
        assert Path(permit).exists()
        if self.failure == 'during_test':
            self.failure = None
            raise RuntimeError('synthetic partial terminal test')
        ids = ['synthetic_%03d'%i for i in range(87)]
        return dict(case_ids=ids, case_dice=[0.5]*87, dice=0.5, final_sha256=artifact['sha256'], test_evaluations=1)

    def synthetic_review(self):
        # Only artificial files created under this CPU test directory.
        assert 'sdcl_controller_cpu_' in str(self.output.resolve())
        assert not (self.output/'final_test_intent.json').exists()
        failure = self.output/'failure.json'
        (self.output/'preserved_failure.json').write_bytes(failure.read_bytes())
        failure.unlink()
        for name, prefix in self.saved['files'].items():
            path = self.output/name
            raw = path.read_bytes() if path.exists() else b''
            assert raw.startswith(prefix)
            if len(raw) > len(prefix):
                (self.output/(name+'.preserved_tail')).write_bytes(raw[len(prefix):])
                self.extra_updates += 2*len(raw[len(prefix):].splitlines())
            path.write_bytes(prefix)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    work = Path(__file__).resolve().parent.parent
    root = Path(tempfile.mkdtemp(prefix='sdcl_controller_cpu_', dir=work)).resolve()
    assert root.is_relative_to(work.resolve())
    tests = []
    ids = ['synthetic_%03d'%i for i in range(87)]
    recipe = dict(prefix_iterations=3, ssl_iterations=4)
    def passed(name): tests.append(dict(name=name, passed=True))
    def rejects(name, action):
        try: action()
        except (ValueError, AssertionError, RuntimeError): passed(name)
        else: raise AssertionError('Not rejected: '+name)
    assert plan(5)['primary_inputs']==3796 and plan(10)['primary_inputs']==3740
    assert plan(5)['prefix_iterations']==648 and plan(10)['prefix_iterations']==1620
    assert plan(5)['original_clock_inputs']==12420 and plan(10)['original_clock_inputs']==12240
    passed('real_two_case_budgets_and_preserved_clock')
    rejects('reject_boolean_labels', lambda: plan(True))
    rejects('reject_early_prefix_transition', lambda: transition(initial(), recipe))
    bad=initial(); bad['ssl']=counts(1,4,4,2)
    rejects('reject_SSL_before_prefix', lambda: validate(bad, recipe))
    bad=initial(); bad['prefix']=counts(1,4,0,1)
    rejects('reject_half_native_update_count', lambda: validate(bad, recipe))
    bad=initial(); bad['prefix']=counts(4,16,0,8)
    rejects('reject_prefix_budget_overrun', lambda: validate(bad, recipe))
    for name, failure, extra in [('prefix_failure',('prefix',2),4), ('before_transition','before_transition',0),
                                 ('transition_save','transition_save',0), ('ssl_failure',('ssl',2),4),
                                 ('strict_reload','strict_reload',0)]:
        out=root/name; b=FakeBackend(out, recipe, failure)
        rejects(name+'_preserved', lambda: TwoStageRun(recipe,out,ids).run(b))
        assert b.test_calls==0 and (out/'failure.json').exists()
        rejects(name+'_unreviewed_resume_refused', lambda: TwoStageRun(recipe,out,ids,resume=True).run(b))
        b.synthetic_review()
        result=TwoStageRun(recipe,out,ids,resume=True).run(b)
        assert result['counts']['prefix']==counts(3,12,0,6)
        assert result['counts']['ssl']==counts(4,16,16,8) and b.test_calls==1
        assert len((out/'pretrain_steps.jsonl').read_text().splitlines())==3
        assert len((out/'steps.jsonl').read_text().splitlines())==4
        assert b.extra_updates==extra
        passed(name+'_resume_exact_phase_counts_and_disclosed_tail')
    out=root/'during_test'; b=FakeBackend(out,recipe,'during_test')
    rejects('partial_test_preserved', lambda: TwoStageRun(recipe,out,ids).run(b))
    rejects('partial_test_never_replayed', lambda: TwoStageRun(recipe,out,ids,resume=True).run(b))
    assert b.test_calls==1
    out=root/'intervals'; long=dict(prefix_iterations=101,ssl_iterations=101); b=FakeBackend(out,long)
    result=TwoStageRun(long,out,ids).run(b)
    assert any(p['phase']=='prefix' and p['prefix']['iterations']==100 for p in b.checkpoints)
    assert any(p['phase']=='ssl' and p['ssl']['iterations']==100 for p in b.checkpoints)
    assert b.test_calls==1 and b.transition_calls==1
    passed('both_phases_100_step_initial_boundary_terminal_checkpoints')
    rejects('completed_terminal_never_retested', lambda: TwoStageRun(long,out,ids,resume=True).run(b))
    # Compile new sources without creating pycache or importing torch.
    source=Path(__file__).resolve().parent
    for p in source.glob('*.py'):
        compile(p.read_bytes(),str(p),'exec')
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    passed('no_torch_numpy_models_cuda_data_or_real_test_loaded')
    value=dict(status='DRAFT_CONTROLLER_CPU_PASS_NOT_GPU_QUALIFIED',tests=tests,test_count=len(tests),
               new_models=0,new_cuda_loads=0,new_optimization_updates=0,new_patient_inputs=0,new_test_evaluations=0,
               source_files={str(p):sha(p) for p in source.glob('*.py')},synthetic_evidence_directory=str(root),
               remaining=['real bounded storage/protocol binding','complete both-phase GPU qualification in separate processes',
                          'independent admission','live observer and terminal acceptance','formal launch after prior group acceptance'])
    with args.out.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(status=value['status'],test_count=len(tests),new_optimization_updates=0,new_test_evaluations=0)))


if __name__=='__main__': main()
