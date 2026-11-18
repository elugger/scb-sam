"""New IO/lifecycle fault tests only; tiny pickle payloads and a MOCK gate.

No torch, patient data, GPU, model, training, real test or production registry.
The previous 98 metadata checks are not rerun here.
"""
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile

from relif_controller import FixedCutoffRun, file_sha
from relif_state import counts, plan, require
from relif_storage import RecoveryStore, read_pin, safe_file

PASSED = []


def check(name, action):
    action()
    PASSED.append(name)


def rejects(action):
    try:
        action()
    except (ValueError, FileExistsError):
        return
    raise AssertionError('Expected rejection')


class MockGate:
    def __init__(self):
        self.held, self.blocked = False, False

    @contextlib.contextmanager
    def original_lock(self):
        assert not self.held
        self.held = True
        try:
            yield
        finally:
            self.held = False

    def capacity_snapshot(self):
        assert self.held
        if self.blocked:
            raise ValueError('Insufficient filesystem capacity: synthetic')
        return dict(mock=True)


def storage_tests(base):
    def fixture(name, bound=65536):
        root = base / name
        job, artifacts = root / 'job', root / 'artifacts'
        job.mkdir(parents=True)
        artifacts.mkdir()
        gate = MockGate()
        validator = lambda s: require(s.get('payload_marker') == 'complete_mock_state', 'Missing mock state')
        store = RecoveryStore(job, artifacts, [artifacts / 'a' / 'final.pth', artifacts / 'b' / 'final.pth'],
                              bound, 65536, 'a'*64, gate, validator, pickle.dump, pickle.load,
                              sync_directory=lambda p: None)
        return store, gate

    def state(store, n):
        return dict(payload_marker='complete_mock_state', contract_sha256='a'*64,
                    execution_role='formal', counts=counts(n), ledger=store.ledger_pin(), mock_value=n)

    store, gate = fixture('normal')
    first = store.checkpoint(state(store, 0))
    commits = [first]
    for n in range(1, 4):
        store.append_step(dict(mock_step=n))
        commits.append(store.checkpoint(state(store, n)))
    check('two_current_generations_and_monotonic_commits', lambda: require(
        [c['generation'] for c in store.available()] == [3, 2], 'Wrong generations'))
    current = store.available()[0]
    review = dict(contract_sha256='a'*64, checkpoint_sha256=current['sha256'], generation=3,
                  independent_resume_review_passed=True, consumed_by_pid=os.getpid())
    check('same_fd_reviewed_pickle_roundtrip', lambda: require(
        store.load_reviewed(current, review) == state(store, 3), 'Roundtrip differs'))
    check('wrong_checkpoint_review_rejected', lambda: rejects(lambda: store.load_reviewed(
        current, dict(review, checkpoint_sha256='b'*64))))
    check('unconsumed_review_rejected', lambda: rejects(lambda: store.load_reviewed(
        current, dict(review, consumed_by_pid=-1))))
    check('engineering_state_rejected', lambda: rejects(lambda: store.checkpoint(
        dict(state(store, 3), execution_role='engineering_larc'))))
    check('wrong_contract_rejected', lambda: rejects(lambda: store.checkpoint(
        dict(state(store, 3), contract_sha256='b'*64))))
    check('missing_state_rejected', lambda: rejects(lambda: store.checkpoint(
        dict(state(store, 3), payload_marker=None))))
    check('ledger_pin_mismatch_rejected', lambda: rejects(lambda: store.checkpoint(
        dict(state(store, 3), ledger=commits[0]['ledger']))))
    check('parent_traversal_rejected', lambda: rejects(lambda: safe_file(store.job/'../escape', store.job)))
    check('outside_root_rejected', lambda: rejects(lambda: safe_file(base/'outside', store.job)))
    check('oversized_ledger_row_rejected', lambda: rejects(lambda: store.append_step({'x':'x'*8192})))
    before = store.ledger_pin()
    gate.blocked = True
    check('mock_capacity_rejects_before_append', lambda: rejects(lambda: store.append_step({'n':4})))
    require(store.ledger_pin() == before, 'Blocked append changed ledger')
    gate.blocked = False
    store.record('formal/evidence.json', {'a':1})
    check('exclusive_evidence_rejected', lambda: rejects(lambda: store.record('formal/evidence.json', {'a':1})))
    check('identical_idempotent_evidence', lambda: store.record('formal/evidence.json', {'a':1}, True))
    check('different_idempotent_evidence_rejected', lambda: rejects(lambda: store.record('formal/evidence.json', {'a':2}, True)))
    with (store.job/'formal/steps.jsonl').open('ab') as f:
        f.write(b'{"partial":')
    check('partial_ledger_never_appended', lambda: rejects(lambda: store.append_step({'n':4})))
    check('partial_ledger_never_loaded', lambda: rejects(lambda: store.load_reviewed(current, review)))
    small, _ = fixture('overflow', 256)
    check('oversized_payload_rejected', lambda: rejects(lambda: small.checkpoint(
        dict(state(small, 0), extra='x'*4096))))
    require(Path(str(small.slots[0])+'.partial').exists() and not small.slots[0].exists(), 'Partial evidence not preserved')
    check('partial_checkpoint_blocks_retry', lambda: rejects(lambda: small.checkpoint(state(small, 0))))
    corrupt, _ = fixture('corrupt')
    corrupt.checkpoint(state(corrupt, 0))
    corrupt.slots[0].write_bytes(b'corrupt')
    check('corrupt_current_generation_blocks_new_save', lambda: rejects(lambda: corrupt.checkpoint(state(corrupt, 0))))
    orphan, _ = fixture('orphan')
    orphan.slots[0].parent.mkdir()
    orphan.slots[0].write_bytes(b'uncommitted')
    check('uncommitted_existing_slot_not_overwritten', lambda: rejects(lambda: orphan.checkpoint(state(orphan, 0))))


class MockBackend:
    def __init__(self, output, fault=None):
        self.execution_role, self.recipe = 'formal', plan(5)
        self.output, self.fault = output, fault
        self.output.mkdir()
        self.progress, self.updated, self.test_calls = counts(0), 0, 0
        self.pairs = [dict(labeled='L', unlabeled='U', schedule_index=i) for i in range(12420)]
        self.checkpoint_positions = []
        self.ids = ['case_%03d'%i for i in range(87)]

    def guard_run_lineage(self, resume):
        require(not resume, 'Mock has no resume review')
        require(not any((self.output/n).exists() for n in ('run_intent.json','failure.json','final_test_intent.json')),
                'Run already used')

    def record(self, name, value, consistent=False):
        path = self.output/name
        with path.open('x', encoding='utf-8') as stream:
            json.dump(value, stream)

    def prepare_initial(self):
        return {k:True for k in ('source_pins_verified','input_pins_verified','initialization_verified',
                                'resource_gate_passed','test_unopened')}

    def checkpoint(self, progress):
        require(progress == self.progress, 'Mock checkpoint counters')
        self.checkpoint_positions.append(progress['iterations'])

    def original_clock(self, i):
        return dict(step=i,total=12420)

    def train_iteration(self, i):
        assert i == self.updated
        if self.fault == 'update_failure' and i == 1:
            raise ValueError('Injected partial forward failure')
        self.updated += 1
        self.progress = counts(self.updated)
        row = dict(labeled_inputs=1, unlabeled_inputs=1, successful_optimizer_updates=1,
                   finite_loss=True, finite_nonzero_gradients=True, actual_case_ids_verified=True,
                   teacher_no_grad=True, skip_count=0, unlabeled_gt_in_loss=False,
                   actual_pair=self.pairs[i], completed_primary_pairs=i+1, gradient_norm=1.,
                   losses={k:.1 for k in ('loss','supervised_ce','supervised_dice','distillation','cgcr','larc')},
                   native_receipt=dict(step=i+1,clock=self.original_clock(i),primary_labeled=1,
                   primary_unlabeled=1,optimizer_updates=1,student_case_forwards=4,foundation_case_forwards=2,
                   sobf_views=2,teacher_case_forwards=0,extra_donor_inputs=0,test_evaluations=0,
                   unlab_gt_in_loss=False,case_ids=dict(labeled='L',unlabeled='U')))
        if self.fault == 'compressed_clock':
            row['native_receipt']['clock']['total'] = 3795
        return row

    def append_step(self, value):
        if self.fault == 'ledger_failure':
            raise ValueError('Injected ledger write failure')

    def verify_training_coverage(self, progress):
        assert progress == counts(3795)
        return dict(actual_case_coverage_verified=True,complete_reference_cycles=11,
                    primary_labeled_inputs=3795,primary_unlabeled_inputs=3795)

    def save_terminal(self, progress):
        path = self.output/'mock_final.bin'
        path.write_bytes(b'tiny_mock_no_model')
        return dict(path=str(path),sha256=file_sha(path),resource_lock_verified=True)

    def strict_reload(self, artifact):
        return dict(strict=True,native_student_identity_verified=True,
                    native_projection_identity_verified=self.fault != 'projection_reload',sha256=artifact['sha256'])

    def verify_test_prerequisites(self):
        require(self.fault != 'import_failure', 'Pinned test module prerequisite failed')

    def guard_test_lineage(self):
        require(not (self.output/'final_test_intent.json').exists(), 'Test already opened')

    def test(self, artifact, permit):
        self.test_calls += 1
        require(self.fault != 'test_failure', 'Injected failure after unique intent')
        rows = [dict(case_id=c,actual_index=i,dice=.5,jaccard=.4,hd95=2.,asd=1.) for i,c in enumerate(self.ids)]
        if self.fault == 'duplicate_case':
            rows[-1]['case_id'] = rows[0]['case_id']
        with (self.output/'final_test_cases.jsonl').open('x') as stream:
            for row in rows:
                stream.write(json.dumps(row)+'\n')
        return dict(case_ids=self.ids,case_dice=[.5]*87,dice=.5,jaccard=.4,hd95=2.,
                    asd=99. if self.fault == 'asd_aggregate' else 1.,final_sha256=artifact['sha256'],
                    test_evaluations=1,distance_unit='voxel')


def controller_tests(base):
    backend = MockBackend(base/'normal_controller')
    run = FixedCutoffRun(backend.recipe, backend.output, backend.ids)
    result = run.run(backend)
    check('full_mock_cutoff_3795_and_single_terminal_test', lambda: require(
        result['counts'] == counts(3795) and backend.updated == 3795 and backend.test_calls == 1
        and backend.checkpoint_positions == [0]+list(range(100,3800,100))+[3795], 'Mock lifecycle mismatch'))
    check('completed_run_cannot_restart', lambda: rejects(lambda: run.run(backend)))
    for fault in ('compressed_clock','update_failure','ledger_failure','projection_reload','import_failure',
                  'test_failure','duplicate_case','asd_aggregate'):
        backend = MockBackend(base/fault, fault)
        check('controller_rejects_'+fault, lambda b=backend: rejects(lambda: FixedCutoffRun(b.recipe,b.output,b.ids).run(b)))
        evidence = json.loads((backend.output/'failure.json').read_text())
        require(evidence['failed_partial_compute_not_assumed_zero'] is True, 'Partial cost hidden')
        if fault in ('compressed_clock','update_failure','ledger_failure','projection_reload','import_failure'):
            require(backend.test_calls == 0 and not (backend.output/'final_test_intent.json').exists(), 'Test prematurely opened')
        else:
            require(backend.test_calls == 1 and (backend.output/'final_test_intent.json').exists(), 'Test intent not preserved')
        rejects(lambda: FixedCutoffRun(backend.recipe,backend.output,backend.ids).run(backend))


def main():
    # Only this uniquely generated owned temporary directory is removed by
    # TemporaryDirectory; no production files, reservations or evidence touched.
    scratch = Path(__file__).resolve().parent / '.cpu_test_scratch'
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='relif_io_cpu_', dir=scratch) as temporary:
        base = Path(temporary).resolve()
        require(base.parent == scratch and base.name.startswith('relif_io_cpu_'),
                'Unexpected test workspace')
        storage_tests(base)
        controller_tests(base)
    print(json.dumps(dict(passed=True,checks=len(PASSED),names=PASSED,
          proof_scope='synthetic lifecycle and tiny pickle IO; mock resource gate; directory fsync mocked',
          model_constructions=0,cuda_loads=0,patient_loads=0,optimizer_updates=0,actual_tests=0,
          production_jobs=0,reservations=0,dispatches=0,
          sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}),indent=2))


if __name__ == '__main__':
    main()
