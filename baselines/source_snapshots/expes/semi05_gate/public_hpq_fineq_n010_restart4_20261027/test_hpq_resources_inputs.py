"""Local synthetic I/O and boundary tests, not remote/GPU qualification."""
import contextlib
import hashlib
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import hpq_inputs as inputs
import hpq_resources as resources
from hpq_contract import INIT_SHA, METRICS_SHA
from test_hpq_contract import splits


def payload():
    return dict(format=resources.WEIGHT_FORMAT, epoch=35, completed_epochs=36,
                candidate_schema='PUBLIC_N005_FINE_Q_V1', development_selection_used=True,
                successful_updates=2448, microbatches=12240, test_evaluations_before_save=35,
                fold=2, seed=2027, selection_rule='fixed_epoch35_native_student',
                exact_resume_supported=False, branch_routing_measured=False,
                weights={name: {'fake': 0} for name in resources.ROOT_NAMES})


class ResourceInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent, prefix='cpu_io_')
        self.root = Path(self.temp.name).resolve()
    def tearDown(self):
        self.assertEqual(self.root.parent, Path(__file__).resolve().parent)
        self.temp.cleanup()
    def file(self, name, raw):
        p = self.root / name
        p.write_bytes(raw)
        return p

    def test_bound_read_accepts_exact_bytes(self):
        raw = b'bounded metadata'
        p = self.file('meta', raw)
        self.assertEqual(inputs.bound_small_bytes(p, hashlib.sha256(raw).hexdigest()), raw)

    def test_changed_metadata_rejected(self):
        p = self.file('meta', b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA'):
            inputs.bound_small_bytes(p, '0' * 64)

    def test_oversized_metadata_rejected(self):
        p = self.file('meta', b'12345')
        with self.assertRaisesRegex(ValueError, 'Oversized'):
            inputs.bound_small_bytes(p, hashlib.sha256(b'12345').hexdigest(), maximum=4)

    def test_directory_not_a_file(self):
        with self.assertRaisesRegex(ValueError, 'regular file'):
            inputs.plain_file(self.root)

    def test_bound_split_actual_order(self):
        expected = splits()
        digests = {}
        for role, ids in expected.items():
            raw = ('\n'.join(ids) + '\n').encode()
            self.file({'train_lab': 'n010_lab.txt', 'train_unlab': 'n010_unlab.txt'}.get(role, role + '.txt'), raw)
            digests[role] = hashlib.sha256(raw).hexdigest()
        with patch.object(inputs, 'SPLIT_ROOT', self.root), patch.object(inputs, 'SPLIT_SHA', digests):
            self.assertEqual(inputs.read_bound_splits(), expected)
            self.file('n010_lab.txt', ('\n'.join(reversed(expected['train_lab'])) + '\n').encode())
            with self.assertRaisesRegex(ValueError, 'SHA'):
                inputs.read_bound_splits()

    def test_valid_sha_does_not_allow_bad_split_boundary(self):
        expected = splits()
        expected['test'][0] = expected['train_lab'][0]
        digests = {}
        for role, ids in expected.items():
            raw = ('\n'.join(ids) + '\n').encode()
            self.file({'train_lab': 'n010_lab.txt', 'train_unlab': 'n010_unlab.txt'}.get(role, role + '.txt'), raw)
            digests[role] = hashlib.sha256(raw).hexdigest()
        with patch.object(inputs, 'SPLIT_ROOT', self.root), patch.object(inputs, 'SPLIT_SHA', digests):
            with self.assertRaisesRegex(ValueError, 'overlap'):
                inputs.read_bound_splits()

    def test_acceptance_is_byte_and_identity_bound(self):
        raw = json.dumps(dict(status='VERIFIED_COMPLETE', final_sha256=INIT_SHA,
                              final_metrics_sha256=METRICS_SHA)).encode()
        p = self.file('acceptance.json', raw)
        with patch.object(inputs, 'ACCEPTANCE', p), patch.object(inputs, 'ACCEPTANCE_SHA', hashlib.sha256(raw).hexdigest()):
            self.assertEqual(inputs.read_accepted_n010()['final_sha256'], INIT_SHA)
            p.write_bytes(raw + b' ')
            with self.assertRaisesRegex(ValueError, 'SHA'):
                inputs.read_accepted_n010()

    def test_resource_backend_rejects_changed_code_before_import(self):
        p = self.file('guard.py', b'raise AssertionError("must not execute")')
        with patch.object(resources, 'BACKEND', p):
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                resources.backend()

    def test_bounded_writer_refuses_excess_before_write(self):
        raw = io.BytesIO()
        writer = resources.BoundedWriter(raw, 4)
        self.assertEqual(writer.write(b'123'), 3)
        with self.assertRaisesRegex(ValueError, 'reservation'):
            writer.write(b'45')
        self.assertEqual(raw.getvalue(), b'123')

    def test_payload_exact_fixed_budget(self):
        resources.validate_final_payload(payload())
        for key, value in (('epoch', 34), ('completed_epochs', 35), ('successful_updates', 2483),
                           ('microbatches', 12419), ('test_evaluations_before_save', 1), ('fold', 1),
                           ('seed', 2028), ('epoch', 35.0)):
            p = payload(); p[key] = value
            with self.assertRaises(ValueError, msg=key):
                resources.validate_final_payload(p)

    def test_payload_cannot_claim_resume_routing_or_best(self):
        for key, value in (('exact_resume_supported', True), ('branch_routing_measured', True),
                           ('selection_rule', 'best'), ('optimizer', {}), ('format', 'old')):
            p = payload(); p[key] = value
            with self.assertRaises(ValueError, msg=key):
                resources.validate_final_payload(p)

    def test_payload_all_six_nonempty_roots(self):
        for name in resources.ROOT_NAMES:
            p = payload(); p['weights'][name] = {}
            with self.assertRaisesRegex(ValueError, 'Empty root'):
                resources.validate_final_payload(p)

    def fake_guard(self, reject=False):
        events = []
        @contextlib.contextmanager
        def lock():
            events.append('locked')
            try: yield
            finally: events.append('unlocked')
        def capacity(required):
            self.assertEqual(events[-1], 'locked')
            self.assertEqual(required, resources.RAM_REQUIRED)
            if reject: raise RuntimeError('No capacity')
            events.append('capacity')
            return {'synthetic': True}
        return types.SimpleNamespace(original_lock=lock, capacity_snapshot=capacity), events

    def test_initializer_guard_holds_lock_across_load(self):
        p = self.file('init', b'fake')
        guard, events = self.fake_guard()
        with patch.object(resources, 'backend', return_value=guard), patch.object(resources, 'INIT', p):
            with resources.initializer_load_guard() as receipt:
                self.assertEqual(events, ['locked', 'capacity'])
                self.assertTrue(receipt['synthetic'])
            self.assertEqual(events[-1], 'unlocked')

    def test_capacity_failure_never_reaches_load(self):
        guard, events = self.fake_guard(reject=True)
        with patch.object(resources, 'backend', return_value=guard):
            with self.assertRaisesRegex(RuntimeError, 'No capacity'):
                with resources.initializer_load_guard():
                    self.fail('Must not load')
        self.assertEqual(events, ['locked', 'unlocked'])

    def save_context(self, save):
        guard, events = self.fake_guard()
        stack = contextlib.ExitStack()
        stack.enter_context(patch.object(resources, 'backend', return_value=guard))
        stack.enter_context(patch.object(resources, 'STORE', self.root))
        stack.enter_context(patch.object(resources, 'fsync_directory'))
        stack.enter_context(patch.dict('sys.modules', {'torch': types.SimpleNamespace(save=save)}))
        return stack, events

    @unittest.skipIf(__import__('os').name == 'nt',
                     'Actual atomic hardlink publication is mandatory in remote Linux CPU tests')
    def test_final_write_requires_capacity_and_does_not_claim_reload(self):
        def save(p, stream):
            self.assertEqual(events, ['locked', 'capacity'])
            stream.write(b'synthetic_weight')
        stack, events = self.save_context(save)
        with stack:
            receipt = resources.save_final(payload())
        self.assertFalse(receipt['strict_reload_verified'])
        self.assertEqual(receipt['sha256'], hashlib.sha256(b'synthetic_weight').hexdigest())
        self.assertFalse((self.root / 'final.pth.partial').exists())

    def test_existing_final_or_partial_never_overwritten(self):
        for name in ('final.pth', 'final.pth.partial'):
            with self.subTest(name=name):
                existing = self.file(name, b'preserve')
                stack, _ = self.save_context(lambda *args: self.fail('Must not serialize'))
                with stack, self.assertRaisesRegex(ValueError, 'Existing final/partial'):
                    resources.save_final(payload())
                self.assertEqual(existing.read_bytes(), b'preserve')
                existing.unlink()  # Only this synthetic fixture file.

    def test_failed_save_preserves_partial(self):
        def save(p, stream):
            stream.write(b'failure_evidence')
            raise RuntimeError('Synthetic serialization failure')
        stack, _ = self.save_context(save)
        with stack, self.assertRaisesRegex(RuntimeError, 'serialization'):
            resources.save_final(payload())
        self.assertEqual((self.root / 'final.pth.partial').read_bytes(), b'failure_evidence')
        self.assertFalse((self.root / 'final.pth').exists())

    def test_reload_rejects_wrong_digest(self):
        self.file('final.pth', b'fake')
        guard, _ = self.fake_guard()
        with patch.object(resources, 'backend', return_value=guard), patch.object(resources, 'STORE', self.root):
            with self.assertRaisesRegex(ValueError, 'changed'):
                with resources.final_reload_guard('0' * 64): self.fail('Must not reload')

    def test_no_replace_publication_refuses_raced_target(self):
        temp = self.file('new.partial', b'new')
        target = self.file('already_final', b'original')
        with self.assertRaises(FileExistsError):
            resources.publish_no_replace(temp, target)
        self.assertEqual(temp.read_bytes(), b'new')
        self.assertEqual(target.read_bytes(), b'original')


if __name__ == '__main__':
    unittest.main()
