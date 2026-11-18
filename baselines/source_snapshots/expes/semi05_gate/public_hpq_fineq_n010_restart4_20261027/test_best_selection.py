"""Synthetic CPU disk tests for dedicated best retention and evidence binding."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import torch
import best_selection as b
import hpq_resources as r
import hpq_safety_state as s
from hpq_contract import INIT_SHA, sha_file
from selection_protocol import policy_sha


def signature(value):
    def plain(x):
        if torch.is_tensor(x):
            return dict(shape=list(x.shape), dtype=str(x.dtype), data=x.detach().cpu().tolist())
        if isinstance(x, dict): return {k: plain(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [plain(v) for v in x]
        return x
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True).encode()).hexdigest()


class BestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent, prefix='best_cpu_')
        self.root = Path(self.temp.name).absolute()
        self.store, self.output = self.root / 'store', self.root / 'output'
        self.output.mkdir()
        (self.output / 'manifest.json').write_text('{}')
        self.roots = {k: torch.nn.Linear(2, 2) for k in ('student', 'head', 'adapter', 'teacher', 'teacher_head', 'teacher_adapter')}
        self.engine = types.SimpleNamespace(torch=torch, output=self.output, cfg={}, spec={'phase': 'formal'},
            t=types.SimpleNamespace(matrix_updates=68, roots=lambda: self.roots),
            state=lambda: signature({k: m.state_dict() for k, m in self.roots.items()}), rng=lambda: 'rng',
            signature=signature)
        self.depth = 0
        self.guard = types.SimpleNamespace(targets=lambda: [(self.store / 'best/final.pth', r.FINAL_BOUND)],
            original_lock=self.lock, capacity_snapshot=self.capacity, MARGIN=1024**3+16*1024**2,
            memory_snapshot=lambda: dict(effective_available=r.RAM_REQUIRED + 1))
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(patch.object(b, 'STORE', self.store))
        self.stack.enter_context(patch.object(b, 'backend', return_value=self.guard))
        if os.name == 'nt':
            original = r.plain_file
            def portable(path):
                st = original(path)
                return types.SimpleNamespace(st_dev=st.st_dev, st_ino=st.st_ino, st_size=st.st_size,
                                             st_blocks=(st.st_size+511)//512)
            self.stack.enter_context(patch.object(r, 'plain_file', side_effect=portable))
            self.stack.enter_context(patch.object(r.os, 'statvfs', create=True,
                return_value=types.SimpleNamespace(f_bavail=64*1024**3, f_frsize=1)))
            self.stack.enter_context(patch.object(b, 'fsync_directory'))
        for epoch in range(3):
            (self.output / ('epoch_%02d_test_metrics.json' % epoch)).write_text(json.dumps({'epoch': epoch}))
            (self.output / ('epoch_%02d_test_cases.jsonl' % epoch)).write_text('{}\n')

    def tearDown(self):
        self.stack.close()
        self.temp.cleanup()

    @contextlib.contextmanager
    def lock(self):
        self.depth += 1
        try: yield
        finally: self.depth -= 1

    def capacity(self, required):
        self.assertEqual(self.depth, 1)
        self.assertEqual(required, r.RAM_REQUIRED)
        target = self.store / 'best/final.pth'
        if Path(str(target)+'.partial').exists(): raise RuntimeError('Existing partial')
        size = target.stat().st_size if target.exists() else 0
        return dict(filesystems=[dict(filesystem_device=self.root.stat().st_dev, probe=str(self.root),
            remaining_permanent=r.FINAL_BOUND-size, atomic_peak=3*1024**3,
            targets=[dict(path=str(target), bound_bytes=r.FINAL_BOUND, actual_bytes=size)])])

    def test_capacity_wait_then_real_best_write_without_replayed_training(self):
        import capacity_wait
        original = self.guard.capacity_snapshot
        calls = [0]
        state, rng = self.engine.state(), self.engine.rng()
        def capacity(required):
            calls[0] += 1
            if calls[0] == 1:
                raise RuntimeError(capacity_wait.PREFIX + json.dumps(dict(margin_bytes=-1)))
            return original(required)
        def sleep(seconds):
            self.assertEqual(self.depth, 0)
            self.assertEqual((self.engine.state(), self.engine.rng()), (state, rng))
            self.assertFalse((self.store / 'best/final.pth.partial').exists())
        self.guard.capacity_snapshot = capacity
        with patch.object(capacity_wait.time, 'sleep', side_effect=sleep):
            receipt = b.save_best(self.engine, 0)
        self.assertTrue(receipt['strict_cpu_readback_verified'])
        self.assertEqual((self.engine.state(), self.engine.rng()), (state, rng))
        self.assertEqual(len((self.output / 'capacity_wait.jsonl').read_text().splitlines()), 2)

    def test_best_survives_replacement_and_endpoint_reload(self):
        first = b.save_best(self.engine, 0)
        self.engine.t.matrix_updates = 136
        with torch.no_grad(): self.roots['student'].weight.add_(.1)
        second = b.save_best(self.engine, 1, first)
        self.assertEqual(second['previous_sha256'], first['sha256'])
        self.assertNotEqual(first['sha256'], second['sha256'])
        self.assertTrue((self.output / 'best_epoch_00.json').exists())
        self.assertEqual(b.verify_endpoint(self.engine, second)['new_evaluations'], 0)

    def test_failed_readback_preserves_previous_and_partial(self):
        first = b.save_best(self.engine, 0)
        self.engine.t.matrix_updates = 136
        with patch('hpq_engine.load_torch', return_value={'corrupt': True}):
            with self.assertRaisesRegex(ValueError, 'differs'): b.save_best(self.engine, 1, first)
        self.assertEqual(sha_file(first['path']), first['sha256'])
        self.assertTrue(Path(first['path']+'.partial').exists())
        self.assertFalse((self.output / 'best_epoch_01.json').exists())

    def test_existing_unowned_best_rejected(self):
        first = b.save_best(self.engine, 0)
        self.engine.t.matrix_updates = 136
        with self.assertRaisesRegex(ValueError, 'Unowned'): b.save_best(self.engine, 1)
        self.assertEqual(sha_file(first['path']), first['sha256'])

    def test_changed_prior_model_rejected(self):
        first = b.save_best(self.engine, 0)
        Path(first['path']).write_bytes(b'changed synthetic fixture')
        self.engine.t.matrix_updates = 136
        with self.assertRaisesRegex(ValueError, 'identity'): b.save_best(self.engine, 1, first)

    def test_qualification_removes_only_own_probe(self):
        self.engine.spec['phase'] = 'bridge'
        proof = b.save_best(self.engine, -1, probe=True)
        self.assertEqual(proof['status'], 'BEST_PROBE_PASS')
        self.assertFalse(self.store.exists())
        self.assertTrue((self.output / 'best_probe.json').is_file())

    def test_endpoint_changed_evaluation_rejected(self):
        first = b.save_best(self.engine, 0)
        (self.output / 'epoch_00_test_metrics.json').write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, 'binding'): b.verify_endpoint(self.engine, first)

    def test_budget_and_scope_rejected(self):
        self.engine.t.matrix_updates = 62
        with self.assertRaises(ValueError): b.save_best(self.engine, 0)
        self.engine.spec['phase'] = 'bridge'
        with self.assertRaises(ValueError): b.save_best(self.engine, 0)

    def test_metadata_rejects_other_epoch_or_supervisor(self):
        value = dict(format=b.FORMAT, epoch=0, updates=68, initializer_sha256=INIT_SHA,
            manifest_sha256='m', metrics_sha256='v', cases_sha256='c', selection_policy_sha256=policy_sha(),
            roots=dict.fromkeys(self.roots, {}), automatic_resume=False)
        b.validate_metadata(value, 0, 68, 'm', 'v', 'c')
        for key, val in [('epoch', 1), ('updates', 136), ('initializer_sha256', 'wrong'),
                         ('selection_policy_sha256', 'wrong'), ('automatic_resume', True), ('roots', {})]:
            with self.assertRaises(ValueError): b.validate_metadata(dict(value, **{key: val}), 0, 68, 'm', 'v', 'c')


if __name__ == '__main__': unittest.main()
