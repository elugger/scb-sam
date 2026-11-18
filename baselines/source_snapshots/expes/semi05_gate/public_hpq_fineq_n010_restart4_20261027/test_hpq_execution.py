"""New execution/dispatch boundary regressions; synthetic CPU, not GPU proof."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import hpq_engine as engine
import hpq_resources as resources
import hpq_preparation as preparation
import launch_hpq as launcher
from hpq_contract import BASE, STORE

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / 'remote_sync/run_fcd_public05_a1b_20261015.py'


def synthetic_row():
    return dict(microbatches=[dict(style=dict(donor1='a', donor2='b', complementary_sum=2.,
                     donor_labels_used=False, native_labeled_supervision=True, native_teacher=True,
                     old_fvc_active=False)) for _ in range(5)])


class ExecutionTests(unittest.TestCase):
    def test_style_valid(self):
        engine.validate_style(synthetic_row(), ['a', 'b'])

    def test_same_donor_rejected(self):
        row = synthetic_row()
        row['microbatches'][0]['style']['donor2'] = 'a'
        with self.assertRaises(ValueError): engine.validate_style(row, ['a', 'b'])

    def test_foreign_donor_rejected(self):
        with self.assertRaises(ValueError): engine.validate_style(synthetic_row(), ['a', 'c'])

    def test_mask_rejected(self):
        row = synthetic_row()
        row['microbatches'][0]['style']['complementary_sum'] = 1.
        with self.assertRaises(ValueError): engine.validate_style(row, ['a', 'b'])

    def test_hidden_gt_rejected(self):
        row = synthetic_row()
        row['microbatches'][0]['style']['donor_labels_used'] = True
        with self.assertRaises(ValueError): engine.validate_style(row, ['a', 'b'])

    def test_old_fvc_rejected(self):
        row = synthetic_row()
        row['microbatches'][0]['style']['old_fvc_active'] = True
        with self.assertRaises(ValueError): engine.validate_style(row, ['a', 'b'])

    def test_no_inherited_legacy_entries(self):
        # Test AST isolation using a small synthetic old module with toxic entries.
        text = 'raise RuntimeError("old module ran")\nclass _Engine:\n'
        text += ''.join('    def ' + name + '(self): return 1\n' for name in engine.METHODS)
        text += '    def formal(self): raise RuntimeError("old formal ran")\n'
        with patch.object(engine, 'bound_small_bytes', return_value=text.encode()):
            cls = engine.frozen_execution_helpers()
        self.assertFalse(hasattr(cls, 'formal'))
        self.assertFalse(hasattr(cls, 'bridge'))
        self.assertFalse(hasattr(cls, 'save'))
        self.assertEqual(set(cls.__dict__) - {'__module__', '__dict__', '__weakref__', '__doc__'}, set(engine.METHODS))

    def test_real_reference_ast_extraction(self):
        # Remote tests use the canonical SHA-pinned reference; local uses its mirror.
        ref = REFERENCE if REFERENCE.exists() else Path(engine.REFERENCE)
        self.assertEqual(engine.sha_file(ref), engine.REFERENCE_SHA)
        with patch.object(engine, 'REFERENCE', ref):
            cls = engine.frozen_execution_helpers()
        self.assertEqual(set(engine.METHODS) & set(cls.__dict__), set(engine.METHODS))

    def test_formal_has_no_old_evaluation(self):
        tree = ast.parse((HERE / 'hpq_engine.py').read_text())
        forbidden = {'_evaluate', '_save_ucp', 'preflight', 'run_paired', 'pretrain', 'semi_train'}
        self.assertFalse([n for n in ast.walk(tree) if isinstance(n, ast.Call)
                          and isinstance(n.func, ast.Attribute) and n.func.attr in forbidden])

    def test_test_only_after_save_and_reload(self):
        tree = ast.parse((HERE / 'selection_runner.py').read_text())
        formal = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
        calls = sorted((n.lineno, n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id)
                       for n in ast.walk(formal) if isinstance(n, ast.Call)
                       and isinstance(n.func, (ast.Name, ast.Attribute)))
        names = [name for _, name in calls]
        self.assertLess(names.index('diagnostic_test'), names.index('save_best'))
        self.assertLess(names.index('save_best'), names.index('verify_endpoint'))
        self.assertEqual(names.count('diagnostic_test'), 1)
        self.assertNotIn('final_test', names)
        loop = next(n for n in formal.body if isinstance(n, ast.For))
        self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr == 'final_test' for n in ast.walk(loop)))

    def test_report_write_exclusive(self):
        with tempfile.TemporaryDirectory(prefix='test_execution_', dir=HERE) as d:
            path = Path(d) / 'report.json'
            engine.write_json(path, {'a': 1})
            with self.assertRaises(FileExistsError): engine.write_json(path, {'a': 2})
            self.assertEqual(json.loads(path.read_text()), {'a': 1})

    def test_torch112_load_compatibility(self):
        seen = []
        def old_load(source, map_location): seen.append((source, map_location)); return 1
        self.assertEqual(engine.load_torch(types.SimpleNamespace(load=old_load), 'x'), 1)
        self.assertEqual(seen, [('x', 'cpu')])

    def test_modern_load_explicit(self):
        def new_load(source, map_location, weights_only=True): return weights_only
        self.assertIs(engine.load_torch(types.SimpleNamespace(load=new_load), 'x'), False)


class DispatchTests(unittest.TestCase):
    def test_paths_fixed(self):
        self.assertEqual(launcher.paths('formal')['store'], Path(STORE))
        self.assertEqual(launcher.paths('bridge')['output'], Path(BASE) / 'bridge_r1')

    def test_invalid_phase(self):
        with self.assertRaises(ValueError): launcher.paths('old_formal')

    def test_any_residual_blocks_dispatch(self):
        for name in launcher.paths('formal'):
            with self.subTest(name=name), tempfile.TemporaryDirectory(prefix='test_dispatch_', dir=HERE) as d:
                targets = {k: Path(d) / k for k in launcher.paths('formal')}
                targets[name].touch()
                with patch.object(launcher, 'paths', return_value=targets):
                    with self.assertRaises(ValueError): launcher.check_fresh('formal')

    def gpu(self, compute=0, util=0, graphics=None):
        display = [] if graphics is None else graphics
        return dict(uuid='GPU_CONFIGURED_AT_RUNTIME',
                    utilization_percent=util, memory_bytes=6*1024**2, graphics=display,
                    process_counts=dict(compute=compute, graphics=len(display)))

    def test_empty_gpu(self):
        with patch.object(launcher, 'read_gpu0', return_value=self.gpu()):
            self.assertEqual(launcher.gpu_snapshot(0)['gpu'], 0)

    def test_busy_gpu(self):
        with patch.object(launcher, 'read_gpu0', return_value=self.gpu(compute=1)):
            with self.assertRaises(ValueError): launcher.gpu_snapshot(0)

    def test_gpu1_rejected_before_telemetry(self):
        with patch.object(launcher, 'read_gpu0') as read:
            with self.assertRaises(ValueError): launcher.gpu_snapshot(1)
            read.assert_not_called()

    def test_nonzero_util_without_pid_rejected(self):
        with patch.object(launcher, 'read_gpu0', return_value=self.gpu(util=10)):
            with self.assertRaises(ValueError): launcher.gpu_snapshot(0)

    def test_small_known_xorg_only(self):
        for display, allowed in [([dict(name='Xorg', memory_bytes=5*1024**2)], True),
                                 ([dict(name='unknown', memory_bytes=5*1024**2)], False),
                                 ([dict(name='Xorg', memory_bytes=64*1024**2)], False)]:
            with patch.object(launcher, 'read_gpu0', return_value=self.gpu(graphics=display)):
                if allowed: launcher.gpu_snapshot(0)
                else:
                    with self.assertRaises(ValueError): launcher.gpu_snapshot(0)

    def test_no_queue_or_resume_cli(self):
        text = (HERE / 'launch_hpq.py').read_text()
        self.assertNotIn("'resume'", text)
        self.assertNotIn("'queue'", text)
        self.assertEqual(text.count('subprocess.Popen('), 1)


class ProbeCreditTests(unittest.TestCase):
    def scenario(self, free=80, dev=1, blocks=1):
        info = types.SimpleNamespace(st_dev=dev, st_ino=10, st_size=20, st_blocks=blocks)
        initial = {'filesystems': [dict(filesystem_device=1, probe='/', remaining_permanent=50,
                                        atomic_peak=30, targets=[])]}
        guard = types.SimpleNamespace(targets=lambda: [], MARGIN=10,
                                      memory_snapshot=lambda: {'effective_available': resources.RAM_REQUIRED})
        return info, initial, guard, types.SimpleNamespace(f_bavail=free, f_frsize=1)

    def test_own_allocated_credit_only(self):
        info, initial, guard, fs = self.scenario()
        with patch.object(resources, 'plain_file', return_value=info), patch.object(resources.os, 'statvfs', return_value=fs, create=True):
            value = resources.probe_read_capacity(guard, initial, Path('probe'), (1, 10, 20))
        self.assertEqual(value['filesystems'][0]['required_bytes'], 70)

    def test_no_cross_filesystem_credit(self):
        info, initial, guard, fs = self.scenario(dev=2)
        with patch.object(resources, 'plain_file', return_value=info), patch.object(resources.os, 'statvfs', return_value=fs, create=True):
            with self.assertRaises(ValueError): resources.probe_read_capacity(guard, initial, Path('probe'), (2, 10, 20))

    def test_sparse_unallocated_bytes_no_credit(self):
        info, initial, guard, fs = self.scenario(blocks=0)
        with patch.object(resources, 'plain_file', return_value=info), patch.object(resources.os, 'statvfs', return_value=fs, create=True):
            with self.assertRaises(ValueError): resources.probe_read_capacity(guard, initial, Path('probe'), (1, 10, 20))

    def test_foreign_inode_rejected(self):
        info, initial, guard, fs = self.scenario()
        with patch.object(resources, 'plain_file', return_value=info):
            with self.assertRaises(ValueError): resources.probe_read_capacity(guard, initial, Path('probe'), (1, 11, 20))

    def test_reload_ram_checked(self):
        info, initial, guard, fs = self.scenario()
        guard.memory_snapshot = lambda: {'effective_available': resources.RAM_REQUIRED - 1}
        with patch.object(resources, 'plain_file', return_value=info), patch.object(resources.os, 'statvfs', return_value=fs, create=True):
            with self.assertRaises(ValueError): resources.probe_read_capacity(guard, initial, Path('probe'), (1, 10, 20))


if __name__ == '__main__': unittest.main()
