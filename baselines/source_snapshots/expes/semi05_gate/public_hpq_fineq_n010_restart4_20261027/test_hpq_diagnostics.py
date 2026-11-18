"""Synthetic CPU regressions for new protocol; no image/test inference."""
import ast
import contextlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import types
import unittest
from unittest.mock import patch
import hpq_diagnostics as d
import hpq_engine as e
import hpq_resources as r

HERE = Path(__file__).resolve().parent


class DiagnosticsTests(unittest.TestCase):
    def test_epoch_range(self):
        self.assertEqual(d.marker_name(34), 'epoch_34_test_started.json')
        self.assertEqual(d.marker_name(35), 'epoch_35_test_started.json')
        for x in (-1, 36, True, 1.0, '0'):
            with self.assertRaises(ValueError): d.marker_name(x)

    def test_exclusive_own_epoch_full_budget_permit(self):
        with tempfile.TemporaryDirectory(dir=str(HERE), prefix='cpu_diag_') as tmp:
            base = Path(tmp); output = base / 'formal_r1'; output.mkdir()
            (output / 'manifest.json').write_text('{}')
            ids = ['T%03d' % n for n in range(87)]
            with patch.object(d, 'BASE', PurePosixPath(base.as_posix())):
                for phase, update in [('bridge', 68), ('formal', 65), ('formal', 67), ('formal', True)]:
                    with self.assertRaises(ValueError): d.claim_diagnostic(output, 0, update, phase, ids)
                marker = d.claim_diagnostic(output, 0, 68, 'formal', ids)
                d.validate_permit(marker, ids)
                with self.assertRaises(FileExistsError): d.claim_diagnostic(output, 0, 68, 'formal', ids)
                with self.assertRaises(ValueError): d.validate_permit(marker, list(reversed(ids)))
                with self.assertRaises(ValueError): d.claim_diagnostic(output, 1, 138, 'formal', ids)
                (output / 'manifest.json').write_text('{"changed":true}')
                with self.assertRaises(ValueError): d.validate_permit(marker, ids)

    def test_prior_requires_result_case_digest(self):
        with tempfile.TemporaryDirectory(dir=str(HERE), prefix='cpu_diag_') as tmp:
            p = Path(tmp)
            (p/'manifest.json').write_text('{}')
            ids = ['T%03d' % n for n in range(87)]
            marker = p/d.marker_name(0); marker.write_text(json.dumps(dict(epoch=0, case_ids=ids)))
            cases = p/'epoch_00_test_cases.jsonl'
            cases.write_text(''.join(json.dumps(dict(case_id=cid, actual_index=i, epoch=0, updates=68))+'\n'
                                     for i, cid in enumerate(ids)))
            row = dict(epoch=0, test_cases=87, updates=68, isolation_verified=True,
                       intent_sha256=d.sha_file(marker), manifest_sha256=d.sha_file(p/'manifest.json'),
                       cases_sha256=d.sha_file(cases))
            result = p/'epoch_00_test_metrics.json'; result.write_text(json.dumps(row))
            d.validate_prior(p, 1)
            cases.write_text('changed\n')
            with self.assertRaises(ValueError): d.validate_prior(p, 1)

    def fake_engine(self, restore=True):
        parameter = types.SimpleNamespace(grad=None)
        class Root:
            training = True
            def modules(self): return [self]
            def parameters(self): return [parameter]
            def eval(self): self.training = False
        root = Root()
        trainer = types.SimpleNamespace(matrix_updates=7, roots=lambda: {'student': root})
        @contextlib.contextmanager
        def scope(*args):
            try: yield
            finally:
                if restore: root.training = True
        trainer.isolated_context = scope
        return types.SimpleNamespace(t=trainer, state=lambda: 'weights-opt-scaler', rng=lambda: 'rng',
                                     signature=repr, torch=types.SimpleNamespace(no_grad=contextlib.nullcontext)), root, parameter

    def test_isolation_restores_modes(self):
        engine, root, _ = self.fake_engine()
        with d.isolated_evaluation(engine): self.assertFalse(root.training)
        self.assertTrue(root.training)

    def test_isolation_checks_even_when_forward_fails(self):
        engine, root, _ = self.fake_engine()
        with self.assertRaisesRegex(RuntimeError, 'forward'):
            with d.isolated_evaluation(engine): raise RuntimeError('forward')
        self.assertTrue(root.training)

    def test_mode_gradient_update_rng_mutations_rejected(self):
        engine, _, _ = self.fake_engine(False)
        with self.assertRaisesRegex(ValueError, 'modes'):
            with d.isolated_evaluation(engine): pass
        for kind in ('grad', 'updates', 'rng', 'state'):
            engine, _, param = self.fake_engine()
            with self.assertRaises(ValueError):
                with d.isolated_evaluation(engine):
                    if kind == 'grad': param.grad = 'changed'
                    elif kind == 'updates': engine.t.matrix_updates += 1
                    elif kind == 'rng': engine.rng = lambda: 'changed'
                    else: engine.state = lambda: 'changed'

    def test_epoch35_is_not_double_tested(self):
        import inspect
        source = (HERE / 'selection_runner.py').read_text()
        self.assertIn('range(36)', source)
        self.assertEqual(source.count('engine.diagnostic_test(epoch)'), 1)
        self.assertNotIn('.final_test(', source)
        self.assertIn('verify_endpoint(engine, selected)', source)

    def test_eval_has_no_backward_optimizer_or_teacher_update(self):
        import inspect
        tree = ast.parse(inspect.getsource(e))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Execution')
        node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate_cases')
        calls = {n.func.attr for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertFalse(calls & {'backward', 'step', 'zero_grad', '_update_teacher'})
        self.assertIn('isolated_evaluation(self)', inspect.getsource(e.Execution.evaluate_cases))

    def test_bridge_qualifies_eval_before_next_step(self):
        import inspect
        source = inspect.getsource(e.Execution.bridge)
        self.assertIn("self.evaluate_cases('train_lab'", source)
        self.assertLess(source.index('evaluation_probe ='), source.index('row = self.checked_step(raw[0])'))
        self.assertIn("self.record(row) == results['active_B'][0]", source)

    def test_resource_retarget_keeps_frozen_backend_and_empty_retired(self):
        import inspect
        source = inspect.getsource(r.backend)
        self.assertIn('original_targets() +', source)
        self.assertIn('verify_retired()', source)
        self.assertEqual(r.STOP_SHA, 'b47a658a9d56d1e1163423961f9a66cc722f60ad07db39789370ecc37177722b')
        self.assertNotEqual(str(r.RETIRED), str(r.STORE))


if __name__ == '__main__': unittest.main()
