import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest

from fixed_endpoint import Endpoint, FixedFinalRun, sha_file
from historical_kernels import (SPECS, load_pinned, class_method, extract_kernel,
                                MT_SOURCE, SDCL_SOURCE, UA_SOURCE)
from method_registry import METHODS, inspect


class FakeBackend:
    """Synthetic control-flow backend, never a real model or qualification."""
    def __init__(self, root):
        self.root, self.calls = Path(root), []
        self.bad_step = None
        self.bad_reload = False
        self.bad_cases = False
        self.bad_save = False
        self.fail_test = False
        self.bad_aggregate = False

    def prepare(self, endpoint):
        self.calls.append("prepare")
        return dict(source_pins_verified=True, input_pins_verified=True,
                    initialization_verified=True, resource_gate_passed=True, test_unopened=True)

    def train_iteration(self, remaining):
        self.calls.append("train")
        step = dict(labeled_inputs=1, unlabeled_inputs=1, finite_loss=True,
                    finite_nonzero_gradients=True, teacher_no_grad=True, skip_count=0,
                    actual_case_ids_verified=True, unlabeled_gt_in_loss=False,
                    successful_optimizer_updates=1)
        if self.bad_step:
            step.update(self.bad_step)
        return step

    def verify_training_coverage(self):
        self.calls.append("coverage")
        return dict(actual_case_coverage_verified=True)

    def save_terminal(self, counts):
        self.calls.append("save")
        p = self.root / "synthetic_final.bin"
        p.write_bytes(b"synthetic-test-only-terminal")
        return dict(path=str(p), sha256=sha_file(p), resource_lock_verified=not self.bad_save)

    def strict_reload(self, artifact):
        self.calls.append("reload")
        return dict(strict=not self.bad_reload, sha256=artifact["sha256"],
                    native_student_identity_verified=True)

    def test(self, artifact, permit):
        self.calls.append("test")
        receipt = json.loads(permit.read_text(encoding="utf-8"))
        assert receipt["test_evaluation"] == 1
        if self.fail_test:
            raise RuntimeError("Synthetic interrupted evaluation")
        return dict(case_ids=["a", "a"] if self.bad_cases else ["a", "b"],
                    case_dice=[0.2, 0.4], dice=0.9 if self.bad_aggregate else 0.3,
                    final_sha256=artifact["sha256"])


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="endpoint_test_", dir=Path(__file__).parent)
        self.root = Path(self.tmp.name)
        self.endpoint = Endpoint("synthetic", 2, 2, 1, ("a", "b"))
        self.backend = FakeBackend(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def execute(self):
        return FixedFinalRun(self.endpoint, self.root / "run").run(self.backend)

    def test_only_terminal_save_reload_then_one_test(self):
        result = self.execute()
        self.assertEqual(self.backend.calls, ["prepare", "train", "train", "coverage", "save", "reload", "test"])
        self.assertEqual(result["counts"]["successful_optimizer_updates"], 2)
        self.assertEqual(result["test_evaluations"], 1)

    def test_no_reuse_of_started_or_failed_output(self):
        self.backend.bad_step = dict(skip_count=1)
        with self.assertRaises(ValueError):
            self.execute()
        with self.assertRaises(FileExistsError):
            self.execute()
        self.assertNotIn("test", self.backend.calls)

    def test_skip_nonfinite_teacher_gt_or_bad_identity_blocks_test(self):
        fields = [dict(skip_count=1), dict(finite_loss=False),
                  dict(finite_nonzero_gradients=False), dict(teacher_no_grad=False),
                  dict(unlabeled_gt_in_loss=True), dict(actual_case_ids_verified=False),
                  dict(successful_optimizer_updates=0), dict(successful_optimizer_updates=True)]
        for i, bad in enumerate(fields):
            backend = FakeBackend(self.root)
            backend.bad_step = bad
            out = self.root / ("run_" + str(i))
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                FixedFinalRun(self.endpoint, out).run(backend)
            self.assertNotIn("test", backend.calls)
            self.assertTrue((out / "failure.json").exists())

    def test_budget_overrun_blocks_save(self):
        self.backend.bad_step = dict(labeled_inputs=3)
        with self.assertRaises(ValueError):
            self.execute()
        self.assertNotIn("save", self.backend.calls)

    def test_failed_reload_or_save_prevents_test(self):
        for i, flag in enumerate(("bad_reload", "bad_save")):
            backend = FakeBackend(self.root)
            setattr(backend, flag, True)
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                FixedFinalRun(self.endpoint, self.root / ("blocked_" + str(i))).run(backend)
            self.assertNotIn("test", backend.calls)

    def test_duplicate_test_case_rejected(self):
        self.backend.bad_cases = True
        with self.assertRaises(ValueError):
            self.execute()
        self.assertFalse((self.root / "run" / "final_metrics.json").exists())

    def test_wrong_case_mean_rejected(self):
        self.backend.bad_aggregate = True
        with self.assertRaises(ValueError):
            self.execute()

    def test_interrupted_test_keeps_intent_and_prevents_retry(self):
        self.backend.fail_test = True
        with self.assertRaises(RuntimeError):
            self.execute()
        self.assertTrue((self.root / "run" / "final_test_intent.json").exists())
        with self.assertRaises(FileExistsError):
            self.execute()
        self.assertEqual(self.backend.calls.count("test"), 1)


class KernelTests(unittest.TestCase):
    def test_scientific_prefix_ast_identical(self):
        for name, (source, cls, method, _) in SPECS.items():
            tree, _ = load_pinned(source)
            original_method = class_method(tree, cls, method)
            loop = next(n for n in ast.walk(original_method) if isinstance(n, ast.For)
                        and ast.unparse(n.iter) == "range(steps)")
            generated, _ = extract_kernel(name)
            prefix = generated.body[0].body[:-1]
            for i, statement in enumerate(prefix):
                self.assertEqual(ast.dump(statement, include_attributes=False),
                                 ast.dump(loop.body[i], include_attributes=False), (name, i))

    def test_no_test_best_resume_or_loop_termination_in_kernels(self):
        for name in SPECS:
            module, _ = extract_kernel(name)
            code = ast.unparse(module)
            compile(module, "synthetic_audit", "exec")
            for forbidden in ("_evaluate(", "_save(", "_load(", "best.pth", "break", "continue"):
                self.assertNotIn(forbidden, code)

    def test_mt_and_sdcl_optimizer_counts(self):
        for name, expected in (("mt_semisam", 1), ("uamt_semisam", 1),
                               ("sdcl", 2), ("sdcl_pretrain", 2)):
            _, report = extract_kernel(name)
            self.assertEqual(report["native_optimizer_calls"], expected)

    def test_uamt_actual_helper_is_pinned_separately(self):
        tree, _ = load_pinned(UA_SOURCE)
        helper = class_method(tree, "UAMTSemiSAMTrainer", "_specialist_consistency_loss")
        self.assertIn("_teacher_uncertainty", ast.unparse(helper))

    def test_only_main_table_methods_scheduled(self):
        self.assertEqual(set(METHODS), {"supervised", "sdcl", "mt_semisam", "uamt_semisam", "sfr", "relif3d", "hpq"})
        with self.assertRaises(ValueError):
            inspect("selftrain", "BY", 5)


if __name__ == "__main__":
    unittest.main()
