"""Independent CPU-only contract tests for the new basic-supervised runner.

These tests never import torch, construct a model, contact the server, or run
training/test inference. Static control-flow checks complement (not replace)
the launcher's live source, resource, process, and checkpoint checks.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


RUNNER_PATH = Path(__file__).with_name("train_public_basic_sup_20261018.py")
SOURCE = RUNNER_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)
SPEC = importlib.util.spec_from_file_location("basic_sup_cpu_subject", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
GUARD_PATH = Path(__file__).with_name("public_basic_sup_guard_20261018.py")
GUARD_SPEC = importlib.util.spec_from_file_location("basic_sup_cpu_guard_subject", GUARD_PATH)
GUARD = importlib.util.module_from_spec(GUARD_SPEC)
GUARD_SPEC.loader.exec_module(GUARD)


def minimal_config():
    return {"project_root": "/project", "output_dir": "output/fresh", "artifact_dir": "artifacts/fresh",
            "fold_cv_dir": "data/fold_2", "train_list": "train.txt", "test_list": "test.txt",
            "sam_checkpoint": "generic_sam.pth", "sam_checkpoint_sha256": "a" * 64,
            "source_sha256": {"runner.py": "b" * 64},
            "input_sha256": {"train.txt": "c" * 64, "test.txt": "d" * 64},
            "resource_guard_module": "public_basic_sup_guard_20261018", "train_count": 17,
            "test_count": 87, "label_percent": 5, "fold": 2, "seed": 2027, "epochs": 60}


class BudgetAndConfigurationTests(unittest.TestCase):
    def test_authorized_arm_budgets(self):
        for percent, count in ((5, 17), (10, 35), (100, 346)):
            with self.subTest(percent=percent):
                cfg = minimal_config()
                cfg.update(label_percent=percent, train_count=count)
                budget = RUNNER.validate_config(cfg)
                self.assertEqual(budget, {"epochs": 60, "train_count": count, "updates_per_epoch": count,
                                        "total_updates": count * 60, "batch_size": 1, "replay_per_epoch": 0})

    def test_invalid_training_counts(self):
        for count in (0, -1, True, 17.0, "17", None):
            with self.subTest(count=count), self.assertRaises(ValueError):
                RUNNER.training_budget(count)

    def test_invalid_epoch_counts(self):
        for epochs in (0, -1, True, 60.0, "60", None):
            with self.subTest(epochs=epochs), self.assertRaises(ValueError):
                RUNNER.training_budget(17, epochs)

    def test_fixed_protocol_values(self):
        for key, value in (("fold", 0), ("fold", True), ("seed", 2026), ("epochs", 59),
                           ("epochs", 61), ("epochs", 60.0), ("test_count", 86), ("label_percent", 20),
                           ("label_percent", True), ("label_percent", 5.0)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                cfg = minimal_config()
                cfg[key] = value
                RUNNER.validate_config(cfg)

    def test_missing_required_fields(self):
        for key in minimal_config():
            with self.subTest(key=key), self.assertRaises(ValueError):
                cfg = minimal_config()
                del cfg[key]
                RUNNER.validate_config(cfg)

    def test_no_hpq_unlabeled_resume_or_best_initialization(self):
        for key, value in (("use_hpq", True), ("use_unlabeled", True), ("resume", True),
                           ("mode", "semi"), ("test_policy", "test_best"), ("batch_size", 2),
                           ("num_workers", 1), ("deep_supervision", True), ("img_size", 96),
                           ("model_type", "other"), ("num_classes", 3), ("init_from", "old_best.pth"),
                           ("coarse_checkpoint", "old.pth"), ("teacher_checkpoint", "old.pth"),
                           ("best_checkpoint", "old.pth"), ("resume_from", "latest.pth")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                cfg = minimal_config()
                cfg[key] = value
                RUNNER.validate_config(cfg)

    def test_frozen_optimizer_and_loss(self):
        for key, value in (("lr", 5e-8), ("lr_encoder_scale", 0.1), ("lr_decoder_scale", 1),
                           ("lr_coarse_scale", 1), ("weight_decay", 0), ("aux_weight", 0),
                           ("beta1", 0.8), ("beta2", 0.99), ("amp_init_scale", 65536.0)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                cfg = minimal_config()
                cfg[key] = value
                RUNNER.validate_config(cfg)

    def test_empty_manifests(self):
        for key in ("source_sha256", "input_sha256"):
            for value in ({}, [], "sha"):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    cfg = minimal_config()
                    cfg[key] = value
                    RUNNER.validate_config(cfg)

    def test_percentage_count_mismatch_refused(self):
        for percent, count in ((5, 35), (10, 34), (10, 17), (100, 433), (100, 17)):
            with self.subTest(percent=percent, count=count), self.assertRaises(ValueError):
                cfg = minimal_config()
                cfg.update(label_percent=percent, train_count=count)
                RUNNER.validate_config(cfg)

    def test_malformed_checkpoint_hash_refused(self):
        for value in ("a" * 63, "a" * 65, "g" * 64, "A" * 64, "", None, 123):
            with self.subTest(value=value), self.assertRaises(ValueError):
                cfg = minimal_config()
                cfg["sam_checkpoint_sha256"] = value
                RUNNER.validate_config(cfg)

    def test_malformed_manifest_hash_refused(self):
        for group in ("source_sha256", "input_sha256"):
            with self.subTest(group=group), self.assertRaises(ValueError):
                cfg = minimal_config()
                cfg[group][next(iter(cfg[group]))] = "bad"
                RUNNER.validate_config(cfg)


class CaseListTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="basic-sup-cpu-", dir=RUNNER_PATH.parent)
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "cases.txt"

    def write_read(self, text):
        self.path.write_text(text, encoding="utf-8")
        return RUNNER.read_case_ids(self.path)

    def test_order_preserved_and_blank_lines_ignored(self):
        self.assertEqual(self.write_read(" b \n\na\nc\n"), ["b", "a", "c"])

    def test_empty_list(self):
        with self.assertRaises(ValueError):
            self.write_read(" \n\n")

    def test_duplicate_ids(self):
        with self.assertRaises(ValueError):
            self.write_read("a\nb\na\n")

    def test_unsafe_ids(self):
        for case in (".", "..", "../other", "a/b", "a\\b", "C:drive", "a\x00b", "image.nii", "image.nii.gz"):
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.write_read(case + "\n")

    def test_disjoint_exact_sets(self):
        self.assertEqual(RUNNER.validate_case_sets(["a", "b"], ["c"], 2, 1),
                         {"train_count": 2, "test_count": 1, "case_sets_disjoint": True})

    def test_overlap_duplicate_empty_and_wrong_counts(self):
        scenarios = [(["a"], ["a"], 1, 1), (["a", "a"], ["b"], 2, 1),
                     (["a"], ["b", "b"], 1, 2), ([], ["b"], 0, 1),
                     (["a"], [], 1, 0), (["a"], ["b"], 2, 1), (["a"], ["b"], 1, 2)]
        for args in scenarios:
            with self.subTest(args=args), self.assertRaises(ValueError):
                RUNNER.validate_case_sets(*args)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="basic-sup-manifest-", dir=RUNNER_PATH.parent)
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        for name, content in (("train.txt", b"train\n"), ("test.txt", b"test\n"), ("sam.pth", b"fake-small-checkpoint")):
            (root / name).write_bytes(content)
        self.cfg = minimal_config()
        self.cfg.update(project_root=str(root), sam_checkpoint="sam.pth",
                        sam_checkpoint_sha256=RUNNER.sha256_file(root / "sam.pth"),
                        source_sha256={str(RUNNER_PATH.resolve()): RUNNER.sha256_file(RUNNER_PATH)},
                        input_sha256={name: RUNNER.sha256_file(root / name) for name in ("train.txt", "test.txt")})

    def test_exact_files_and_own_source_bound(self):
        checked = RUNNER.verify_manifests(self.cfg)
        self.assertEqual(len(checked), 4)
        self.assertEqual(checked[str(RUNNER_PATH.resolve())]["kind"], "source_sha256")

    def test_source_mismatch_refused(self):
        self.cfg["source_sha256"][str(RUNNER_PATH.resolve())] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "SHA mismatch"):
            RUNNER.verify_manifests(self.cfg)

    def test_missing_own_binding_refused(self):
        self.cfg["source_sha256"] = {}
        with self.assertRaisesRegex(RuntimeError, "Runner source"):
            RUNNER.verify_manifests(self.cfg)

    def test_changed_list_refused(self):
        Path(self.cfg["project_root"], "train.txt").write_text("changed\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "SHA mismatch"):
            RUNNER.verify_manifests(self.cfg)

    def test_unbound_test_list_refused(self):
        del self.cfg["input_sha256"]["test.txt"]
        with self.assertRaisesRegex(RuntimeError, "test_list is not bound"):
            RUNNER.verify_manifests(self.cfg)

    def test_checkpoint_mismatch_refused(self):
        self.cfg["sam_checkpoint_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "checkpoint SHA mismatch"):
            RUNNER.verify_manifests(self.cfg)


class FakeTensor:
    def __init__(self, values, ndim=4):
        self.values, self.ndim = list(values), ndim
        self.shape = (1, len(self.values), 1, 1)

    def clone(self):
        return FakeTensor(self.values, self.ndim)

    def detach(self):
        return self

    def long(self):
        return self

    def dim(self):
        return self.ndim

    def squeeze(self, _axis):
        return FakeTensor(self.values, self.ndim - 1)

    def __eq__(self, other):
        return FakeTensor([value == other for value in self.values], self.ndim)

    def __or__(self, other):
        return FakeTensor([a or b for a, b in zip(self.values, other.values)], self.ndim)

    def all(self):
        return all(self.values)

    def sum(self):
        return types.SimpleNamespace(item=lambda: sum(self.values))


class StrictDatasetTests(unittest.TestCase):
    def dataset(self, label_values):
        reads = []

        class FakeSitkImage:
            def __init__(self, name):
                self.name = name
            def GetOrigin(self):
                return (0, 0, 0)
            def GetDirection(self):
                return (1, 0, 0)

        def read_image(name):
            reads.append(name)
            return FakeSitkImage(name)

        class FakeBaseDataset:
            def __init__(self, _root, ids, transform, threshold, split):
                self.image_paths = [case + "_image" for case in ids]
                self.label_paths = [case + "_label" for case in ids]
                self.transform = transform
            def __getitem__(self, _index):
                raise AssertionError("Parent replacement-capable __getitem__ must never be called")

        sitk = types.ModuleType("SimpleITK")
        sitk.ReadImage = read_image
        tio = types.ModuleType("torchio")
        tio.ScalarImage = types.SimpleNamespace(from_sitk=lambda _image: types.SimpleNamespace(data=FakeTensor([1, 2])))
        tio.LabelMap = types.SimpleNamespace(from_sitk=lambda _image: types.SimpleNamespace(data=FakeTensor(label_values)))
        tio.Subject = lambda **kwargs: types.SimpleNamespace(**kwargs)
        semi = types.ModuleType("expes.common.semi_data")
        semi.CaseListDataset = FakeBaseDataset
        semi.build_train_transform = lambda _size, random_flip: lambda subject: subject
        semi.build_eval_transform = lambda _size: lambda subject: subject
        modules = {"SimpleITK": sitk, "torchio": tio, "expes": types.ModuleType("expes"),
                   "expes.common": types.ModuleType("expes.common"), "expes.common.semi_data": semi}
        with patch.dict(sys.modules, modules):
            dataset = RUNNER.make_strict_dataset(minimal_config(), ["alpha", "beta"], "test")
        return dataset, reads

    def test_zero_foreground_keeps_requested_case(self):
        dataset, reads = self.dataset([0, 0])
        row = dataset[1]
        self.assertEqual(row["case_id"], "beta")
        self.assertEqual(row["requested_index"], 1)
        self.assertEqual(row["foreground_voxels"], 0)
        self.assertEqual(reads, ["beta_image", "beta_label"])

    def test_small_foreground_keeps_requested_case(self):
        dataset, reads = self.dataset([0, 1])
        row = dataset[0]
        self.assertEqual(row["case_id"], "alpha")
        self.assertEqual(row["foreground_voxels"], 1)
        self.assertEqual(reads, ["alpha_image", "alpha_label"])

    def test_nonbinary_label_refused(self):
        dataset, _ = self.dataset([0, 2])
        with self.assertRaisesRegex(RuntimeError, "Non-binary"):
            dataset[0]


class StateHashTests(unittest.TestCase):
    class Tensor:
        def __init__(self, data=b"1234", dtype="float32", shape=(1,)):
            self.data, self.dtype, self.shape = data, dtype, shape
        def detach(self):
            return self
        def cpu(self):
            return self
        def contiguous(self):
            return self
        def reshape(self, _size):
            return self
        def view(self, _dtype):
            return self
        def numpy(self):
            return self.data

    def hash(self, values):
        fake_torch = types.ModuleType("torch")
        fake_torch.uint8 = "uint8"
        with patch.dict(sys.modules, {"torch": fake_torch}):
            return RUNNER.state_dict_sha256(values)

    def test_state_hash_order_independent(self):
        a, b = self.Tensor(b"aaaa"), self.Tensor(b"bbbb")
        self.assertEqual(self.hash({"a": a, "b": b}), self.hash({"b": b, "a": a}))

    def test_state_hash_detects_bytes_and_name_changes(self):
        expected = self.hash({"a": self.Tensor()})
        self.assertNotEqual(expected, self.hash({"b": self.Tensor()}))
        self.assertNotEqual(expected, self.hash({"a": self.Tensor(b"4321")}))

    def test_state_hash_detects_shape_and_dtype_changes(self):
        expected = self.hash({"a": self.Tensor()})
        self.assertNotEqual(expected, self.hash({"a": self.Tensor(dtype="int32")}))
        self.assertNotEqual(expected, self.hash({"a": self.Tensor(shape=(1, 1))}))


class StaticScientificBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.run = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == "run")
        self.calls = [node for node in ast.walk(self.run) if isinstance(node, ast.Call)]

    def call_named(self, name):
        return [node for node in self.calls if
                (isinstance(node.func, ast.Name) and node.func.id == name) or
                (isinstance(node.func, ast.Attribute) and node.func.attr == name)]

    def test_no_heavy_top_level_imports(self):
        forbidden = ("torch", "torchio", "numpy", "SimpleITK", "segment_anything", "expes")
        for node in TREE.body:
            if isinstance(node, ast.Import):
                self.assertFalse(any(alias.name.startswith(forbidden) for alias in node.names))
            elif isinstance(node, ast.ImportFrom):
                self.assertFalse((node.module or "").startswith(forbidden))

    def test_no_hpq_selftrain_teacher_or_pseudo_imports(self):
        imports = [node.module or "" for node in ast.walk(TREE) if isinstance(node, ast.ImportFrom)]
        self.assertFalse(any(any(token in name.lower() for token in ("paired_fcd", "selftrain", "uncertainty", "weak_strong"))
                             for name in imports))
        called = [ast.unparse(node.func).lower() for node in self.calls]
        self.assertFalse(any(any(token in name for token in ("teacher", "pseudo_loss", "context_loss", "ema_update"))
                             for name in called))

    def test_single_fresh_basic_model_construction(self):
        builders = self.call_named("build_auto_sam_seg")
        self.assertEqual(len(builders), 1)
        kwargs = {item.arg: item.value for item in builders[0].keywords}
        self.assertIsNone(ast.literal_eval(kwargs["coarse_checkpoint"]))
        self.assertEqual(ast.literal_eval(kwargs["model_type"]), "vit_b_ori")
        self.assertFalse(ast.literal_eval(kwargs["deep_supervision"]))
        seed_calls = self.call_named("seed") + self.call_named("manual_seed") + self.call_named("manual_seed_all")
        fixed_before_build = [node for node in seed_calls if node.lineno < builders[0].lineno and node.args
                              and isinstance(node.args[0], ast.Constant) and node.args[0].value == 2027]
        self.assertGreaterEqual(len(fixed_before_build), 4)

    def test_initialization_identity_before_any_dataset_construction(self):
        writes = [node for node in self.call_named("atomic_json") if node.args and
                  "initialization.json" in ast.unparse(node.args[0])]
        self.assertEqual(len(writes), 1)
        self.assertLess(writes[0].lineno, min(node.lineno for node in self.call_named("make_strict_dataset")))
        payload = ast.unparse(writes[0].args[1])
        self.assertIn("full_student_state_sha256", payload)
        self.assertIn("coarse_state_sha256", payload)
        self.assertIn("sam_checkpoint_sha256", payload)

    def test_fixed_amp_scale_and_unverified_continuation_disclosed(self):
        scalers = self.call_named("GradScaler")
        self.assertEqual(len(scalers), 1)
        kwargs = {item.arg: item.value for item in scalers[0].keywords}
        self.assertEqual(ast.literal_eval(kwargs["init_scale"]), 1024.0)
        literals = [node.value for node in ast.walk(self.run) if isinstance(node, ast.Constant)]
        self.assertIn("continuation_replay_verified", literals)
        self.assertNotIn("public_basic_sup_exact_epoch_boundary_v1", literals)

    def test_single_test_dataset_and_metric_call_after_training(self):
        epoch_loop = next(node for node in ast.walk(self.run) if isinstance(node, ast.For)
                          and isinstance(node.target, ast.Name) and node.target.id == "epoch")
        self.assertEqual(ast.unparse(epoch_loop.iter), "range(60)")
        test_builders = [node for node in self.call_named("make_strict_dataset") if
                         len(node.args) == 3 and isinstance(node.args[2], ast.Constant) and node.args[2].value == "test"]
        self.assertEqual(len(test_builders), 1)
        self.assertGreater(test_builders[0].lineno, epoch_loop.end_lineno)
        metrics = self.call_named("binary_case_metrics")
        self.assertEqual(len(metrics), 1)
        self.assertGreater(metrics[0].lineno, test_builders[0].lineno)

    def test_final_saved_and_strict_reloaded_before_test(self):
        final_saves = [node for node in self.call_named("guarded_torch_save") if any(
            kw.arg == "save_kind" and isinstance(kw.value, ast.Constant) and kw.value.value == "final" for kw in node.keywords)]
        self.assertEqual(len(final_saves), 1)
        loads = self.call_named("load_state_dict")
        self.assertEqual(len(loads), 1)
        self.assertTrue(any(kw.arg == "strict" and ast.literal_eval(kw.value) is True for kw in loads[0].keywords))
        test_builder = next(node for node in self.call_named("make_strict_dataset") if
                            isinstance(node.args[-1], ast.Constant) and node.args[-1].value == "test")
        self.assertLess(final_saves[0].lineno, loads[0].lineno)
        self.assertLess(loads[0].lineno, test_builder.lineno)

    def test_test_marker_exclusive_and_no_best_artifact(self):
        marker_opens = [node for node in self.call_named("open") if isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name) and node.func.value.id == "evaluation_marker"]
        self.assertEqual(len(marker_opens), 1)
        self.assertEqual(ast.literal_eval(marker_opens[0].args[0]), "x")
        literals = [node.value for node in ast.walk(self.run) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
        self.assertNotIn("best.pth", literals)
        self.assertIn("fixed_epoch_60_student", literals)

    def test_data_loader_covers_all_samples_without_drop(self):
        loaders = self.call_named("DataLoader")
        self.assertEqual(len(loaders), 2)
        for node in loaders:
            kwargs = {item.arg: item.value for item in node.keywords}
            self.assertEqual(ast.literal_eval(kwargs["batch_size"]), 1)
            self.assertEqual(ast.literal_eval(kwargs["num_workers"]), 0)
            self.assertFalse(ast.literal_eval(kwargs["drop_last"]))

    def test_strict_loader_does_not_recurse_or_randomly_replace(self):
        dataset_factory = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == "make_strict_dataset")
        getter = next(node for node in ast.walk(dataset_factory) if isinstance(node, ast.FunctionDef) and node.name == "__getitem__")
        call_names = [ast.unparse(node.func) for node in ast.walk(getter) if isinstance(node, ast.Call)]
        self.assertFalse(any("__getitem__" in name or "randint" in name or "choice" in name for name in call_names))


class GuardTargetAndLockTests(unittest.TestCase):
    def test_exact_six_target_set(self):
        observed = GUARD.targets()
        expected = [(GUARD.STORE / arm / name, bound) for arm in ("sup05", "sup10", "sup100")
                    for name, bound in (("latest.pth", 3 * GUARD.GIB), ("final.pth", GUARD.GIB))]
        self.assertEqual(observed, expected)
        self.assertEqual(len({path for path, _ in observed}), 6)

    def test_canonical_targets_for_all_three_arms(self):
        for percent, arm in ((5, "sup05"), (10, "sup10"), (100, "sup100")):
            for name in ("latest.pth", "final.pth"):
                target = GUARD.STORE / arm / name
                cfg = {"label_percent": percent, "artifact_dir": str(target.parent)}
                self.assertEqual(GUARD.validate_target(target, cfg), target)

    def test_wrong_arm_name_parent_and_config_are_refused(self):
        cfg = {"label_percent": 5, "artifact_dir": str(GUARD.STORE / "sup05")}
        for target in (GUARD.STORE / "sup10/final.pth", GUARD.STORE / "sup05/best.pth",
                       GUARD.STORE / "other/final.pth", GUARD.STORE / "sup05/sub/final.pth"):
            with self.subTest(target=str(target)), self.assertRaises(RuntimeError):
                GUARD.validate_target(target, cfg)
        with self.assertRaisesRegex(RuntimeError, "artifact path mismatch"):
            GUARD.validate_target(GUARD.STORE / "sup05/final.pth", {"label_percent": 5, "artifact_dir": "other"})

    def test_ancestor_symlink_is_refused(self):
        target = GUARD.STORE / "sup05/final.pth"
        for bad_path in (target, target.parent, GUARD.STORE):
            with self.subTest(path=str(bad_path)), patch.object(Path, "is_symlink", lambda path: path == bad_path):
                with self.assertRaisesRegex(RuntimeError, "Symlink"):
                    GUARD.no_symlink(target)

    def test_exact_lock_identity_and_descriptor_pass(self):
        metadata = types.SimpleNamespace(st_mode=stat.S_IFREG | 0o600, st_dev=2049, st_ino=168299140, st_nlink=1)
        with patch.object(GUARD, "no_symlink"), patch.object(Path, "stat", return_value=metadata), \
                patch.object(GUARD.os, "fstat", return_value=metadata):
            GUARD.check_lock(19)

    def test_changed_lock_device_inode_links_or_type_refused(self):
        good = {"st_mode": stat.S_IFREG | 0o600, "st_dev": 2049, "st_ino": 168299140, "st_nlink": 1}
        for key, value in (("st_mode", stat.S_IFDIR), ("st_dev", 2050), ("st_ino", 168299141), ("st_nlink", 2)):
            metadata = types.SimpleNamespace(**dict(good, **{key: value}))
            with self.subTest(key=key), patch.object(GUARD, "no_symlink"), patch.object(Path, "stat", return_value=metadata):
                with self.assertRaisesRegex(RuntimeError, "Original save.lock identity changed"):
                    GUARD.check_lock()

    def test_lock_fd_replacement_refused(self):
        metadata = types.SimpleNamespace(st_mode=stat.S_IFREG, st_dev=2049, st_ino=168299140, st_nlink=1)
        wrong = types.SimpleNamespace(st_dev=2049, st_ino=1, st_nlink=1)
        with patch.object(GUARD, "no_symlink"), patch.object(Path, "stat", return_value=metadata), \
                patch.object(GUARD.os, "fstat", return_value=wrong):
            with self.assertRaisesRegex(RuntimeError, "Lock fd changed"):
                GUARD.check_lock(19)

    def test_lock_rechecked_before_and_after_locked_work(self):
        calls = []
        fake_fcntl = types.ModuleType("fcntl")
        fake_fcntl.LOCK_EX = 2
        fake_fcntl.flock = lambda fd, flag: calls.append(("flock", fd, flag))
        with patch.dict(sys.modules, {"fcntl": fake_fcntl}), patch.object(GUARD, "check_lock", side_effect=lambda fd=None: calls.append(("check", fd))), \
                patch.object(GUARD.os, "O_NOFOLLOW", 0x20000, create=True), patch.object(GUARD.os, "open", return_value=19), \
                patch.object(GUARD.os, "close", side_effect=lambda fd: calls.append(("close", fd))):
            with GUARD.original_lock() as descriptor:
                self.assertEqual(descriptor, 19)
                calls.append(("work", descriptor))
        self.assertEqual(calls, [("check", None), ("flock", 19, 2), ("check", 19), ("work", 19),
                                 ("check", 19), ("close", 19)])


class GuardCapacityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="basic-sup-guard-", dir=RUNNER_PATH.parent)
        self.addCleanup(self.temp.cleanup)
        self.store = Path(self.temp.name)
        for arm in ("sup05", "sup10", "sup100"):
            (self.store / arm).mkdir()
        for name, value in (("STORE", self.store), ("BOUNDS", {"latest.pth": 30, "final.pth": 10}), ("MARGIN", 7)):
            context = patch.object(GUARD, name, value)
            context.start()
            self.addCleanup(context.stop)
        memory = patch.object(GUARD, "memory_snapshot", return_value={"effective_available": 1000, "cgroup_ancestors": []})
        memory.start()
        self.addCleanup(memory.stop)
        fs = patch.object(GUARD.os, "statvfs", return_value=types.SimpleNamespace(f_bavail=1000, f_frsize=1), create=True)
        fs.start()
        self.addCleanup(fs.stop)

    def test_remaining_growth_and_shared_atomic_peak(self):
        (self.store / "sup05/latest.pth").write_bytes(b"12345")
        result = GUARD.capacity_snapshot(required_ram=100)
        self.assertEqual(len(result["filesystems"]), 1)
        group = result["filesystems"][0]
        self.assertEqual(group["remaining_permanent"], 115)
        self.assertEqual(group["atomic_peak"], 30)
        self.assertEqual(group["required_bytes"], 152)
        self.assertEqual(group["margin_bytes"], 848)

    def test_partial_refused_on_any_of_six_targets(self):
        for target, _ in GUARD.targets():
            partial = Path(str(target) + ".partial")
            partial.write_bytes(b"partial")
            try:
                with self.subTest(target=str(target)), self.assertRaisesRegex(RuntimeError, "Existing atomic partial"):
                    GUARD.capacity_snapshot(required_ram=100)
            finally:
                partial.unlink()

    def test_bound_overflow_refused(self):
        (self.store / "sup05/final.pth").write_bytes(b"x" * 11)
        with self.assertRaisesRegex(RuntimeError, "bound exceeded"):
            GUARD.capacity_snapshot(required_ram=100)

    def test_hardlinked_checkpoint_refused(self):
        target = self.store / "sup05/final.pth"
        target.write_bytes(b"one")
        original_stat = Path.stat

        def fake_stat(path, *args, **kwargs):
            value = original_stat(path, *args, **kwargs)
            if path == target:
                return types.SimpleNamespace(st_mode=value.st_mode, st_dev=value.st_dev, st_ino=value.st_ino,
                                             st_size=value.st_size, st_nlink=2)
            return value

        with patch.object(Path, "stat", fake_stat), self.assertRaisesRegex(RuntimeError, "Invalid checkpoint target"):
            GUARD.capacity_snapshot(required_ram=100)

    def test_directory_checkpoint_refused(self):
        (self.store / "sup05/final.pth").mkdir()
        with self.assertRaisesRegex(RuntimeError, "Invalid checkpoint target"):
            GUARD.capacity_snapshot(required_ram=100)

    def test_ram_shortfall_refused(self):
        with self.assertRaisesRegex(RuntimeError, "Insufficient RAM"):
            GUARD.capacity_snapshot(required_ram=1001)

    def test_filesystem_space_cannot_be_pooled(self):
        class Probe:
            def __init__(self, device):
                self.device = device
            def stat(self):
                return types.SimpleNamespace(st_dev=self.device)
            def __str__(self):
                return "probe" + str(self.device)

        def parent_probe(path):
            return Probe(1 if path.name == "sup05" else 2)

        with patch.object(GUARD, "existing_parent", side_effect=parent_probe), \
                patch.object(GUARD.os, "statvfs", side_effect=lambda name: types.SimpleNamespace(
                    f_bavail=76 if name == "probe1" else 10000, f_frsize=1), create=True):
            with self.assertRaisesRegex(RuntimeError, "Insufficient filesystem capacity"):
                GUARD.capacity_snapshot(required_ram=100)
        with patch.object(GUARD, "existing_parent", side_effect=parent_probe), \
                patch.object(GUARD.os, "statvfs", return_value=types.SimpleNamespace(f_bavail=1000, f_frsize=1), create=True):
            result = GUARD.capacity_snapshot(required_ram=100)
            self.assertEqual(sorted(group["required_bytes"] for group in result["filesystems"]), [77, 117])


class GuardMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="basic-sup-memory-", dir=RUNNER_PATH.parent)
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.proc, self.cgroup = root / "proc", root / "cgroups"
        (self.proc / "self").mkdir(parents=True)
        self.cgroup.mkdir()
        (self.proc / "meminfo").write_text("MemAvailable: 1000 kB\n", encoding="utf-8")

    def write_group(self, path, maximum, current, v1=False):
        path.mkdir(parents=True, exist_ok=True)
        (path / ("memory.limit_in_bytes" if v1 else "memory.max")).write_text(str(maximum), encoding="utf-8")
        (path / ("memory.usage_in_bytes" if v1 else "memory.current")).write_text(str(current), encoding="utf-8")

    def snapshot(self):
        return GUARD.memory_snapshot(self.proc, self.cgroup)

    def write_mounts(self, rows):
        def encode(value):
            return str(value).replace("\\", r"\134").replace(" ", r"\040")
        lines = [f"{20 + index} 1 0:{index + 1} {root} {encode(point)} rw - {kind} cgroup {options}"
                 for index, (kind, root, point, options) in enumerate(rows)]
        (self.proc / "self/mountinfo").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_v2_all_ancestors_and_tightest_remaining(self):
        (self.proc / "self/cgroup").write_text("0::/parent/child\n", encoding="utf-8")
        self.write_group(self.cgroup, "max", 100)
        self.write_group(self.cgroup / "parent", 900, 800)
        self.write_group(self.cgroup / "parent/child", 600, 100)
        snapshot = self.snapshot()
        self.assertEqual(snapshot["host_mem_available"], 1024000)
        self.assertEqual(snapshot["effective_available"], 100)
        self.assertEqual(len(snapshot["cgroup_ancestors"]), 3)

    def test_v1_all_ancestors(self):
        (self.proc / "self/cgroup").write_text("2:cpu:/unused\n4:memory:/parent/child\n", encoding="utf-8")
        base = self.cgroup / "memory"
        self.write_group(base, 900, 200, v1=True)
        self.write_group(base / "parent", 800, 200, v1=True)
        self.write_group(base / "parent/child", 600, 100, v1=True)
        self.assertEqual(self.snapshot()["effective_available"], 500)

    def test_host_memory_still_caps_unlimited_groups(self):
        (self.proc / "self/cgroup").write_text("0::/\n", encoding="utf-8")
        self.write_group(self.cgroup, "max", 123)
        self.assertEqual(self.snapshot()["effective_available"], 1024000)

    def test_usage_exceeding_limit_has_zero_available(self):
        (self.proc / "self/cgroup").write_text("0::/\n", encoding="utf-8")
        self.write_group(self.cgroup, 100, 120)
        self.assertEqual(self.snapshot()["effective_available"], 0)

    def test_missing_nonroot_memory_controller_refused(self):
        (self.proc / "self/cgroup").write_text("0::/missing\n", encoding="utf-8")
        self.write_group(self.cgroup, "max", 100)
        with self.assertRaisesRegex(RuntimeError, "Missing memory controller"):
            self.snapshot()

    def test_invalid_numeric_limits_or_usage_refused(self):
        (self.proc / "self/cgroup").write_text("0::/\n", encoding="utf-8")
        for maximum, current in ((0, 10), (-1, 10), (100, -1)):
            self.write_group(self.cgroup, maximum, current)
            with self.subTest(maximum=maximum, current=current), self.assertRaises(RuntimeError):
                self.snapshot()

    def test_unlimited_group_negative_usage_refused(self):
        (self.proc / "self/cgroup").write_text("0::/\n", encoding="utf-8")
        self.write_group(self.cgroup, "max", -1)
        with self.assertRaisesRegex(RuntimeError, "memory"):
            self.snapshot()

    def test_parent_traversal_membership_refused(self):
        (self.proc / "self/cgroup").write_text("0::/parent/../../outside\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "Invalid cgroup membership"):
            self.snapshot()

    def test_hybrid_actual_unified_mount_without_memory_is_recorded(self):
        (self.proc / "self/cgroup").write_text("0::/user.slice/session.scope\n4:memory:/\n", encoding="utf-8")
        memory_base, unified = self.cgroup / "memory", self.cgroup / "unified"
        self.write_group(memory_base, 1000, 400, v1=True)
        unified.mkdir()
        (unified / "cgroup.controllers").write_text("cpu pids io\n", encoding="utf-8")
        self.write_mounts([("cgroup", "/", memory_base, "rw,memory"), ("cgroup2", "/", unified, "rw")])
        snapshot = self.snapshot()
        self.assertEqual(snapshot["effective_available"], 600)
        self.assertTrue(snapshot["mountinfo_used"])
        self.assertEqual(len(snapshot["cgroup_ancestors"]), 1)
        self.assertEqual(len(snapshot["non_memory_hierarchies"]), 1)
        self.assertEqual(snapshot["non_memory_hierarchies"][0]["path"], str(unified / "user.slice/session.scope"))
        self.assertIn("verified_v1_memory", snapshot["non_memory_hierarchies"][0]["reason"])

    def test_hybrid_without_mountinfo_explicit_unified_fallback(self):
        (self.proc / "self/cgroup").write_text("0::/user.slice/session.scope\n4:memory:/\n", encoding="utf-8")
        self.write_group(self.cgroup / "memory", 800, 300, v1=True)
        unified = self.cgroup / "unified"
        unified.mkdir()
        (unified / "cgroup.controllers").write_text("cpu pids\n", encoding="utf-8")
        snapshot = self.snapshot()
        self.assertEqual(snapshot["effective_available"], 500)
        self.assertFalse(snapshot["mountinfo_used"])
        self.assertEqual(snapshot["non_memory_hierarchies"][0]["mountpoint"], str(unified))

    def test_v2_missing_memory_cannot_hide_missing_v1(self):
        (self.proc / "self/cgroup").write_text("0::/user.slice\n", encoding="utf-8")
        (self.cgroup / "cgroup.controllers").write_text("cpu pids\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "no verified v1"):
            self.snapshot()

    def test_v2_existing_memory_limit_not_ignored_by_controller_list(self):
        (self.proc / "self/cgroup").write_text("0::/\n", encoding="utf-8")
        (self.cgroup / "cgroup.controllers").write_text("cpu\n", encoding="utf-8")
        self.write_group(self.cgroup, 1000, 700)
        self.assertEqual(self.snapshot()["effective_available"], 300)

    def test_mapped_mount_root_preserves_visible_ancestor_limits(self):
        (self.proc / "self/cgroup").write_text("0::/host_group/child\n", encoding="utf-8")
        mounted = self.cgroup / "visible"
        self.write_group(mounted, 900, 800)
        self.write_group(mounted / "child", 800, 100)
        (mounted / "cgroup.controllers").write_text("memory cpu\n", encoding="utf-8")
        self.write_mounts([("cgroup2", "/host_group", mounted, "rw")])
        snapshot = self.snapshot()
        self.assertEqual(snapshot["effective_available"], 100)
        self.assertEqual(len(snapshot["cgroup_ancestors"]), 2)

    def test_broad_mount_preferred_over_bind_that_hides_ancestor(self):
        (self.proc / "self/cgroup").write_text("4:memory:/parent/child\n", encoding="utf-8")
        memory_base = self.cgroup / "memory"
        self.write_group(memory_base, 900, 850, v1=True)
        self.write_group(memory_base / "parent", 800, 100, v1=True)
        self.write_group(memory_base / "parent/child", 700, 100, v1=True)
        self.write_mounts([("cgroup", "/", memory_base, "rw,memory"),
                           ("cgroup", "/parent", self.cgroup / "deeper_bind", "rw,memory")])
        self.assertEqual(self.snapshot()["effective_available"], 50)

    def test_unmapped_membership_refused(self):
        (self.proc / "self/cgroup").write_text("4:memory:/other\n", encoding="utf-8")
        self.write_mounts([("cgroup", "/host_group", self.cgroup / "memory", "rw,memory")])
        with self.assertRaisesRegex(RuntimeError, "No actual cgroup mount"):
            self.snapshot()

    def test_mountpoint_escaped_spaces_decoded(self):
        (self.proc / "self/cgroup").write_text("4:memory:/\n", encoding="utf-8")
        mounted = self.cgroup / "memory with space"
        self.write_group(mounted, 500, 100, v1=True)
        self.write_mounts([("cgroup", "/", mounted, "rw,memory")])
        self.assertEqual(self.snapshot()["effective_available"], 400)


class ImportedScientificDependencyTests(unittest.TestCase):
    def setUp(self):
        run = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == "run")
        self.binding_loop = next(node for node in ast.walk(run) if isinstance(node, ast.For)
                                 and isinstance(node.target, ast.Name) and node.target.id == "module_name")
        self.names = ast.literal_eval(self.binding_loop.iter)
        self.paths = {name: (RUNNER_PATH.parent / (name.replace(".", "_") + ".cpython-310.pyc")).resolve()
                      for name in self.names}
        self.checked = {str(path): {"kind": "source_sha256"} for path in self.paths.values()}

    def execute_binding(self):
        code = compile(ast.Module(body=[self.binding_loop], type_ignores=[]), "<cpu-scientific-bind-test>", "exec")
        importer = types.SimpleNamespace(import_module=lambda name: types.SimpleNamespace(__file__=str(self.paths[name])))
        exec(code, {"Path": Path, "importlib": importer, "checked": self.checked})

    def test_actual_pyc_module_files_are_accepted_when_bound(self):
        self.execute_binding()

    def test_unbound_actual_pyc_module_refused(self):
        del self.checked[str(self.paths["segment_anything.auto_sam_seg"])]
        with self.assertRaisesRegex(RuntimeError, "Scientific dependency is not SHA-bound"):
            self.execute_binding()

    def test_module_bound_only_as_input_is_refused(self):
        self.checked[str(self.paths["segment_anything.modeling.sam3D_cascade"])]["kind"] = "input_sha256"
        with self.assertRaisesRegex(RuntimeError, "Scientific dependency is not SHA-bound"):
            self.execute_binding()


class GuardAtomicSaveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="basic-sup-atomic-", dir=RUNNER_PATH.parent)
        self.addCleanup(self.temp.cleanup)
        self.store = Path(self.temp.name)
        self.arm = self.store / "sup05"
        self.arm.mkdir()
        self.cfg = {"label_percent": 5, "artifact_dir": str(self.arm)}
        fake_torch = types.ModuleType("torch")
        fake_torch.save = lambda payload, handle: handle.write(payload)
        contexts = [patch.object(GUARD, "STORE", self.store),
                    patch.object(GUARD, "BOUNDS", {"latest.pth": 30, "final.pth": 10}),
                    patch.object(GUARD, "MARGIN", 7), patch.dict(sys.modules, {"torch": fake_torch}),
                    patch.object(GUARD, "original_lock", side_effect=lambda: contextlib.nullcontext(19)),
                    patch.object(GUARD, "capacity_snapshot", return_value={"mock_cpu_capacity": True}),
                    patch.object(GUARD.os, "O_DIRECTORY", 0x10000, create=True),
                    patch.object(GUARD.os, "open", return_value=19), patch.object(GUARD.os, "close"),
                    patch.object(GUARD.os, "fsync")]
        for context in contexts:
            context.start()
            self.addCleanup(context.stop)

    def test_atomic_final_small_mock_payload_roundtrip(self):
        target = self.arm / "final.pth"
        receipt = GUARD.guarded_torch_save(payload=b"final", target=target, config=self.cfg, save_kind="final")
        self.assertEqual(target.read_bytes(), b"final")
        self.assertEqual(receipt["sha256"], hashlib.sha256(b"final").hexdigest())
        self.assertEqual(receipt["bytes"], 5)
        self.assertFalse(Path(str(target) + ".partial").exists())
        GUARD.capacity_snapshot.assert_called_once_with(27)

    def test_existing_final_never_overwritten(self):
        target = self.arm / "final.pth"
        target.write_bytes(b"old")
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            GUARD.guarded_torch_save(payload=b"new", target=target, config=self.cfg, save_kind="final")
        self.assertEqual(target.read_bytes(), b"old")

    def test_latest_replacement_uses_one_atomic_copy(self):
        target = self.arm / "latest.pth"
        target.write_bytes(b"old")
        GUARD.guarded_torch_save(payload=b"new", target=target, config=self.cfg, save_kind="latest")
        self.assertEqual(target.read_bytes(), b"new")
        self.assertFalse(Path(str(target) + ".partial").exists())

    def test_serialization_overflow_preserves_partial(self):
        target = self.arm / "final.pth"
        with self.assertRaisesRegex(RuntimeError, "partial preserved"):
            GUARD.guarded_torch_save(payload=b"x" * 11, target=target, config=self.cfg, save_kind="final")
        self.assertFalse(target.exists())
        self.assertEqual(Path(str(target) + ".partial").read_bytes(), b"x" * 11)

    def test_existing_partial_is_never_overwritten(self):
        target = self.arm / "final.pth"
        partial = Path(str(target) + ".partial")
        partial.write_bytes(b"old partial")
        with self.assertRaises(FileExistsError):
            GUARD.guarded_torch_save(payload=b"new", target=target, config=self.cfg, save_kind="final")
        self.assertEqual(partial.read_bytes(), b"old partial")
        self.assertFalse(target.exists())

    def test_kind_mismatch_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "kind mismatch"):
            GUARD.guarded_torch_save(payload=b"new", target=self.arm / "final.pth", config=self.cfg, save_kind="latest")

    def test_load_gate_uses_actual_saved_final_size(self):
        target = self.arm / "final.pth"
        target.write_bytes(b"five!")
        GUARD.check_load_capacity(target=target, config=self.cfg)
        GUARD.capacity_snapshot.assert_called_once_with(17)


class TorchVersionCompatibilityTests(unittest.TestCase):
    def test_old_load_signature_omits_unsupported_weights_only(self):
        calls = []
        sentinel = object()

        def old_load(path, map_location=None, pickle_module=None, **pickle_load_args):
            calls.append((path, map_location, pickle_load_args))
            return sentinel

        with patch.dict(sys.modules, {"torch": types.SimpleNamespace(load=old_load)}):
            result = RUNNER.load_trusted_checkpoint("trusted.pth", map_location="cpu")
        self.assertIs(result, sentinel)
        self.assertEqual(calls, [("trusted.pth", "cpu", {})])

    def test_new_load_signature_explicit_false_single_read(self):
        calls = []
        sentinel = object()

        def new_load(path, map_location=None, *, weights_only=True):
            calls.append((path, map_location, weights_only))
            return sentinel

        with patch.dict(sys.modules, {"torch": types.SimpleNamespace(load=new_load)}):
            result = RUNNER.load_trusted_checkpoint("trusted.pth", map_location="cpu")
        self.assertIs(result, sentinel)
        self.assertEqual(calls, [("trusted.pth", "cpu", False)])

    def test_load_internal_error_not_retried_or_reinterpreted(self):
        calls = []

        def broken_load(path, map_location=None, *, weights_only=True):
            calls.append((path, map_location, weights_only))
            raise TypeError("synthetic deserialization failure")

        with patch.dict(sys.modules, {"torch": types.SimpleNamespace(load=broken_load)}):
            with self.assertRaisesRegex(TypeError, "synthetic deserialization failure"):
                RUNNER.load_trusted_checkpoint("trusted.pth")
        self.assertEqual(len(calls), 1)

    class Optimizer:
        def __init__(self):
            self.calls = []

        def step(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return "original return"

    def test_step_counter_without_hook_preserves_single_call_args_and_return(self):
        optimizer, counter = self.Optimizer(), {"count": 0}
        self.assertFalse(hasattr(optimizer, "register_step_post_hook"))
        original = RUNNER.install_optimizer_step_counter(optimizer, counter)
        value = optimizer.step("closure", special=7)
        self.assertEqual(value, "original return")
        self.assertEqual(optimizer.calls, [(("closure",), {"special": 7})])
        self.assertEqual(counter["count"], 1)
        optimizer.step = original
        optimizer.step()
        self.assertEqual(counter["count"], 1)
        self.assertEqual(len(optimizer.calls), 2)

    def test_step_counter_amp_skip_remains_zero_and_success_adds_one(self):
        optimizer, counter = self.Optimizer(), {"count": 0}
        RUNNER.install_optimizer_step_counter(optimizer, counter)

        def scaler_step(found_inf):
            if not found_inf:
                return optimizer.step()

        scaler_step(found_inf=True)
        self.assertEqual(counter["count"], 0)
        self.assertEqual(optimizer.calls, [])
        scaler_step(found_inf=False)
        self.assertEqual(counter["count"], 1)
        self.assertEqual(len(optimizer.calls), 1)

    def test_step_counter_exception_is_not_a_successful_step(self):
        calls = []

        def failing_step():
            calls.append(1)
            raise RuntimeError("synthetic optimizer failure")

        optimizer, counter = types.SimpleNamespace(step=failing_step), {"count": 0}
        RUNNER.install_optimizer_step_counter(optimizer, counter)
        with self.assertRaisesRegex(RuntimeError, "synthetic optimizer failure"):
            optimizer.step()
        self.assertEqual(counter["count"], 0)
        self.assertEqual(len(calls), 1)

    def test_runner_calls_single_compatibility_load_and_retains_exact_step_gate(self):
        run = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == "run")
        run_calls = [node for node in ast.walk(run) if isinstance(node, ast.Call)]
        loads = [node for node in run_calls if ast.unparse(node.func) == "load_trusted_checkpoint"]
        self.assertEqual(len(loads), 1)
        self.assertIn("final_path", ast.unparse(loads[0]))
        installs = [node for node in run_calls if ast.unparse(node.func) == "install_optimizer_step_counter"]
        self.assertEqual(len(installs), 1)
        self.assertIn("step_calls['count'] != before_steps + 1", ast.unparse(run))
        self.assertFalse(any("register_step_post_hook" in ast.unparse(node.func) for node in run_calls))
        all_direct_loads = [node for node in ast.walk(TREE) if isinstance(node, ast.Call) and ast.unparse(node.func) == "torch.load"]
        self.assertEqual(len(all_direct_loads), 1)
        helper = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == "load_trusted_checkpoint")
        self.assertLessEqual(helper.lineno, all_direct_loads[0].lineno)
        self.assertGreaterEqual(helper.end_lineno, all_direct_loads[0].lineno)


class InitializationAndLaunchBindingTests(unittest.TestCase):
    def test_uniform_torch_thread_caps_recorded(self):
        calls = {ast.unparse(node.func): node for node in ast.walk(TREE)
                 if isinstance(node, ast.Call) and ast.unparse(node.func) in
                 ("torch.set_num_threads", "torch.set_num_interop_threads")}
        self.assertEqual(ast.literal_eval(calls["torch.set_num_threads"].args[0]), 4)
        self.assertEqual(ast.literal_eval(calls["torch.set_num_interop_threads"].args[0]), 1)
        self.assertIn("uniform_engineering_cpu_thread_cap_not_per_arm_tuning", SOURCE)

    class Tensor:
        def __init__(self, values=(1, 2), shape=(2,), dtype="float32"):
            self.values, self.shape, self.dtype = values, shape, dtype

        def detach(self):
            return self

        def cpu(self):
            return self

        def to(self, *, dtype):
            return type(self)(self.values, self.shape, dtype)

    def audit(self, actual, expected, shas=None):
        fake_torch = types.SimpleNamespace(Tensor=self.Tensor,
            load=lambda *args, **kwargs: {"state": expected},
            equal=lambda left, right: left.values == right.values and left.dtype == right.dtype)
        sam = types.SimpleNamespace(state_dict=lambda: actual)
        with patch.dict(sys.modules, {"torch": fake_torch}), patch.object(
                RUNNER, "sha256_file", side_effect=shas or ["a"*64, "a"*64]):
            return RUNNER.audit_sam_initialization(sam, "generic.pth", "a"*64, lambda checkpoint: checkpoint["state"])

    def test_generic_sam_audit_checks_every_value_without_repair(self):
        result = self.audit({"weight": self.Tensor()}, {"weight": self.Tensor()})
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["all_values_checked"])
        self.assertEqual(result["tensor_values_checked"], 1)
        self.assertFalse(result["model_reloaded_or_repaired"])
        self.assertEqual((result["model_forward_calls"], result["training_updates"], result["test_evaluations"]), (0, 0, 0))

    def test_generic_sam_audit_refuses_missing_unexpected_shape_value_and_nontensor(self):
        variants = [({"weight": self.Tensor()}, {"other": self.Tensor()}, "missing_keys"),
                    ({"weight": self.Tensor()}, {"weight": self.Tensor(shape=(1, 2))}, "shape_mismatches"),
                    ({"weight": self.Tensor()}, {"weight": self.Tensor(values=(3, 4))}, "value_mismatches"),
                    ({"weight": self.Tensor()}, {"weight": "not tensor"}, "invalid_entries")]
        for actual, expected, key in variants:
            with self.subTest(key=key):
                result = self.audit(actual, expected)
                self.assertEqual(result["status"], "FAIL")
                self.assertTrue(result[key])

    def test_generic_sam_audit_records_expected_load_dtype_conversion(self):
        result = self.audit({"weight": self.Tensor()}, {"weight": self.Tensor(dtype="float16")})
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["dtype_conversions"][0]["key"], "weight")

    def test_generic_sam_audit_rejects_checkpoint_change_before_or_during(self):
        for shas in (["b"*64], ["a"*64, "b"*64]):
            with self.subTest(shas=shas), self.assertRaisesRegex(RuntimeError, "checkpoint changed"):
                self.audit({"weight": self.Tensor()}, {"weight": self.Tensor()}, shas)

    def test_generic_sam_audit_fail_is_before_dataset_and_no_second_model_load(self):
        run_node = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == "run")
        audit_call = next(node for node in ast.walk(run_node) if isinstance(node, ast.Call) and ast.unparse(node.func) == "audit_sam_initialization")
        datasets = [node for node in ast.walk(run_node) if isinstance(node, ast.Call) and ast.unparse(node.func) == "make_strict_dataset"]
        fail_if = next(node for node in ast.walk(run_node) if isinstance(node, ast.If) and 'initialization_audit[\'status\']' in ast.unparse(node.test))
        self.assertTrue(any(isinstance(node, ast.Raise) for node in ast.walk(fail_if)))
        self.assertLess(audit_call.lineno, min(node.lineno for node in datasets))
        self.assertLess(fail_if.lineno, min(node.lineno for node in datasets))
        audit_node = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == "audit_sam_initialization")
        self.assertFalse(any(isinstance(node, ast.Call) and "load_state_dict" in ast.unparse(node.func) for node in ast.walk(audit_node)))

    def test_main_verifies_exact_config_bytes_before_parse_or_run(self):
        with tempfile.TemporaryDirectory(dir=RUNNER_PATH.parent) as tmp:
            config_path = Path(tmp)/"config.json"
            config_path.write_text(json.dumps(minimal_config()), encoding="utf-8")
            base = ["runner", "--config", str(config_path), "--config-sha256", "0"*64,
                    "--gpu", "0", "--gpu-claim-fd", "42", "--gpu-claim-device", "1", "--gpu-claim-inode", "2"]
            with patch.object(sys, "argv", base), patch.object(RUNNER, "validate_config") as validate, \
                    patch.object(RUNNER, "validate_gpu_claim") as claim, patch.object(RUNNER, "run") as run:
                with self.assertRaisesRegex(RuntimeError, "Configuration bytes"):
                    RUNNER.main()
                validate.assert_not_called()
                claim.assert_not_called()
                run.assert_not_called()
            base[4] = hashlib.sha256(config_path.read_bytes()).hexdigest()
            with patch.object(sys, "argv", base), patch.object(RUNNER, "validate_gpu_claim", return_value={"verified": True}) as claim, \
                    patch.object(RUNNER, "run") as run, patch.object(RUNNER.os, "close") as close, \
                    patch.dict(os.environ, {}), patch.object(sys, "path", list(sys.path)):
                RUNNER.main()
                claim.assert_called_once_with(minimal_config(), 0, 42, 1, 2)
                run.assert_called_once_with(minimal_config(), base[4], 0, gpu_claim={"verified": True})
                close.assert_called_once_with(42)

    def claim_fixture(self):
        tmp = tempfile.TemporaryDirectory(dir=RUNNER_PATH.parent)
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name).resolve()
        control = root/"control"
        control.mkdir()
        path = control/"gpu0.lock"
        path.write_bytes(b"")
        fd = os.open(path, os.O_RDONLY)
        self.addCleanup(os.close, fd)
        info = path.stat()
        fake_fcntl = types.SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=unittest.mock.Mock())
        return {"project_root": str(root), "output_dir": str(root/"sup05")}, path, fd, info, fake_fcntl

    def test_gpu_claim_retains_exact_regular_single_link_inode(self):
        cfg, path, fd, info, fake = self.claim_fixture()
        with patch.dict(sys.modules, {"fcntl": fake}):
            result = RUNNER.validate_gpu_claim(cfg, 0, fd, info.st_dev, info.st_ino)
        self.assertTrue(result["exclusive_claim_retained"])
        self.assertEqual(result["path"], str(path))
        fake.flock.assert_called_once_with(fd, 6)

    def test_gpu_claim_rejects_stdio_wrong_inode_and_ancestor_symlink(self):
        cfg, path, fd, info, fake = self.claim_fixture()
        with patch.dict(sys.modules, {"fcntl": fake}):
            with self.assertRaisesRegex(RuntimeError, "dedicated"):
                RUNNER.validate_gpu_claim(cfg, 0, 2, info.st_dev, info.st_ino)
            with self.assertRaisesRegex(RuntimeError, "identity differs"):
                RUNNER.validate_gpu_claim(cfg, 0, fd, info.st_dev, info.st_ino+1)
            with patch.object(Path, "is_symlink", autospec=True, side_effect=lambda item: item == path.parent):
                with self.assertRaisesRegex(RuntimeError, "symlink"):
                    RUNNER.validate_gpu_claim(cfg, 0, fd, info.st_dev, info.st_ino)
        fake.flock.assert_not_called()

    def test_gpu_claim_rejects_hardlink_and_fd_path_mismatch(self):
        cfg, path, fd, info, fake = self.claim_fixture()
        wrong = types.SimpleNamespace(st_mode=stat.S_IFREG, st_nlink=2, st_dev=info.st_dev, st_ino=info.st_ino)
        with patch.dict(sys.modules, {"fcntl": fake}), patch.object(RUNNER.os, "fstat", return_value=wrong):
            with self.assertRaisesRegex(RuntimeError, "single-link"):
                RUNNER.validate_gpu_claim(cfg, 0, fd, info.st_dev, info.st_ino)
            wrong.st_nlink, wrong.st_ino = 1, info.st_ino+1
            with self.assertRaisesRegex(RuntimeError, "identity differs"):
                RUNNER.validate_gpu_claim(cfg, 0, fd, info.st_dev, info.st_ino)
        fake.flock.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
