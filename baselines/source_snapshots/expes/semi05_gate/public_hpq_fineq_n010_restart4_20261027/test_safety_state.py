import contextlib
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import torch
import hpq_safety_state as s


class SafetyStateTest(unittest.TestCase):
    def test_four_unique_registered_pieces(self):
        rows = s.targets()
        self.assertEqual(len(rows), 4)
        self.assertEqual(len(set(p for p, _ in rows)), 4)
        self.assertTrue(all(p.name == "final.pth" and b == 2 * 1024 ** 3 for p, b in rows))

    def test_cpu_copy_does_not_alias_live_tensor(self):
        value = torch.ones(2)
        result = s.cpu_tree(torch, {"a": [value]})
        result["a"][0].zero_()
        self.assertTrue(torch.equal(value, torch.ones(2)))

    def test_registered_aliases_are_copied_only_once(self):
        module = torch.nn.Module()
        module.a = torch.nn.Linear(4, 3)
        module.same_a = module.a
        original = module.state_dict()
        copied = s.cpu_tree(torch, original)
        self.assertEqual(copied["a.weight"].data_ptr(), copied["same_a.weight"].data_ptr())
        self.assertNotEqual(copied["a.weight"].data_ptr(), original["a.weight"].data_ptr())
        with __import__("io").BytesIO() as stream:
            torch.save(copied, stream)
            stream.seek(0)
            loaded = torch.load(stream)
        self.assertEqual(loaded["a.weight"].data_ptr(), loaded["same_a.weight"].data_ptr())
        self.assertTrue(torch.equal(loaded["a.weight"], original["a.weight"]))

    def test_distinct_offsets_and_distinct_root_parameters_are_not_conflated(self):
        value = torch.arange(12).reshape(3, 4)
        second_root = value.clone()
        copied = s.cpu_tree(torch, {"a": value[0], "b": value[1], "c": second_root[0]})
        self.assertTrue(torch.equal(copied["a"], value[0]))
        self.assertTrue(torch.equal(copied["b"], value[1]))
        self.assertNotEqual(copied["a"].data_ptr(), copied["b"].data_ptr())
        self.assertNotEqual(copied["a"].data_ptr(), copied["c"].data_ptr())

    def test_same_storage_different_shapes_and_strides_keep_values(self):
        value = torch.arange(12).reshape(3, 4)
        copied = s.cpu_tree(torch, [value, value.t(), value.reshape(2, 6)])
        self.assertEqual([tuple(t.shape) for t in copied], [(3, 4), (4, 3), (2, 6)])
        for a, b in zip(copied, [value, value.t(), value.reshape(2, 6)]):
            self.assertTrue(torch.equal(a, b))

    def exercise(self, callback):
        with tempfile.TemporaryDirectory(prefix="safety_test_", dir=Path(__file__).parent) as folder:
            root = Path(folder).absolute()
            output = root / "output"
            output.mkdir()
            engine = types.SimpleNamespace(torch=torch, output=output, cfg={}, spec={"phase": "formal"},
                manifest={"preparation_sha256": "a" * 64}, t=types.SimpleNamespace(matrix_updates=68),
                state=lambda: "state", rng=lambda: "rng",
                signature=lambda x: hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest())
            guard = types.SimpleNamespace(original_lock=contextlib.nullcontext,
                                           capacity_snapshot=lambda n: {"required_ram": n})
            payload = lambda e, epoch: {n: {"part": n, "epoch": epoch} for n in ("weights", "optimizer")}
            with patch.object(s, "STORE", root / "store"), patch.object(s, "backend", return_value=guard), \
                 patch.object(s, "pieces", side_effect=payload), patch.object(s, "fsync_directory"), \
                 patch.object(s, "probe_read_capacity", return_value={"passes": True}):
                callback(engine)

    def test_two_slots_and_generation_receipts(self):
        def run(e):
            r0 = s.save_epoch(e, 0)
            e.t.matrix_updates = 136
            r1 = s.save_epoch(e, 1)
            saved_slot1 = {k: s.sha_file(v["path"]) for k, v in r1["parts"].items()}
            e.t.matrix_updates = 204
            r2 = s.save_epoch(e, 2)
            self.assertEqual((r0["slot"], r1["slot"], r2["slot"]), (0, 1, 0))
            self.assertEqual(saved_slot1, {k: s.sha_file(v["path"]) for k, v in r1["parts"].items()})
            self.assertFalse(r2["automatic_resume"])
            self.assertFalse(r2["continuation_replay_verified"])
        self.exercise(run)

    def test_repeated_generation_rejected(self):
        def run(e):
            s.save_epoch(e, 0)
            with self.assertRaises(ValueError):
                s.save_epoch(e, 0)
        self.exercise(run)

    def test_foreign_target_rejected(self):
        def run(e):
            path = s.targets()[0][0]
            path.parent.mkdir(parents=True)
            path.touch()
            with self.assertRaises(ValueError):
                s.save_epoch(e, 0)
        self.exercise(run)

    def test_tampered_previous_slot_rejected(self):
        def run(e):
            r = s.save_epoch(e, 0)
            with Path(r["parts"]["weights"]["path"]).open("ab") as stream:
                stream.write(b"corrupt")
            e.t.matrix_updates = 204
            with self.assertRaises(ValueError):
                s.save_epoch(e, 2)
        self.exercise(run)

    def test_serialization_failure_preserves_previous_slot(self):
        def run(e):
            r = s.save_epoch(e, 0)
            e.t.matrix_updates = 136
            with patch.object(torch, "save", side_effect=RuntimeError("injected")):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    s.save_epoch(e, 1)
            for row in r["parts"].values():
                self.assertEqual(s.sha_file(row["path"]), row["sha256"])
            self.assertFalse((e.output / "safety_epoch_01.json").exists())
            self.assertTrue(list(Path(s.STORE).rglob("*.partial")))
        self.exercise(run)


if __name__ == "__main__":
    unittest.main()
