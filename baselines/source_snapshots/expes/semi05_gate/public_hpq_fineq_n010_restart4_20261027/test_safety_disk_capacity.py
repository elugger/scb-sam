"""Real disk serialization through the unmocked own-partial capacity checker.

Synthetic models only; no dataset, GPU, training or evaluation.
"""
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
import hpq_resources as r
import hpq_safety_state as s


class SafetyDiskCapacityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="safety_real_disk_", dir=Path(__file__).parent)
        self.root = Path(self.temp.name).absolute()
        self.store = self.root / "store"
        self.output = self.root / "output"
        self.output.mkdir()
        self.lock_depth = 0
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(patch.object(s, "STORE", self.store))
        self.paths = s.targets()
        for p, _ in self.paths:
            p.parent.mkdir(parents=True, exist_ok=True)
        self.guard = types.SimpleNamespace(
            targets=lambda: list(self.paths),
            original_lock=self.lock,
            capacity_snapshot=self.capacity,
            memory_snapshot=lambda: {"effective_available": r.RAM_REQUIRED + 1},
            MARGIN=1024 ** 3 + 16 * 1024 ** 2)
        self.stack.enter_context(patch.object(s, "backend", return_value=self.guard))
        self.stack.enter_context(patch.object(s, "pieces", side_effect=lambda e, epoch: {
            name: {"part": name, "epoch": epoch, "updates": e.t.matrix_updates}
            for name in ("weights", "optimizer")}))
        if os.name == "nt":
            # Only OS facilities unavailable on Windows are shimmed. The real
            # capacity checker, exclusive writes, loads, renames and hashes run.
            original_plain = r.plain_file
            def portable_plain(path):
                st = original_plain(path)
                return types.SimpleNamespace(st_dev=st.st_dev, st_ino=st.st_ino,
                    st_size=st.st_size, st_blocks=(st.st_size + 511) // 512)
            self.stack.enter_context(patch.object(r, "plain_file", side_effect=portable_plain))
            self.stack.enter_context(patch.object(r.os, "statvfs", create=True,
                return_value=types.SimpleNamespace(f_bavail=64 * 1024 ** 3, f_frsize=1)))
            self.stack.enter_context(patch.object(s, "fsync_directory"))
        self.engine = types.SimpleNamespace(torch=torch, output=self.output, cfg={},
            spec={"phase": "formal"}, manifest={"preparation_sha256": "a" * 64},
            t=types.SimpleNamespace(matrix_updates=68), state=lambda: "state",
            rng=lambda: "rng",
            signature=lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest())

    def tearDown(self):
        self.stack.close()
        self.temp.cleanup()

    @contextlib.contextmanager
    def lock(self):
        self.lock_depth += 1
        try:
            yield
        finally:
            self.lock_depth -= 1

    def capacity(self, required):
        self.assertEqual(self.lock_depth, 1)
        self.assertEqual(required, r.RAM_REQUIRED)
        targets = []
        for p, bound in self.paths:
            if Path(str(p) + ".partial").exists():
                raise RuntimeError("Existing atomic partial")
            actual = p.stat().st_size if p.exists() else 0
            targets.append({"path": str(p), "actual_bytes": actual, "bound_bytes": bound})
        return {"filesystems": [{"filesystem_device": self.root.stat().st_dev,
            "probe": str(self.root), "targets": targets,
            "remaining_permanent": sum(x["bound_bytes"] - x["actual_bytes"] for x in targets),
            "atomic_peak": 3 * 1024 ** 3}]}

    def partial(self, path, data=b"real bytes"):
        with path.open("xb") as stream:
            stream.write(data)
        st = path.stat()
        return (st.st_dev, st.st_ino, st.st_size)

    def test_real_three_generations_write_reload_receipt_and_alternate(self):
        first = s.save_epoch(self.engine, 0)
        self.engine.t.matrix_updates = 136
        second = s.save_epoch(self.engine, 1)
        second_hashes = {k: r.sha_file(v["path"]) for k, v in second["parts"].items()}
        self.engine.t.matrix_updates = 204
        third = s.save_epoch(self.engine, 2)
        self.assertEqual([x["slot"] for x in (first, second, third)], [0, 1, 0])
        self.assertEqual(second_hashes, {k: r.sha_file(v["path"]) for k, v in second["parts"].items()})
        self.assertEqual(len(list(self.output.glob("safety_epoch_*.json"))), 3)
        self.assertFalse(list(self.store.rglob("*.partial")))
        self.assertFalse(third["automatic_resume"])
        self.assertFalse(third["continuation_replay_verified"])
        for row in third["parts"].values():
            self.assertEqual(row["capacity_before_reload"]["owned_target"], row["path"])

    def test_bridge_uses_same_real_disk_path_without_final_or_partial_residue(self):
        result = s.memory_probe(self.engine)
        self.assertEqual(result["disk_roundtrip"]["status"],
                         "REGISTERED_OWN_PARTIAL_DISK_ROUNDTRIP_VERIFIED")
        self.assertEqual(set(result["disk_roundtrip"]["parts"]), {"weights", "optimizer"})
        self.assertTrue(all(x["own_verified_probe_removed"]
                            for x in result["disk_roundtrip"]["parts"].values()))
        self.assertFalse([p for p in self.store.rglob("*") if p.is_file()])
        self.assertEqual(result["new_test_forwards"], 0)

    def test_bridge_restores_fresh_formal_store_absence(self):
        fresh = self.root / "never_created_formal_store"
        with patch.object(s, "STORE", fresh):
            self.paths = s.targets()
            result = s.memory_probe(self.engine)
            self.assertFalse(fresh.exists())
            self.assertEqual(set(result["disk_roundtrip"]["parts"]), {"weights", "optimizer"})

    def test_reproduces_r2_rejection_when_own_target_not_declared(self):
        with self.guard.original_lock():
            initial = self.capacity(r.RAM_REQUIRED)
            target = self.paths[0][0]
            partial = Path(str(target) + ".partial")
            identity = self.partial(partial)
            with self.assertRaisesRegex(ValueError, "Concurrent permanent partial"):
                r.probe_read_capacity(self.guard, initial, partial, identity)
            self.assertTrue(partial.exists())

    def test_other_registered_partial_is_still_rejected(self):
        with self.guard.original_lock():
            initial = self.capacity(r.RAM_REQUIRED)
            own, other = self.paths[0][0], self.paths[1][0]
            partial = Path(str(own) + ".partial")
            identity = self.partial(partial)
            self.partial(Path(str(other) + ".partial"))
            with self.assertRaisesRegex(ValueError, "Concurrent permanent partial"):
                r.probe_read_capacity(self.guard, initial, partial, identity, owned_target=own)

    def test_wrong_owner_and_unregistered_owner_rejected(self):
        with self.guard.original_lock():
            initial = self.capacity(r.RAM_REQUIRED)
            own = self.paths[0][0]
            partial = Path(str(own) + ".partial")
            identity = self.partial(partial)
            for wrong in (self.paths[1][0], self.store / "unregistered.pth"):
                with self.assertRaises(ValueError):
                    r.probe_read_capacity(self.guard, initial, partial, identity, owned_target=wrong)

    def test_inode_size_mismatch_rejected(self):
        with self.guard.original_lock():
            initial = self.capacity(r.RAM_REQUIRED)
            own = self.paths[0][0]
            partial = Path(str(own) + ".partial")
            dev, inode, size = self.partial(partial)
            for identity in ((dev, inode + 1, size), (dev, inode, size + 1)):
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    r.probe_read_capacity(self.guard, initial, partial, identity, owned_target=own)

    def test_old_permanent_mutation_rejected(self):
        own = self.paths[0][0]
        own.write_bytes(b"old")
        with self.guard.original_lock():
            initial = self.capacity(r.RAM_REQUIRED)
            partial = Path(str(own) + ".partial")
            identity = self.partial(partial)
            own.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Joint target changed"):
                r.probe_read_capacity(self.guard, initial, partial, identity, owned_target=own)

    def test_memory_gate_not_relaxed(self):
        with self.guard.original_lock():
            initial = self.capacity(r.RAM_REQUIRED)
            own = self.paths[0][0]
            partial = Path(str(own) + ".partial")
            identity = self.partial(partial)
            with patch.object(self.guard, "memory_snapshot", return_value={"effective_available": 0}):
                with self.assertRaisesRegex(ValueError, "RAM"):
                    r.probe_read_capacity(self.guard, initial, partial, identity, owned_target=own)

    def test_capacity_gate_not_relaxed(self):
        with self.guard.original_lock():
            initial = self.capacity(r.RAM_REQUIRED)
            own = self.paths[0][0]
            partial = Path(str(own) + ".partial")
            identity = self.partial(partial)
            with patch.object(r.os, "statvfs", create=True,
                              return_value=types.SimpleNamespace(f_bavail=0, f_frsize=1)):
                with self.assertRaisesRegex(ValueError, "filesystem capacity"):
                    r.probe_read_capacity(self.guard, initial, partial, identity, owned_target=own)


if __name__ == "__main__":
    unittest.main()
