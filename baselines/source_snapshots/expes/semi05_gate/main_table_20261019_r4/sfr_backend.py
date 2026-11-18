"""SFR fixed-final backend candidate; no CLI or implicit dispatch.

The unchanged resource registry accepts only basename final.pth. Each SFR
artifact therefore has an exact, separately enrolled stage directory ending in
final.pth. All writes/reloads/probe deletion use the original resource backend
and lock; this module never creates or replaces save.lock.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time

from budget_protocol import BUDGETS
from fixed_endpoint import exclusive_json
from runtime_data import ROOT, BoundCases, bind_case_files, bound_splits, schedule
import runtime_resources as resources
from sfr_runtime_components import (build_bound_sam, ema_parameters_only, freeze_prefix,
    lora_native_state, sliding_predict, strict_lora_reload, verify_native_sources)
from sfr_stage_protocol import CANVAS, OFFICIAL_COMMIT, PATCH, ssl_clock, stage_plan
from sfr_training_core import NativeBoundTrainingData, SFRTrainingCore

ARTIFACT_BASE = ROOT / "expes/artifacts/FCD_A_MATRIX_20261006/MAIN_TABLE_FIXED_FINAL_20261019"
CONTROL_BASE = ROOT / "expes/work_dir/semi05_gate/MAIN_TABLE_FIXED_FINAL_20261019"
MIB = 1024 ** 2
BOUNDS = dict(prefix=64 * MIB, final=64 * MIB, qualification_probe=256 * MIB)
ORIGINAL_VIT_B_BYTES = 375042383
OWN_SOURCE_NAMES = ("sfr_backend.py", "sfr_training_core.py", "sfr_runtime_components.py",
    "sfr_stage_protocol.py", "budget_protocol.py", "fixed_endpoint.py", "runtime_data.py",
    "runtime_resources.py", "official_compat.py", "dataset_pins.json")


def target_plan(dataset, ratio, tag):
    if dataset not in ("BY", "Public433") or ratio not in (5, 10) or not re.fullmatch(r"r[1-9][0-9]*", tag):
        raise ValueError("SFR job outside the frozen closed scope")
    relative = Path("sfr") / dataset / ("semi%02d" % ratio) / "fold2" / tag
    base = ARTIFACT_BASE / relative
    return {"prefix": dict(path=str(base / "sam_prefix/final.pth"), bound=BOUNDS["prefix"]),
            "final": dict(path=str(base / "final.pth"), bound=BOUNDS["final"]),
            "qualification_probe": dict(path=str(base / "qualification_probe/final.pth"),
                                         bound=BOUNDS["qualification_probe"])}


def validate_contract_shape(c):
    plan = stage_plan(c["dataset"], c["ratio"])
    expected_targets = target_plan(c["dataset"], c["ratio"], c["tag"])
    if (c.get("method") != "sfr" or c.get("fold") != 2 or c.get("seed") != 2027
            or c.get("gpu") != 1 or c.get("selection") != "terminal_after_fixed_exposure"
            or c.get("test_evaluations") != 1 or c.get("early_stop") is not False
            or c.get("exact_resume_supported") is not False
            or c.get("primary_pairs") != plan["ssl_labeled_inputs"]
            or c.get("sfr_stages") != plan or c.get("sfr_targets") != expected_targets
            or c.get("target") != expected_targets["final"]["path"]
            or c.get("official_commit") != OFFICIAL_COMMIT):
        raise ValueError("SFR frozen stage/science/target contract differs")
    relative = Path("sfr") / c["dataset"] / ("semi%02d" % c["ratio"]) / "fold2" / c["tag"]
    if Path(c["output"]) != CONTROL_BASE / relative / "formal":
        raise ValueError("SFR output is outside the exact canonical job")
    init = c.get("foundation", {})
    if (init.get("family") != "original SAM 2D ViT-B" or init.get("bytes") != ORIGINAL_VIT_B_BYTES
            or not re.fullmatch(r"[0-9a-f]{64}", init.get("sha256", ""))
            or init.get("source_url") != "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            or not Path(init.get("path", "")).is_absolute()
            or Path(init["path"]).name != "sam_vit_b_01ec64.pth"
            or ROOT not in Path(init["path"]).parents):
        raise ValueError("SFR original foundation pin is missing or substituted")
    code = Path(c["official_code_root"])
    if not code.is_absolute() or ROOT not in code.parents:
        raise ValueError("SFR official code must be independently deployed inside the authorized project")
    return plan


def prospective_contract(dataset, ratio, tag, official_code_root, foundation_pin):
    """Read-only contract construction, not enrollment/qualification/dispatch.

    The foundation_pin must come from the verified download/provenance receipt,
    never from this function guessing a digest from a filename. It deliberately
    does not call the shared r3 prepare function or bind its Turbo initializer.
    """
    stages = stage_plan(dataset, ratio)
    targets = target_plan(dataset, ratio, tag)
    code = Path(official_code_root)
    here = Path(__file__).resolve().parent
    official = sorted(code.rglob("*.py"))
    paths = [here / name for name in OWN_SOURCE_NAMES] + official + [ROOT / "expes/common/seg_metrics.py"]
    sources = [dict(path=str(path), bytes=path.stat().st_size, sha256=resources.sha(path))
               for path in sorted(set(paths))]
    splits = bound_splits(dataset, ratio)
    relative = Path("sfr") / dataset / ("semi%02d" % ratio) / "fold2" / tag
    value = dict(method="sfr", dataset=dataset, ratio=ratio, tag=tag, gpu=1, fold=2, seed=2027,
        target=targets["final"]["path"], sfr_targets=targets, sfr_stages=stages,
        primary_pairs=stages["ssl_labeled_inputs"], output=str(CONTROL_BASE / relative / "formal"),
        foundation=dict(foundation_pin), official_commit=OFFICIAL_COMMIT,
        official_code_root=str(code), official_source_paths=[str(path) for path in official], sources=sources,
        splits=splits, case_files=bind_case_files(dataset, splits),
        selection="terminal_after_fixed_exposure", test_evaluations=1, early_stop=False,
        exact_resume_supported=False, comparison_scope="primary SSL 3D inputs only; required SAM prefix and auxiliaries separate")
    verify_sfr_contract(value)
    return value


def verify_sfr_contract(c):
    """Does not reuse r2's Turbo-specific verifier or read test image contents."""
    validate_contract_shape(c)
    rows = c["sources"]
    paths = [row["path"] for row in rows]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate SFR source binding")
    for row in rows:
        path = Path(row["path"])
        if (not path.is_absolute() or not path.is_file() or path.is_symlink()
                or any(p.is_symlink() for p in path.parents)
                or path.stat().st_size != row["bytes"] or resources.sha(path) != row["sha256"]):
            raise ValueError("SFR source binding changed: " + str(path))
    here = Path(__file__).resolve().parent
    required = {str(here / name) for name in OWN_SOURCE_NAMES}
    code = Path(c["official_code_root"])
    official = {str(path) for path in code.rglob("*.py")}
    required |= official
    required.add(str(ROOT / "expes/common/seg_metrics.py"))
    if not official or not required.issubset(set(paths)):
        raise ValueError("SFR source/metric dependency inventory is incomplete")
    if set(c["official_source_paths"]) != official:
        raise ValueError("Unbound/changed SFR official Python dependency set")
    verify_native_sources(code)
    splits = bound_splits(c["dataset"], c["ratio"])
    if splits != c["splits"] or bind_case_files(c["dataset"], splits) != c["case_files"]:
        raise ValueError("SFR actual split/case file identity changed")
    init = c["foundation"]
    checkpoint = Path(init["path"])
    if (not checkpoint.is_file() or checkpoint.is_symlink() or any(p.is_symlink() for p in checkpoint.parents)
            or checkpoint.stat().st_size != init["bytes"] or resources.sha(checkpoint) != init["sha256"]):
        raise ValueError("SFR original SAM checkpoint changed")


def enroll_contract_targets(c):
    """Explicit preparation hook. New dispatch code must call this before bind.

    Existing final.pth-only registry semantics are preserved. A failed enrollment
    leaves already reserved entries intact and grants no launch permission.
    """
    validate_contract_shape(c)
    return {role: resources.enroll_target(Path(row["path"]), row["bound"])
            for role, row in c["sfr_targets"].items()}


def tensor_state_signature(state):
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _identity(path):
    stat = Path(path).stat()
    return dict(device=stat.st_dev, inode=stat.st_ino, bytes=stat.st_size, links=stat.st_nlink)


def _same_tree(torch, left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and set(left) == set(right) and all(_same_tree(torch, left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(_same_tree(torch, a, b) for a, b in zip(left, right))
    return left == right


class CountedOptimizer:
    def __init__(self, optimizer):
        self.inner, self.calls = optimizer, 0
    @property
    def param_groups(self):
        return self.inner.param_groups
    def zero_grad(self):
        self.inner.zero_grad()
    def step(self):
        self.inner.step()
        self.calls += 1
    def state_dict(self):
        return self.inner.state_dict()
    def load_state_dict(self, state):
        return self.inner.load_state_dict(state)


class SFRBackend:
    """FixedFinalRun-compatible backend, requiring a new qualified dispatcher."""
    def __init__(self, contract, output, verify_contract=None):
        self.contract, self.output = contract, Path(output)
        self.external_verifier = verify_contract
        self.plan = validate_contract_shape(contract)
        self.budget = next(b for b in BUDGETS if b.dataset == contract["dataset"] and b.ratio == contract["ratio"])
        self.targets = contract["sfr_targets"]
        self.target = Path(self.targets["final"]["path"])
        self.initialized = False
        self.prefix_artifact = self.final_artifact = None
        self.reload_verified = False
        self.cycle_receipts = []
        self.test_started = False
        self.measured_extra_engineering_optimizer_updates = 0

    def _verify(self):
        verify_sfr_contract(self.contract)
        if self.external_verifier is not None and self.external_verifier is not verify_sfr_contract:
            self.external_verifier(self.contract)

    def _guard(self, role):
        row = self.targets[role]
        return resources.resource_backend(Path(row["path"]), row["bound"])

    def _enrollment_locked(self):
        rows = json.loads(resources.REGISTRY.read_text())
        if len({r["path"] for r in rows}) != len(rows):
            raise ValueError("Duplicate shared target registration")
        mapped = {r["path"]: r["bound"] for r in rows}
        if any(mapped.get(row["path"]) != row["bound"] for row in self.targets.values()):
            raise ValueError("Every exact SFR prefix/final/probe target must be enrolled before initialization")

    def _capacity(self, role="final"):
        guard = self._guard(role)
        with guard.original_lock():
            self._enrollment_locked()
            return guard.capacity_snapshot(resources.RAM_REQUIRED)

    def _save(self, role, payload):
        self._capacity(role)
        artifact = resources.save_terminal(Path(self.targets[role]["path"]), payload)
        artifact["role"] = role
        artifact["identity"] = _identity(artifact["path"])
        return artifact

    def _reload(self, role, artifact, verifier):
        target = Path(self.targets[role]["path"])
        if artifact["path"] != str(target) or artifact.get("role") != role:
            raise ValueError("Wrong exact SFR artifact role/path")
        guard = self._guard(role)
        with guard.original_lock():
            self._enrollment_locked()
            guard.capacity_snapshot(resources.RAM_REQUIRED)
            if (target.is_symlink() or any(p.is_symlink() for p in target.parents)
                    or resources.sha(target) != artifact["sha256"]
                    or _identity(target) != artifact["identity"] or target.stat().st_nlink != 1):
                raise ValueError("SFR artifact identity/SHA changed before strict reload")
            loaded = self.torch.load(str(target), map_location=self.device)
            verifier(loaded)
            if resources.sha(target) != artifact["sha256"] or _identity(target) != artifact["identity"]:
                raise ValueError("SFR artifact changed during strict reload")
        return loaded

    def _append(self, filename, row):
        with (self.output / filename).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _endpoint(self, endpoint):
        if (endpoint.method != "sfr" or endpoint.labeled_inputs != self.budget.primary_pairs
                or endpoint.unlabeled_inputs != self.budget.primary_pairs
                or endpoint.optimizer_updates_per_iteration != 1
                or tuple(endpoint.test_case_ids) != tuple(self.contract["splits"]["test"])):
            raise ValueError("SFR actual fixed endpoint differs")

    def prepare(self, endpoint, perform_prefix=True):
        if self.initialized:
            raise ValueError("SFR backend cannot be prepared/reused twice")
        self._verify()
        self._endpoint(endpoint)
        capacity = self._capacity()
        for row in self.targets.values():
            if Path(row["path"]).exists() or Path(row["path"] + ".partial").exists():
                raise ValueError("Existing SFR artifact/partial; inspect instead of restarting")
        import numpy as np
        import torch
        self.np, self.torch = np, torch
        self.device = torch.device("cuda:0")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("SFR worker must expose exactly the designated GPU1")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
            raise ValueError("SFR worker is not bound to authorized GPU1")
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        random.seed(2027)
        np.random.seed(2027)
        torch.manual_seed(2027)
        torch.cuda.manual_seed_all(2027)
        self.splits = self.contract["splits"]
        self.pairs = schedule(self.budget, self.splits)
        bound = BoundCases(self.contract["dataset"], self.splits, training_flip=False,
                           file_bindings=self.contract["case_files"])
        init = self.contract["foundation"]
        # Keep foundation construction under the same RAM/FS check as peers.
        guard = self._guard("prefix")
        with guard.original_lock():
            self._enrollment_locked()
            guard.capacity_snapshot(resources.RAM_REQUIRED)
            sam, self.foundation_receipt = build_bound_sam(self.contract["official_code_root"],
                                    init["path"], init["sha256"], self.device)
        self.core = SFRTrainingCore(self.contract["official_code_root"], sam, self.foundation_receipt,
            self.contract["dataset"], self.contract["ratio"], self.splits["train_lab"], self.pairs, self.device)
        self.core.prefix_optimizer = CountedOptimizer(self.core.prefix_optimizer)
        self.data = NativeBoundTrainingData(bound, self.core.ops)
        self.initialized = True
        self.output.mkdir(parents=True, exist_ok=True)
        self.initial_sam_signature = tensor_state_signature(sam.state_dict())
        if perform_prefix:
            self._run_prefix()
        return dict(source_pins_verified=True, input_pins_verified=True, initialization_verified=True,
            resource_gate_passed=True, test_unopened=True, capacity=capacity,
            foundation=self.foundation_receipt, native_stages=self.plan,
            prefix_executed=perform_prefix, prefix_artifact=self.prefix_artifact,
            initialization_kind="original_2D_SAM_ViT-B_LoRA4_then_fresh_VNet_MT_seed2027",
            gt_roi_used=True, total_compute_matched=False)

    def _run_prefix(self):
        if any((self.output / name).exists() for name in ("sfr_prefix_steps.jsonl", "sfr_prefix_terminal.json")):
            raise ValueError("Existing SFR prefix ledger; no implicit resume")
        started = time.perf_counter()
        for identities in self.core.prefix_schedule:
            row = self.core.finetune_iteration(self.data.batch("train_lab", identities))
            self.torch.cuda.synchronize()
            self._append("sfr_prefix_steps.jsonl", row)
        if self.core.prefix_optimizer.calls != self.plan["prefix_optimizer_updates"]:
            raise ValueError("Actual prefix optimizer calls differ")
        payload = self.core.prefix_terminal_payload()
        self.prefix_signature = tensor_state_signature(payload["native_lora_state"])
        payload["native_lora_signature"] = self.prefix_signature
        payload["ledger_sha256"] = resources.sha(self.output / "sfr_prefix_steps.jsonl")
        self.prefix_artifact = self._save("prefix", payload)
        def accept(loaded):
            if (loaded["native_lora_signature"] != self.prefix_signature
                    or tensor_state_signature(loaded["native_lora_state"]) != self.prefix_signature
                    or loaded["ledger_sha256"] != payload["ledger_sha256"]):
                raise ValueError("Prefix strict disk state/ledger differs")
            self.core.begin_retraining(loaded)
        # begin_retraining verifies exact stage/count/foundation plus tensor
        # identity, loads strictly, freezes SAM and builds native fresh VNet/MT.
        self._reload("prefix", self.prefix_artifact, accept)
        self.core.optimizer = CountedOptimizer(self.core.optimizer)
        self.initial_student_signature = tensor_state_signature(self.core.student.state_dict())
        self.prefix_wall_seconds = time.perf_counter() - started
        exclusive_json(self.output / "sfr_prefix_terminal.json", dict(status="PREFIX_FIXED_TERMINAL_VERIFIED",
            artifact=self.prefix_artifact, native_lora_signature=self.prefix_signature,
            plan=self.plan, ledger_sha256=payload["ledger_sha256"], strict_disk_reload=True,
            sam_frozen=True, test_evaluations=0, wall_seconds=self.prefix_wall_seconds,
            actual_optimizer_updates=self.plan["prefix_optimizer_updates"]))

    def train_iteration(self, remaining):
        if not self.prefix_artifact or min(remaining.values()) < 2:
            raise ValueError("SFR requires the verified prefix and complete native 2+2 tail")
        pairs = self.pairs[self.core.ssl_position:self.core.ssl_position + 2]
        before = self.core.optimizer.calls
        row = self.core.retrain_iteration(self.data.batch("train_lab", [p["labeled"] for p in pairs]),
                                         self.data.batch("train_unlab", [p["unlabeled"] for p in pairs]))
        self.torch.cuda.synchronize()
        if self.core.optimizer.calls - before != 1 or not math.isfinite(row["loss"]):
            raise ValueError("Measured SFR optimizer/loss receipt differs")
        row.update(finite_loss=True, finite_nonzero_gradients=True, actual_case_ids_verified=True,
                   successful_optimizer_updates=self.core.optimizer.calls - before)
        # Native batches can cross an odd-length coverage-cycle boundary.
        complete = self.core.ssl_position // self.budget.primary_pairs_per_cycle
        while len(self.cycle_receipts) < complete:
            epoch = len(self.cycle_receipts)
            start = epoch * self.budget.primary_pairs_per_cycle
            actual = [p for r in self.core.ssl_rows for p in r["actual_pairs"]][start:start + self.budget.primary_pairs_per_cycle]
            self.cycle_receipts.append(self._verify_cycle(epoch, actual))
        return row

    def _verify_cycle(self, epoch, actual):
        if (len(actual) != self.budget.primary_pairs_per_cycle
                or set(p["labeled"] for p in actual) != set(self.splits["train_lab"])
                or set(p["unlabeled"] for p in actual) != set(self.splits["train_unlab"])
                or any(p["epoch"] != epoch for p in actual)
                or sum(p["unlabeled_replay"] for p in actual) != self.budget.replay_per_cycle):
            raise ValueError("Actual SFR cycle coverage differs")
        return dict(epoch=epoch, actual_case_coverage_verified=True,
                    unlabeled_replay=self.budget.replay_per_cycle)

    def verify_training_coverage(self):
        actual = [p for row in self.core.ssl_rows for p in row["actual_pairs"]]
        if (actual != self.pairs or self.core.ssl_position != self.budget.primary_pairs
                or len(self.cycle_receipts) != 36 or self.core.optimizer.calls != self.plan["ssl_optimizer_updates"]):
            raise ValueError("SFR full actual budget/optimizer/case ledger differs")
        return dict(actual_case_coverage_verified=True, cycles=self.cycle_receipts, native_stages=self.plan,
                    primary_labeled_inputs=len(actual), primary_unlabeled_inputs=len(actual),
                    prefix_artifact=self.prefix_artifact, prefix_wall_seconds=self.prefix_wall_seconds,
                    ssl_teacher_input_forwards=sum(r["teacher_input_forwards"] for r in self.core.ssl_rows),
                    ssl_sam_stitched_board_forwards=sum(r["sam_stitched_board_forwards"] for r in self.core.ssl_rows))

    def save_terminal(self, counts):
        if counts != dict(labeled_inputs=self.budget.primary_pairs, unlabeled_inputs=self.budget.primary_pairs,
                          iterations=self.plan["ssl_iterations"], successful_optimizer_updates=self.plan["ssl_iterations"]):
            raise ValueError("SFR terminal controller counts differ")
        self.verify_training_coverage()
        payload = self.core.terminal_payload()
        self.final_signature = tensor_state_signature(payload["student_state"])
        if self.final_signature == self.initial_student_signature:
            raise ValueError("SFR native terminal student did not change")
        payload.update(counts=counts, native_student_signature=self.final_signature,
                       prefix_artifact=self.prefix_artifact, prefix_wall_seconds=self.prefix_wall_seconds)
        self.final_artifact = self._save("final", payload)
        return self.final_artifact

    def strict_reload(self, artifact):
        def accept(loaded):
            if (loaded["method"] != "sfr" or loaded["plan"] != self.plan
                    or loaded["foundation"] != self.foundation_receipt
                    or loaded["prefix_artifact"] != self.prefix_artifact
                    or loaded["native_student_signature"] != self.final_signature
                    or tensor_state_signature(loaded["student_state"]) != self.final_signature):
                raise ValueError("SFR final payload identity differs")
            self.core.student.load_state_dict(loaded["student_state"], strict=True)
            if tensor_state_signature(self.core.student.state_dict()) != self.final_signature:
                raise ValueError("SFR strict native student reload differs")
        self._reload("final", artifact, accept)
        self.reload_verified = True
        return dict(strict=True, sha256=artifact["sha256"], native_student_identity_verified=True)

    def test(self, artifact, permit):
        if self.test_started or not self.reload_verified or artifact != self.final_artifact:
            raise ValueError("SFR unique test gate is not open")
        self._verify()
        permit = Path(permit)
        expected = dict(selection="terminal_after_fixed_exposure", final_sha256=artifact["sha256"],
                        test_case_ids=list(self.splits["test"]), test_evaluation=1)
        if permit != self.output / "final_test_intent.json" or json.loads(permit.read_text()) != expected:
            raise ValueError("SFR test permit identity differs")
        # Exclusive durable start prevents another process/backend instance from
        # reusing an existing intent after a partial evaluation.
        exclusive_json(self.output / "sfr_test_started.json", dict(final_sha256=artifact["sha256"],
                       pid=os.getpid(), test_evaluations=1))
        self.test_started = True
        if (self.output / "final_test_cases.jsonl").exists():
            raise ValueError("Existing SFR test cases; preserve rather than repeat")
        from expes.common.seg_metrics import binary_case_metrics
        data = BoundCases(self.contract["dataset"], self.splits, test_permit=permit, training_flip=False,
                          file_bindings=self.contract["case_files"])
        self.core.student.eval()
        rows = []
        for identity in self.splits["test"]:
            case = data.load("test", identity)
            volume = case["image"][0].detach().cpu().numpy()
            std = float(self.np.std(volume))
            if not bool(self.np.isfinite(volume).all()) or not math.isfinite(std) or std <= 0:
                raise ValueError("SFR nonfinite/constant test case; no substitution")
            volume = self.torch.from_numpy(((volume - self.np.mean(volume)) / std).copy()).float().to(self.device)
            prediction, _, coverage = sliding_predict(self.core.student, volume)
            row = dict(case_id=identity, actual_index=case["actual_index"], gt_roi_used=True,
                inference_coverage=coverage,
                **binary_case_metrics(prediction.cpu(), case["label"], spacing=(1., 1., 1.)))
            self._append("final_test_cases.jsonl", row)
            rows.append(row)
        return dict(case_ids=[r["case_id"] for r in rows], case_dice=[r["dice"] for r in rows],
            **{key: sum(r[key] for r in rows) / len(rows) for key in ("dice", "jaccard", "hd95", "asd")},
            final_sha256=artifact["sha256"], test_evaluations=1, distance_unit="voxel",
            prefix_artifact=self.prefix_artifact, native_stages=self.plan, gt_roi_used=True,
            inference="complete_shared_ROI_native_112x112x80_windows_stride18_18_4")

    def _delete_own_probe(self, artifact):
        target = Path(self.targets["qualification_probe"]["path"])
        guard = self._guard("qualification_probe")
        with guard.original_lock():
            self._enrollment_locked()
            guard.capacity_snapshot(resources.RAM_REQUIRED)
            if (artifact["path"] != str(target) or artifact.get("role") != "qualification_probe"
                    or target.is_symlink() or _identity(target) != artifact["identity"]
                    or target.stat().st_nlink != 1 or resources.sha(target) != artifact["sha256"]):
                raise ValueError("Own qualification probe identity differs; preserve")
            target.unlink()
            fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        return dict(exact_own_probe_deleted=True, bytes=artifact["bytes"], sha256=artifact["sha256"],
                    original_lock_verified=True, target_reservation_retained=True)

    def qualify(self, endpoint):
        """One real native update per stage, then exact update-zero restoration.

        This does not forge a complete prefix or call begin_retraining on a
        one-update prefix. The independent SSL engineering witness uses the
        same native models/ops/noise/optimizer and is never a formal result.
        """
        self.output.mkdir(parents=True, exist_ok=False)
        exclusive_json(self.output / "sfr_qualification_intent.json", dict(stages=["sam_finetuning", "vnet_retraining"],
                       test_evaluations=0, formal_optimizer_updates=0))
        try:
            preparation = self.prepare(endpoint, perform_prefix=False)
            torch = self.torch
            from official_compat import load_sfr_architecture
            VNet = load_sfr_architecture(self.contract["official_code_root"])
            random.seed(2027)
            self.np.random.seed(2027)
            torch.manual_seed(2027)
            student = VNet(n_channels=1, n_classes=2, normalization="batchnorm", has_dropout=True,
                           pyramid_has_dropout=True).to(self.device)
            teacher = VNet(n_channels=1, n_classes=2, normalization="batchnorm", has_dropout=True,
                           pyramid_has_dropout=True).to(self.device)
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
            optimizer = CountedOptimizer(torch.optim.SGD(student.parameters(), lr=.01, momentum=.9, weight_decay=.0001))
            flags = {name: p.requires_grad for name, p in self.core.sam.named_parameters()}
            initial = dict(sam_compact={k: v.cpu() for k, v in lora_native_state(self.core.sam).items()},
                student=_cpu_state(student), teacher=_cpu_state(teacher),
                prefix_optimizer=copy.deepcopy(self.core.prefix_optimizer.state_dict()),
                ssl_optimizer=copy.deepcopy(optimizer.state_dict()),
                random=random.getstate(), numpy=self.np.random.get_state(), cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(), sam_flags=flags)
            signatures = dict(sam=tensor_state_signature(self.core.sam.state_dict()),
                              student=tensor_state_signature(initial["student"]), teacher=tensor_state_signature(initial["teacher"]))
            prefix = self.core.finetune_iteration(self.data.batch("train_lab", self.core.prefix_schedule[0]))
            self.measured_extra_engineering_optimizer_updates = self.core.prefix_optimizer.calls
            self._append("sfr_qualification_steps.jsonl", prefix)
            freeze_prefix(self.core.sam)
            pairs = self.pairs[:2]
            lab = self.data.batch("train_lab", [p["labeled"] for p in pairs])
            unlab = self.data.batch("train_unlab", [p["unlabeled"] for p in pairs])
            volume = torch.cat([lab["image"], unlab["image"]]).to(self.device)
            labels = lab["label"].to(self.device)
            student.train()
            teacher.train()
            noise = torch.clamp(torch.randn_like(volume) * .1, -.2, .2)
            logits = student(volume)[0]
            with torch.no_grad():
                teacher_logits = teacher(volume + noise)[0]
            _, pseudo = self.core.ops.pseudo_labels(self.core.sam, volume[2:])
            loss, parts = self.core.ops.retrain_loss(logits, teacher_logits, labels, pseudo, 0, self.plan["ssl_iterations"])
            norm = self.core._step(loss, optimizer, student)
            self.measured_extra_engineering_optimizer_updates += optimizer.calls
            ema_parameters_only(student, teacher, 0)
            torch.cuda.synchronize()
            if any(p.grad is not None for p in teacher.parameters()) or any(p.grad is not None for p in self.core.sam.parameters()):
                raise ValueError("SFR qualification frozen branch gradient")
            ssl = dict(stage="vnet_retraining_engineering_witness", actual_pairs=pairs,
                       labeled_inputs=2, unlabeled_inputs=2, successful_optimizer_updates=optimizer.calls,
                       loss=float(loss.detach()), gradient_norm=norm, teacher_no_grad=True,
                       sam_no_grad=True, unlabeled_gt_in_loss=False, test_evaluations=0)
            self._append("sfr_qualification_steps.jsonl", ssl)
            del volume, labels, logits, teacher_logits, pseudo, loss, parts
            def restore(state):
                strict_lora_reload(self.core.sam, state["sam_compact"])
                for name, p in self.core.sam.named_parameters():
                    p.requires_grad_(state["sam_flags"][name])
                    p.grad = None
                self.core.sam.train()
                student.load_state_dict(state["student"], strict=True)
                teacher.load_state_dict(state["teacher"], strict=True)
                for model in (student, teacher):
                    for p in model.parameters():
                        p.grad = None
                self.core.prefix_optimizer.load_state_dict(state["prefix_optimizer"])
                optimizer.load_state_dict(state["ssl_optimizer"])
                if (tensor_state_signature(self.core.sam.state_dict()) != signatures["sam"]
                        or tensor_state_signature(student.state_dict()) != signatures["student"]
                        or tensor_state_signature(teacher.state_dict()) != signatures["teacher"]
                        or not _same_tree(torch, self.core.prefix_optimizer.state_dict(), state["prefix_optimizer"])
                        or not _same_tree(torch, optimizer.state_dict(), state["ssl_optimizer"])):
                    raise ValueError("SFR update-zero native model/optimizer restoration differs")
                random.setstate(state["random"])
                self.np.random.set_state(state["numpy"])
                torch.set_rng_state(state["cpu_rng"].cpu())
                torch.cuda.set_rng_state(state["cuda_rng"].cpu())
                actual_np = self.np.random.get_state()
                if (random.getstate() != state["random"] or actual_np[0] != state["numpy"][0]
                        or not self.np.array_equal(actual_np[1], state["numpy"][1])
                        or actual_np[2:] != state["numpy"][2:]
                        or not torch.equal(torch.get_rng_state(), state["cpu_rng"].cpu())
                        or not torch.equal(torch.cuda.get_rng_state(), state["cuda_rng"].cpu())):
                    raise ValueError("SFR qualification RNG restoration differs")
            restore(initial)
            probe = self._save("qualification_probe", dict(format="sfr_own_update_zero_probe_v1", states=initial,
                               signatures=signatures, foundation_sha256=self.contract["foundation"]["sha256"]))
            def accept(loaded):
                if (loaded["format"] != "sfr_own_update_zero_probe_v1" or loaded["signatures"] != signatures
                        or loaded["foundation_sha256"] != self.contract["foundation"]["sha256"]):
                    raise ValueError("SFR probe native identity differs")
                restore(loaded["states"])
            self._reload("qualification_probe", probe, accept)
            exclusive_json(self.output / "sfr_qualification_roundtrip.json", dict(artifact=probe,
                           actual_disk_roundtrip=True, native_state_signatures=signatures,
                           restored_model_optimizer_rng=True))
            deleted = self._delete_own_probe(probe)
            self.core.prefix_rows = []
            self.core.prefix_optimizer.calls = 0
            result = dict(status="QUALIFIED", preparation=preparation,
                actual_extra_engineering_updates=self.measured_extra_engineering_optimizer_updates,
                prefix_receipt=prefix, ssl_receipt=ssl, new_test_evaluations=0,
                restored_model_optimizer_rng=True, actual_disk_roundtrip=True,
                probe=probe, probe_cleanup=deleted, native_state_signatures=signatures,
                complete_prefix_executed=False, qualification_not_formal_training=True,
                formal_worker_must_start_fresh=True)
            exclusive_json(self.output / "sfr_qualification_verified.json", result)
            return result
        except BaseException as error:
            if hasattr(self, "core"):
                self.measured_extra_engineering_optimizer_updates = max(self.measured_extra_engineering_optimizer_updates,
                    getattr(self.core.prefix_optimizer, "calls", 0) + (optimizer.calls if "optimizer" in locals() else 0))
            exclusive_json(self.output / "sfr_qualification_failure.json", dict(error=repr(error),
                actual_extra_engineering_optimizer_updates=self.measured_extra_engineering_optimizer_updates,
                formal_optimizer_updates=0, new_test_evaluations=0, failed_attempt_preserved=True))
            raise
