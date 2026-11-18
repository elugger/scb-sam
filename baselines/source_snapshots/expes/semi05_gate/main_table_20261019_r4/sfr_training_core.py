"""Two-stage SFR update engine; deliberately has no dispatch/file/test actions.

The caller must provide bound real cases and an independently SHA-bound original
SAM ViT-B model, persist each returned receipt, and supply the existing original
resource lock for all prefix/terminal writes. This engine does not grant a job
qualification, a disk-roundtrip assertion, or permission to access test data.
"""
import math
import random

from sfr_runtime_components import (NativeOps, ema_parameters_only, freeze_prefix,
                                    lora_native_state, strict_lora_reload)
from sfr_stage_protocol import CANVAS, PATCH, finetune_batches, prefix_lr, ssl_clock, stage_plan


class NativeBoundTrainingData:
    """Use the existing identity-bound reader with the native SFR augmentation."""
    def __init__(self, bound_cases, ops):
        if bound_cases.training_flip:
            raise ValueError("Construct BoundCases(training_flip=False); avoid duplicate augmentation")
        if bound_cases.test_permit is not None:
            raise ValueError("SFR training reader must not hold a test permit")
        self.bound_cases, self.ops = bound_cases, ops

    def batch(self, role, identities):
        if role not in ("train_lab", "train_unlab"):
            raise ValueError("Training reader cannot load test")
        cases = [self.ops.training_sample(self.bound_cases.load(role, case)) for case in identities]
        images = self.ops.torch.stack([case["image"] for case in cases])
        labels = None if role == "train_unlab" else self.ops.torch.stack([case["label"] for case in cases])
        if role == "train_unlab" and any(case["label"] is not None for case in cases):
            raise ValueError("Unlabeled GT crossed native training reader")
        return dict(image=images, label=labels, case_ids=list(identities),
                    actual_indices=[case["actual_index"] for case in cases],
                    role=role, shared_gt_roi_used=all(case["gt_roi_used"] for case in cases),
                    native_patch=list(PATCH), unlab_gt_returned=False)


class SFRTrainingCore:
    """Native fine-tuning then native 3D SSL, with explicit input identities.

    This class is an unqualified candidate integration component. In particular,
    passing a terminal payload into begin_retraining verifies its in-memory
    identity only; the caller must first save and strictly reload it under the
    existing lock and bind that artifact in its contract/ledger.
    """
    def __init__(self, code_root, sam, foundation_receipt, dataset, ratio,
                 labeled_case_ids, primary_pairs, device):
        self.plan = stage_plan(dataset, ratio)
        self.ops = NativeOps(code_root)
        self.torch = self.ops.torch
        self.sam, self.device = sam, self.torch.device(device)
        self.foundation_receipt = dict(foundation_receipt)
        if (foundation_receipt.get("family") != "original SAM 2D ViT-B"
                or len(foundation_receipt.get("sha256", "")) != 64
                or foundation_receipt.get("image_size") != CANVAS
                or foundation_receipt.get("lora_rank") != 4):
            raise ValueError("A separately bound native foundation receipt is required")
        if next(sam.parameters()).device != self.device:
            raise ValueError("SAM is not on the explicit worker device")
        from budget_protocol import BUDGETS
        budget = next(b for b in BUDGETS if b.dataset == dataset and b.ratio == ratio)
        self.labeled = tuple(labeled_case_ids)
        self.pairs = list(primary_pairs)
        if len(self.labeled) != budget.labeled_cases or len(set(self.labeled)) != len(self.labeled):
            raise ValueError("Actual labeled population differs from the frozen budget")
        if len(self.pairs) != budget.primary_pairs:
            raise ValueError("Actual SSL pair schedule differs from the frozen budget")
        if any(p["labeled"] not in self.labeled or p["unlabeled"] in self.labeled for p in self.pairs):
            raise ValueError("SFR pair roles differ or overlap")
        if len({p["unlabeled"] for p in self.pairs}) != budget.unlabeled_cases:
            raise ValueError("SFR scheduled unlabeled population differs")
        self.prefix_schedule = list(finetune_batches(self.labeled))
        self.prefix_rows, self.ssl_rows = [], []
        self.prefix_complete, self.ssl_position = False, 0
        self.sam.train()
        params = [p for p in self.sam.parameters() if p.requires_grad]
        if not params:
            raise ValueError("No native LoRA/prompt/decoder parameters are trainable")
        self.prefix_optimizer = self.torch.optim.AdamW(params, lr=.001, betas=(.9, .999), weight_decay=.1)
        self.stage = "sam_finetuning"

    def _step(self, loss, optimizer, model):
        torch = self.torch
        if not bool(torch.isfinite(loss)):
            raise ValueError("SFR nonfinite loss; preserve the failed attempt")
        optimizer.zero_grad()
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise ValueError("SFR missing/nonfinite gradients; do not step or silently skip")
        norm = math.sqrt(sum(float(g.detach().double().square().sum()) for g in gradients))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("SFR zero/nonfinite gradient norm")
        optimizer.step()
        return norm

    def finetune_iteration(self, batch):
        if self.stage != "sam_finetuning" or len(self.prefix_rows) >= self.plan["prefix_iterations"]:
            raise ValueError("SFR prefix is not accepting another update")
        index = len(self.prefix_rows)
        expected = self.prefix_schedule[index]
        if batch["role"] != "train_lab" or tuple(batch["case_ids"]) != expected or batch["label"] is None:
            raise ValueError("Prefix actual case identities differ from the next fixed batch")
        images, labels = batch["image"], batch["label"]
        x, y = self.ops.finetune_board(images, labels, self.device)
        lr = prefix_lr(index)
        for group in self.prefix_optimizer.param_groups:
            group["lr"] = lr
        self.sam.train()
        output = self.sam(x, False, CANVAS)["masks"]
        loss, parts = self.ops.finetune_loss(output, y)
        norm = self._step(loss, self.prefix_optimizer, self.sam)
        row = dict(stage="sam_finetuning", iteration=index + 1,
                   actual_labeled_case_ids=list(expected), labeled_inputs=len(expected),
                   successful_optimizer_updates=1, sam_stitched_board_forwards=len(expected),
                   lr=lr, loss=float(loss.detach()),
                   losses={k: float(v.detach()) for k, v in parts.items()}, gradient_norm=norm,
                   native_one_case_tail=len(expected) == 1, test_evaluations=0)
        self.prefix_rows.append(row)
        return row

    def prefix_terminal_payload(self):
        if (len(self.prefix_rows) != self.plan["prefix_iterations"]
                or sum(r["labeled_inputs"] for r in self.prefix_rows) != self.plan["prefix_labeled_inputs"]
                or [tuple(r["actual_labeled_case_ids"]) for r in self.prefix_rows] != self.prefix_schedule):
            raise ValueError("SFR required native prefix is incomplete")
        self.prefix_complete = True
        return dict(method="sfr", stage="sam_finetuning_terminal", plan=self.plan,
                    foundation=self.foundation_receipt, selection="fixed_terminal_6000_no_test",
                    native_lora_state=lora_native_state(self.sam),
                    successful_optimizer_updates=len(self.prefix_rows),
                    actual_labeled_inputs=sum(r["labeled_inputs"] for r in self.prefix_rows),
                    disk_roundtrip_verified=False)

    def begin_retraining(self, reloaded_terminal_payload):
        if self.stage != "sam_finetuning" or not self.prefix_complete:
            raise ValueError("Complete and persist the prefix before the native 3D stage")
        payload = reloaded_terminal_payload
        if (payload.get("method") != "sfr" or payload.get("stage") != "sam_finetuning_terminal"
                or payload.get("plan") != self.plan or payload.get("foundation") != self.foundation_receipt
                or payload.get("selection") != "fixed_terminal_6000_no_test"
                or payload.get("successful_optimizer_updates") != self.plan["prefix_iterations"]
                or payload.get("actual_labeled_inputs") != self.plan["prefix_labeled_inputs"]):
            raise ValueError("SFR prefix terminal identity differs")
        current = lora_native_state(self.sam)
        state = payload["native_lora_state"]
        if set(current) != set(state) or any(not self.torch.equal(current[k], state[k].to(current[k].device))
                                           for k in current):
            raise ValueError("Loaded prefix is not this run's fixed terminal state")
        strict_lora_reload(self.sam, state)
        freeze_prefix(self.sam)
        # Official retraining is a second script with a freshly seeded RNG.
        random.seed(2027)
        self.ops.np.random.seed(2027)
        self.torch.manual_seed(2027)
        from official_compat import load_sfr_architecture
        VNet = load_sfr_architecture(self.ops.code_root)
        self.student = VNet(n_channels=1, n_classes=2, normalization="batchnorm", has_dropout=True,
                            pyramid_has_dropout=True).to(self.device)
        self.teacher = VNet(n_channels=1, n_classes=2, normalization="batchnorm", has_dropout=True,
                            pyramid_has_dropout=True).to(self.device)
        # Preserve independent teacher initialization and the first alpha=0
        # parameter update. BN buffers retain the native teacher forward state.
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.optimizer = self.torch.optim.SGD(self.student.parameters(), lr=.01,
                                               momentum=.9, weight_decay=.0001)
        self.prefix_optimizer = None
        self.stage = "vnet_retraining"
        return dict(stage=self.stage, prefix_identity_matches=True, frozen_sam=True,
                    fresh_stage_seed=2027, native_teacher_initialization="independent",
                    prefix_disk_verification_required_from_outer_worker=True)

    def retrain_iteration(self, labeled_batch, unlabeled_batch):
        if self.stage != "vnet_retraining" or self.ssl_position >= len(self.pairs):
            raise ValueError("SFR SSL is not accepting another update")
        pairs = self.pairs[self.ssl_position:self.ssl_position + 2]
        if (len(pairs) != 2 or labeled_batch["role"] != "train_lab"
                or unlabeled_batch["role"] != "train_unlab"
                or labeled_batch["case_ids"] != [p["labeled"] for p in pairs]
                or unlabeled_batch["case_ids"] != [p["unlabeled"] for p in pairs]
                or unlabeled_batch["label"] is not None):
            raise ValueError("SFR actual native batch identities/roles differ")
        self.ops._native_batch(labeled_batch["image"])
        self.ops._native_batch(unlabeled_batch["image"])
        torch = self.torch
        volume = torch.cat([labeled_batch["image"], unlabeled_batch["image"]]).to(self.device)
        labels = labeled_batch["label"].to(self.device)
        index = self.ssl_position // 2
        clock = ssl_clock(index, self.plan["ssl_iterations"])
        for group in self.optimizer.param_groups:
            group["lr"] = clock["lr"]
        self.student.train()
        self.teacher.train()
        noise = torch.clamp(torch.randn_like(volume) * .1, -.2, .2)
        logits = self.student(volume)[0]
        with torch.no_grad():
            teacher_logits = self.teacher(volume + noise)[0]
        _, pseudo = self.ops.pseudo_labels(self.sam, volume[2:])
        loss, parts = self.ops.retrain_loss(logits, teacher_logits, labels, pseudo,
                                           index, self.plan["ssl_iterations"])
        norm = self._step(loss, self.optimizer, self.student)
        if any(p.grad is not None for p in self.teacher.parameters()) or any(p.grad is not None for p in self.sam.parameters()):
            raise ValueError("Frozen SFR teacher/SAM received a gradient")
        ema_parameters_only(self.student, self.teacher, index)
        self.ssl_position += 2
        row = dict(stage="vnet_retraining", iteration=index + 1,
                   labeled_inputs=2, unlabeled_inputs=2, successful_optimizer_updates=1,
                   actual_pairs=pairs, teacher_input_forwards=4, sam_stitched_board_forwards=2,
                   completed_primary_pairs=self.ssl_position, lr=clock["lr"],
                   loss=float(loss.detach()), gradient_norm=norm, teacher_no_grad=True,
                   sam_no_grad=True, unlabeled_gt_in_loss=False, skip_count=0,
                   losses={k: float(v.detach()) if hasattr(v, "detach") else v for k, v in parts.items()},
                   test_evaluations=0)
        self.ssl_rows.append(row)
        return row

    def terminal_payload(self):
        actual_pairs = [pair for row in self.ssl_rows for pair in row["actual_pairs"]]
        if (self.stage != "vnet_retraining" or self.ssl_position != len(self.pairs)
                or len(self.ssl_rows) != self.plan["ssl_iterations"] or actual_pairs != self.pairs):
            raise ValueError("SFR complete fixed-budget input/optimizer ledger is required")
        self.stage = "terminal_requires_outer_save_reload_test"
        return dict(method="sfr", stage="fixed_ssl_terminal", plan=self.plan,
                    foundation=self.foundation_receipt, primary_labeled_inputs=self.ssl_position,
                    primary_unlabeled_inputs=self.ssl_position, optimizer_updates=len(self.ssl_rows),
                    student_state={k: v.detach().clone() for k, v in self.student.state_dict().items()},
                    selection="fixed_terminal_no_test_selection", test_evaluations=0,
                    outer_original_lock_save_strict_reload_unique_test_required=True)
