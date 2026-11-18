"""Audited native SFR components for a future isolated, qualified worker.

No CLI, launch, CUDA initialization, checkpoint download, or test-set access is
performed by importing this module. It is not connected to the shared entry.
The outer source/data/weight binding, resource lock, prefix endpoint, unique
test gate and actual GPU qualification remain mandatory before a real run.
"""
import ast
import hashlib
import importlib
import math
from pathlib import Path
import random

from sfr_stage_protocol import CANVAS, PATCH, inference_windows, prefix_lr, ssl_clock

# Exact local checkout bytes; upload these bytes unchanged with this adapter.
SOURCE_SHA256 = {
    "utils/util.py": "5340a033ce6820c5702c686cb5b2d96c3e9e012d566229a92c9c47ca08ced485",
    "utils/bezier_curve.py": "25116ef4b81512eb0ed61e134963fc60cdf847f97e836dd0d261c7aa53f70b9b",
    "utils/losses.py": "fbe80ddde1305963c57022aca18e2a949e3d97c6c52eb29f54c62e2ef200aea8",
    "dataloaders/dataset.py": "8ec6d6cd26b59f3a502e08f9c6d354dc33d543e11b1d8c1e3f32647bd4f30ddc",
    "test_util_singleclass.py": "2d9d035cae2f306dc461e7151798b88520758f1ff98ab45f7d5eb43fb3cf5dfb",
    "train_finetuning.py": "0498a68c41fc47362f6c14022b5ab3e452233fb9e46c79b790efcd04691cc0b7",
    "train_retraining.py": "aab06e59686b538ca8b3e73dfe2b83faa9cc248dc6c423f11bf6ec7fdc0f29f0",
}


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_native_sources(code_root):
    root = Path(code_root).resolve(strict=True)
    rows = []
    for relative, expected in SOURCE_SHA256.items():
        path = root / relative
        actual = _sha(path)
        if actual != expected:
            raise ValueError("SFR pinned source differs: " + relative)
        rows.append(dict(path=relative, size=path.stat().st_size, sha256=actual))
    return rows


def _definitions(path, names, scope):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in selected} != set(names):
        raise ValueError("Official definition set changed")
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), str(path), "exec"), scope)


class NativeOps:
    """Load unchanged audited definitions without executing official run loops."""
    def __init__(self, code_root):
        self.code_root = Path(code_root)
        self.source_receipt = verify_native_sources(code_root)
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from scipy.special import comb
        self.np, self.torch, self.F = np, torch, F
        scope = dict(np=np, torch=torch, nn=nn, F=F, random=random, comb=comb)
        _definitions(self.code_root / "utils/bezier_curve.py",
                     ("bernstein_poly", "bezier_curve", "nonlinear_transformation_r2"), scope)
        _definitions(self.code_root / "utils/util.py",
                     ("Spread_bs_aug", "Spread_bs_tensor", "RandomRotFlip_2d"), scope)
        _definitions(self.code_root / "dataloaders/dataset.py", ("RandomRotFlip", "RandomCrop"), scope)
        _definitions(self.code_root / "utils/losses.py", ("DiceLoss", "softmax_mse_loss"), scope)
        self.scope = scope
        self.dice_loss = scope["DiceLoss"](2)

    def training_sample(self, bound_sample):
        """128-cube shared GT ROI -> native normalized/augmented 112x112x80.

        BoundCases must be constructed with training_flip=False. The native
        rotation/flip is applied here once. Unlabeled labels are never accepted;
        zeros provide only the shape required by the unchanged crop function.
        """
        torch, np = self.torch, self.np
        role = bound_sample["role"]
        if role not in ("train_lab", "train_unlab"):
            raise ValueError("SFR training adapter cannot open test")
        image = bound_sample["image"]
        label = bound_sample["label"]
        if tuple(image.shape) != (1, 128, 128, 128):
            raise ValueError("Expected already-bound shared 128-cube ROI")
        if role == "train_unlab" and label is not None:
            raise ValueError("Unlabeled GT crossed SFR boundary")
        if role == "train_lab" and (label is None or tuple(label.shape) != (128, 128, 128)):
            raise ValueError("Labeled SFR sample is missing its actual target")
        a = image.detach().cpu().numpy()[0].copy()
        std = float(np.std(a))
        if not bool(np.isfinite(a).all()) or not math.isfinite(std) or std <= 0:
            raise ValueError("Nonfinite or constant case; no silent replacement")
        a = (a - np.mean(a)) / std
        y = np.zeros(a.shape, dtype=np.int64) if label is None else label.cpu().numpy().copy()
        sample = self.scope["RandomRotFlip"]()(dict(image=a, label=y))
        sample = self.scope["RandomCrop"](PATCH)(sample)
        x = torch.from_numpy(sample["image"].copy()).float().unsqueeze(0)
        target = None if label is None else torch.from_numpy(sample["label"].copy()).long()
        return dict(bound_sample, image=x, label=target, native_patch=list(PATCH),
                    native_rotflip_applied=True, normalization="whole_shared_ROI_mean_std",
                    shared_gt_roi_disclosed=bool(bound_sample.get("gt_roi_used", False)),
                    unlab_gt_returned=False)

    def finetune_board(self, images, labels, device):
        self._native_batch(images, minimum=1, maximum=2)
        if tuple(labels.shape) != (images.shape[0],) + PATCH:
            raise ValueError("Prefix target shape differs")
        if not bool(((labels == 0) | (labels == 1)).all()):
            raise ValueError("Nonbinary prefix target")
        # Keep the original per-slice nt=2/nonlinear=.5 and board rotations.
        result = self.scope["Spread_bs_aug"](images.detach().cpu().numpy().copy(),
                    labels.detach().cpu().numpy().copy(), CANVAS, 2, .5)
        board, target = result[:2]
        np = self.np
        rotations = np.array([np.random.randint(0, 4) for _ in range(board.shape[0])])
        axes = np.array([np.random.randint(0, 2) for _ in range(board.shape[0])])
        for i in range(board.shape[0]):
            board[i] = self.scope["RandomRotFlip_2d"](board[i], rotations[i], axes[i])
            target[i] = self.scope["RandomRotFlip_2d"](target[i], rotations[i], axes[i])
        # Replace deprecated np.long and hard-coded .cuda with explicit dtype/device.
        x = self.torch.from_numpy(board.astype(np.float32)).unsqueeze(1).repeat(1, 3, 1, 1).to(device)
        y = self.torch.from_numpy(target.astype(np.int64)).to(device)
        return x, y

    def _native_batch(self, images, minimum=2, maximum=2):
        if (images.ndim != 5 or images.shape[1] != 1 or tuple(images.shape[-3:]) != PATCH
                or not minimum <= images.shape[0] <= maximum):
            raise ValueError("SFR native patch/batch differs; truncation is forbidden")
        if not bool(self.torch.isfinite(images).all()):
            raise ValueError("Nonfinite SFR image")

    def stitch_unlabeled(self, images):
        """No GT argument exists. Official label input is a shape-only zero array."""
        self._native_batch(images)
        cpu = images.detach().cpu()
        dummy = self.torch.zeros((images.shape[0],) + PATCH, dtype=self.torch.long)
        result = self.scope["Spread_bs_tensor"](cpu, dummy, CANVAS)
        return result[0], tuple(result[2:])

    def unstitch(self, board, metadata):
        """Recover every native voxel for BxCx1024x1024 probability/logit boards."""
        if board.ndim != 4 or tuple(board.shape[-2:]) != (CANVAS, CANVAS):
            raise ValueError("SFR output board shape differs")
        nr, nc, pw, ph, ps, pww, phh, sl, sw = metadata
        if nr * nc < PATCH[-1] + 2 * ps:
            raise ValueError("SFR board would lose real slices")
        if pww < 0 or phh < 0:
            board = self.F.pad(board, (abs(phh), abs(phh), abs(pww), abs(pww)))
            pww, phh = 0, 0
        slices = []
        for index in range(ps, ps + PATCH[-1]):
            row, col = divmod(index, nc)
            tile = board[:, :, pww + row * sl:pww + (row + 1) * sl,
                               phh + col * sw:phh + (col + 1) * sw]
            slices.append(tile[:, :, pw:sl - pw if pw else sl, ph:sw - ph if ph else sw])
        recovered = self.torch.stack(slices, 2).permute(0, 1, 4, 3, 2).contiguous()
        if tuple(recovered.shape[2:]) != PATCH:
            raise ValueError("SFR reconstruction differs from all 80 native slices")
        return recovered

    def pseudo_labels(self, frozen_sam, images):
        if frozen_sam.training or any(p.requires_grad for p in frozen_sam.parameters()):
            raise ValueError("SFR SAM must be frozen after the prefix terminal handoff")
        board, metadata = self.stitch_unlabeled(images)
        x = board.unsqueeze(1).repeat(1, 3, 1, 1).to(images.device)
        with self.torch.no_grad():
            masks = frozen_sam(x, False, CANVAS)["masks"]
            if tuple(masks.shape) != (images.shape[0], 2, CANVAS, CANVAS):
                raise ValueError("Native SFR requires explicit two-class SAM logits")
            probability = self.unstitch(self.F.softmax(masks, dim=1), metadata)
        if not bool(self.torch.isfinite(probability).all()):
            raise ValueError("Nonfinite SAM pseudo labels")
        return probability, probability.argmax(1).long()

    def finetune_loss(self, masks, labels):
        if masks.shape[1] != 2 or masks.shape[0] != labels.shape[0] or masks.shape[2:] != labels.shape[1:]:
            raise ValueError("SFR prefix logits/target differ")
        ce = self.F.cross_entropy(masks, labels, ignore_index=255)
        dice = self.dice_loss(self.F.softmax(masks, dim=1), labels.unsqueeze(1))
        # Native Dice keeps ignored-board pixels in the prediction denominator.
        return .5 * (ce + dice), dict(ce=ce, dice=dice)

    def retrain_loss(self, logits, teacher_logits, labeled_targets, sam_targets, completed, target_iterations):
        if logits.ndim != 5 or logits.shape[0] != 4 or logits.shape[1] != 2:
            raise ValueError("SFR retraining requires native 2 labeled + 2 unlabeled")
        if teacher_logits.shape != logits.shape or teacher_logits.requires_grad:
            raise ValueError("Teacher must match and have no graph")
        if labeled_targets.shape != logits[:2, 0].shape or sam_targets.shape != logits[2:, 0].shape:
            raise ValueError("SFR target shape differs")
        if sam_targets.requires_grad or labeled_targets.requires_grad:
            raise ValueError("SFR targets must be detached")
        if not bool((((labeled_targets == 0) | (labeled_targets == 1)).all()
                     & ((sam_targets == 0) | (sam_targets == 1)).all())):
            raise ValueError("Nonbinary SFR target")
        probability = self.F.softmax(logits, dim=1)
        lab = .5 * (self.F.cross_entropy(logits[:2], labeled_targets)
                    + self.dice_loss(probability[:2], labeled_targets.unsqueeze(1)))
        unlab = .5 * (self.F.cross_entropy(logits[2:], sam_targets)
                      + self.dice_loss(probability[2:], sam_targets.unsqueeze(1)))
        distance = self.scope["softmax_mse_loss"](logits, teacher_logits).mean()
        clock = ssl_clock(completed, target_iterations)
        consistency = clock["consistency_weight"] * distance
        return lab + unlab + consistency, dict(supervised=lab, pseudo=unlab,
            consistency=consistency, consistency_distance=distance, **clock)


def freeze_prefix(sam):
    sam.eval()
    for parameter in sam.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


def ema_parameters_only(student, teacher, zero_based_step):
    """Preserve the official parameter-only update (including native BN behavior)."""
    if type(zero_based_step) is not int or zero_based_step < 0:
        raise ValueError("Invalid actual EMA clock")
    a = dict(student.named_parameters())
    b = dict(teacher.named_parameters())
    if list(a) != list(b) or any(a[k].shape != b[k].shape for k in a):
        raise ValueError("SFR student/teacher identity mismatch")
    alpha = min(1 - 1 / (zero_based_step + 1), .99)
    import torch
    with torch.no_grad():
        for key in a:
            b[key].mul_(alpha).add_(a[key], alpha=1 - alpha)


def lora_native_state(sam):
    """Official compact LoRA+prompt+decoder payload; no frozen encoder copy."""
    state = {"w_a_%03d" % i: p.weight.detach().clone() for i, p in enumerate(sam.w_As)}
    state.update({"w_b_%03d" % i: p.weight.detach().clone() for i, p in enumerate(sam.w_Bs)})
    state.update({k: v.detach().clone() for k, v in sam.sam.state_dict().items()
                  if "prompt_encoder" in k or "mask_decoder" in k})
    return state


def strict_lora_reload(sam, state):
    expected = lora_native_state(sam)
    if set(state) != set(expected) or any(state[k].shape != expected[k].shape
            or state[k].dtype != expected[k].dtype for k in expected):
        raise ValueError("SFR compact endpoint keys/shape/dtype differ")
    import torch
    if any(not bool(torch.isfinite(v).all()) for v in state.values()):
        raise ValueError("SFR compact endpoint nonfinite")
    with torch.no_grad():
        for i, module in enumerate(sam.w_As):
            module.weight.copy_(state["w_a_%03d" % i])
        for i, module in enumerate(sam.w_Bs):
            module.weight.copy_(state["w_b_%03d" % i])
    full = sam.sam.state_dict()
    full.update({k: v for k, v in state.items() if "prompt_encoder" in k or "mask_decoder" in k})
    sam.sam.load_state_dict(full, strict=True)
    reloaded = lora_native_state(sam)
    if any(not torch.equal(reloaded[k], state[k].to(reloaded[k].device)) for k in state):
        raise ValueError("SFR compact endpoint strict roundtrip differs")


def build_bound_sam(code_root, checkpoint, expected_sha256, device):
    """Build only when an externally verified original SAM ViT-B pin is supplied.

    Retain the official two-class decoder adaptation. The native broad fallback
    is replaced by exact permitted-key/shape checks, so missing encoder weights
    cannot become an unnoticed random foundation model. No file is downloaded.
    """
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_sha256)):
        raise ValueError("An independently bound original SAM ViT-B SHA256 is required")
    checkpoint = Path(checkpoint).resolve(strict=True)
    if not checkpoint.is_file() or _sha(checkpoint) != expected_sha256:
        raise ValueError("Original foundation checkpoint identity differs")
    verify_native_sources(code_root)
    from official_compat import load_sfr_lora
    lora, import_receipt = load_sfr_lora(code_root)
    native = importlib.import_module("segment_anything_lora.build_sam")
    import torch
    sam, embedding = native.sam_model_registry["vit_b"](image_size=CANVAS, num_classes=1,
                         checkpoint=None, pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1])
    source = torch.load(str(checkpoint), map_location="cpu")
    if not isinstance(source, dict) or not all(isinstance(v, torch.Tensor) for v in source.values()):
        raise ValueError("Original SAM ViT-B must be a plain tensor state dictionary")
    current = sam.state_dict()
    excluded = ("mask_tokens", "output_hypernetworks_mlps", "iou_prediction_head")
    transferable = {k for k in current if not any(name in k for name in excluded)}
    if not transferable.issubset(source) or any(source[k].shape != current[k].shape for k in transferable):
        raise ValueError("Original 1024 ViT-B transferable tensors are incomplete or wrong shape")
    unexpected = set(source) - set(current)
    if any(not any(name in k for name in excluded) for k in unexpected):
        raise ValueError("Unexpected non-decoder foundation tensors")
    if any(not bool(torch.isfinite(source[k]).all()) for k in transferable):
        raise ValueError("Nonfinite foundation tensor")
    current.update({k: source[k] for k in transferable})
    sam.load_state_dict(current, strict=True)
    model = lora.LoRA_Sam(sam, 4).to(device)
    receipt = dict(checkpoint=str(checkpoint), sha256=expected_sha256, size=checkpoint.stat().st_size,
                   family="original SAM 2D ViT-B", image_size=CANVAS, lora_rank=4,
                   transferred_keys=len(transferable),
                   native_reinitialized_decoder_keys=sorted(set(current) - transferable),
                   excluded_original_decoder_keys=sorted(set(source) - transferable),
                   embedding_size=embedding, import_receipt=import_receipt)
    return model, receipt


def sliding_predict(student, normalized_volume):
    """Original 18/18/4 softmax tiling on the entire shared ROI, no labels."""
    import torch
    import torch.nn.functional as F
    if student.training or normalized_volume.ndim != 3:
        raise ValueError("SFR inference requires eval model and one complete 3D ROI")
    plan = inference_windows(tuple(normalized_volume.shape))
    pads = plan["padding"]
    volume = F.pad(normalized_volume, tuple(v for pair in reversed(pads) for v in pair))
    score = torch.zeros((2,) + tuple(volume.shape), device=volume.device, dtype=torch.float32)
    count = torch.zeros_like(volume, dtype=torch.float32)
    with torch.no_grad():
        for x in plan["starts"][0]:
            for y in plan["starts"][1]:
                for z in plan["starts"][2]:
                    region = (slice(x, x + PATCH[0]), slice(y, y + PATCH[1]), slice(z, z + PATCH[2]))
                    output = student(volume[region][None, None])[0]
                    if tuple(output.shape) != (1, 2) + PATCH:
                        raise ValueError("Native VNet inference shape differs")
                    score[(slice(None),) + region] += F.softmax(output, dim=1)[0].float()
                    count[region] += 1
    if not bool((count > 0).all()) or not bool(torch.isfinite(score).all()):
        raise ValueError("Incomplete/nonfinite SFR inference coverage")
    score /= count.unsqueeze(0)
    original = tuple(slice(p[0], p[0] + n) for p, n in zip(pads, normalized_volume.shape))
    score = score[(slice(None),) + original]
    return score.argmax(0), score, dict(plan, all_original_voxels_covered=True,
                                      minimum_coverage=int(count.min().item()))
