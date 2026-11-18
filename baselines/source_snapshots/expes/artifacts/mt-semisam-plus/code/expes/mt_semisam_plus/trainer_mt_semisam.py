"""MT + SemiSAM+: 3D U-Net specialist + frozen SAM-Med3D generalist.

Training logic follows SemiSAM-main/code_semisam+/SemiSAM_train_MT_3D.py.
Eval protocol matches selftrain/uamt: every N epochs on fold test, best by Dice.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
from torch.nn.modules.loss import CrossEntropyLoss
from tqdm import tqdm

from expes.common.config import save_config
from expes.common.ramps import sigmoid_rampup
from expes.common.semi_data import (
    CaseListDataset,
    build_eval_transform,
    build_train_transform,
    read_case_list,
    split_list_path,
)
from expes.common.seg_metrics import aggregate_metrics, binary_case_metrics, logits_to_binary
from expes.mt_semisam_plus.semisam_generalist import SemiSAMGeneralist
from utils.cascade_train_utils import init_seeds, normalize_batch
from utils.data_loader import Union_Dataloader

ROOT = Path(__file__).resolve().parents[2]
SEMISAM_CODE = ROOT / "SemiSAM-main" / "code_semisam+"

# Cap host-thread RAM blow-up from BLAS / OpenMP
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
torch.set_num_threads(2)


class DiceLoss(nn.Module):
    """Same as SemiSAM+ utils.losses.DiceLoss."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes

    def _one_hot(self, target: torch.Tensor) -> torch.Tensor:
        oh = []
        for i in range(self.n_classes):
            oh.append(target == i)
        return torch.cat(oh, dim=1).float()

    def _dice(self, score: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        smooth = 1e-5
        inter = torch.sum(score * target)
        return 1.0 - (2 * inter + smooth) / (
            torch.sum(score * score) + torch.sum(target * target) + smooth
        )

    def forward(self, soft: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # soft: (B,C,...); target: (B,1,...) or (B,...)
        if target.dim() == soft.dim() - 1:
            target = target.unsqueeze(1)
        target = self._one_hot(target)
        loss = soft.new_tensor(0.0)
        for i in range(self.n_classes):
            loss = loss + self._dice(soft[:, i], target[:, i])
        return loss / self.n_classes


def _import_unet_3d():
    p = str(SEMISAM_CODE)
    if p not in sys.path:
        sys.path.insert(0, p)
    from networks.unet_3D import unet_3D  # type: ignore

    return unet_3D


class MTSemiSAMTrainer:
    method_tag = "MT-SemiSAM+"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        exp = cfg["experiment"]
        if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(exp.get("gpu", "0"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.output_dir = Path(exp["output_dir"])
        if not self.output_dir.is_absolute():
            self.output_dir = ROOT / self.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, self.output_dir / "config.yaml")

        init_seeds(int(exp.get("seed", 2026)))
        self._setup_logging()

        self.best_dice = 0.0
        self.best_epoch = -1
        self.global_step = 0
        self._steps_per_epoch = 32
        self.norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)

        self.student = self._build_unet().to(self.device)
        self.teacher = self._build_unet().to(self.device)
        for p in self.teacher.parameters():
            p.detach_()
        self._update_teacher(alpha=0.0, global_step=0)

        mcfg = cfg.get("model", {})
        scfg = cfg.get("semi", {})
        self.generalist = SemiSAMGeneralist(
            checkpoint=mcfg.get("sam_checkpoint", "ckpt/sam_med3d_turbo.pth"),
            device=self.device,
            prompt=str(scfg.get("prompt", "unc")),
            num_clicks=int(scfg.get("sam_num_clicks", 10)),
        )
        logging.info(
            "Generalist SAM-Med3D loaded (frozen), prompt=%s clicks=%d",
            scfg.get("prompt", "unc"),
            int(scfg.get("sam_num_clicks", 10)),
        )

        self.ce_loss = CrossEntropyLoss()
        self.dice_loss = DiceLoss(int(mcfg.get("num_classes", 2)))
        self._build_optimizer()
        self._build_dataloaders()

    def _setup_logging(self) -> None:
        log_path = self.output_dir / "log.txt"
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(message)s",
            datefmt="%H:%M:%S",
            force=True,
            handlers=[
                logging.FileHandler(str(log_path), mode="a", encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )

    def _build_unet(self) -> nn.Module:
        unet_3D = _import_unet_3d()
        n_cls = int(self.cfg.get("model", {}).get("num_classes", 2))
        return unet_3D(n_classes=n_cls, in_channels=1)

    def _build_optimizer(self) -> None:
        tcfg = self.cfg["train"]
        self.base_lr = float(tcfg.get("lr", 0.01))
        self.optimizer = torch.optim.SGD(
            self.student.parameters(),
            lr=self.base_lr,
            momentum=float(tcfg.get("momentum", 0.9)),
            weight_decay=float(tcfg.get("weight_decay", 1e-4)),
        )

    def _split_paths(self, fold_idx: int) -> tuple[Path, Path]:
        dcfg = self.cfg["dataset"]
        split_root = Path(dcfg["split_root"])
        if not split_root.is_absolute():
            split_root = ROOT / split_root
        fold_cv_dir = Path(dcfg["fold_cv_dir_template"].format(fold=fold_idx))
        if not fold_cv_dir.is_absolute():
            fold_cv_dir = ROOT / fold_cv_dir
        return split_root, fold_cv_dir

    def _build_dataloaders(self) -> None:
        dcfg = self.cfg["dataset"]
        tcfg = self.cfg["train"]
        fold_idx = int(dcfg["fold"])
        split_root, fold_cv_dir = self._split_paths(fold_idx)
        img_size = int(dcfg.get("img_size", 128))
        threshold = int(dcfg.get("foreground_threshold", 1000))

        lab_names = read_case_list(split_list_path(split_root, fold_idx, "train_lab"))
        unlab_names = read_case_list(split_list_path(split_root, fold_idx, "train_unlab"))
        test_names = read_case_list(split_list_path(split_root, fold_idx, "test"))

        train_tf = build_train_transform(img_size, random_flip=True)
        eval_tf = build_eval_transform(img_size)
        self.lab_set = CaseListDataset(fold_cv_dir, lab_names, train_tf, threshold, "train")
        self.unlab_set = CaseListDataset(fold_cv_dir, unlab_names, train_tf, threshold, "train")
        self.test_set = CaseListDataset(fold_cv_dir, test_names, eval_tf, threshold, "test")

        nw = int(tcfg.get("num_workers", 0))
        lbs = int(tcfg.get("labeled_batch_size", 1))
        bs = int(tcfg.get("batch_size", 1))
        # pin_memory=False: dual-job host RAM is the OOM bottleneck
        self.lab_loader = Union_Dataloader(
            dataset=self.lab_set,
            batch_size=lbs,
            shuffle=True,
            num_workers=nw,
            pin_memory=False,
            drop_last=True,
        )
        self.unlab_loader = Union_Dataloader(
            dataset=self.unlab_set,
            batch_size=bs,
            shuffle=True,
            num_workers=nw,
            pin_memory=False,
            drop_last=True,
        )
        self.test_loader = Union_Dataloader(
            dataset=self.test_set,
            batch_size=1,
            shuffle=False,
            num_workers=nw,
            pin_memory=False,
            drop_last=False,
        )
        logging.info(
            "Data fold=%d lab=%d unlab=%d test=%d",
            fold_idx,
            len(self.lab_set),
            len(self.unlab_set),
            len(self.test_set),
        )

    def _update_teacher(self, alpha: float, global_step: int) -> None:
        a = min(1.0 - 1.0 / (global_step + 1), float(alpha))
        for t_p, s_p in zip(self.teacher.parameters(), self.student.parameters()):
            t_p.data.mul_(a).add_((1.0 - a) * s_p.data)

    def _add_noise(self, x: torch.Tensor) -> torch.Tensor:
        scfg = self.cfg.get("semi", {})
        std = float(scfg.get("noise_std", 0.1))
        clamp_v = float(scfg.get("noise_clamp", 0.2))
        if std <= 0:
            return x
        return x + torch.clamp(torch.randn_like(x) * std, -clamp_v, clamp_v)

    def _prepare_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        image = normalize_batch(batch["image"], self.norm_transform).to(self.device)
        label = batch["label"].to(self.device)
        if label.dim() == 4 and label.shape[0] != image.shape[0]:
            # (C,D,H,W) single sample already batched by loader differently
            pass
        if label.dim() == 3:
            label = label.unsqueeze(0)
        return image, label

    def _consistency_weight(self, iter_num: int) -> float:
        scfg = self.cfg.get("semi", {})
        base = float(scfg.get("consistency", 0.1))
        ramp = float(scfg.get("consistency_rampup", 200.0))
        # SemiSAM+ uses iter//150 as epoch proxy; configurable for short runs.
        div = max(1, int(scfg.get("ramp_iter_div", 150)))
        return base * sigmoid_rampup(iter_num // div, ramp)

    def _sam_consistency_weight(self, iter_num: int, max_iterations: int) -> float:
        scfg = self.cfg.get("semi", {})
        base = float(scfg.get("consistency", 0.1))
        ramp = float(scfg.get("consistency_rampup", 200.0))
        div = max(1, int(scfg.get("ramp_iter_div", 150)))
        # Inverted ramp: stronger SAM early (SemiSAM+)
        return base * sigmoid_rampup((max_iterations - iter_num) // div, ramp)

    def _poly_lr(self, iter_num: int, max_iterations: int) -> float:
        return self.base_lr * (1.0 - iter_num / max(1, max_iterations)) ** 0.9

    def _specialist_consistency_loss(
        self,
        outputs: torch.Tensor,
        outputs_soft: torch.Tensor,
        ema_output: torch.Tensor,
        ema_output_soft: torch.Tensor,
        unlabeled: torch.Tensor,
        labeled_bs: int,
        max_iterations: int,
    ) -> torch.Tensor:
        """MT: mean MSE between student and noisy-teacher soft preds."""
        del outputs, ema_output, unlabeled, max_iterations  # unused in plain MT
        return torch.mean((outputs_soft[labeled_bs:] - ema_output_soft) ** 2)

    def _sam_consistency_loss(
        self,
        outputs_soft: torch.Tensor,
        volume_batch: torch.Tensor,
        labeled_bs: int,
    ) -> torch.Tensor:
        """SemiSAM+ generalist consistency (confidence-aware when prompt=unc).

        Only unlabeled volumes enter the generalist (loss already ignores labeled).
        """
        scfg = self.cfg.get("semi", {})
        prompt = str(scfg.get("prompt", "unc"))
        vol_u = volume_batch[labeled_bs:]
        soft_u = outputs_soft[labeled_bs:, 1:2].detach()
        sam_mask, uncsam = self.generalist(vol_u, soft_u)
        # inference_mode tensors are not autograd-safe; detach+clone for loss math
        sam_mask = sam_mask.detach().float().clone()
        if uncsam is not None:
            uncsam = uncsam.detach().float().clone()
        samseg_soft = torch.cat((1.0 - sam_mask, sam_mask), dim=1)

        if prompt == "unc" and uncsam is not None:
            dist = (outputs_soft[labeled_bs:] - samseg_soft) ** 2
            sam_consistency = torch.mean(dist * uncsam) / (torch.mean(uncsam) + 1e-8) + torch.mean(
                uncsam
            )
        else:
            sam_consistency = torch.mean((outputs_soft[labeled_bs:] - samseg_soft) ** 2)
        # NOTE: original had sum(x)/(sum(x)+eps) on a scalar (~1, kills grads); omit.
        return sam_consistency

    @torch.no_grad()
    def _evaluate(self) -> Dict[str, float]:
        self.student.eval()
        case_metrics = []
        for batch in self.test_loader:
            image = normalize_batch(batch["image"], self.norm_transform).to(self.device)
            label = batch["label"]
            logits = self.student(image)
            pred = logits_to_binary(logits)[0]
            gt = label.squeeze().cpu().numpy()
            case_metrics.append(binary_case_metrics(pred, gt))
        self.student.train()
        return aggregate_metrics(case_metrics)

    def _log_metrics(self, tag: str, metrics: Dict[str, float]) -> None:
        logging.info(
            "%s dice=%.4f jaccard=%.4f hd95=%.4f asd=%.4f",
            tag,
            metrics["dice"],
            metrics["jaccard"],
            metrics["hd95"],
            metrics["asd"],
        )

    def _save(self, name: str, epoch: int, extra: Optional[Dict[str, Any]] = None) -> None:
        path = self.output_dir / name
        payload: Dict[str, Any] = {
            "student": self.student.state_dict(),
            "teacher": self.teacher.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "best_dice": self.best_dice,
            "best_epoch": self.best_epoch,
            "global_step": self.global_step,
            "cfg": self.cfg,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def _try_load_resume(self) -> Optional[Dict[str, Any]]:
        """Load last.pth (preferred) or best.pth for crash resume."""
        if not bool(self.cfg.get("train", {}).get("resume", True)):
            return None
        for name in ("last.pth", "best.pth"):
            path = self.output_dir / name
            if not path.is_file():
                continue
            try:
                state = torch.load(str(path), map_location=self.device)
            except Exception as e:
                logging.warning("Failed to load %s: %s", path, e)
                continue
            if not isinstance(state, dict) or "student" not in state:
                continue
            # Skip fresh init markers (epoch < 0) for last; best may be real
            if name == "last.pth" and int(state.get("epoch", -1)) < 0:
                continue
            logging.info("Resume candidate: %s epoch=%s step=%s best=%.4f",
                         name, state.get("epoch"), state.get("global_step"),
                         float(state.get("best_dice", 0.0)))
            return state
        return None

    def _apply_resume(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.student.load_state_dict(state["student"])
        if "teacher" in state:
            self.teacher.load_state_dict(state["teacher"])
        else:
            self._update_teacher(alpha=0.0, global_step=0)
        if "optimizer" in state:
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except Exception as e:
                logging.warning("Optimizer state not restored: %s", e)
        self.best_dice = float(state.get("best_dice", 0.0))
        self.best_epoch = int(state.get("best_epoch", state.get("epoch", -1)))
        self.global_step = int(state.get("global_step", 0))
        start_epoch = int(state.get("epoch", -1)) + 1
        no_improve = int(state.get("no_improve", 0))
        logging.info(
            "Resumed weights: start_epoch=%d step=%d best=%.4f @%d no_improve=%d",
            start_epoch,
            self.global_step,
            self.best_dice,
            self.best_epoch,
            no_improve,
        )
        return {"start_epoch": max(0, start_epoch), "no_improve": no_improve}

    def semi_train(self) -> None:
        tcfg = self.cfg["train"]
        scfg = self.cfg["semi"]
        semi_epochs = int(scfg.get("semi_epochs", 100))
        eval_interval = int(tcfg.get("eval_interval", 5))
        ema_decay = float(self.cfg.get("ema", {}).get("alpha", 0.99))
        labeled_bs = int(tcfg.get("labeled_batch_size", 1))

        lab_cycles = int(scfg.get("lab_cycles_per_epoch", 2))
        unlab_cap = int(scfg.get("unlab_cap_per_epoch", 40))
        steps = max(
            max(1, len(self.lab_loader)) * max(1, lab_cycles),
            min(max(1, len(self.unlab_loader)), max(1, unlab_cap)),
        )
        self._steps_per_epoch = steps
        max_iterations = int(scfg.get("max_iterations", semi_epochs * steps))

        resume_state = self._try_load_resume()
        start_epoch = 0
        no_improve = 0
        if resume_state is not None and int(resume_state.get("epoch", -1)) >= 0:
            info = self._apply_resume(resume_state)
            start_epoch = int(info["start_epoch"])
            no_improve = int(info["no_improve"])
            logging.info("Skip init_eval (resuming)")
        else:
            metrics0 = self._evaluate()
            self.best_dice = float(metrics0["dice"])
            self.best_epoch = -1
            self._save("init_eval.pth", epoch=-1)
            self._save("best.pth", epoch=-1)
            self._log_metrics("init_eval", metrics0)

        logging.info(
            "%s start dice=%.4f epochs=%d start_epoch=%d steps/ep=%d "
            "max_iter=%d prompt=%s early_stop_patience=%d min_epoch=%d",
            self.method_tag,
            self.best_dice,
            semi_epochs,
            start_epoch,
            steps,
            max_iterations,
            scfg.get("prompt", "unc"),
            int(scfg.get("early_stop_patience", 0)),
            int(scfg.get("early_stop_min_epoch", 0)),
        )

        early_pat = int(scfg.get("early_stop_patience", 0))
        early_min_ep = int(scfg.get("early_stop_min_epoch", 0))

        for epoch in tqdm(range(start_epoch, semi_epochs), desc=self.method_tag, ncols=70):
            lab_iter = cycle(self.lab_loader)
            unlab_iter = cycle(self.unlab_loader)
            losses, sups, cons, sams = [], [], [], []

            self.student.train()
            self.teacher.train()
            for _ in range(steps):
                img_l, lab_l = self._prepare_batch(next(lab_iter))
                img_u, _ = self._prepare_batch(next(unlab_iter))
                # Concat labeled + unlabeled (SemiSAM TwoStream layout)
                volume_batch = torch.cat([img_l, img_u], dim=0)
                label_batch = lab_l  # only first labeled_bs used
                if label_batch.dim() == 4 and label_batch.shape[1] == 1:
                    label_batch = label_batch.squeeze(1)

                unlabeled = volume_batch[labeled_bs:]
                ema_inputs = self._add_noise(unlabeled)

                outputs = self.student(volume_batch)
                outputs_soft = torch.softmax(outputs, dim=1)
                with torch.no_grad():
                    ema_output = self.teacher(ema_inputs)
                    ema_output_soft = torch.softmax(ema_output, dim=1)

                loss_ce = self.ce_loss(outputs[:labeled_bs], label_batch[:labeled_bs].long())
                loss_dice = self.dice_loss(
                    outputs_soft[:labeled_bs], label_batch[:labeled_bs].unsqueeze(1)
                )
                supervised_loss = 0.5 * (loss_dice + loss_ce)

                consistency_weight = self._consistency_weight(self.global_step)
                consistency_loss = self._specialist_consistency_loss(
                    outputs,
                    outputs_soft,
                    ema_output,
                    ema_output_soft,
                    unlabeled,
                    labeled_bs,
                    max_iterations,
                )

                sam_consistency = self._sam_consistency_loss(
                    outputs_soft, volume_batch, labeled_bs
                )
                consistency_weight_sam = self._sam_consistency_weight(
                    self.global_step, max_iterations
                )
                sam_scale = float(scfg.get("sam_consistency_scale", 0.1))
                sam_con_loss = sam_scale * consistency_weight_sam * sam_consistency

                loss = supervised_loss + consistency_weight * consistency_loss + sam_con_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                self._update_teacher(ema_decay, self.global_step)

                lr_ = self._poly_lr(self.global_step, max_iterations)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr_

                self.global_step += 1
                losses.append(float(loss.item()))
                sups.append(float(supervised_loss.item()))
                cons.append(float((consistency_weight * consistency_loss).item()))
                sams.append(float(sam_con_loss.item()))

                # Drop graph refs early; SAM branch is RAM-heavy over long runs
                del outputs, outputs_soft, ema_output, ema_output_soft, loss
                del volume_batch, label_batch, unlabeled, ema_inputs
                if self.global_step % 10 == 0:
                    gc.collect()
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()

                if self.global_step >= max_iterations:
                    break

            logging.info(
                "epoch=%d step=%d loss=%.4f sup=%.4f mt=%.4f sam=%.4f lr=%.6f",
                epoch,
                self.global_step,
                float(np.mean(losses)),
                float(np.mean(sups)),
                float(np.mean(cons)),
                float(np.mean(sams)),
                self.optimizer.param_groups[0]["lr"],
            )
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            # Soft host-RAM brake: if process RSS is huge, hint OS to release
            try:
                import resource

                rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                # Linux ru_maxrss is KB
                if rss_kb > 20 * 1024 * 1024:  # >20GB peak
                    gc.collect()
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
            except Exception:
                pass

            if epoch % eval_interval == 0 or epoch == semi_epochs - 1:
                metrics = self._evaluate()
                self._log_metrics("semi_eval", metrics)
                if metrics["dice"] > self.best_dice:
                    self.best_dice = metrics["dice"]
                    self.best_epoch = epoch
                    self._save("best.pth", epoch=epoch, extra={"no_improve": 0})
                    no_improve = 0
                    logging.info(
                        "saved best.pth epoch=%d dice=%.4f", epoch, self.best_dice
                    )
                else:
                    no_improve += 1
                    logging.info(
                        "no improve (cur=%.4f best=%.4f streak=%d/%s min_ep=%d)",
                        metrics["dice"],
                        self.best_dice,
                        no_improve,
                        early_pat if early_pat > 0 else "-",
                        early_min_ep,
                    )
                    if (
                        early_pat > 0
                        and no_improve >= early_pat
                        and epoch >= early_min_ep
                    ):
                        logging.info(
                            "early stop: %d evals without dice improve (patience=%d) "
                            "best=%.4f @ epoch=%d step=%d",
                            no_improve,
                            early_pat,
                            self.best_dice,
                            self.best_epoch,
                            self.global_step,
                        )
                        self._save(
                            "last.pth",
                            epoch=epoch,
                            extra={"no_improve": no_improve},
                        )
                        break
                    if (
                        early_pat > 0
                        and no_improve >= early_pat
                        and epoch < early_min_ep
                    ):
                        logging.info(
                            "early-stop patience met but epoch=%d < min_epoch=%d; continue",
                            epoch,
                            early_min_ep,
                        )
                # Crash-resume checkpoint every eval
                self._save("last.pth", epoch=epoch, extra={"no_improve": no_improve})
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

            if self.global_step >= max_iterations:
                logging.info("reached max_iterations=%d", max_iterations)
                metrics = self._evaluate()
                self._log_metrics("end_eval", metrics)
                if metrics["dice"] > self.best_dice:
                    self.best_dice = metrics["dice"]
                    self.best_epoch = epoch
                    self._save("best.pth", epoch=epoch, extra={"no_improve": 0})
                    logging.info(
                        "saved best.pth at end epoch=%d dice=%.4f", epoch, self.best_dice
                    )
                self._save("last.pth", epoch=epoch, extra={"no_improve": no_improve})
                break
            # Persist last even on non-eval epochs (lightweight interval)
            if epoch % max(1, eval_interval) == (eval_interval - 1):
                pass  # already saved on eval
            elif (epoch + 1) % 10 == 0:
                self._save("last.pth", epoch=epoch, extra={"no_improve": no_improve})

    def run(self) -> Dict[str, float]:
        logging.info("%s run output=%s", self.method_tag, self.output_dir)
        mode = str(self.cfg.get("train", {}).get("mode", "semi")).lower()
        if mode == "sup":
            self.sup_train()
        else:
            self.semi_train()

        # Report metrics from best checkpoint (not last weights)
        best_path = self.output_dir / "best.pth"
        if best_path.is_file():
            state = torch.load(str(best_path), map_location=self.device)
            if isinstance(state, dict) and "student" in state:
                self.student.load_state_dict(state["student"])
                logging.info(
                    "Loaded best.pth for final metrics (epoch=%s dice=%.4f)",
                    state.get("epoch"),
                    float(state.get("best_dice", self.best_dice)),
                )

        metrics = self._evaluate()
        self._log_metrics("final", metrics)
        out = self.output_dir / "final_metrics.txt"
        with open(out, "w", encoding="utf-8") as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v:.6f}\n")
            f.write(f"method: {self.method_tag}\n")
            f.write(f"prompt: {self.cfg.get('semi', {}).get('prompt', 'unc')}\n")
            f.write(f"best_dice: {self.best_dice:.6f}\n")
            f.write(f"best_epoch: {self.best_epoch}\n")
            f.write(f"global_step: {self.global_step}\n")
            f.write(
                f"early_stop_patience: {int(self.cfg.get('semi', {}).get('early_stop_patience', 0))}\n"
            )
            extra = self._final_metrics_extra()
            for k, v in extra.items():
                f.write(f"{k}: {v}\n")
        logging.info("Saved %s", out)
        return metrics

    def _final_metrics_extra(self) -> Dict[str, Any]:
        return {}

    def sup_train(self) -> None:
        """Fully supervised 3D U-Net on labeled set only (Sup-* arms)."""
        tcfg = self.cfg["train"]
        scfg = self.cfg.get("semi", {})
        epochs = int(scfg.get("semi_epochs", tcfg.get("pretrain_epochs", 100)))
        eval_interval = int(tcfg.get("eval_interval", 5))
        steps = max(1, len(self.lab_loader)) * int(scfg.get("lab_cycles_per_epoch", 4))
        max_iterations = int(scfg.get("max_iterations", epochs * steps))
        self._steps_per_epoch = steps

        metrics0 = self._evaluate()
        self.best_dice = float(metrics0["dice"])
        self._save("best.pth", epoch=-1)
        self._log_metrics("init_eval", metrics0)

        for epoch in tqdm(range(epochs), desc="mt_semisam+_sup", ncols=70):
            lab_iter = cycle(self.lab_loader)
            self.student.train()
            losses = []
            for _ in range(steps):
                img_l, lab_l = self._prepare_batch(next(lab_iter))
                if lab_l.dim() == 4 and lab_l.shape[1] == 1:
                    lab_l = lab_l.squeeze(1)
                outputs = self.student(img_l)
                outputs_soft = torch.softmax(outputs, dim=1)
                loss_ce = self.ce_loss(outputs, lab_l.long())
                loss_dice = self.dice_loss(outputs_soft, lab_l.unsqueeze(1))
                loss = 0.5 * (loss_ce + loss_dice)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                lr_ = self._poly_lr(self.global_step, max_iterations)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr_
                self.global_step += 1
                losses.append(float(loss.item()))
                if self.global_step >= max_iterations:
                    break
            logging.info(
                "sup epoch=%d step=%d loss=%.4f lr=%.6f",
                epoch,
                self.global_step,
                float(np.mean(losses)),
                self.optimizer.param_groups[0]["lr"],
            )
            if epoch % eval_interval == 0 or epoch == epochs - 1:
                metrics = self._evaluate()
                self._log_metrics("sup_eval", metrics)
                if metrics["dice"] > self.best_dice:
                    self.best_dice = metrics["dice"]
                    self.best_epoch = epoch
                    self._save("best.pth", epoch=epoch)
                    logging.info("saved best.pth epoch=%d dice=%.4f", epoch, self.best_dice)
            if self.global_step >= max_iterations:
                break
