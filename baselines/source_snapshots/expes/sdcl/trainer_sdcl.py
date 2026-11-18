"""SDCL trainer on BY-125 (strict: VNet + ResVNet, pre-train + self-train)."""

from __future__ import annotations

import gc
import logging
import os
import sys
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
from torch.utils.data import DataLoader
from tqdm import tqdm

from expes.common.config import save_config
from expes.common.semi_data import (
    CaseListDataset,
    build_eval_transform,
    build_train_transform,
    read_case_list,
    split_list_path,
)
from expes.common.seg_metrics import aggregate_metrics, binary_case_metrics, logits_to_binary
from expes.sdcl.sdcl_networks import build_ema_vnet, build_resvnet, build_vnet
from expes.sdcl.sdcl_ops import (
    context_mask,
    forward_logits,
    get_cut_mask,
    get_xor_region,
    mix_loss,
    mix_max_kl_loss,
    mix_mse_loss,
    update_ema,
)
from utils.cascade_train_utils import init_seeds, normalize_batch

ROOT = Path(__file__).resolve().parents[2]
SDCL_CODE = ROOT / "SDCL-master" / "code"
if str(SDCL_CODE) not in sys.path:
    sys.path.insert(0, str(SDCL_CODE))
from utils.losses import mask_DiceLoss  # noqa: E402

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
torch.set_num_threads(2)


class ReversedCaseListDataset(CaseListDataset):
    """Same cases as parent list but reversed order (SDCL lab/unlab pair B)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_paths = list(reversed(self.image_paths))
        self.label_paths = list(reversed(self.label_paths))


class SDCLTrainer:
    method_tag = "SDCL"

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

        self.student_v = build_vnet().to(self.device)
        self.student_r = build_resvnet().to(self.device)
        self.ema_v = build_ema_vnet().to(self.device)

        scfg = cfg.get("semi", {})
        self.mask_ratio = float(scfg.get("mask_ratio", 2 / 3))
        self.u_weight = float(scfg.get("u_weight", 0.5))
        self.mse_w = float(scfg.get("mse_weight", 0.5))
        self.kl_w = float(scfg.get("kl_weight", 0.05))
        self.ema_alpha = float(cfg.get("ema", {}).get("alpha", 0.99))

        self.dice_loss = mask_DiceLoss(nclass=2)
        self.ce_none = nn.CrossEntropyLoss(reduction="none")

        self._build_optimizers()
        self._build_dataloaders()

        self.best_dice = 0.0
        self.best_epoch = -1
        self.global_step = 0
        self.norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)

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

    def _build_optimizers(self) -> None:
        tcfg = cfg_train = self.cfg["train"]
        lr = float(tcfg.get("lr", 1e-3))
        self.opt_v = torch.optim.Adam(self.student_v.parameters(), lr=lr)
        self.opt_r = torch.optim.Adam(self.student_r.parameters(), lr=lr)

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
        nw = int(tcfg.get("num_workers", 0))
        bs = int(tcfg.get("batch_size", 2))

        lab_names = read_case_list(split_list_path(split_root, fold_idx, "train_lab"))
        unlab_names = read_case_list(split_list_path(split_root, fold_idx, "train_unlab"))
        test_names = read_case_list(split_list_path(split_root, fold_idx, "test"))

        train_tf = build_train_transform(img_size, random_flip=True)
        eval_tf = build_eval_transform(img_size)

        def _dl(dataset, shuffle=False):
            return DataLoader(
                dataset,
                batch_size=bs,
                shuffle=shuffle,
                num_workers=nw,
                pin_memory=False,
                drop_last=True,
            )

        self.lab_a = _dl(CaseListDataset(fold_cv_dir, lab_names, train_tf, threshold, "train"))
        self.lab_b = _dl(
            ReversedCaseListDataset(fold_cv_dir, lab_names, train_tf, threshold, "train")
        )
        self.unlab_a = _dl(CaseListDataset(fold_cv_dir, unlab_names, train_tf, threshold, "train"))
        self.unlab_b = _dl(
            ReversedCaseListDataset(fold_cv_dir, unlab_names, train_tf, threshold, "train")
        )
        self.test_loader = DataLoader(
            CaseListDataset(fold_cv_dir, test_names, eval_tf, threshold, "test"),
            batch_size=1,
            shuffle=False,
            num_workers=nw,
            pin_memory=False,
            drop_last=False,
        )
        logging.info(
            "Data fold=%d lab=%d unlab=%d test=%d bs=%d",
            fold_idx,
            len(lab_names),
            len(unlab_names),
            len(test_names),
            bs,
        )

    def _prepare(self, batch: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        img = normalize_batch(batch["image"], self.norm_transform).to(self.device)
        lab = batch["label"].to(self.device)
        if lab.dim() == 3:
            lab = lab.unsqueeze(0)
        if lab.dim() == 4 and lab.shape[1] == 1:
            lab = lab.squeeze(1)
        return img, lab.long()

    @torch.no_grad()
    def _ensemble_logits(self, image: torch.Tensor) -> torch.Tensor:
        self.student_v.eval()
        self.student_r.eval()
        lv = forward_logits(self.student_v, image)
        lr = forward_logits(self.student_r, image)
        return 0.5 * (F.softmax(lv, dim=1) + F.softmax(lr, dim=1))

    @torch.no_grad()
    def _evaluate(self) -> Dict[str, float]:
        case_metrics = []
        for batch in self.test_loader:
            image = normalize_batch(batch["image"], self.norm_transform).to(self.device)
            label = batch["label"]
            prob = self._ensemble_logits(image)
            pred = (prob[:, 1:2] > 0.5).squeeze(1).cpu().numpy().astype(bool)
            gt = label.squeeze().cpu().numpy()
            case_metrics.append(binary_case_metrics(pred, gt))
        self.student_v.train()
        self.student_r.train()
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
        payload: Dict[str, Any] = {
            "student_v": self.student_v.state_dict(),
            "student_r": self.student_r.state_dict(),
            "ema_v": self.ema_v.state_dict(),
            "opt_v": self.opt_v.state_dict(),
            "opt_r": self.opt_r.state_dict(),
            "epoch": epoch,
            "best_dice": self.best_dice,
            "best_epoch": self.best_epoch,
            "global_step": self.global_step,
            "phase": extra.get("phase", "self") if extra else "self",
            "cfg": self.cfg,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, self.output_dir / name)

    def _try_load_resume(self) -> Optional[Dict[str, Any]]:
        if not bool(self.cfg.get("train", {}).get("resume", True)):
            return None
        for name in ("last.pth", "best.pth", "pretrain_best.pth"):
            path = self.output_dir / name
            if not path.is_file():
                continue
            try:
                state = torch.load(str(path), map_location=self.device)
            except Exception as e:
                logging.warning("load %s failed: %s", name, e)
                continue
            if not isinstance(state, dict) or "student_v" not in state:
                continue
            logging.info("Resume from %s epoch=%s phase=%s", name, state.get("epoch"), state.get("phase"))
            return state
        return None

    def _apply_resume(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.student_v.load_state_dict(state["student_v"])
        self.student_r.load_state_dict(state["student_r"])
        if "ema_v" in state:
            self.ema_v.load_state_dict(state["ema_v"])
        if "opt_v" in state:
            self.opt_v.load_state_dict(state["opt_v"])
        if "opt_r" in state:
            self.opt_r.load_state_dict(state["opt_r"])
        self.best_dice = float(state.get("best_dice", 0.0))
        self.best_epoch = int(state.get("best_epoch", -1))
        self.global_step = int(state.get("global_step", 0))
        return {
            "start_epoch": int(state.get("epoch", -1)) + 1,
            "phase": str(state.get("phase", "pre")),
            "no_improve": int(state.get("no_improve", 0)),
        }

    def _pretrain_step(self, img_a, lab_a, img_b, lab_b) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            img_mask, _ = context_mask(img_a, self.mask_ratio)
        volume = img_a * img_mask + img_b * (1 - img_mask)
        label = lab_a * img_mask + lab_b * (1 - img_mask)

        out_v = forward_logits(self.student_v, volume)
        loss_ce = F.cross_entropy(out_v, label)
        loss_dice = self.dice_loss(out_v, label)
        loss_v = 0.5 * (loss_ce + loss_dice)

        out_r = forward_logits(self.student_r, volume)
        loss_ce2 = F.cross_entropy(out_r, label)
        loss_dice2 = self.dice_loss(out_r, label)
        loss_r = 0.5 * (loss_ce2 + loss_dice2)
        return loss_v, loss_r

    def pretrain(self, start_epoch: int = 0) -> None:
        scfg = self.cfg["semi"]
        tcfg = self.cfg["train"]
        epochs = int(scfg.get("pre_epochs", 81))
        eval_interval = int(tcfg.get("eval_interval", 5))
        cycles = max(1, int(scfg.get("lab_cycles_per_epoch", 4)))
        # Prefers train.* for pretrain-only baselines; falls back to semi.*
        early_pat = int(
            tcfg.get(
                "early_stop_patience",
                scfg.get("early_stop_patience", 0),
            )
        )
        early_min = int(
            tcfg.get(
                "early_stop_min_epoch",
                scfg.get("early_stop_min_epoch", 0),
            )
        )
        no_improve = 0

        if start_epoch == 0:
            metrics0 = self._evaluate()
            self.best_dice = float(metrics0["dice"])
            self._save("init_eval.pth", epoch=-1, extra={"phase": "pre"})
            self._log_metrics("init_eval", metrics0)

        logging.info(
            "sdcl_pre epochs=%d eval_interval=%d early_stop=patience%d/min%d",
            epochs,
            eval_interval,
            early_pat,
            early_min,
        )

        for epoch in tqdm(range(start_epoch, epochs), desc="sdcl_pre", ncols=70):
            self.student_v.train()
            self.student_r.train()
            it_a, it_b = cycle(self.lab_a), cycle(self.lab_b)
            steps = max(len(self.lab_a), len(self.lab_b)) * cycles
            losses_v, losses_r = [], []
            for _ in range(steps):
                img_a, lab_a = self._prepare(next(it_a))
                img_b, lab_b = self._prepare(next(it_b))
                loss_v, loss_r = self._pretrain_step(img_a, lab_a, img_b, lab_b)

                self.opt_v.zero_grad(set_to_none=True)
                # SemiSAM+ shares SAM term across dual students → need retain_graph
                loss_v.backward(retain_graph=bool(getattr(self, "retain_graph_first_backward", False)))
                self.opt_v.step()

                self.opt_r.zero_grad(set_to_none=True)
                loss_r.backward()
                self.opt_r.step()

                self.global_step += 1
                losses_v.append(float(loss_v.item()))
                losses_r.append(float(loss_r.item()))
                del loss_v, loss_r
                if self.global_step % 5 == 0 and self.device.type == "cuda":
                    torch.cuda.empty_cache()

            logging.info(
                "pre epoch=%d step=%d loss_v=%.4f loss_r=%.4f",
                epoch,
                self.global_step,
                float(np.mean(losses_v)),
                float(np.mean(losses_r)),
            )

            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            if epoch % eval_interval == 0 or epoch == epochs - 1:
                metrics = self._evaluate()
                self._log_metrics("pre_eval", metrics)
                if metrics["dice"] > self.best_dice:
                    self.best_dice = metrics["dice"]
                    self.best_epoch = epoch
                    no_improve = 0
                    self._save("pretrain_best.pth", epoch=epoch, extra={"phase": "pre", "no_improve": 0})
                    logging.info("saved pretrain_best.pth dice=%.4f", self.best_dice)
                else:
                    no_improve += 1
                    logging.info(
                        "pre no improve %d/%d (cur=%.4f best=%.4f@ep%d)",
                        no_improve,
                        early_pat,
                        metrics["dice"],
                        self.best_dice,
                        self.best_epoch,
                    )
                    if (
                        early_pat > 0
                        and no_improve >= early_pat
                        and epoch >= early_min
                    ):
                        logging.info(
                            "pretrain early stop at epoch=%d best=%.4f@ep%d",
                            epoch,
                            self.best_dice,
                            self.best_epoch,
                        )
                        self._save(
                            "last.pth",
                            epoch=epoch,
                            extra={"phase": "pre", "no_improve": no_improve},
                        )
                        break
                self._save("last.pth", epoch=epoch, extra={"phase": "pre"})

        logging.info("Pretrain done best_dice=%.4f", self.best_dice)
        # Seed best.pth so self-train resume / queue completion always have it
        pb = self.output_dir / "pretrain_best.pth"
        if pb.is_file():
            state = torch.load(str(pb), map_location=self.device)
            self.student_v.load_state_dict(state["student_v"])
            self.student_r.load_state_dict(state["student_r"])
            self.ema_v.load_state_dict(state["student_v"])
            self._save("best.pth", epoch=int(state.get("epoch", self.best_epoch)), extra={"phase": "pre"})
            logging.info("Seeded best.pth from pretrain_best.pth dice=%.4f", self.best_dice)
        # Persist phase=self immediately so a kill during early self-train resumes
        # into self (not pre). epoch=-1 → start_epoch=0 on next resume.
        self._save("last.pth", epoch=-1, extra={"phase": "self", "no_improve": 0})
        logging.info("Marked last.pth phase=self for self-train resume")

    def _load_pretrain_for_self(self) -> None:
        path = self.output_dir / "pretrain_best.pth"
        if not path.is_file():
            logging.warning("No pretrain_best.pth; continuing from current weights")
            return
        state = torch.load(str(path), map_location=self.device)
        self.student_v.load_state_dict(state["student_v"])
        self.student_r.load_state_dict(state["student_r"])
        self.ema_v.load_state_dict(state["student_v"])
        logging.info("Loaded pretrain_best.pth for self-train")

    def _self_train_step(
        self,
        img_a,
        lab_a,
        img_b,
        lab_b,
        unimg_a,
        _unlab_a,
        unimg_b,
        _unlab_b,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            unout_a = forward_logits(self.ema_v, unimg_a)
            unout_b = forward_logits(self.ema_v, unimg_b)
            plab_a = get_cut_mask(unout_a)
            plab_b = get_cut_mask(unout_b)
            img_mask, loss_mask = context_mask(img_a, self.mask_ratio)

        mixl = unimg_a * img_mask + img_b * (1 - img_mask)
        mixu = img_a * img_mask + unimg_b * (1 - img_mask)

        out_l = forward_logits(self.student_v, mixl)
        out_u = forward_logits(self.student_v, mixu)
        loss_l = mix_loss(out_l, plab_a, lab_b, loss_mask, u_weight=self.u_weight, unlab=True)
        loss_u = mix_loss(out_u, lab_a, plab_b, loss_mask, u_weight=self.u_weight)

        out_l2 = forward_logits(self.student_r, mixl)
        out_u2 = forward_logits(self.student_r, mixu)
        loss_l2 = mix_loss(out_l2, plab_a, lab_b, loss_mask, u_weight=self.u_weight, unlab=True)
        loss_u2 = mix_loss(out_u2, lab_a, plab_b, loss_mask, u_weight=self.u_weight)

        with torch.no_grad():
            diff1 = get_xor_region(out_l, out_l2)
            diff2 = get_xor_region(out_u, out_u2)

        def _corr(out, img_l, patch_l, mask, unlab=False, diff_mask=None):
            mse = mix_mse_loss(
                out, img_l, patch_l, mask, u_weight=self.u_weight, unlab=unlab, diff_mask=diff_mask
            )
            kl = mix_max_kl_loss(
                out, img_l, patch_l, mask, u_weight=self.u_weight, unlab=unlab, diff_mask=diff_mask
            )
            return mse, kl

        mse_v = _corr(out_l, plab_a, lab_b, loss_mask, unlab=True, diff_mask=diff1)[0]
        mse_v += _corr(out_u, lab_a, plab_b, loss_mask, diff_mask=diff2)[0]
        kl_v = _corr(out_l, plab_a, lab_b, loss_mask, unlab=True, diff_mask=diff1)[1]
        kl_v += _corr(out_u, lab_a, plab_b, loss_mask, diff_mask=diff2)[1]

        mse_r = _corr(out_l2, plab_a, lab_b, loss_mask, unlab=True, diff_mask=diff1)[0]
        mse_r += _corr(out_u2, lab_a, plab_b, loss_mask, diff_mask=diff2)[0]
        kl_r = _corr(out_l2, plab_a, lab_b, loss_mask, unlab=True, diff_mask=diff1)[1]
        kl_r += _corr(out_u2, lab_a, plab_b, loss_mask, diff_mask=diff2)[1]

        loss_v = (loss_l + loss_u) + self.mse_w * mse_v + self.kl_w * kl_v
        loss_r = (loss_l2 + loss_u2) + self.mse_w * mse_r + self.kl_w * kl_r
        return loss_v, loss_r

    def self_train(self, start_epoch: int = 0, no_improve: int = 0) -> None:
        scfg = self.cfg["semi"]
        tcfg = self.cfg["train"]
        epochs = int(scfg.get("self_epochs", 276))
        eval_interval = int(tcfg.get("eval_interval", 5))
        early_pat = int(scfg.get("early_stop_patience", 20))
        early_min = int(scfg.get("early_stop_min_epoch", 200))
        cycles = max(1, int(scfg.get("unlab_cycles_per_epoch", 2)))

        # Only load pretrain at the true start of self-train — never on resume
        if start_epoch == 0:
            self._load_pretrain_for_self()
        else:
            logging.info(
                "Resume self-train from epoch=%d (skip pretrain reload) best=%.4f",
                start_epoch,
                self.best_dice,
            )

        for epoch in tqdm(range(start_epoch, epochs), desc="sdcl_self", ncols=70):
            self.student_v.train()
            self.student_r.train()
            self.ema_v.train()

            it = zip(
                cycle(self.lab_a),
                cycle(self.lab_b),
                cycle(self.unlab_a),
                cycle(self.unlab_b),
            )
            steps = max(len(self.unlab_a), len(self.lab_a)) * cycles
            losses_v, losses_r = [], []

            for _ in range(steps):
                (ba, bb, ua, ub) = next(it)
                img_a, lab_a = self._prepare(ba)
                img_b, lab_b = self._prepare(bb)
                unimg_a, unlab_a = self._prepare(ua)
                unimg_b, unlab_b = self._prepare(ub)

                loss_v, loss_r = self._self_train_step(
                    img_a, lab_a, img_b, lab_b, unimg_a, unlab_a, unimg_b, unlab_b
                )

                self.opt_v.zero_grad(set_to_none=True)
                loss_v.backward()
                self.opt_v.step()

                self.opt_r.zero_grad(set_to_none=True)
                loss_r.backward()
                self.opt_r.step()

                update_ema(self.student_v, self.ema_v, self.ema_alpha)
                self.global_step += 1
                losses_v.append(float(loss_v.item()))
                losses_r.append(float(loss_r.item()))
                del loss_v, loss_r, img_a, lab_a, img_b, lab_b, unimg_a, unlab_a, unimg_b, unlab_b
                if self.global_step % 5 == 0:
                    gc.collect()
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()

            logging.info(
                "self epoch=%d step=%d loss_v=%.4f loss_r=%.4f",
                epoch,
                self.global_step,
                float(np.mean(losses_v)),
                float(np.mean(losses_r)),
            )
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            if epoch % eval_interval == 0 or epoch == epochs - 1:
                metrics = self._evaluate()
                self._log_metrics("self_eval", metrics)
                if metrics["dice"] > self.best_dice:
                    self.best_dice = metrics["dice"]
                    self.best_epoch = epoch
                    no_improve = 0
                    self._save("best.pth", epoch=epoch, extra={"phase": "self", "no_improve": 0})
                    logging.info("saved best.pth dice=%.4f", self.best_dice)
                else:
                    no_improve += 1
                    logging.info(
                        "no improve cur=%.4f best=%.4f streak=%d/%d min_ep=%d",
                        metrics["dice"],
                        self.best_dice,
                        no_improve,
                        early_pat,
                        early_min,
                    )
                    if early_pat > 0 and no_improve >= early_pat and epoch >= early_min:
                        logging.info("early stop at epoch=%d", epoch)
                        self._save("last.pth", epoch=epoch, extra={"phase": "self", "no_improve": no_improve})
                        return
                self._save("last.pth", epoch=epoch, extra={"phase": "self", "no_improve": no_improve})
            else:
                # Persist every epoch so kill/relaunch does not lose progress between evals.
                self._save("last.pth", epoch=epoch, extra={"phase": "self", "no_improve": no_improve})

    def _finalize_pretrain_only(self) -> Dict[str, float]:
        pb = self.output_dir / "pretrain_best.pth"
        if pb.is_file():
            state = torch.load(str(pb), map_location=self.device)
            self.student_v.load_state_dict(state["student_v"])
            self.student_r.load_state_dict(state["student_r"])
            if "best_dice" in state:
                self.best_dice = float(state["best_dice"])
            if "epoch" in state:
                self.best_epoch = int(state.get("best_epoch", state["epoch"]))
        metrics = self._evaluate()
        self._log_metrics("final_pretrain_only", metrics)
        out = self.output_dir / "final_metrics.txt"
        dice = round(metrics["dice"], 4)
        with open(out, "w", encoding="utf-8") as f:
            f.write(f"dice: {dice:.4f}\n")
            f.write(f"dice_pct: {dice * 100:.2f}%\n")
            for k in ("jaccard", "hd95", "asd"):
                f.write(f"{k}: {metrics[k]:.4f}\n")
            f.write(f"method: {self.method_tag}\n")
            f.write("phase: pretrain_only\n")
            f.write(f"best_dice: {self.best_dice:.4f}\n")
            f.write(f"best_epoch: {self.best_epoch}\n")
            f.write(f"global_step: {self.global_step}\n")
        logging.info("Saved %s (pretrain-only)", out)
        return metrics

    def run(self, pretrain_only: Optional[bool] = None) -> Dict[str, float]:
        if pretrain_only is None:
            pretrain_only = bool(self.cfg.get("train", {}).get("pretrain_only", False))
        logging.info(
            "%s run output=%s pretrain_only=%s",
            self.method_tag,
            self.output_dir,
            pretrain_only,
        )
        resume = self._try_load_resume()
        phase = "pre"
        start_epoch = 0
        no_improve = 0
        if resume is not None:
            info = self._apply_resume(resume)
            phase = info["phase"]
            start_epoch = info["start_epoch"]
            no_improve = info["no_improve"]

        pre_epochs = int(self.cfg.get("semi", {}).get("pre_epochs", 81))
        # Defensive: last.pth left at final pre epoch → treat as ready for self
        if phase == "pre" and start_epoch >= pre_epochs:
            logging.info(
                "Resume past pre_epochs (%d>=%d); advancing to self-train",
                start_epoch,
                pre_epochs,
            )
            phase = "self"
            start_epoch = 0
            no_improve = 0
            self._save("last.pth", epoch=-1, extra={"phase": "self", "no_improve": 0})

        if phase == "pre":
            self.pretrain(start_epoch=start_epoch)
            if pretrain_only:
                return self._finalize_pretrain_only()
            phase = "self"
            start_epoch = 0

        if phase == "self":
            self.self_train(start_epoch=start_epoch, no_improve=no_improve)

        best_path = self.output_dir / "best.pth"
        if not best_path.is_file():
            # Fallback: pretrain_best is often the true best under early self plateau
            for cand in ("pretrain_best.pth", "last.pth"):
                cp = self.output_dir / cand
                if cp.is_file():
                    import shutil

                    shutil.copy2(cp, best_path)
                    logging.warning("best.pth missing; copied from %s", cand)
                    break
        if best_path.is_file():
            state = torch.load(str(best_path), map_location=self.device)
            self.student_v.load_state_dict(state["student_v"])
            self.student_r.load_state_dict(state["student_r"])
            if "best_dice" in state:
                self.best_dice = float(state["best_dice"])
            if "epoch" in state:
                self.best_epoch = int(state.get("best_epoch", state["epoch"]))

        metrics = self._evaluate()
        self._log_metrics("final", metrics)
        out = self.output_dir / "final_metrics.txt"
        dice = round(metrics["dice"], 4)
        with open(out, "w", encoding="utf-8") as f:
            f.write(f"dice: {dice:.4f}\n")
            f.write(f"dice_pct: {dice * 100:.2f}%\n")
            for k in ("jaccard", "hd95", "asd"):
                f.write(f"{k}: {metrics[k]:.4f}\n")
            f.write(f"method: {self.method_tag}\n")
            f.write(f"best_dice: {self.best_dice:.4f}\n")
            f.write(f"best_epoch: {self.best_epoch}\n")
            f.write(f"global_step: {self.global_step}\n")
        logging.info("Saved %s", out)
        return metrics
