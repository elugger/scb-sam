"""Self-Training trainer for AutoSAMSeg (semi-V L0 baseline, model swapped)."""

from __future__ import annotations

import logging
import os
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torchio as tio
from torch.cuda import amp
from tqdm import tqdm

from expes.common.config import save_config
from expes.common.losses import confidence_mask_from_logits, pseudo_loss, softmax_mse_loss
from expes.common.ramps import sigmoid_rampup
from expes.common.semi_data import (
    CaseListDataset,
    build_eval_transform,
    build_train_transform,
    make_case_dataloader,
    make_pseudo_dataloader,
    read_case_list,
    split_list_path,
)
from expes.common.seg_metrics import aggregate_metrics, binary_case_metrics, logits_to_binary
from segment_anything.auto_sam_seg import AutoSAMSeg, build_auto_sam_seg
from utils.auto_sam_loss import build_supervised_loss_fns, cascade_supervised_loss
from utils.cascade_train_utils import compute_batch_dice, init_seeds, normalize_batch


ROOT = Path(__file__).resolve().parents[2]


class SelfTrainTrainer:
    """Self-training on AutoSAMSeg: Sup pretrain + Semi with EMA-teacher hard pseudo labels."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        exp = cfg["experiment"]
        self.output_dir = Path(exp["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, self.output_dir / "config.yaml")

        os.environ["CUDA_VISIBLE_DEVICES"] = str(exp.get("gpu", "0"))
        init_seeds(int(exp.get("seed", 2026)))

        self._setup_logging()
        self.best_dice = 0.0
        self.best_epoch = -1
        self.best_pretrain_dice = 0.0
        self.best_pretrain_epoch = -1
        self.best_semi_dice = 0.0
        self.best_semi_epoch = -1
        self.norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
        self.main_loss_fn, self.coarse_loss_fn = build_supervised_loss_fns()

        self.student = self._build_model().to(self.device)
        self.teacher = self._build_model().to(self.device)
        self._update_teacher(alpha=0.0)

        self._build_optimizer()
        self._build_dataloaders()
        self.scaler = amp.GradScaler(enabled=(self.device.type == "cuda"))

    def _setup_logging(self) -> None:
        log_path = self.output_dir / "log.txt"
        logging.basicConfig(
            filename=str(log_path),
            level=logging.INFO,
            format="[%(asctime)s] %(message)s",
            datefmt="%H:%M:%S",
            force=True,
        )
        logging.getLogger().addHandler(logging.StreamHandler())

    def _build_model(self) -> AutoSAMSeg:
        mcfg = self.cfg.get("model", {})
        dcfg = self.cfg.get("dataset", {})
        sam_ckpt = mcfg.get("sam_checkpoint")
        if sam_ckpt and not os.path.isabs(sam_ckpt):
            sam_ckpt = str(ROOT / sam_ckpt)
        if sam_ckpt and not os.path.isfile(sam_ckpt):
            sam_ckpt = None
        model = build_auto_sam_seg(
            model_type=mcfg.get("model_type", "vit_b_ori"),
            sam_checkpoint=sam_ckpt,
            coarse_checkpoint=None,
            img_size=int(dcfg.get("img_size", 128)),
            num_classes=int(mcfg.get("num_classes", 2)),
            deep_supervision=bool(mcfg.get("deep_supervision", False)),
            device=str(self.device),
        )
        model.set_train_phase("e2e")
        return model

    def _build_optimizer(self, base_lr: Optional[float] = None) -> None:
        tcfg = self.cfg["train"]
        lr = float(tcfg["lr"] if base_lr is None else base_lr)
        sam = self.student.cascade.sam
        coarse = self.student.cascade.coarse_head
        self.optimizer = torch.optim.AdamW(
            [
                {"params": sam.image_encoder.parameters(), "lr": lr * float(tcfg.get("lr_encoder_scale", 1.0))},
                {"params": sam.prompt_encoder.parameters(), "lr": lr * float(tcfg.get("lr_decoder_scale", 0.1))},
                {"params": sam.mask_decoder.parameters(), "lr": lr * float(tcfg.get("lr_decoder_scale", 0.1))},
                {"params": coarse.parameters(), "lr": lr * float(tcfg.get("lr_coarse_scale", 1.25))},
            ],
            betas=(float(tcfg.get("beta1", 0.9)), float(tcfg.get("beta2", 0.999))),
            weight_decay=float(tcfg.get("weight_decay", 0.1)),
        )
        logging.info("Optimizer base_lr=%.2e", lr)

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

        self.lab_set = CaseListDataset(fold_cv_dir, lab_names, train_tf, threshold, split="train")
        self.unlab_set = CaseListDataset(fold_cv_dir, unlab_names, eval_tf, threshold, split="train")
        self.test_set = CaseListDataset(fold_cv_dir, test_names, eval_tf, threshold, split="test")

        labeled_bs = int(tcfg.get("labeled_batch_size", tcfg.get("batch_size", 1)))
        self.lab_loader = make_case_dataloader(
            self.lab_set,
            batch_size=labeled_bs,
            shuffle=True,
            num_workers=int(tcfg.get("num_workers", 2)),
            drop_last=True,
        )
        self.unlab_loader = make_case_dataloader(
            self.unlab_set,
            batch_size=1,
            shuffle=False,
            num_workers=int(tcfg.get("num_workers", 2)),
            drop_last=False,
        )
        self.test_loader = make_case_dataloader(
            self.test_set,
            batch_size=1,
            shuffle=False,
            num_workers=int(tcfg.get("num_workers", 2)),
            drop_last=False,
        )

        logging.info(
            "Data fold=%d lab=%d unlab=%d test=%d",
            fold_idx,
            len(self.lab_set),
            len(self.unlab_set),
            len(self.test_set),
        )

    def _update_teacher(self, alpha: float) -> None:
        for t_p, s_p in zip(self.teacher.parameters(), self.student.parameters()):
            t_p.data.mul_(alpha).add_((1.0 - alpha) * s_p.data)

    def _consistency_weight(self, epoch: int) -> float:
        """Sigmoid ramp-up in [0, 1]; caller multiplies lambda_cons once."""
        scfg = self.cfg["semi"]
        ramp = int(scfg.get("consistency_rampup", 40))
        return sigmoid_rampup(epoch, ramp)

    def _perturb_input(self, image: torch.Tensor, noise_std: float) -> torch.Tensor:
        """Gaussian input noise for Self-Training (needed for ViT/LN: no BN/dropout gap)."""
        if noise_std <= 0:
            return image
        return image + torch.randn_like(image) * noise_std

    def _prepare_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        image = normalize_batch(batch["image"], self.norm_transform).to(self.device)
        label = batch["label"].to(self.device)
        if label.dim() == 4:
            label = label.unsqueeze(1)
        return image, label

    def _supervised_step(self, image: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        with amp.autocast(enabled=(self.device.type == "cuda")):
            logits, coarse_logits = self.student(image, return_aux=True)
            aux_w = float(self.cfg["train"].get("aux_weight", 0.3))
            return cascade_supervised_loss(
                logits,
                coarse_logits,
                label.float(),
                self.main_loss_fn,
                self.coarse_loss_fn,
                aux_weight=aux_w,
            )

    def _forward_main_logits(self, model: AutoSAMSeg, image: torch.Tensor) -> torch.Tensor:
        with amp.autocast(enabled=(self.device.type == "cuda")):
            return model(image)

    @torch.no_grad()
    def _evaluate(self) -> Dict[str, float]:
        self.student.eval()
        case_metrics = []
        for batch in self.test_loader:
            image, label = self._prepare_batch(batch)
            logits = self._forward_main_logits(self.student, image)
            pred = logits_to_binary(logits)
            gt = (label > 0).squeeze(1).detach().cpu().numpy().astype(bool)
            for i in range(pred.shape[0]):
                case_metrics.append(binary_case_metrics(pred[i], gt[i], spacing=(1.0, 1.0, 1.0)))
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

    def _save(self, name: str, full: bool = False, epoch: int = -1) -> Path:
        path = self.output_dir / name
        payload: Dict[str, Any] = {
            "student": self.student.state_dict(),
            "best_dice": self.best_dice,
            "best_epoch": epoch if epoch >= 0 else self.best_epoch,
            "cfg": self.cfg,
        }
        # Only keep teacher+optim on true "best" milestones to avoid multi-GB disk stalls.
        if full:
            payload["teacher"] = self.teacher.state_dict()
            payload["optimizer_state_dict"] = self.optimizer.state_dict()
        torch.save(payload, str(path))
        return path

    def _load(self, path: str | Path) -> None:
        state = torch.load(str(path), map_location=self.device)
        if "student" in state:
            self.student.load_state_dict(state["student"])
        elif "model_state_dict" in state:
            self.student.load_state_dict(state["model_state_dict"])
        else:
            self.student.load_state_dict(state)

        if "teacher" in state:
            self.teacher.load_state_dict(state["teacher"])
        else:
            # FS / Sup-20 ckpt has no EMA teacher → copy student
            self._update_teacher(alpha=0.0)

        if "optimizer_state_dict" in state:
            try:
                self.optimizer.load_state_dict(state["optimizer_state_dict"])
            except Exception as exc:
                logging.warning("Skip optimizer load from %s: %s", path, exc)

    def pretrain(self) -> Path:
        tcfg = self.cfg["train"]
        epochs = int(tcfg["pretrain_epochs"])
        use_dice = bool(tcfg.get("use_dice", True))
        dice_w = float(tcfg.get("dice_weight", 0.5))
        eval_interval = int(tcfg.get("eval_interval", 5))
        logging.info(
            "pretrain epochs=%d eval_interval=%d select=test_dice",
            epochs,
            eval_interval,
        )

        for epoch in range(1, epochs + 1):
            self.student.train()
            losses, dices = [], []

            for batch in self.lab_loader:
                image, label = self._prepare_batch(batch)
                loss = self._supervised_step(image, label)

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self._update_teacher(float(self.cfg["ema"]["alpha"]))

                with torch.no_grad():
                    logits = self._forward_main_logits(self.student, image)
                    dice = compute_batch_dice(logits, label.float())

                losses.append(loss.item())
                dices.append(dice)

            logging.info(
                "pretrain epoch=%d loss=%.4f dice=%.4f",
                epoch,
                float(np.mean(losses)),
                float(np.mean(dices)),
            )

            if epoch % eval_interval == 0 or epoch == epochs:
                metrics = self._evaluate()
                self._log_metrics("pretrain_eval", metrics)
                if metrics["dice"] > self.best_pretrain_dice:
                    self.best_pretrain_dice = metrics["dice"]
                    self.best_pretrain_epoch = epoch
                    self.best_dice = metrics["dice"]
                    self.best_epoch = epoch
                    # student-only to avoid multi-GB disk stalls; teacher synced on load
                    self._save("pretrain_best.pth", full=False, epoch=epoch)
                    logging.info(
                        "saved pretrain_best.pth epoch=%d dice=%.4f",
                        epoch,
                        self.best_pretrain_dice,
                    )

        return self.output_dir / "pretrain_best.pth"

    @torch.no_grad()
    def _refresh_pseudo_loader(self):
        """Generate confident pseudo labels with EMA teacher (more stable than student)."""
        self.teacher.eval()
        scfg = self.cfg["semi"]
        thr = float(scfg.get("confidence_threshold", 0.9))
        tcfg = self.cfg["train"]
        use_teacher = bool(scfg.get("pseudo_from_teacher", True))
        plab_model = self.teacher if use_teacher else self.student
        plab_model.eval()

        imgs, pseudo, masks, gts = [], [], [], []
        plab_dice = 0.0

        for batch in self.unlab_loader:
            image, label = self._prepare_batch(batch)
            logits = self._forward_main_logits(plab_model, image)
            conf_mask, pred = confidence_mask_from_logits(logits, thr)

            imgs.append(image.cpu())
            pseudo.append(pred.cpu())
            masks.append(conf_mask.cpu())
            gts.append(label.cpu())

            pred_fg = (pred == 1).float()
            gt_fg = (label > 0).float()
            inter = (pred_fg * gt_fg).sum()
            denom = pred_fg.sum() + gt_fg.sum() + 1e-6
            plab_dice += float((2 * inter / denom).item())

        n = max(1, len(self.unlab_loader))
        plab_dice /= n
        conf_ratio = float(torch.cat([m.flatten() for m in masks]).mean().item()) if masks else 0.0
        logging.info(
            "pseudo_label_dice=%.4f conf_ratio=%.4f from=%s thr=%.2f",
            plab_dice,
            conf_ratio,
            "teacher" if use_teacher else "student",
            thr,
        )

        pseudo_loader = make_pseudo_dataloader(
            imgs,
            pseudo,
            masks,
            batch_size=int(tcfg.get("batch_size", 1)),
            num_workers=int(tcfg.get("num_workers", 0)),
            repeat=int(scfg.get("pseudo_repeat", 2)),
        )
        return pseudo_loader, plab_dice

    def semi_train(self, ckpt_path: Optional[str | Path] = None) -> None:
        if ckpt_path is not None and Path(ckpt_path).exists():
            self._load(ckpt_path)
            logging.info("Loaded checkpoint %s", ckpt_path)

        tcfg = self.cfg["train"]
        scfg = self.cfg["semi"]
        use_dice = bool(tcfg.get("use_dice", True))
        dice_w = float(tcfg.get("dice_weight", 0.5))
        semi_epochs = int(scfg.get("semi_epochs", 201))
        eval_interval = int(tcfg.get("eval_interval", 5))
        refresh_step = int(scfg.get("pseudo_refresh_step", 10))

        # Lower / override LR for semi so unsupervised terms do not wipe Sup-20.
        semi_lr = scfg.get("semi_lr", None)
        if semi_lr is not None:
            self._build_optimizer(base_lr=float(semi_lr))
            logging.info("Semi optimizer rebuilt with semi_lr=%.2e", float(semi_lr))

        # Floor = Sup-20: evaluate BEFORE any semi update; Semi must not report worse.
        base_metrics = self._evaluate()
        self.best_semi_dice = float(base_metrics["dice"])
        self.best_semi_epoch = -1  # -1 = inherited from Sup-20
        self.best_dice = self.best_semi_dice
        self.best_epoch = -1
        self._save("best.pth", full=False, epoch=-1)
        logging.info(
            "semi start (Sup-20 floor) dice=%.4f jaccard=%.4f hd95=%.4f asd=%.4f → init best.pth",
            base_metrics["dice"],
            base_metrics["jaccard"],
            base_metrics["hd95"],
            base_metrics["asd"],
        )
        # Early stop (opt-in): consecutive evals without beating best; min_epoch blocks early dip.
        es_patience = int(scfg.get("early_stop_patience", 0))
        es_min_epoch = int(scfg.get("early_stop_min_epoch", 60))
        no_improve_evals = 0
        logging.info(
            "semi_train epochs=%d eval_interval=%d select=test_dice floor=sup20 noise=%.3f "
            "early_stop=patience%d/min%d",
            semi_epochs,
            eval_interval,
            float(scfg.get("input_noise_std", 0.0)),
            es_patience,
            es_min_epoch,
        )

        # Refresh inside the loop only (epoch % refresh_step == 0), avoid double cost at ep0.
        pseudo_loader = None

        for epoch in tqdm(range(semi_epochs), desc="semi", ncols=70):
            if epoch % refresh_step == 0 or pseudo_loader is None:
                pseudo_loader, plab_dice = self._refresh_pseudo_loader()
                logging.info("refresh pseudo dice=%.4f", plab_dice)

            self.student.train()
            self.teacher.eval()
            cw = self._consistency_weight(epoch)
            sup_losses, cons_losses, pseudo_losses = [], [], []

            lab_iter = cycle(self.lab_loader)
            pseudo_iter = cycle(pseudo_loader)
            steps = max(len(self.lab_loader), len(pseudo_loader))

            noise_std = float(scfg.get("input_noise_std", 0.0))
            lambda_cons = float(scfg.get("lambda_cons", 1.0))
            # Dead when no input gap: skip teacher forward (same as cons≈0, ~1/3 less compute).
            run_cons = noise_std > 0.0 and lambda_cons > 0.0 and cw > 0.0
            noise_teacher = bool(scfg.get("noise_on_teacher", False))
            cons_temp = float(scfg.get("cons_temperature", 1.0))

            for _ in range(steps):
                batch_l = next(lab_iter)
                img_l, lab_l = self._prepare_batch(batch_l)

                img_u, plab_u, mask_u = next(pseudo_iter)
                # Pseudo cache already Z-normalized in _refresh_pseudo_loader — do NOT normalize again.
                if img_u.dim() == 4:
                    img_u = img_u.unsqueeze(0)
                img_u = img_u.to(self.device, non_blocking=True)
                if plab_u.dim() == 4:
                    plab_u = plab_u.unsqueeze(0)
                if mask_u.dim() == 4:
                    mask_u = mask_u.unsqueeze(0)
                plab_u = plab_u.to(self.device, non_blocking=True).long()
                mask_u = mask_u.to(self.device, non_blocking=True).float()

                img_s = self._perturb_input(img_u, noise_std)

                with amp.autocast(enabled=(self.device.type == "cuda")):
                    loss_sup = float(scfg.get("lambda_sup", 1.0)) * self._supervised_step(img_l, lab_l)

                    logits_s = self._forward_main_logits(self.student, img_s)
                    if run_cons:
                        img_t = self._perturb_input(img_u, noise_std) if noise_teacher else img_u
                        with torch.no_grad():
                            logits_t = self._forward_main_logits(self.teacher, img_t)
                        loss_cons = (
                            lambda_cons
                            * cw
                            * softmax_mse_loss(logits_s, logits_t, temperature=cons_temp)
                        )
                    else:
                        loss_cons = logits_s.new_zeros(())

                    loss_pseudo = float(scfg.get("lambda_pseudo", 1.0)) * pseudo_loss(
                        logits_s,
                        plab_u,
                        mask_u,
                        use_dice,
                        dice_w,
                    )
                    loss = loss_sup + loss_cons + loss_pseudo

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self._update_teacher(float(self.cfg["ema"]["alpha"]))

                sup_losses.append(float(loss_sup.item()))
                cons_losses.append(float(loss_cons.item()))
                pseudo_losses.append(float(loss_pseudo.item()))

            logging.info(
                "semi epoch=%d sup=%.4f cons=%.6f pseudo=%.4f cw=%.4f noise=%.3f steps=%d run_cons=%s",
                epoch,
                float(np.mean(sup_losses)),
                float(np.mean(cons_losses)),
                float(np.mean(pseudo_losses)),
                cw,
                noise_std,
                steps,
                run_cons,
            )

            if epoch % eval_interval == 0 or epoch == semi_epochs - 1:
                metrics = self._evaluate()
                self._log_metrics("semi_eval", metrics)
                if metrics["dice"] > self.best_semi_dice:
                    self.best_semi_dice = metrics["dice"]
                    self.best_semi_epoch = epoch
                    self.best_dice = metrics["dice"]
                    self.best_epoch = epoch
                    no_improve_evals = 0
                    self._save("best.pth", full=False, epoch=epoch)
                    logging.info(
                        "saved best.pth (semi) epoch=%d dice=%.4f (gain vs Sup-20 floor)",
                        epoch,
                        self.best_semi_dice,
                    )
                else:
                    no_improve_evals += 1
                    logging.info(
                        "no improve (cur=%.4f best=%.4f floor/sup20) streak=%d/%d min_ep=%d",
                        metrics["dice"],
                        self.best_semi_dice,
                        no_improve_evals,
                        es_patience,
                        es_min_epoch,
                    )
                    if (
                        es_patience > 0
                        and no_improve_evals >= es_patience
                        and epoch >= es_min_epoch
                    ):
                        logging.info(
                            "semi early stop at epoch=%d best=%.4f@ep%d",
                            epoch,
                            self.best_semi_dice,
                            self.best_semi_epoch,
                        )
                        return
    def run(self, skip_pretrain: bool = False, init_from: Optional[str] = None) -> Dict[str, float]:
        """Sup-20 == pretrain; Semi-20 continues from that checkpoint."""
        import shutil

        mode = self.cfg["train"].get("mode", "semi")
        logging.info(
            "MT run mode=%s output=%s skip_pretrain=%s init_from=%s",
            mode,
            self.output_dir,
            skip_pretrain,
            init_from,
        )
        logging.info(
            "Protocol: Sup-20 == MT pretrain on train_lab; "
            "Semi-20 continues from Sup-20 weights (pretrain_best / --init_from)."
        )

        if skip_pretrain or init_from:
            ckpt = Path(init_from) if init_from else (self.output_dir / "pretrain_best.pth")
            if not ckpt.is_absolute():
                ckpt = ROOT / ckpt
            if not ckpt.exists():
                raise FileNotFoundError(f"init checkpoint not found: {ckpt}")
            self._load(ckpt)
            metrics = self._evaluate()
            self.best_pretrain_dice = metrics["dice"]
            try:
                st = torch.load(str(ckpt), map_location="cpu")
                self.best_pretrain_epoch = int(st.get("best_epoch", st.get("epoch", -1)))
            except Exception:
                self.best_pretrain_epoch = -1
            logging.info(
                "Loaded Sup-20/init dice=%.4f from %s → continue semi (no ckpt rewrite)",
                metrics["dice"],
                ckpt,
            )
            if mode == "semi":
                self.semi_train(None)  # already loaded
            else:
                self._save("best.pth", full=False)
        else:
            self.pretrain()
            src = self.output_dir / "pretrain_best.pth"
            if src.exists() and not (self.output_dir / "sup20_best.pth").exists():
                import shutil

                shutil.copy2(src, self.output_dir / "sup20_best.pth")
            logging.info(
                "Sup-20 == pretrain_best dice=%.4f @ ep%d",
                self.best_pretrain_dice,
                self.best_pretrain_epoch,
            )
            if mode == "semi":
                self.semi_train(src)
            else:
                self._save("best.pth", full=False)

        metrics = self._evaluate()
        self._log_metrics("final", metrics)

        out = self.output_dir / "final_metrics.txt"
        with open(out, "w", encoding="utf-8") as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v:.6f}\n")
            f.write(f"mode: {mode}\n")
            f.write(f"sup20_dice: {self.best_pretrain_dice:.6f}\n")
            f.write(f"sup20_epoch: {self.best_pretrain_epoch}\n")
            f.write(f"semi20_dice: {self.best_semi_dice:.6f}\n")
            f.write(f"semi20_epoch: {self.best_semi_epoch}\n")
            f.write("protocol: Sup-20==pretrain; Semi-20 continues from Sup-20\n")
        logging.info("Saved %s", out)
        return metrics
