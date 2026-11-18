"""Launch one independent C0/C1 job; no auto-generated method search."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import yaml


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--init-from", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--gpu", required=True, choices=("0", "1"))
    parser.add_argument("--control", required=True, choices=("C0", "C1"))
    parser.add_argument("--expected-floor", type=float, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-result")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    # CUDA visibility is fixed before any model/torch import.
    import torch
    from expes.semi05_gate.paired_wsv_control_20260908 import PairedWSVControl
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA device required; no CPU fallback")
    torch.set_num_threads(2)
    output = Path(args.output_dir)
    if any((output / name).exists() for name in ("config.yaml", "manifest.json", "best.pth", "log.txt")):
        raise RuntimeError("Refusing an already initialized run directory")
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    allowed = Path("/var/tmp/zc_asnet_paired_20260908").resolve()
    if allowed not in checkpoint_dir.parents:
        raise RuntimeError("Checkpoint directory must be a named child of the project working store")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if any(checkpoint_dir.iterdir()):
        raise RuntimeError("Checkpoint directory is not empty")
    if shutil.disk_usage(checkpoint_dir).free < 8*1024**3:
        raise RuntimeError("Insufficient checkpoint working space")
    if not args.preflight_only:
        if not args.preflight_result:
            raise RuntimeError("A successful ratio-matched preflight result is required")
        evidence = json.loads(Path(args.preflight_result).read_text())
        if (evidence.get("status") != "PASS"
                or evidence.get("evaluation_difference", 1) > 1e-8
                or abs(evidence["floor"]["dice"]-args.expected_floor) > .0001):
            raise RuntimeError("Invalid preflight evidence")
    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["experiment"].update(gpu=args.gpu, output_dir=str(output), seed=2026)
    cfg["dataset"]["fold"] = 2
    cfg["model"]["deep_supervision"] = False
    scfg = cfg["semi"]
    for key in list(scfg):
        if key.startswith(("duw_", "fclr_")):
            scfg.pop(key)
    scfg.update(ucp_epochs=36, ucp_fg_start=.92, ucp_fg_end=.70, ucp_ramp_epochs=12,
                ucp_lambda_max=.15, ucp_pseudo_ramp_epochs=8, ucp_accumulation=5,
                ucp_encoder_lr=5e-8, ucp_decoder_lr=2e-7, ucp_coarse_lr=2e-7,
                ucp_ema_alpha=.999, ucp_noise_std=.05, ucp_early_stop_min_epochs=36,
                ucp_early_stop_patience=36, ws_flip_mode="independent", ws_intensity_scale=.10,
                ws_intensity_shift=.05, ws_consistency_weight=.10,
                paired_refresh_labels=args.control == "C1", paired_workers=2,
                paired_checkpoint_dir=str(checkpoint_dir))
    output.mkdir(parents=True, exist_ok=True)
    sources = ["expes/semi05_gate/"+name for name in (
        "paired_rng_data_20260908.py", "paired_wsv_control_20260908.py",
        "train_paired_wsv_control_20260908.py", "weak_strong_view_ucp_trainer.py",
        "uncertainty_curriculum_trainer.py")]
    split_dir = Path(cfg["dataset"]["split_root"]) / "fold2"
    manifest = dict(control=args.control, preflight_only=args.preflight_only,
                    init_from=str(Path(args.init_from).resolve()), init_sha256=sha256(args.init_from),
                    source_hashes={name: sha256(name) for name in sources},
                    split_hashes={name: sha256(split_dir/name) for name in ("train_lab.txt", "train_unlab.txt", "test.txt")},
                    device=torch.cuda.get_device_name(0), visible_physical_gpu=args.gpu,
                    checkpoint_working_store=str(checkpoint_dir), checkpoint_archive_pending=True,
                    protocol="legacy label-localized ROI retained; no new GT-derived targets",
                    independent_run_only=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    trainer = PairedWSVControl(cfg)
    if args.preflight_only:
        trainer.preflight(args.init_from, args.expected_floor)
    else:
        trainer.run_paired(args.init_from, args.expected_floor)


if __name__ == "__main__":
    main()
