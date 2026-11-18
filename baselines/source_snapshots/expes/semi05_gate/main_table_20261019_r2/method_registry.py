"""The current main table only; controls outside it are not scheduled."""
from pathlib import Path
import subprocess
from budget_protocol import BUDGETS
from historical_kernels import extract_kernel

HERE = Path(__file__).resolve().parent
SOURCES = HERE.parent / "baseline_sources_20261019"
METHODS = {
    "supervised": dict(label="Supervised (our cascade)", status="existing_public_results_verify_for_reuse"),
    "sdcl": dict(label="SDCL", kernel="sdcl", pretrain_kernel="sdcl_pretrain",
                 native_labeled=4, native_unlabeled=4, native_optimizers=2),
    "mt_semisam": dict(label="SemiSAM+ (MT)", kernel="mt_semisam",
                       native_labeled=1, native_unlabeled=1, native_optimizers=1),
    "uamt_semisam": dict(label="SemiSAM+ (UA-MT)", kernel="uamt_semisam",
                         native_labeled=1, native_unlabeled=1, native_optimizers=1),
    "sfr": dict(label="SFR", repo="SFR", commit="dc32304b6ba7a6f9ecbcea5a30e9eebe98510ad6",
                blocked=["missing networks/utils.py and networks/networks_other.py",
                         "SAM ViT-B original checkpoint identity not bound",
                         "SAM fine-tuning and 3D retraining stage/data adaptation not qualified"]),
    "relif3d": dict(label="ReLiF-3D", repo="ReLiF_3D", commit="bbda984aa337520c009091d4a02d2e104bb176d6",
                    blocked=["single-channel SAM output incompatible with foreground slice",
                             "detached feature hook and projection optimizer require scientific verification",
                             "original SAM-Med3D checkpoint identity not bound"]),
    "hpq": dict(label="HPQ (ours)", status="existing_run_read_only"),
}


def inspect(method, dataset, ratio):
    if method not in METHODS:
        raise ValueError("Method outside current main-table scope")
    b = next((b for b in BUDGETS if b.dataset == dataset and b.ratio == ratio), None)
    if b is None:
        raise ValueError("Dataset/ratio outside authorized scope")
    result = dict(method=method, budget=b.record(), **METHODS[method])
    if "kernel" in result:
        _, result["kernel_audit"] = extract_kernel(result["kernel"])
        if "pretrain_kernel" in result:
            _, result["pretrain_audit"] = extract_kernel(result["pretrain_kernel"])
        result["native_iteration_plan"] = b.iteration_plan(result["native_labeled"], result["native_unlabeled"])
        result["status"] = "UPDATE_KERNEL_AND_ENDPOINT_CONTROLLER_IMPLEMENTED_NOT_GPU_QUALIFIED"
        result["pending"] = ["bound runtime data/model/init adapter", "full resource target enrollment",
                              "CPU tensor smoke and independent numerical qualification"]
    if "repo" in result:
        repo = SOURCES / result["repo"]
        actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        if actual != result["commit"]:
            raise ValueError("Official source commit changed")
        result["status"] = "SOURCE_PINNED_IMPLEMENTATION_BLOCKED"
    result["launch_ready"] = False
    return result
