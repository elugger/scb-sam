"""Public-only constructor adaptation; every loss/forward method stays frozen.

Original style constructor hardcodes BY's 100 training IDs. This isolated class
inserts a constructor into the MRO that performs the exact original setup and
hook registration, except it validates 17 + 329/330 IDs from the Public contract.
No original class/global/source is patched. Explicit original parent invocation
skips only the BY-specific constructor which this body faithfully reproduces.
"""
from pathlib import Path
from expes.common.semi_data import read_case_list
from expes.semi05_gate import fcd_public05_contract_20261015 as contract
from expes.semi05_gate.paired_zea_cross_scale_half_20260911 import PairedCrossScaleHalf
from expes.semi05_gate.paired_l_cross_style_complement_20261001 import PairedCrossStyleComplement
from expes.semi05_gate.paired_fcd_a1b_20261009 import QualifiedStyleBalanced


class PublicCrossStyleComplement(PairedCrossStyleComplement):
    def __init__(self, cfg):
        assert cfg["semi"]["cross_style_complement"] == "two-donor-full-decoder"
        PairedCrossScaleHalf.__init__(self, cfg)
        split = Path(cfg["dataset"]["split_root"]) / f"fold{cfg['dataset']['fold']}"
        self._style_donors = read_case_list(split / "train_unlab.txt")
        counts = contract.budget(cfg["dataset"]["fold"])
        assert cfg["dataset"]["split_root"] == contract.SPLIT_ROOT
        assert cfg["dataset"]["fold_cv_dir_template"] == contract.DATA_TEMPLATE
        assert len(self._style_donors) == counts["unlab_count"]
        assert len(read_case_list(split / "train_lab.txt")) == counts["lab_count"]
        assert len(self._style_donors) >= 2
        self._style_root = Path(cfg["dataset"]["fold_cv_dir_template"].format(
            fold=cfg["dataset"]["fold"])) / "train/imagesTr"
        self._style_amp_cache = {}
        self._b_mask = None
        self._b_branch = None
        self._b_capture = False
        self._b_captured_embeddings = []
        self._b_last = None

        # CrossScale's adapter hook is registered during super().__init__. This
        # later hook therefore masks the adapted embedding delivered to the
        # complete main mask decoder and residual refinement path.
        def complementary_mask(_module, _inputs, embedding):
            if self._b_mask is None:
                return embedding
            mask = self._b_mask.to(device=embedding.device, dtype=embedding.dtype)
            if tuple(mask.shape) != (1, embedding.shape[1], 1, 1, 1):
                raise RuntimeError(f"Invalid complementary mask {tuple(mask.shape)}")
            # Reentrant activation checkpointing executes the first pass under
            # no_grad and recomputes with gradients during backward. Capture
            # only the latter so the audit observes the true routed gradient.
            if self._b_capture and embedding.requires_grad:
                embedding.retain_grad()
                self._b_captured_embeddings.append(
                    dict(branch=int(self._b_branch), embedding=embedding, mask=mask.detach().clone()))
            return embedding * mask

        self._b_mask_handle = self.student.cascade.image_encoder.register_forward_hook(
            complementary_mask)



class PublicQualifiedStyleBalanced(QualifiedStyleBalanced, PublicCrossStyleComplement):
    """Only MRO insertion; scientific methods inherited unchanged."""
    pass

