"""SDCL dual backbones: VNet (student A) + ResVNet (student B)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
SDCL_CODE = ROOT / "SDCL-master" / "code"
if str(SDCL_CODE) not in sys.path:
    sys.path.insert(0, str(SDCL_CODE))

from networks.ResVNet import ResVNet  # noqa: E402
from pancreas.Vnet import VNet  # noqa: E402


def build_vnet() -> nn.Module:
    return VNet(n_channels=1, n_classes=2, normalization="instancenorm", has_dropout=True)


def build_resvnet() -> nn.Module:
    return ResVNet(n_channels=1, n_classes=2, normalization="instancenorm", has_dropout=True)


def build_ema_vnet() -> nn.Module:
    net = build_vnet()
    for p in net.parameters():
        p.detach_()
    return net
