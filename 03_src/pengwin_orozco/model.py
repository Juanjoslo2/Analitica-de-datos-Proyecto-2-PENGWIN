"""model.py. Ensamblado del intento propio: backbone (FundidoraPC + CBAM) → cabeza DFL.

Semana 9 = solo detección (el enunciado pide backbone, CBAM y cabeza de detección
con overfit y primeras cajas razonables). La clasificación y la segmentación se
añadirán encima del mismo backbone en la semana 10.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from pengwin_orozco.backbone import build_backbone
from pengwin_orozco.detector import DFLGridHead


class PengwinOrozcoNet(nn.Module):
    def __init__(self, model_cfg: Dict, in_channels: int = 3):
        super().__init__()
        self.backbone = build_backbone(model_cfg)
        if self.backbone.stride != int(model_cfg.get("det_stride", 8)):
            raise ValueError("det_stride debe coincidir con el stride del backbone (8)")
        self.det_head = DFLGridHead(
            in_channels=self.backbone.out_channels,
            num_classes=int(model_cfg.get("num_classes", 3)),
            stride=self.backbone.stride,
            reg_max=int(model_cfg.get("reg_max", 32)),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        _, _, c4 = self.backbone(x)
        det_scores, det_dist = self.det_head(c4)
        return {"det_scores": det_scores, "det_dist": det_dist}


def build_model(cfg: Dict, log=print) -> PengwinOrozcoNet:
    return PengwinOrozcoNet(cfg["model"])


def count_parameters(model: nn.Module) -> Dict[str, int]:
    groups = {"backbone": 0, "det_head": 0}
    for n, p in model.named_parameters():
        groups[n.split(".")[0]] += p.numel()
    groups["total"] = sum(groups.values())
    return groups