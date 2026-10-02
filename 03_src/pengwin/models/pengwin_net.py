"""pengwin_net.py.

Modelo multitarea completo: backbone compartido (FundidoraPC extendida + CBAM) → 3 cabezas.

    entrada (B, 3, 256, 256)  cortes (z−Δ, z, z+Δ) en [0, 1]
        │
    FundidoraBackbone ──► C1 s2 · C2 s4 · C3 s8 (CBAM) · C4 s16 (CBAM)
        │                                   └──── bifurcación ────┐
        ├─ C4 ─────────────────────────► ClassificationHead  → cls_logits (B, 3)
        ├─ C3 + C4 → TopDownNeck → P3 ─► GridDetectionHead   → det_scores (B, 3, 32, 32)
        │                                                      det_ltrb   (B, 3, 4, 32, 32)
        └─ P3 + C2 + C1 ───────────────► SegmentationHead    → seg_logits (B, 4, 256, 256)
                                                               edge_logits (B, 1, 256, 256)
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from pengwin.models.backbone import build_backbone, load_fundidora_weights
from pengwin.models.heads import ClassificationHead, GridDetectionHead, SegmentationHead, TopDownNeck


class PengwinNet(nn.Module):
    def __init__(self, model_cfg: Dict, in_channels: int = 3):
        super().__init__()
        self.backbone = build_backbone(model_cfg, in_channels)
        c1, c2, c3, c4 = self.backbone.widths
        neck = int(model_cfg.get("neck_channels", 128))
        n_cls = int(model_cfg.get("num_classes", 3))
        self.stride = int(model_cfg.get("det_stride", 8))
        if self.stride != self.backbone.strides[2]:
            raise ValueError("det_stride debe coincidir con el stride de C3 (8)")
        self.neck = TopDownNeck(c3, c4, neck)
        self.cls_head = ClassificationHead(c4, n_cls)
        self.det_head = GridDetectionHead(neck, n_cls, self.stride)
        self.seg_head = SegmentationHead(neck, c2, c1, num_classes=n_cls + 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        c1, c2, c3, c4 = self.backbone(x)
        p3 = self.neck(c3, c4)
        det_scores, det_ltrb = self.det_head(p3)
        seg_logits, edge_logits = self.seg_head(p3, c2, c1, x.shape[-2:])
        return {
            "cls_logits": self.cls_head(c4),
            "det_scores": det_scores,
            "det_ltrb": det_ltrb,
            "seg_logits": seg_logits,
            "edge_logits": edge_logits,
        }

    def head_parameters(self):
        """Parámetros de cuello y cabezas (siempre desde cero)."""
        return [p for n, p in self.named_parameters() if not n.startswith("backbone.")]


def build_model(cfg: Dict, pretrained_path: str | None = None, log=print) -> PengwinNet:
    """Construye el modelo según ``cfg['model']``; si ``pretrained`` es verdadero, carga
    los pesos de FundidoraPC SOLO en el backbone (las cabezas quedan aleatorias)."""
    m = cfg["model"]
    model = PengwinNet(m)
    if m.get("pretrained", False):
        path = pretrained_path or m.get("pretrained_weights")
        if not path:
            raise ValueError("model.pretrained=true pero no hay model.pretrained_weights")
        report = load_fundidora_weights(model.backbone, path)
        log(f"[backbone] pesos de {path}: cargados {report['loaded']} | omitidos {report['skipped']}")
    return model


def count_parameters(model: nn.Module) -> Dict[str, int]:
    groups = {"backbone": 0, "neck": 0, "cls_head": 0, "det_head": 0, "seg_head": 0}
    for n, p in model.named_parameters():
        groups[n.split(".")[0]] += p.numel()
    groups["total"] = sum(groups.values())
    return groups
