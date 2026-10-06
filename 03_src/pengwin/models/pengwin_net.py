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
                                                               core_logits (B, 3, 256, 256)*
                                                               dist_logits (B, 1, 256, 256)**
                                                               role_logits (B, 3, 256, 256)***

(*) solo con ``model.seg_outputs: [semantic4, core3]`` [F2B1]; (**) solo con
``model.seg_outputs: [semantic4, dist]`` [F2B2]; (***) solo con ``role3`` en ``model.seg_outputs``
[y4xul]. Con ``model.seg_fullres_skip: true`` la cabeza de segmentación recibe además C0 (bloque 1
antes del pooling, stride 1) y ``model.pool`` elige el pooling del backbone. Las salidas de la cabeza de segmentación las
elige ``model.seg_outputs``; ``cls_logits``, ``det_scores`` y ``det_ltrb`` no cambian nunca.
Si no hay cabeza de borde binaria, ``edge_logits`` se deriva de ``core_logits`` o de
``dist_logits`` (ver ``SegmentationHead``).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from pengwin.models.backbone import build_backbone, load_fundidora_weights
from pengwin.models.heads import (
    DEFAULT_SEG_OUTPUTS, ClassificationHead, GridDetectionHead, SegmentationHead, TopDownNeck,
)


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
        self.neck = TopDownNeck(c3, c4, neck, gamma=bool(model_cfg.get("neck_gamma", False)))
        self.cls_head = ClassificationHead(c4, n_cls)
        self.det_head = GridDetectionHead(neck, n_cls, self.stride)
        self.seg_outputs = tuple(model_cfg.get("seg_outputs", DEFAULT_SEG_OUTPUTS) or DEFAULT_SEG_OUTPUTS)
        self.fullres_skip = bool(model_cfg.get("seg_fullres_skip", False))
        self.seg_head = SegmentationHead(neck, c2, c1, num_classes=n_cls + 1,
                                         spatial_dropout=float(model_cfg.get("seg_spatial_dropout", 0.0)),
                                         outputs=self.seg_outputs, c0=c1 if self.fullres_skip else 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.fullres_skip:
            c0, (c1, c2, c3, c4) = self.backbone.forward_with_stem(x)
        else:
            c0, (c1, c2, c3, c4) = None, self.backbone(x)
        p3 = self.neck(c3, c4)
        det_scores, det_ltrb = self.det_head(p3)
        out = {
            "cls_logits": self.cls_head(c4),
            "det_scores": det_scores,
            "det_ltrb": det_ltrb,
        }
        out.update(self.seg_head(p3, c2, c1, x.shape[-2:], c0))
        return out

    def gammas(self) -> Dict[str, float]:
        """γ aprendidos: cuánto usa la red cada componente (ver ``scripts/component_contribution.py``).

        - residual bN: media de |γ| de la última BN del bloque residual (arranca en 0).
        - CBAM bN: γ de la mezcla con la identidad (arranca en 0; solo con ``cbam_gamma``).
        - cuello C4: γ del contexto profundo en P3 (arranca en 1; solo con ``neck_gamma``).
        """
        out = {}
        for k, st in enumerate(self.backbone.stages, start=1):
            if hasattr(st.res, "bn_b"):
                out[f"residual b{k}"] = float(st.res.bn_b.weight.detach().abs().mean())
            if getattr(st.cbam, "gamma", None) is not None:
                out[f"CBAM b{k}"] = float(st.cbam.gamma.detach())
        if self.neck.gamma is not None:
            out["cuello C4"] = float(self.neck.gamma.detach())
        return out

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
