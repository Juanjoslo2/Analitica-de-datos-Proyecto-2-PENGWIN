"""detector.py.

Cabeza de detección propia con DFL (Distribution Focal Loss) sobre un grid
anchor-free de stride 8 [enunciado §4.1: "grid propio y NMS propio"].

Idea (implementada desde cero, no copiada):
    Cada celda del grid 32×32 predice, para cada región k (SA, coxal izq., coxal
    der.):
        s_k            logit de presencia (se entrena con focal loss)
        dist_k (l,t,r,b)  UNA DISTRIBUCIÓN de ``reg_max`` bins por cada borde,
                          en unidades de celda; la distancia usada es la
                          esperanza de la distribución (precisión subpixel).

La regresión "clásica" de un número (p. ej. exp() → l,t,r,b) no puede expresar
medios píxeles; DFL sí, y es especialmente útil aquí porque el 39 % de las cajas
del sacro mide < 16 px [EDA §8] (una celda = 8 px).

Asignación (``assign_dfl_targets``):
    positiva    centro de celda dentro de la caja Y a menos de ``radius`` celdas
                de su centro (center sampling). Cajas diminutas: la celda del
                centro, garantizando ≥ 1 positivo.
    ignorar     celdas dentro de una caja ``ignore`` (lado < 4 px): sin pérdida.
    negativa    el resto.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from pengwin_orozco.boxes import batched_nms


def cell_centers(grid_size: int, stride: int, device=None) -> torch.Tensor:
    """(G, G, 2) con (cx, cy) en píxeles de la entrada."""
    c = (torch.arange(grid_size, device=device, dtype=torch.float32) + 0.5) * stride
    cy, cx = torch.meshgrid(c, c, indexing="ij")
    return torch.stack([cx, cy], dim=-1)


def assign_dfl_targets(
    boxes: torch.Tensor,
    present: torch.Tensor,
    ignore: torch.Tensor,
    grid_size: int,
    stride: int,
    radius: float = 1.5,
) -> Dict[str, torch.Tensor]:
    """boxes (B, K, 4), present/ignore (B, K) → pos/valid (B, K, G, G), ltrb_cells (B, K, 4, G, G).

    ``ltrb_cells`` son las distancias en UNIDADES DE CELDA (px / stride) para todas
    las celdas; la pérdida DFL las recorta a [0, reg_max − 1].
    """
    ctr = cell_centers(grid_size, stride, boxes.device)
    cx, cy = ctr[..., 0], ctr[..., 1]
    x0, y0, x1, y1 = (boxes[..., i, None, None] for i in range(4))
    inside = (cx > x0) & (cx < x1) & (cy > y0) & (cy < y1)
    bcx, bcy = (x0 + x1) / 2, (y0 + y1) / 2
    near = ((cx - bcx).abs() < radius * stride) & ((cy - bcy).abs() < radius * stride)
    usable = (present.bool() & ~ignore.bool())[..., None, None]
    pos = inside & near & usable

    # Cajas diminutas sin ninguna celda positiva: la celda que contiene el centro
    j = (bcx / stride).floor().long().clamp(0, grid_size - 1)[..., 0, 0]
    i = (bcy / stride).floor().long().clamp(0, grid_size - 1)[..., 0, 0]
    empty = usable[..., 0, 0] & ~pos.flatten(2).any(-1)
    if empty.any():
        b_idx, k_idx = torch.nonzero(empty, as_tuple=True)
        pos[b_idx, k_idx, i[b_idx, k_idx], j[b_idx, k_idx]] = True

    valid = ~(inside & (ignore.bool() & present.bool())[..., None, None])
    ltrb_px = torch.stack([cx - x0, cy - y0, x1 - cx, y1 - cy], dim=2)
    return {"pos": pos, "valid": valid | pos, "ltrb_cells": ltrb_px / stride}


def dfl_expectation(dist_logits: torch.Tensor, stride: int) -> torch.Tensor:
    """(B, K, 4, R, G, G) → (B, K, 4, G, G) distancias esperadas en píxeles."""
    r = dist_logits.shape[3]
    probs = dist_logits.softmax(dim=3)
    bins = torch.arange(r, device=probs.device, dtype=probs.dtype).view(1, 1, 1, r, 1, 1)
    return (probs * bins).sum(dim=3) * stride


def decode(
    score_logits: torch.Tensor,
    dist_logits: torch.Tensor,
    stride: int,
    image_size: int,
    reg_max: int,
    score_threshold: float = 0.3,
    nms_iou: float = 0.5,
    max_per_class: int | None = 1,
    pre_nms_top_k: int = 100,
) -> List[Dict[str, torch.Tensor]]:
    """Salidas de la cabeza → lista (por imagen) de {boxes (N,4), scores, labels}.

    NMS propio por clase; como máximo ``max_per_class`` cajas por región y corte.
    """
    B, K, G, _ = score_logits.shape
    scores = torch.sigmoid(score_logits.float()).flatten(2)
    ltrb = dfl_expectation(dist_logits.float(), stride)                     # (B, K, 4, G, G)
    ctr = cell_centers(G, stride, score_logits.device)
    l, t, r, b = ltrb.unbind(dim=2)
    boxes = torch.stack([ctr[..., 0] - l, ctr[..., 1] - t, ctr[..., 0] + r, ctr[..., 1] + b], dim=-1)
    boxes = boxes.clamp(0, image_size).flatten(2, 3)                        # (B, K, G·G, 4)
    out = []
    for bi in range(B):
        bx, sc, lb = [], [], []
        for k in range(K):
            s = scores[bi, k]
            top = s.topk(min(pre_nms_top_k, s.numel())).indices
            top = top[s[top] > score_threshold]
            bx.append(boxes[bi, k, top])
            sc.append(s[top])
            lb.append(torch.full_like(top, k))
        bx, sc, lb = torch.cat(bx), torch.cat(sc), torch.cat(lb)
        keep = batched_nms(bx, sc, lb, nms_iou, max_per_class)
        out.append({"boxes": bx[keep], "scores": sc[keep], "labels": lb[keep]})
    return out


class DFLGridHead(nn.Module):
    """Cabeza anchor-free con DFL sobre el mapa C4 (stride 8).

    Salidas:
        scores (B, K, G, G)              logits de presencia por región
        dist   (B, K, 4, reg_max, G, G)  logits de la distribución por borde
    """

    def __init__(self, in_channels: int, num_classes: int = 3, stride: int = 8,
                 reg_max: int = 32, tower_depth: int = 2, prior: float = 0.01):
        super().__init__()
        self.num_classes = num_classes
        self.stride = stride
        self.reg_max = reg_max

        def tower():
            layers = []
            for _ in range(tower_depth):
                layers += [nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                           nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True)]
            return nn.Sequential(*layers)

        self.cls_tower = tower()
        self.box_tower = tower()
        self.score = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.box = nn.Conv2d(in_channels, num_classes * 4 * reg_max, 3, padding=1)
        # Prior 1 % de celdas positivas: que los negativos no dominen al inicio
        nn.init.constant_(self.score.bias, -torch.log(torch.tensor((1 - prior) / prior)))
        # Inicializar la distribución con pico suave en el bin 2 (≈ 2 celdas = 16 px),
        # equivalente a la idea de "cajas pequeñas al inicio" del diseño compartido
        bias = self.box.bias.view(num_classes, 4, reg_max)
        peak = torch.arange(reg_max, dtype=torch.float32)
        nn.init.constant_(bias, 0.0)
        bias.data.copy_(bias.data + (-((peak - 2.0) ** 2) * 0.5).view(1, 1, -1))

    def forward(self, c4: torch.Tensor):
        b, _, g, _ = c4.shape
        scores = self.score(self.cls_tower(c4))
        dist = self.box(self.box_tower(c4)).view(b, self.num_classes, 4, self.reg_max, g, g)
        return scores, dist