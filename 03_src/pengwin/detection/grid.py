"""grid.py.

Detección por grid propio, anchor-free, stride 8 [DD §2, EDA §8].

Con entrada 256×256 el mapa P3 tiene 32×32 celdas; el centro de la celda (i, j) está en
c = ((j + 0,5)·8, (i + 0,5)·8) px. Cada celda predice, para CADA región k:
    s_k        puntaje (logit) de que la celda pertenezca al centro de la caja de k
    l, t, r, b distancias (px) del centro de la celda a los cuatro lados de la caja de k

Las cajas son por clase (12 canales y no 4) porque las envolventes del sacro y de cada
coxal se solapan en la articulación sacroilíaca: una misma celda puede ser positiva
para dos regiones con cajas distintas.

Asignación (``assign_targets``):
    positiva  centro de celda dentro de la caja Y a menos de ``radius`` celdas de su centro
              (center sampling, como FCOS). Si la caja es tan pequeña que ninguna celda
              cumple, se usa la celda que contiene su centro: toda caja tiene ≥ 1 positivo.
    ignorar   celdas dentro de una caja marcada ``ignore`` (lado < 4 px): sin pérdida.
    negativa  todo lo demás.
Decodificación (``decode``): sigmoide → umbral → caja = centro ± (l, t, r, b) → recorte
al lienzo → NMS propio por clase → como máximo ``max_per_class`` cajas por región.
"""

from __future__ import annotations

from typing import Dict, List

import torch

from pengwin.detection.nms import batched_nms


def cell_centers(grid_size: int, stride: int, device=None) -> torch.Tensor:
    """(G, G, 2) con (cx, cy) en píxeles de la entrada."""
    c = (torch.arange(grid_size, device=device, dtype=torch.float32) + 0.5) * stride
    cy, cx = torch.meshgrid(c, c, indexing="ij")
    return torch.stack([cx, cy], dim=-1)


def assign_targets(
    boxes: torch.Tensor,
    present: torch.Tensor,
    ignore: torch.Tensor,
    grid_size: int,
    stride: int,
    radius: float = 1.5,
) -> Dict[str, torch.Tensor]:
    """boxes (B, K, 4), present/ignore (B, K) -> objetivos densos sobre el grid.

    Devuelve ``pos`` y ``valid`` bool (B, K, G, G) y ``ltrb`` float (B, K, 4, G, G).
    """
    ctr = cell_centers(grid_size, stride, boxes.device)               # (G, G, 2)
    cx, cy = ctr[..., 0], ctr[..., 1]
    x0, y0, x1, y1 = (boxes[..., i, None, None] for i in range(4))   # (B, K, 1, 1)
    inside = (cx > x0) & (cx < x1) & (cy > y0) & (cy < y1)           # (B, K, G, G)
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
    ltrb = torch.stack([cx - x0, cy - y0, x1 - cx, y1 - cy], dim=2)   # (B, K, 4, G, G)
    return {"pos": pos, "valid": valid | pos, "ltrb": ltrb}


def ltrb_to_boxes(ltrb: torch.Tensor, stride: int) -> torch.Tensor:
    """(B, K, 4, G, G) distancias -> (B, K, G, G, 4) cajas x0, y0, x1, y1."""
    g = ltrb.shape[-1]
    ctr = cell_centers(g, stride, ltrb.device)
    l, t, r, b = ltrb.unbind(dim=2)
    return torch.stack([ctr[..., 0] - l, ctr[..., 1] - t, ctr[..., 0] + r, ctr[..., 1] + b], dim=-1)


@torch.no_grad()
def decode(
    score_logits: torch.Tensor,
    ltrb: torch.Tensor,
    stride: int,
    image_size: int,
    score_threshold: float = 0.3,
    nms_iou: float = 0.5,
    max_per_class: int | None = 1,
    pre_nms_top_k: int = 100,
) -> List[Dict[str, torch.Tensor]]:
    """Salidas de la cabeza -> lista (una por imagen) de {boxes (N,4), scores (N,), labels (N,)}."""
    B, K, G, _ = score_logits.shape
    scores = torch.sigmoid(score_logits.float()).flatten(2)          # (B, K, G·G)
    boxes = ltrb_to_boxes(ltrb.float(), stride).clamp(0, image_size).flatten(2, 3)  # (B, K, G·G, 4)
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
