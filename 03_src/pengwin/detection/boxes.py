"""boxes.py.

Operaciones de cajas implementadas por el equipo (sin torchvision.ops) [enunciado §3.1].
Formato por defecto: esquinas (x0, y0, x1, y1) en coordenadas continuas de píxel,
igual que ``targets.region_boxes_2d``. Todas las funciones aceptan tensores (..., 4).
"""

from __future__ import annotations

import torch


def box_area(b: torch.Tensor) -> torch.Tensor:
    return (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)


def box_iou(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """IoU de todos contra todos: a (N, 4), b (M, 4) -> (N, M)."""
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    return inter / (union + eps)


def paired_iou_giou(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7):
    """IoU y GIoU elemento a elemento: a, b (..., 4) -> (iou, giou), cada uno (...).

    GIoU = IoU − |C \\ (A ∪ B)| / |C|, con C la caja mínima que encierra a A y B.
    A diferencia del IoU, sigue dando gradiente cuando las cajas no se tocan.
    """
    lt = torch.maximum(a[..., :2], b[..., :2])
    rb = torch.minimum(a[..., 2:], b[..., 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    union = box_area(a) + box_area(b) - inter
    iou = inter / (union + eps)
    c_lt = torch.minimum(a[..., :2], b[..., :2])
    c_rb = torch.maximum(a[..., 2:], b[..., 2:])
    c_area = (c_rb - c_lt).clamp(min=0).prod(-1)
    return iou, iou - (c_area - union) / (c_area + eps)


def xyxy_to_cxcywh(b: torch.Tensor) -> torch.Tensor:
    return torch.stack([(b[..., 0] + b[..., 2]) / 2, (b[..., 1] + b[..., 3]) / 2,
                        b[..., 2] - b[..., 0], b[..., 3] - b[..., 1]], dim=-1)


def cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    return torch.stack([b[..., 0] - b[..., 2] / 2, b[..., 1] - b[..., 3] / 2,
                        b[..., 0] + b[..., 2] / 2, b[..., 1] + b[..., 3] / 2], dim=-1)
