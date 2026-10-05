"""boxes.py. Operaciones de cajas propias (IoU, GIoU, NMS), sin torchvision.ops."""

from __future__ import annotations

import torch


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """(…, 4) en x0, y0, x1, y1 → área, clampada para cajas invertidas."""
    w = (boxes[..., 2] - boxes[..., 0]).clamp(min=0.0)
    h = (boxes[..., 3] - boxes[..., 1]).clamp(min=0.0)
    return w * h


def box_iou(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """IoU entre todos los pares: (N, 4) × (M, 4) → (N, M)."""
    x1 = torch.maximum(a[:, None, 0], b[None, :, 0])
    y1 = torch.maximum(a[:, None, 1], b[None, :, 1])
    x2 = torch.minimum(a[:, None, 2], b[None, :, 2])
    y2 = torch.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    return inter / (union + eps)


def paired_iou_giou(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7):
    """IoU y GIoU elemento a elemento sobre tensores de la misma forma (…, 4).

    GIoU penaliza también la caja envolvente vacía: da gradiente útil incluso
    cuando las cajas no se tocan (mejor que IoU puro para entrenar la regresión).
    """
    x1 = torch.minimum(a[..., 0], b[..., 0])
    y1 = torch.minimum(a[..., 1], b[..., 1])
    x2 = torch.maximum(a[..., 2], b[..., 2])
    y2 = torch.maximum(a[..., 3], b[..., 3])
    c_area = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)
    inter = (torch.minimum(a[..., 2], b[..., 2]) - torch.maximum(a[..., 0], b[..., 0])).clamp(min=0.0) * \
            (torch.minimum(a[..., 3], b[..., 3]) - torch.maximum(a[..., 1], b[..., 1])).clamp(min=0.0)
    union = box_area(a) + box_area(b) - inter
    iou = inter / (union + eps)
    giou = iou - (c_area - union) / (c_area + eps)
    return iou, giou


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5,
        max_keep: int | None = None) -> torch.Tensor:
    """NMS voraz propio. Devuelve los índices originales en orden de score descendente.

    Se precalcula la matriz de IoU una sola vez (viable: el grid es ≤ 1024 celdas).
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep = []
    if max_keep is not None and max_keep <= 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    ious = box_iou(boxes[order], boxes[order])
    for i in range(order.numel()):
        if max_keep is not None and len(keep) >= max_keep:
            break
        if any(ious[i, kept].max().item() > iou_threshold for kept in keep):
            continue
        keep.append(i)
    return order[torch.tensor(keep, device=boxes.device)]


def batched_nms(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor,
                iou_threshold: float = 0.5, max_per_class: int | None = None) -> torch.Tensor:
    """NMS independiente por clase; ``max_per_class`` limita las cajas por clase."""
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    keep_all = []
    for c in labels.unique():
        idx = torch.nonzero(labels == c, as_tuple=False).flatten()
        keep_all.append(idx[nms(boxes[idx], scores[idx], iou_threshold, max_per_class)])
    out = torch.cat(keep_all) if keep_all else torch.empty(0, dtype=torch.long, device=boxes.device)
    return out[scores[out].argsort(descending=True)]