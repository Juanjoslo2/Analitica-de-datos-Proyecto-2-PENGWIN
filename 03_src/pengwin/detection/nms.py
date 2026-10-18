"""nms.py.

Supresión de no máximos propia [enunciado §3.1, DD §2].

``nms`` es la versión voraz clásica: se toma la caja de mayor puntaje, se eliminan las que
la solapan con IoU > umbral y se repite. Se vectoriza calculando la matriz de IoU UNA vez
(N ≤ 1024 celdas del grid de stride 8, así que la matriz cabe sin problema) y recorriendo
las cajas en orden; cada paso solo lee una fila de la matriz.

``batched_nms`` aplica la NMS por clase: una caja del sacro nunca suprime una del coxal,
aunque se solapen en la articulación sacroilíaca.
"""

from __future__ import annotations

import torch

from pengwin.detection.boxes import box_iou


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5, max_keep: int | None = None) -> torch.Tensor:
    """Índices (en ``boxes``) de las cajas conservadas, en orden de puntaje descendente."""
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    iou = box_iou(boxes[order], boxes[order])
    suppressed = torch.zeros(order.numel(), dtype=torch.bool, device=boxes.device)
    keep = []
    for i in range(order.numel()):
        if suppressed[i]:
            continue
        keep.append(i)
        if max_keep is not None and len(keep) >= max_keep:
            break
        suppressed |= iou[i] > iou_threshold      # incluye i mismo (IoU 1), ya está en keep
    return order[torch.tensor(keep, dtype=torch.long, device=boxes.device)]


def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.5,
    max_per_class: int | None = None,
) -> torch.Tensor:
    """NMS independiente por clase; devuelve índices ordenados por puntaje."""
    keep = [torch.nonzero(labels == c).flatten()[nms(boxes[labels == c], scores[labels == c], iou_threshold, max_per_class)]
            for c in labels.unique()]
    if not keep:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    keep = torch.cat(keep)
    return keep[scores[keep].argsort(descending=True)]
