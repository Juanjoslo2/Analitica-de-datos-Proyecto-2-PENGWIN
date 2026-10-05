"""losses.py. Pérdida de detección: focal (presencia) + DFL (distribución) + (1 − GIoU)."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from pengwin_orozco.boxes import paired_iou_giou
from pengwin_orozco.detector import assign_dfl_targets, dfl_expectation


def sigmoid_focal_loss(logits: torch.Tensor, targets: torch.Tensor,
                       alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Focal loss (Lin et al., 2017) elemento a elemento, sin reducir."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    a_t = alpha * targets + (1 - alpha) * (1 - targets)
    return a_t * (1 - p_t) ** gamma * ce


def dfl_loss(dist_logits: torch.Tensor, target_cells: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """DFL (Li et al., 2021): CE sobre los bins con objetivo "integral" por partes.

    Para cada celda positiva, el valor continuo v (en celdas, ya recortado a
    [0, reg_max − 1]) se reparte entre el bin inferior lb = ⌊v⌋ y el superior
    ub = lb + 1 con pesos (1 − frac) y frac. Es una CE de dos puntos que permite
    al modelo representar valores subpixel.
    """
    v = target_cells.float().clamp(0, dist_logits.shape[3] - 1)
    lb = v.floor().long()
    ub = (lb + 1).clamp(max=dist_logits.shape[3] - 1)
    frac = (v - lb).unsqueeze(3)
    logp = dist_logits.log_softmax(dim=3)
    p_lb = logp.gather(3, lb.unsqueeze(3))
    p_ub = logp.gather(3, ub.unsqueeze(3))
    loss = -((1 - frac) * p_lb + frac * p_ub)
    pos4 = pos.unsqueeze(2)                                  # (B, K, 1, G, G)
    return (loss * pos4.unsqueeze(3)).sum() / pos.sum().clamp(min=1).float()


def detection_loss(
    scores: torch.Tensor,
    dist_logits: torch.Tensor,
    boxes: torch.Tensor,
    present: torch.Tensor,
    ignore: torch.Tensor,
    stride: int,
    focal: Dict,
    reg_max: int,
) -> Dict[str, torch.Tensor]:
    """Pérdida completa de detección para un batch (semana 9: solo esta cabeza).

    L_det = focal(obj) + λ_dfl·DFL + (1 − GIoU) sobre las celdas positivas.
    """
    g = scores.shape[-1]
    t = assign_dfl_targets(boxes, present, ignore, g, stride, focal.get("center_radius", 1.5))
    pos, valid, target_cells = t["pos"], t["valid"], t["ltrb_cells"]
    n_pos = pos.sum().clamp(min=1).float()
    l_obj = (sigmoid_focal_loss(scores.float(), pos.float(), focal.get("alpha", 0.25), focal.get("gamma", 2.0))
             * valid).sum() / n_pos
    l_dfl = dfl_loss(dist_logits.float(), target_cells, pos)
    if pos.any():
        ltrb_px = dfl_expectation(dist_logits.float(), stride)
        g_, k_, i_, j_ = torch.nonzero(pos, as_tuple=True)
        pred = torch.stack([
            cell_cx(g, stride, j_, dist_logits.device) - ltrb_px[g_, k_, 0, i_, j_],
            cell_cy(g, stride, i_, dist_logits.device) - ltrb_px[g_, k_, 1, i_, j_],
            cell_cx(g, stride, j_, dist_logits.device) + ltrb_px[g_, k_, 2, i_, j_],
            cell_cy(g, stride, i_, dist_logits.device) + ltrb_px[g_, k_, 3, i_, j_],
        ], dim=-1)
        gt = boxes[g_, k_]
        iou, giou = paired_iou_giou(pred, gt)
        l_giou = (1 - giou).mean()
        mean_iou = iou.detach().mean()
    else:
        l_giou = dist_logits.sum() * 0.0
        mean_iou = torch.zeros((), device=scores.device)
    return {"det_obj": l_obj, "det_dfl": l_dfl, "det_giou": l_giou,
            "det": l_obj + l_dfl + l_giou, "det_pos_iou": mean_iou}


def cell_cx(grid_size: int, stride: int, j: torch.Tensor, device) -> torch.Tensor:
    return (j.float() + 0.5) * stride


def cell_cy(grid_size: int, stride: int, i: torch.Tensor, device) -> torch.Tensor:
    return (i.float() + 0.5) * stride


class DetectionLoss(nn.Module):
    """Envuelve ``detection_loss`` con la config; strided para gradientes. Semana 9:
    la pérdida es solo de detección (sin clasificación ni segmentación todavía)."""

    def __init__(self, cfg: Dict):
        super().__init__()
        self.stride = int(cfg["model"].get("det_stride", 8))
        self.reg_max = int(cfg["model"].get("reg_max", 32))
        lc = cfg["loss"]
        self.focal = {"alpha": lc.get("focal_alpha", 0.25), "gamma": lc.get("focal_gamma", 2.0),
                      "center_radius": lc.get("center_radius", 1.5)}

    def forward(self, out: Dict[str, torch.Tensor], batch: Dict) -> Dict[str, torch.Tensor]:
        parts = detection_loss(
            out["det_scores"], out["det_dist"],
            batch["boxes"], batch["present"], batch["ignore"],
            self.stride, self.focal, self.reg_max,
        )
        parts["total"] = parts["det"]
        return parts