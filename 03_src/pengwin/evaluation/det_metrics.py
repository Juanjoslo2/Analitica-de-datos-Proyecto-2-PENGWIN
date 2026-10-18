"""det_metrics.py.

Métricas de detección y segmentación propias [enunciado §5].

Detección (una caja GT por región y corte):
    AP por clase   área bajo la curva precisión-recall con envolvente monótona
                   (todos los puntos, como VOC ≥ 2010 y COCO). Una predicción es TP si
                   IoU ≥ umbral con una GT de su clase aún no emparejada; las GT marcadas
                   ``ignore`` no cuentan, y una predicción que cae sobre ellas no es FP.
    mAP@0.50       media de AP sobre las clases con al menos una GT.
    mAP@[.50:.95]  media de mAP en los umbrales 0,50, 0,55, …, 0,95.
    IoU promedio   por cada GT, IoU con la predicción de mayor puntaje de su clase en el
                   mismo corte (0 si no hay ninguna) → promedio. Penaliza las omisiones.
Segmentación:
    Dice/IoU por clase acumulados sobre todos los píxeles del conjunto (no por corte, para
    que los cortes con poco hueso no pesen igual que los que tienen mucho).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch

from pengwin.data.targets import REGION_NAMES
from pengwin.detection.boxes import box_iou

IOU_THRESHOLDS = np.round(np.arange(0.50, 0.96, 0.05), 2)


def average_precision(tp: np.ndarray, scores: np.ndarray, n_gt: int) -> float:
    if n_gt == 0:
        return float("nan")
    if tp.size == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tp = tp[order].astype(float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-12)
    r = np.concatenate([[0.0], recall, [1.0]])
    p = np.concatenate([[1.0], precision, [0.0]])
    p = np.maximum.accumulate(p[::-1])[::-1]               # envolvente monótona
    idx = np.nonzero(r[1:] != r[:-1])[0]
    return float(((r[idx + 1] - r[idx]) * p[idx + 1]).sum())


class DetectionEvaluator:
    """Acumula predicciones y GT corte a corte; ``compute()`` devuelve las métricas."""

    def __init__(self, num_classes: int = 3, class_names: Sequence[str] = REGION_NAMES):
        self.k = num_classes
        self.names = list(class_names)
        self.preds: List[Dict[str, np.ndarray]] = []
        self.gts: List[Dict[str, np.ndarray]] = []

    def add(self, pred: Dict[str, torch.Tensor], boxes: torch.Tensor, present: torch.Tensor, ignore: torch.Tensor) -> None:
        """``pred`` = salida de ``grid.decode`` para un corte; GT (K, 4), (K,), (K,)."""
        self.preds.append({k: v.detach().cpu().numpy() for k, v in pred.items()})
        self.gts.append({"boxes": boxes.detach().cpu().numpy(), "present": present.detach().cpu().numpy() > 0.5,
                         "ignore": ignore.detach().cpu().numpy().astype(bool)})

    def _class_matches(self, c: int, thr: float):
        tps, scores, n_gt = [], [], 0
        for pred, gt in zip(self.preds, self.gts):
            sel = pred["labels"] == c
            pb, ps = pred["boxes"][sel], pred["scores"][sel]
            has_gt = gt["present"][c]
            if has_gt and not gt["ignore"][c]:
                n_gt += 1
            if pb.shape[0] == 0:
                continue
            order = np.argsort(-ps, kind="stable")
            pb, ps = pb[order], ps[order]
            iou = box_iou(torch.from_numpy(pb), torch.from_numpy(gt["boxes"][c:c + 1])).numpy()[:, 0] if has_gt else np.zeros(len(pb))
            matched = False
            for j in range(len(pb)):
                if has_gt and iou[j] >= thr and gt["ignore"][c]:
                    continue                                    # cae sobre una GT ignorada: ni TP ni FP
                hit = has_gt and not matched and iou[j] >= thr
                matched |= hit
                tps.append(hit)
                scores.append(ps[j])
        return np.array(tps, bool), np.array(scores, float), n_gt

    def compute(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        ap = np.full((len(IOU_THRESHOLDS), self.k), np.nan)
        for ti, thr in enumerate(IOU_THRESHOLDS):
            for c in range(self.k):
                tp, sc, n_gt = self._class_matches(c, float(thr))
                ap[ti, c] = average_precision(tp, sc, n_gt)
        out["mAP@0.50"] = float(np.nanmean(ap[0])) if not np.all(np.isnan(ap[0])) else float("nan")
        out["mAP@[.50:.95]"] = float(np.nanmean(np.nanmean(ap, axis=1))) if not np.all(np.isnan(ap)) else float("nan")
        for c, name in enumerate(self.names):
            out[f"AP50_{name}"] = float(ap[0, c])

        ious = []
        for pred, gt in zip(self.preds, self.gts):
            for c in range(self.k):
                if not gt["present"][c] or gt["ignore"][c]:
                    continue
                sel = pred["labels"] == c
                if not sel.any():
                    ious.append(0.0)
                    continue
                best = pred["boxes"][sel][np.argmax(pred["scores"][sel])]
                ious.append(float(box_iou(torch.from_numpy(best[None]), torch.from_numpy(gt["boxes"][c:c + 1]))[0, 0]))
        out["IoU_promedio"] = float(np.mean(ious)) if ious else float("nan")
        return out


class SegmentationEvaluator:
    """Dice e IoU por clase acumulando intersección y tamaños sobre todos los píxeles."""

    def __init__(self, num_classes: int = 4, class_names: Sequence[str] = ("fondo",) + tuple(REGION_NAMES)):
        self.n = num_classes
        self.names = list(class_names)
        self.inter = np.zeros(num_classes)
        self.pred_sum = np.zeros(num_classes)
        self.gt_sum = np.zeros(num_classes)

    def add(self, pred_labels: torch.Tensor, gt_labels: torch.Tensor) -> None:
        p = pred_labels.flatten().cpu()
        g = gt_labels.flatten().cpu()
        self.inter += torch.bincount(g[p == g], minlength=self.n).numpy()
        self.pred_sum += torch.bincount(p, minlength=self.n).numpy()
        self.gt_sum += torch.bincount(g, minlength=self.n).numpy()

    def compute(self) -> Dict[str, float]:
        dice = 2 * self.inter / np.maximum(self.pred_sum + self.gt_sum, 1)
        iou = self.inter / np.maximum(self.pred_sum + self.gt_sum - self.inter, 1)
        out = {f"dice_{n}": float(d) for n, d in zip(self.names, dice)}
        out.update({f"iou_{n}": float(v) for n, v in zip(self.names, iou)})
        out["dice_hueso"] = float(dice[1:].mean())
        out["iou_hueso"] = float(iou[1:].mean())
        return out
