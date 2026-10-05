"""metrics.py. Métricas de detección propias: AP/mAP estilo VOC 2010+/COCO e IoU promedio.

Semana 9: solo detección → estas son las métricas de la prueba de correctitud.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

IOU_THRESHOLDS = np.round(np.arange(0.50, 0.96, 0.05), 2)     # 0.50 … 0.95


def _iou_single(a: np.ndarray, b: np.ndarray) -> float:
    """IoU entre dos cajas x0, y0, x1, y1 (numpy)."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(inter, area_a + area_b - inter)


def average_precision(tp: np.ndarray, scores: np.ndarray, n_gt: int) -> float:
    """AP con la curva PR exacta (precisión con envolvente monótona hacia atrás).

    NaN si no hay GT; 0 si hay GT pero ninguna predicción.
    """
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores)
    tp = tp[order]
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(1 - tp)
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    rec = tp_cum / n_gt
    prec = np.concatenate([[1.0], prec, [0.0]])
    rec = np.concatenate([[0.0], rec, [1.0]])
    prec = np.maximum.accumulate(prec[::-1])[::-1]              # envolvente monótona
    idx = np.where(rec[1:] != rec[:-1])[0]
    return float(np.sum((rec[idx + 1] - rec[idx]) * prec[idx + 1]))


class DetectionEvaluator:
    """Acumula predicciones por corte y clase; una GT por clase y corte (máx 1 caja).

    Las GT ``ignore`` (cajas < 4 px) no cuentan como GT: las predicciones que caen
    sobre ellas se descartan (ni TP ni FP) para no penalizar a la red por lo que
    el propio diseño considera no evaluable.

    El emparejado TP/FP se hace POR UMBRAL de IoU (como COCO): una predicción con
    IoU 0,6 es TP a @0,5 pero FP a @0,75 — así se computa mAP@[.50:.95].
    """

    def __init__(self, num_classes: int = 3):
        self.num_classes = num_classes
        self._rows: List[Dict] = []

    def add(self, pred: Dict[str, np.ndarray], boxes: np.ndarray, present: np.ndarray, ignore: np.ndarray):
        """pred = {boxes (N,4), scores (N,), labels (N,)} en numpy."""
        self._rows.append({
            "pred": pred,
            "gt": np.asarray(boxes, float),
            "present": np.asarray(present) > 0.5,
            "ignore": np.asarray(ignore) > 0.5,
        })

    def compute(self, thresholds=IOU_THRESHOLDS) -> Dict[str, float]:
        aps = np.full((len(thresholds), self.num_classes), np.nan)
        iou_by_class: Dict[int, List[float]] = {c: [] for c in range(self.num_classes)}
        for c in range(self.num_classes):
            n_gt = 0
            pre_rows = []                                       # por corte: (preds ordenadas, gt_box, modo)
            for r in self._rows:
                m = r["pred"]["labels"] == c
                p = np.asarray(r["pred"]["boxes"])[m]
                s = np.asarray(r["pred"]["scores"])[m]
                order = np.argsort(-s)
                p, s = p[order], s[order]
                gt_box = r["gt"][c]
                if r["present"][c] and not r["ignore"][c]:
                    n_gt += 1
                    iou_by_class[c].append(max([_iou_single(b, gt_box) for b in p], default=0.0))
                    pre_rows.append((p, s, gt_box, "gt"))
                elif r["present"][c]:                           # GT ignorada: descartar preds que la tocan
                    keep = ~np.array([_iou_single(b, gt_box) > 0.0 for b in p])
                    pre_rows.append((p[keep], s[keep], gt_box, "ignore"))
                else:
                    pre_rows.append((p, s, gt_box, "none"))
            for ti, thr in enumerate(thresholds):
                tp_all, sc_all = [], []
                for p, s, gt_box, mode in pre_rows:
                    if mode == "gt":
                        matched = False
                        for b, sc in zip(p, s):
                            hit = _iou_single(b, gt_box) >= thr
                            tp_all.append(1.0 if hit and not matched else 0.0)
                            matched = matched or hit
                            sc_all.append(float(sc))
                    else:
                        tp_all.extend([0.0] * len(p))
                        sc_all.extend(s.tolist())
                aps[ti, c] = average_precision(np.asarray(tp_all, float), np.asarray(sc_all, float), n_gt)
        with np.errstate(invalid="ignore"):
            map50 = float(np.nanmean(aps[0]))
            map50_95 = float(np.nanmean(aps))
        iou_flat = [v for lst in iou_by_class.values() for v in lst]
        return {
            "mAP@0.50": map50,
            "mAP@[.50:.95]": map50_95,
            **{f"AP50_cls{c}": float(np.nanmean(aps[0, c])) for c in range(self.num_classes)},
            "IoU_promedio": float(np.mean(iou_flat)) if iou_flat else float("nan"),
        }