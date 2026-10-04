"""fragment_metrics.py.

Métricas por fragmento (instancia) en la grilla nativa [enunciado §5, DD §3].

Emparejamiento: dentro de cada región, cada fragmento GT se empareja con a lo sumo un
fragmento predicho maximizando el IoU total (asignación húngara). Un fragmento GT sin pareja,
o con IoU 0, cuenta con Dice = IoU = 0: si un fragmento se fusionó con otro, se penaliza.

Por fragmento GT se reporta:
    dice, iou              con su pareja (0 si no tiene)
    recuperado             IoU ≥ 0,5
    volumen_cm3            del GT, para estratificar (< 5, 5-20, > 20 cm³)
    es_principal
Y por cada secundario recuperado: distancia GT vs. distancia predicha (``separation_table``).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from pengwin.measurement.separation import REGION_NAMES, separation_table


def overlap_counts(gt: np.ndarray, pred: np.ndarray, n_labels: int = 31) -> np.ndarray:
    """Matriz (31, 31) de vóxeles en común entre cada etiqueta GT (filas) y predicha (columnas)."""
    return np.bincount(gt.astype(np.int64).ravel() * n_labels + pred.astype(np.int64).ravel(),
                       minlength=n_labels * n_labels).reshape(n_labels, n_labels)


def match_fragments(gt: np.ndarray, pred: np.ndarray, spacing_zyx: Sequence[float]) -> List[Dict]:
    """Una fila por fragmento GT con su pareja predicha y sus métricas."""
    vox_cm3 = float(np.prod(spacing_zyx)) / 1000.0
    M = overlap_counts(gt, pred)
    gt_size, pred_size = M.sum(1), M.sum(0)
    rows = []
    for r in (1, 2, 3):
        g_ids = [i for i in range((r - 1) * 10 + 1, r * 10 + 1) if gt_size[i] > 0]
        p_ids = [j for j in range((r - 1) * 10 + 1, r * 10 + 1) if pred_size[j] > 0]
        if not g_ids:
            continue
        iou = np.zeros((len(g_ids), max(len(p_ids), 1)))
        for a, i in enumerate(g_ids):
            for b, j in enumerate(p_ids):
                inter = M[i, j]
                iou[a, b] = inter / (gt_size[i] + pred_size[j] - inter) if inter else 0.0
        pair = {}
        if p_ids:
            ra, cb = linear_sum_assignment(-iou)
            pair = {g_ids[a]: p_ids[b] for a, b in zip(ra, cb) if iou[a, b] > 0}
        for a, i in enumerate(g_ids):
            j = pair.get(i)
            v = iou[a, p_ids.index(j)] if j is not None else 0.0
            rows.append({"region": REGION_NAMES[r], "gt_label": i, "pred_label": j,
                         "es_principal": i == (r - 1) * 10 + 1, "volumen_cm3": gt_size[i] * vox_cm3,
                         "iou": float(v), "dice": float(2 * v / (1 + v)) if v > 0 else 0.0, "recuperado": v >= 0.5})
    return rows


def distance_comparison(gt: np.ndarray, pred: np.ndarray, spacing_zyx: Sequence[float], matches: List[Dict]) -> List[Dict]:
    """Distancia al principal en GT y en la predicción para cada secundario GT."""
    d_gt = {row["label"]: row for row in separation_table(gt, spacing_zyx)}
    d_pred = {row["label"]: row for row in separation_table(pred, spacing_zyx)}
    out = []
    for m in matches:
        if m["es_principal"] or m["gt_label"] not in d_gt:
            continue
        g = d_gt[m["gt_label"]]
        p = d_pred.get(m["pred_label"]) if m["pred_label"] is not None and m["recuperado"] else None
        out.append({"region": m["region"], "gt_label": m["gt_label"], "volumen_cm3": m["volumen_cm3"],
                    "dist_gt_mm": g["dist_mm"], "contacto_gt": g["in_contact"],
                    "dist_pred_mm": p["dist_mm"] if p else np.nan,
                    "error_mm": (p["dist_mm"] - g["dist_mm"]) if p else np.nan, "medido": p is not None})
    return out


def summarize(frag_rows: List[Dict], dist_rows: List[Dict]) -> Dict[str, float]:
    import pandas as pd

    f = pd.DataFrame(frag_rows)
    sec = f[~f.es_principal]
    out = {"n_fragmentos_gt": len(f), "dice_fragmento": f.dice.mean(), "iou_fragmento": f.iou.mean(),
           "recuperados_%": 100 * f.recuperado.mean(),
           "dice_principal": f[f.es_principal].dice.mean(),
           "dice_secundario": sec.dice.mean() if len(sec) else np.nan,
           "recuperados_secundarios_%": 100 * sec.recuperado.mean() if len(sec) else np.nan}
    bins = pd.cut(f.volumen_cm3, [0, 5, 20, np.inf], labels=["<5 cm3", "5-20 cm3", ">20 cm3"])
    for name, g in f.groupby(bins, observed=False):
        out[f"dice_{name}"] = g.dice.mean() if len(g) else np.nan
        out[f"n_{name}"] = len(g)
    if dist_rows:
        d = pd.DataFrame(dist_rows)
        dm = d[d.medido]
        out["secundarios_con_distancia_%"] = 100 * d.medido.mean()
        out["mae_distancia_mm"] = dm.error_mm.abs().mean() if len(dm) else np.nan
        # la media la domina un solo fragmento mal medido (p. ej. principal partido en dos): se reporta también la mediana
        out["mediana_error_mm"] = dm.error_mm.abs().median() if len(dm) else np.nan
        for name, g in dm.groupby("contacto_gt"):
            out["mae_mm_" + ("en_contacto" if name else "separados")] = g.error_mm.abs().mean()
    return {k: float(v) if v is not None else np.nan for k, v in out.items()}
