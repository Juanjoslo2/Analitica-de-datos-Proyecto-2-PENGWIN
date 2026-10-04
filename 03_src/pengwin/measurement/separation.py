"""separation.py.

Distancia de separación de cada fragmento conminuto respecto al principal de su hueso
[enunciado §3.3, EDA §7].

    d = mín sobre los vóxeles del fragmento de  EDT(complemento del principal)
con ``scipy.ndimage.distance_transform_edt(..., sampling=spacing)`` y el spacing del header
en orden (z, y, x). La EDT mide de centro a centro de vóxel: dos fragmentos que se tocan dan
≈ 1 vóxel, no 0. Por eso se marca "en contacto" cuando d ≤ diagonal del vóxel.

Se aplica igual a la predicción y al ground truth, para medir cuánto error transfiere la
segmentación a la distancia.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy import ndimage as ndi

REGION_NAMES = {1: "SA", 2: "LI", 3: "RI"}


def separation_table(labels: np.ndarray, spacing_zyx: Sequence[float]) -> List[Dict]:
    """Una fila por fragmento secundario: región, etiqueta, volumen, distancia (mm), en contacto."""
    spacing_zyx = tuple(float(s) for s in spacing_zyx)
    diag = float(np.sqrt(np.sum(np.square(spacing_zyx))))
    vox_cm3 = float(np.prod(spacing_zyx)) / 1000.0
    counts = np.bincount(labels.ravel(), minlength=31)
    rows = []
    for r in (1, 2, 3):
        main_id = (r - 1) * 10 + 1
        ids = [i for i in range((r - 1) * 10 + 1, r * 10 + 1) if counts[i] > 0]
        if counts[main_id] == 0 or len(ids) < 2:
            continue
        region = (labels >= (r - 1) * 10 + 1) & (labels <= r * 10)
        obj = ndi.find_objects(region.astype(np.uint8))[0]
        sl = tuple(slice(max(s.start - 2, 0), min(s.stop + 2, n)) for s, n in zip(obj, labels.shape))
        lab = labels[sl]
        dt = ndi.distance_transform_edt(lab != main_id, sampling=spacing_zyx)
        for i in ids:
            if i == main_id:
                continue
            d = float(dt[lab == i].min())
            rows.append({"region": REGION_NAMES[r], "label": i, "volume_cm3": counts[i] * vox_cm3,
                         "dist_mm": d, "in_contact": d <= diag + 1e-6})
    return rows
