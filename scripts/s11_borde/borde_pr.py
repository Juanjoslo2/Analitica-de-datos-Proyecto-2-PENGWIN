"""Semana 11 · precisión y recall del borde predicho en un fold, contra ``edge.npy`` (dilatación 2).

Es la misma medición que se hizo en el cribado de la Fase 1 (``reports/tuning/borde_cv_*.json``), para
ver si un candidato cambia la COBERTURA del borde, que es el cuello de botella medido (recall ≈ 0,19).
Uso (dentro del contenedor, después de ``cv_predict.py``):
    python scripts/s11_borde/borde_pr.py <corrida> <fold>   ->  reports/tuning/borde_<corrida>.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "03_src")
from pengwin.data.slice_cache import load_case_cache, load_edge_cache  # noqa: E402

NAME, FOLD = sys.argv[1], int(sys.argv[2])
casos = json.loads(Path("reports/tuning/cv_folds.json").read_text(encoding="utf-8"))["folds"][FOLD]
thr = [0.1, 0.2, 0.3, 0.5]
tp = {t: 0 for t in thr}
fp, fn = dict(tp), dict(tp)
pm = []
for cid in casos:
    e = np.load(f"/data/oof/{NAME}/{cid}.npz")["edge"].astype(np.float32) / 255.0
    d = Path("/data/data_processed") / cid
    hueso = np.asarray(load_case_cache(d)[1]) > 0
    g = (np.asarray(load_edge_cache(d, name="edge.npy")) > 0) & hueso
    if g.any():
        pm.append(float(np.mean(e[g])))
    for t in thr:
        p = (e >= t) & hueso
        tp[t] += int((p & g).sum())
        fp[t] += int((p & ~g).sum())
        fn[t] += int((~p & g).sum())
res = {"corrida": NAME, "fold": FOLD, "referencias": {"edge.npy": {
    "P_media_en_borde_real": round(float(np.mean(pm)), 4),
    "por_umbral": {str(t): {"precision": round(tp[t] / max(tp[t] + fp[t], 1), 4),
                            "recall": round(tp[t] / max(tp[t] + fn[t], 1), 4)} for t in thr}}}}
Path(f"reports/tuning/borde_{NAME}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1))
