"""ensemble_oof.py.

Ensamble de predicciones fuera de fold (OOF) de varios modelos, para medir cuánto del error
viene de la VARIANZA entre corridas (semana 10, 2026-10-06).

La Fase 1 mostró que cambiar solo la semilla mueve el Dice por fragmento más que cualquier
hiperparámetro. Si promediar modelos sube la métrica, el problema es de varianza y se ataca
con técnicas de un solo modelo (EMA/SWA de pesos); si no la sube, el límite es de sesgo.

Por caso combina los ``<case>.npz`` de cada modelo (mismo fold):
    edge      media de P(borde)
    semantic  voto por mayoría de la región (empate -> el primer modelo de la lista)
y escribe ``<out>/<case>.npz`` con el mismo formato que ``cv_predict.py``, así lo lee
``tune_postprocess.py`` sin cambios.

Uso:
    python scripts/ensemble_oof.py --models cv_c00_base_e20,cv_c00b_seed1337_e20 \
        --oof-root /data/oof --folds-file reports/tuning/cv_folds.json --out ens2_base
"""

import argparse
import json
from pathlib import Path

import numpy as np


def combine(paths):
    zs = [np.load(p) for p in paths]
    edge = np.mean([z["edge"].astype(np.float32) for z in zs], axis=0)
    sem = np.stack([z["semantic"] for z in zs])                     # (M, Z, H, W)
    votos = np.stack([(sem == r).sum(0) for r in range(4)])          # (4, Z, H, W)
    # argmax devuelve el primer máximo: en empate gana la clase menor; se corrige con el primer modelo
    gana = votos.argmax(0).astype(np.uint8)
    empate = (votos == votos.max(0, keepdims=True)).sum(0) > 1
    gana[empate] = sem[0][empate]
    return {"semantic": gana, "edge": np.round(edge).astype(np.uint8)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="prefijos de corrida sin _f{k}, separados por coma")
    ap.add_argument("--oof-root", type=Path, required=True)
    ap.add_argument("--folds-file", type=Path, required=True)
    ap.add_argument("--out", required=True, help="prefijo de salida: se crea <oof-root>/<out>_f{k}")
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    folds = json.loads(args.folds_file.read_text(encoding="utf-8"))["folds"]
    for k, cases in enumerate(folds):
        out = args.oof_root / f"{args.out}_f{k}"
        out.mkdir(parents=True, exist_ok=True)
        for cid in cases:
            arr = combine([args.oof_root / f"{m}_f{k}" / f"{cid}.npz" for m in models])
            np.savez_compressed(out / f"{cid}.npz", **arr)
        (out / "metricas.json").write_text(json.dumps({"ensamble": models, "fold": k, "casos": cases}, indent=1),
                                           encoding="utf-8")
        print(f"fold {k}: {len(cases)} casos -> {out}", flush=True)


if __name__ == "__main__":
    main()
