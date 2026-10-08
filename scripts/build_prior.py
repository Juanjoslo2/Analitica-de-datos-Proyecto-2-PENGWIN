"""build_prior.py.

Máscara previa de la segunda pasada [y4xul]: la región que un modelo predijo para cada caso SIN
haberlo visto. Se escribe junto al caché de 256 px:

    <cache-dir>/<caso>/prior_sem.npy     (Z, 256, 256) uint8  región predicha (0 fondo, 1 SA, 2 LI, 3 RI)
    <cache-dir>/<caso>/prior_boxes.npy   (Z, 3, 4) int16      caja de cada región en cada corte

Las fuentes son las predicciones fuera de fold de ``scripts/cv_predict.py``: el caso del fold k
viene del modelo entrenado sin el fold k. Así, al entrenar la segunda pasada, ningún caso lleva una
máscara producida por un modelo que lo tuvo en su entrenamiento, y la máscara tiene los errores
reales que tendrá en inferencia. **Nunca se usa el ground truth** para entrenar.

Uso:
    python scripts/build_prior.py --cache-dir /ruta/cache --folds-file reports/tuning/cv_folds.json \
        --oof "/ruta/oof/y4_fullres_role_f{fold}_e20"
    python scripts/build_prior.py --cache-dir /ruta/cache --oof-dir /ruta/oof/test --cases 001 003   # test
    python scripts/build_prior.py --cache-dir /ruta/cache --from-labels --cases 047   # SOLO pruebas de humo
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "03_src"))

from pengwin.data.slice_cache import load_case_cache  # noqa: E402
from pengwin.data.targets import region_of  # noqa: E402
from pengwin.data.two_pass import PRIOR_BOXES, PRIOR_SEM, prior_boxes  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--folds-file", type=Path, default=None)
    ap.add_argument("--oof", default=None, help="carpeta de predicciones con {fold}")
    ap.add_argument("--oof-dir", type=Path, default=None, help="una sola carpeta de predicciones (con --cases)")
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--from-labels", action="store_true",
                    help="usa la región del ground truth: SOLO para pruebas de humo del código, nunca para entrenar")
    args = ap.parse_args()

    fuentes = {}
    if args.from_labels:
        print("AVISO: máscara previa tomada del ground truth. Válido solo para probar el código.", flush=True)
        fuentes = {c: None for c in (args.cases or [])}
    elif args.oof_dir is not None:
        fuentes = {c: args.oof_dir / f"{c}.npz" for c in (args.cases or [p.stem for p in sorted(args.oof_dir.glob("*.npz"))])}
    else:
        folds = json.loads(args.folds_file.read_text(encoding="utf-8"))["folds"]
        for k, casos in enumerate(folds):
            for c in casos:
                fuentes[c] = Path(args.oof.format(fold=k)) / f"{c}.npz"
        if args.cases:
            fuentes = {c: f for c, f in fuentes.items() if c in set(args.cases)}

    for n, (cid, f) in enumerate(sorted(fuentes.items()), 1):
        d = args.cache_dir / cid
        if f is None:
            sem = region_of(np.asarray(load_case_cache(d)[1]))
        else:
            sem = np.load(f)["semantic"]
        b = prior_boxes(sem)
        np.save(d / PRIOR_SEM, sem.astype(np.uint8))
        np.save(d / PRIOR_BOXES, b)
        print(f"[{n}/{len(fuentes)}] {cid}: {int((b[..., 2] > b[..., 0]).sum())} recortes", flush=True)


if __name__ == "__main__":
    main()
