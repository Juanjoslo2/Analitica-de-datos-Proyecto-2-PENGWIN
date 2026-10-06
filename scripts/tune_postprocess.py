"""tune_postprocess.py.

Ajuste del posproceso de fragmentos con validación cruzada, usando las predicciones fuera de
fold (``cv_predict.py``). No reentrena: cada combinación de parámetros se evalúa sobre las
mismas predicciones, en la grilla del modelo (GT = ``label.npy`` del caché).

Para cada combinación y cada fold se calculan las métricas por fragmento juntando todos los
fragmentos del fold (``fragment_metrics.summarize``). El resumen reporta la media y la desviación
entre folds. El test no se toca.

Uso (grilla cartesiana; los parámetros no listados quedan en ``--base``):
    python scripts/tune_postprocess.py --name pp_etapa1 --cache-dir /data/data_processed \
        --oof /data/oof/cv_base_f{fold} --folds-file reports/tuning/cv_folds.json \
        --grid "seed_depth_mm=3,4,5,6,7;edge_threshold=0.1,0.2,0.3,0.5" \
        --base "method=edt;seed_min_cm3=0.02;edge_weight=5"
Salida: reports/tuning/<name>.csv (una fila por combinación y fold) y <name>.json (media ± desv.).
"""

import argparse
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "03_src"))

from pengwin.data.slice_cache import load_case_cache  # noqa: E402
from pengwin.data.targets import DIST_MM_PER_LEVEL  # noqa: E402
from pengwin.evaluation.fragment_metrics import distance_comparison, match_fragments, summarize  # noqa: E402
from pengwin.postprocess.instances import separate_instances  # noqa: E402

METRICS = ["dice_fragmento", "iou_fragmento", "dice_principal", "dice_secundario", "recuperados_secundarios_%",
           "mae_distancia_mm", "mediana_error_mm"]


def parse_spec(spec: str) -> dict:
    out = {}
    for part in filter(None, spec.split(";")):
        k, v = part.split("=")
        vals = [x if k == "method" else float(x) for x in v.split(",")]
        out[k.strip()] = vals
    return out


def run_case(task):
    """Evalúa todas las combinaciones en un caso (se carga una sola vez)."""
    case, fold, oof_dir, cache_dir, combos = task
    z = np.load(Path(oof_dir) / f"{case}.npz")
    sem, edge = z["semantic"], z["edge"].astype(np.float32) / 255.0
    # P(núcleo) de los modelos con la salida core3 (F2B1); los demás .npz no la traen
    core = z["core"].astype(np.float32) / 255.0 if "core" in z.files else None
    # Distancia predicha a la fractura en mm (F2B2); los .npz anteriores no la traen y los
    # métodos de siempre no la usan: puramente aditivo
    dist = z["dist"].astype(np.float32) * DIST_MM_PER_LEVEL if "dist" in z.files else None
    role = z["role"].astype(np.float32) / 255.0 if "role" in z.files else None      # [y4xul]
    _, label, meta = load_case_cache(Path(cache_dir) / case)
    gt = np.asarray(label)
    sp = (meta["spacing_zyx"][0], meta["pixel_mm"], meta["pixel_mm"])
    out = []
    for i, c in enumerate(combos):
        lab = separate_instances(sem, edge, sp, c.get("edge_threshold", 0.2), c.get("min_fragment_cm3", 0.1),
                                 method=c.get("method", "edt"), seed_depth_mm=c.get("seed_depth_mm", 5.0),
                                 seed_min_cm3=c.get("seed_min_cm3", 0.02), edge_weight=c.get("edge_weight", 5.0),
                                 core=core, core_threshold=c.get("core_threshold", 0.5),
                                 core_seed_depth_mm=c.get("core_seed_depth_mm", 0.0),
                                 hmax_h_mm=c.get("hmax_h_mm", 1.5),
                                 dist=dist, dist_max_mm=c.get("dist_max_mm", 8.0),
                                 role=role, role_threshold=c.get("role_threshold", 0.5),
                                 role_seed_depth_mm=c.get("role_seed_depth_mm", 1.5),
                                 role_smooth_mm=c.get("role_smooth_mm", 0.0))
        fr = match_fragments(gt, lab, sp)
        dr = distance_comparison(gt, lab, sp, fr)
        out.append((i, fold, fr, dr))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--oof", required=True, help="carpeta de predicciones con {fold}, p. ej. /data/oof/cv_base_f{fold}")
    ap.add_argument("--folds-file", type=Path, required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--base", default="method=edt;edge_threshold=0.2;seed_depth_mm=5;seed_min_cm3=0.02;edge_weight=5")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--only-folds", default=None, help="p. ej. 0,1,2: evalúa solo esos folds")
    args = ap.parse_args()

    base = {k: v[0] for k, v in parse_spec(args.base).items()}
    grid = parse_spec(args.grid)
    keys = list(grid)
    combos = [{**base, **dict(zip(keys, vals))} for vals in itertools.product(*(grid[k] for k in keys))]
    folds = json.loads(args.folds_file.read_text(encoding="utf-8"))["folds"]
    use = set(range(len(folds))) if not args.only_folds else {int(x) for x in args.only_folds.split(",")}
    tasks = [(c, k, args.oof.format(fold=k), str(args.cache_dir), combos) for k, f in enumerate(folds) if k in use for c in f]
    print(f"{len(combos)} combinaciones × {len(tasks)} casos", flush=True)

    acc = {}
    with Pool(args.jobs) as pool:
        for res in pool.imap_unordered(run_case, tasks):
            for i, fold, fr, dr in res:
                a = acc.setdefault((i, fold), ([], []))
                a[0].extend(fr)
                a[1].extend(dr)
    rows = []
    for (i, fold), (fr, dr) in sorted(acc.items()):
        s = summarize(fr, dr)
        rows.append({"combo": i, "fold": fold, **{k: combos[i][k] for k in combos[i]}, **{m: s.get(m) for m in METRICS}})
    df = pd.DataFrame(rows)
    out_dir = REPO / "reports" / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{args.name}.csv", index=False)

    params = list(combos[0])
    agg = df.groupby("combo").agg(**{f"{m}_media": (m, "mean") for m in METRICS}, **{f"{m}_desv": (m, "std") for m in METRICS})
    agg = pd.concat([pd.DataFrame(combos)[params], agg], axis=1).sort_values("dice_fragmento_media", ascending=False)
    (out_dir / f"{args.name}.json").write_text(json.dumps({"grid": args.grid, "base": args.base, "folds": sorted(use),
                                                           "ranking": agg.to_dict(orient="records")}, indent=1),
                                               encoding="utf-8")
    show = params[1:] + ["dice_fragmento_media", "dice_fragmento_desv", "iou_fragmento_media",
                         "recuperados_secundarios_%_media", "mediana_error_mm_media"]
    print(agg[[c for c in show if c in agg]].head(12).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
