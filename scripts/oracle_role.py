"""oracle_role.py.

Techo del método ``role`` ANTES de gastar GPU [y4xul]: se le da al posproceso el papel real de cada
vóxel (principal / secundario, del ground truth) y se mide el Dice por fragmento que saldría si la
red lo predijera sin error. Mismo criterio que los oráculos de la Fase 2
(``reports/tuning/fase2/RESUMEN_FASE2.md`` §3): si el techo no supera el objetivo, no se entrena.

Además mide cuánto aguanta el método un papel imperfecto: en una fracción ``--drop`` de los cortes
con secundario, el papel se borra (todo "principal"), que es el fallo esperable de una red 2D que
decide corte a corte.

Uso:
    python scripts/oracle_role.py --cache-dir /ruta/cache --folds-file reports/tuning/cv_folds.json
Salida: reports/tuning/y4xul/oraculo_role.json y una tabla por pantalla (media ± desv. entre folds).
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
from pengwin.data.targets import region_of, role_target  # noqa: E402
from pengwin.evaluation.fragment_metrics import distance_comparison, match_fragments, summarize  # noqa: E402
from pengwin.postprocess.instances import separate_instances  # noqa: E402

METRICS = ["dice_fragmento", "iou_fragmento", "dice_principal", "dice_secundario", "recuperados_secundarios_%",
           "dice_<5 cm3", "mediana_error_mm"]


def run_case(task):
    case, fold, cache_dir, combos, seed = task
    _, label, meta = load_case_cache(Path(cache_dir) / case)
    gt = np.asarray(label)
    sp = (meta["spacing_zyx"][0], meta["pixel_mm"], meta["pixel_mm"])
    sem = region_of(gt)
    role_gt = (role_target(gt) == 2).astype(np.float32)
    sin_borde = np.zeros(gt.shape, np.float32)
    con_sec = np.nonzero(role_gt.reshape(len(gt), -1).any(1))[0]
    out = []
    for i, (depth, drop, smooth) in enumerate(combos):
        role = role_gt
        if drop > 0 and len(con_sec):
            rng = np.random.default_rng((seed, int(case), int(round(drop * 100))))
            borrar = rng.choice(con_sec, size=int(round(drop * len(con_sec))), replace=False)
            role = role_gt.copy()
            role[borrar] = 0.0
        lab = separate_instances(sem, sin_borde, sp, method="role", role=role, role_seed_depth_mm=depth,
                                 role_smooth_mm=smooth)
        fr = match_fragments(gt, lab, sp)
        out.append((i, fold, fr, distance_comparison(gt, lab, sp, fr)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--folds-file", type=Path, default=REPO / "reports" / "tuning" / "cv_folds.json")
    ap.add_argument("--depths", default="1,2", help="role_seed_depth_mm a probar")
    ap.add_argument("--drop", default="0,0.2", help="fracción de cortes con secundario cuyo papel se borra")
    ap.add_argument("--smooth", default="0,1,2,3", help="role_smooth_mm a probar")
    ap.add_argument("--only-folds", default=None)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    combos = list(itertools.product(*([float(x) for x in a.split(",")] for a in (args.depths, args.drop, args.smooth))))
    folds = json.loads(args.folds_file.read_text(encoding="utf-8"))["folds"]
    use = set(range(len(folds))) if not args.only_folds else {int(x) for x in args.only_folds.split(",")}
    tasks = [(c, k, str(args.cache_dir), combos, args.seed) for k, f in enumerate(folds) if k in use for c in f]
    print(f"{len(combos)} combinaciones x {len(tasks)} casos", flush=True)

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
        rows.append({"role_seed_depth_mm": combos[i][0], "drop": combos[i][1], "role_smooth_mm": combos[i][2], "fold": fold,
                     **{m: s.get(m) for m in METRICS}})
    df = pd.DataFrame(rows)
    agg = df.groupby(["drop", "role_smooth_mm", "role_seed_depth_mm"]).agg(**{m: (m, "mean") for m in METRICS},
                                                         dice_fragmento_desv=("dice_fragmento", "std")).reset_index()
    out_dir = REPO / "reports" / "tuning" / "y4xul"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "oraculo_role.json").write_text(json.dumps({"folds": sorted(use), "por_fold": rows,
                                                           "resumen": agg.to_dict(orient="records")}, indent=1),
                                               encoding="utf-8")
    pd.set_option("display.width", 200)
    print(agg.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
