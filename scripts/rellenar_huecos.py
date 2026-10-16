"""Cierre de los huecos de hueso delgado: ¿cambia las métricas? (03_semana11 §1.2).

La grilla de 256 px abre huecos en el ala ilíaca que la etiqueta nativa no tiene (§1.1) y el modelo
los copia. ``postprocess.fill_holes_mm`` los cierra con un cierre morfológico por fragmento
(``instances.fill_thin_holes``). Este script mide qué hace eso con las mismas predicciones fuera de
fold con las que se eligió ``configs/tuned.yaml`` (85 casos de train+val, 5 folds; el test no se
toca):

- métricas por fragmento contra la etiqueta NATIVA (como ``evaluate_fragments.py`` en el test) y,
  aparte, contra la de 256 px del caché (como ``tune_postprocess.py``), que tiene los huecos;
- túneles del fragmento principal de cada hueso: predicción (grilla) contra etiqueta nativa.

Regla, fijada antes de correr: se adopta el radio más pequeño que (1) no baje el Dice ni el IoU por
fragmento en la grilla nativa más de 0,002 en la media de los folds y (2) deje los túneles de más de
los principales coxales (suma de max(pred − nativo, 0)) en la mitad o menos que sin cierre. Si con
ese radio el sacro queda con menos túneles que la etiqueta nativa en más principales que sin cierre
(forámenes reales cerrados), se aplica solo a los coxales. Si ningún radio cumple, no se adopta.

    python scripts/rellenar_huecos.py --cache-dir D:/PENGWIN/data_processed --data-dir D:/PENGWIN/01_data \
        --oof D:/PENGWIN/oof/tta_base_f{fold}
Salida: reports/eval/relleno_huecos.json y relleno_huecos_casos.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "03_src"))
sys.path.insert(0, str(REPO / "scripts"))

from huecos_remuestreo import tuneles  # noqa: E402
from pengwin.data.data_loader import load_and_standardize_mha  # noqa: E402
from pengwin.data.slice_cache import load_case_cache  # noqa: E402
from pengwin.evaluation.fragment_metrics import distance_comparison, match_fragments, summarize  # noqa: E402
from pengwin.inference.volume import to_native  # noqa: E402
from pengwin.postprocess.instances import fill_thin_holes, instance_kwargs, resolve_postprocess, separate_instances  # noqa: E402
from pengwin.utils.config import load_config  # noqa: E402

PRINCIPALES = {1: "SA", 11: "LI", 21: "RI"}
REGIONES = {"todas": (1, 2, 3), "coxales": (2, 3)}
METRICAS = ["dice_fragmento", "iou_fragmento", "dice_principal", "dice_secundario", "recuperados_secundarios_%",
            "mediana_error_mm"]


def caso(tarea):
    cid, fold, oof_dir, cache_dir, labels_dir, variantes, pp = tarea
    t0 = time.time()
    z = np.load(Path(oof_dir) / f"{cid}.npz")
    sem, edge = z["semantic"], z["edge"].astype(np.float32) / 255.0
    _, gt_g, meta = load_case_cache(Path(cache_dir) / cid)
    gt_g = np.asarray(gt_g)
    sp = (meta["spacing_zyx"][0], meta["pixel_mm"], meta["pixel_mm"])
    gt_n, sp_n, _ = load_and_standardize_mha(Path(labels_dir) / f"{cid}.mha", is_label=True)
    kw = instance_kwargs(pp)
    kw.update(fill_holes_mm=0.0)
    lab0 = separate_instances(sem, edge, sp, **kw)
    tun_nat = {f: tuneles(gt_n == f) for f in PRINCIPALES if (gt_n == f).any()}
    vox_cm3 = float(np.prod(sp)) / 1000.0
    out = []
    for nombre, radio, regiones in variantes:
        lab = fill_thin_holes(lab0, sp, radio, REGIONES[regiones]) if radio > 0 else lab0
        nat = to_native(lab, meta, order=0)
        fr = match_fragments(gt_n, nat, sp_n)
        dr = distance_comparison(gt_n, nat, sp_n, fr)
        fr_g = match_fragments(gt_g, lab, sp)
        dr_g = distance_comparison(gt_g, lab, sp, fr_g)
        tun = {PRINCIPALES[f]: (tun_nat[f], tuneles(lab == f) if (lab == f).any() else 0) for f in tun_nat}
        out.append({"caso": cid, "fold": fold, "variante": nombre, "fr": fr, "dr": dr, "fr_g": fr_g, "dr_g": dr_g,
                    "tuneles": tun, "añadido_cm3": float((lab != lab0).sum()) * vox_cm3})
    print(f"{cid} (fold {fold}) {time.time() - t0:.0f} s  " + "  ".join(
        f"{o['variante']}: LI {o['tuneles'].get('LI', ('-', '-'))[1]}/RI {o['tuneles'].get('RI', ('-', '-'))[1]}"
        for o in out) + f"  · nativo LI {tun_nat.get(11, '-')}/RI {tun_nat.get(21, '-')}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--oof", required=True, help="carpeta con {fold}, p. ej. D:/PENGWIN/oof/tta_base_f{fold}")
    ap.add_argument("--folds-file", type=Path, default=REPO / "reports" / "tuning" / "cv_folds.json")
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "tuned.yaml")
    ap.add_argument("--radios", default="1.5,2,3", help="radios en mm (todas las regiones)")
    ap.add_argument("--coxales", default="2,3", help="radios que se prueban también solo en los coxales")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    pp = resolve_postprocess(load_config(args.config).get("postprocess"))
    variantes = [("sin cierre", 0.0, "todas")]
    variantes += [(f"r {float(r):g} mm", float(r), "todas") for r in args.radios.split(",") if r]
    variantes += [(f"r {float(r):g} mm coxales", float(r), "coxales") for r in args.coxales.split(",") if r]
    folds = json.loads(args.folds_file.read_text(encoding="utf-8"))["folds"]
    tareas = [(c, k, args.oof.format(fold=k), str(args.cache_dir), str(args.data_dir / "PENGWIN_CT_train_labels"),
               variantes, pp) for k, f in enumerate(folds) for c in f]
    print(f"posproceso {pp['instance_method']} · {len(variantes)} variantes × {len(tareas)} casos", flush=True)
    filas = []
    with Pool(args.jobs) as pool:
        for res in pool.imap_unordered(caso, tareas):
            filas += res

    # Métricas por fold juntando los fragmentos del fold (igual que tune_postprocess.py)
    resumen = []
    for (var, fold), g in pd.DataFrame(filas).groupby(["variante", "fold"]):
        nat = summarize(sum(g.fr, []), sum(g.dr, []))
        grid = summarize(sum(g.fr_g, []), sum(g.dr_g, []))
        resumen.append({"variante": var, "fold": fold, **{f"{m}": nat[m] for m in METRICAS},
                        **{f"{m}_256": grid[m] for m in ("dice_fragmento", "iou_fragmento")},
                        "añadido_cm3_por_caso": g["añadido_cm3"].mean()})
    rf = pd.DataFrame(resumen)
    base = rf[rf.variante == "sin cierre"].set_index("fold")
    tabla = {}
    for var, g in rf.groupby("variante"):
        g = g.set_index("fold")
        d = {m: g[m] - base[m] for m in ("dice_fragmento", "iou_fragmento", "dice_fragmento_256", "iou_fragmento_256")}
        tabla[var] = {**{m: round(float(g[m].mean()), 4) for m in g.columns if m != "variante"},
                      **{f"delta_{m}": round(float(v.mean()), 4) for m, v in d.items()},
                      **{f"delta_{m}_desv": round(float(v.std()), 4) for m, v in d.items()},
                      "folds_dice_a_favor": int((d["dice_fragmento"] > 0).sum())}

    tun = []
    for f in filas:
        for hueso, (n, p) in f["tuneles"].items():
            tun.append({"variante": f["variante"], "caso": f["caso"], "hueso": hueso, "nativo": n, "pred": p})
    tt = pd.DataFrame(tun)
    for var, g in tt.groupby("variante"):
        for hueso, h in g.groupby("hueso"):
            tabla[var][f"tuneles_{hueso}"] = {"nativo": int(h.nativo.sum()), "pred": int(h.pred.sum()),
                                              "de_mas": int((h.pred - h.nativo).clip(lower=0).sum()),
                                              "de_menos": int((h.nativo - h.pred).clip(lower=0).sum()),
                                              "principales_con_de_mas": int((h.pred > h.nativo).sum()),
                                              "principales_con_de_menos": int((h.pred < h.nativo).sum()),
                                              "principales": len(h)}

    out_dir = REPO / "reports" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{k: v for k, v in f.items() if k not in ("fr", "dr", "fr_g", "dr_g", "tuneles")} for f in filas]) \
        .merge(tt.pivot_table(index=["variante", "caso"], columns="hueso", values=["nativo", "pred"]).reset_index()
               .pipe(lambda d: d.set_axis(["variante", "caso"] + [f"tuneles_{a}_{b}" for a, b in d.columns[2:]], axis=1)),
               on=["variante", "caso"]).to_csv(out_dir / f"relleno_huecos{args.tag}_casos.csv", index=False)
    (out_dir / f"relleno_huecos{args.tag}.json").write_text(json.dumps(
        {"que_mide": __doc__.split("\n\n")[1], "oof": args.oof, "posproceso": pp, "por_fold": resumen, "variantes": tabla},
        indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    cols = ["dice_fragmento", "delta_dice_fragmento", "delta_dice_fragmento_desv", "folds_dice_a_favor", "iou_fragmento",
            "delta_iou_fragmento", "delta_dice_fragmento_256", "añadido_cm3_por_caso"]
    print(pd.DataFrame(tabla).T[cols].to_string())
    for var, t in tabla.items():
        print(var, {h: t.get(f"tuneles_{h}") for h in ("SA", "LI", "RI")})


if __name__ == "__main__":
    main()
