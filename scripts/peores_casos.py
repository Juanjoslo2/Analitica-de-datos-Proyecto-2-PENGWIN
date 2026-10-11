"""peores_casos.py.

Galería de los 5 peores casos de test (Dice por fragmento) con un diagnóstico medido de cada
fragmento real que se perdió (§7 del enunciado: "galería de los cinco peores casos con hipótesis").

Para cada secundario real no recuperado (IoU < 0,5) se mira qué predijo el modelo en SUS vóxeles:
    absorbido   la mayoría quedó dentro del principal de su mismo hueso (se fusionó)
    partido     la mayoría quedó en otro fragmento predicho del mismo hueso (corte en otro sitio)
    otro hueso  la mayoría se asignó a otra región (error de la cabeza semántica)
    sin hueso   la mayoría quedó como fondo (no se segmentó)
Junto con su volumen y si está en contacto con el principal en el GT, eso sostiene la hipótesis.

Requiere las predicciones guardadas por evaluate_fragments.py --save-dir.

Uso:
    python scripts/peores_casos.py --pred-dir D:/PENGWIN/pred_tuned --data-dir D:/PENGWIN/01_data \
        --report reports/fragments/v2_last_test_tuned_superficie
Salida: reports/figures/semana11/peor_<caso>.png y reports/eval/peores_casos.json
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "03_src"))

from pengwin.data.data_loader import apply_bone_window, load_and_standardize_mha  # noqa: E402
from pengwin.measurement.separation import separation_table  # noqa: E402

BASE = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}
NOMBRE = {1: "SA", 2: "LI", 3: "RI"}


def colorear(lab):
    rgba = np.zeros(lab.shape + (4,), np.float32)
    for i in np.unique(lab):
        if i == 0:
            continue
        r = (int(i) - 1) // 10 + 1
        k = (int(i) - 1) % 10 + 1
        c = np.array(to_rgb(BASE[r]))
        c = c if k == 1 else c + (1 - c) * min(0.25 + 0.15 * (k - 2), 0.7)
        rgba[lab == i] = (*c, 0.55)
    return rgba


def diagnostico(gt, pred, i):
    m = gt == i
    r = (i - 1) // 10 + 1
    v = pred[m]
    vals, cnt = np.unique(v, return_counts=True)
    j = int(vals[np.argmax(cnt)])
    frac = float(cnt.max() / cnt.sum())
    if j == 0:
        return "sin hueso", frac
    rj = (j - 1) // 10 + 1
    if rj != r:
        return "otro hueso", frac
    return ("absorbido", frac) if j == (r - 1) * 10 + 1 else ("partido", frac)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True, help="prefijo del reporte de evaluate_fragments (sin extensión)")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()
    rep = json.loads(args.report.with_suffix(".json").read_text(encoding="utf-8"))
    frag = pd.read_csv(f"{args.report}_fragmentos.csv", dtype={"case_id": str})
    frag["case_id"] = frag.case_id.str.zfill(3)
    peores = sorted(rep["por_caso"], key=lambda c: rep["por_caso"][c]["dice_fragmento"])[:args.n]
    fig_dir = REPO / "reports" / "figures" / "semana11"
    fig_dir.mkdir(parents=True, exist_ok=True)
    salida = {}
    for cid in peores:
        img_path = next(args.data_dir.glob(f"PENGWIN_CT_train_images*/{cid}.mha"))
        hu, sp, _ = load_and_standardize_mha(img_path)
        gt, _, _ = load_and_standardize_mha(args.data_dir / "PENGWIN_CT_train_labels" / f"{cid}.mha", is_label=True)
        pred, _, _ = load_and_standardize_mha(args.pred_dir / f"{cid}_pred.mha", is_label=True)
        contacto = {r["label"]: r for r in separation_table(gt, sp)}
        f = frag[frag.case_id == cid]
        perdidos = []
        for _, row in f[(~f.es_principal) & (~f.recuperado)].iterrows():
            i = int(row.gt_label)
            tipo, frac = diagnostico(gt, pred, i)
            c = contacto.get(i, {})
            perdidos.append({"fragmento": f"{NOMBRE[(i - 1) // 10 + 1]}-{(i - 1) % 10 + 1}", "gt_label": i,
                             "volumen_cm3": round(float(row.volumen_cm3), 2), "iou": round(float(row.iou), 3),
                             "en_contacto": bool(c.get("in_contact", False)),
                             "dist_gt_mm": round(float(c["dist_mm"]), 2) if c else None,
                             "que_paso": tipo, "fraccion": round(frac, 2)})
        prin = f[f.es_principal]
        n_pred = int(len(np.unique(pred)) - 1)
        salida[cid] = {"dice_fragmento": rep["por_caso"][cid]["dice_fragmento"],
                       "iou_fragmento": rep["por_caso"][cid]["iou_fragmento"],
                       "n_gt": int(len(f)), "n_pred": n_pred,
                       "dice_principales": round(float(prin.dice.mean()), 3), "secundarios_perdidos": perdidos}
        # corte: el de mayor área del secundario perdido más grande (o del peor fragmento)
        objetivo = max(perdidos, key=lambda p: p["volumen_cm3"])["gt_label"] if perdidos else int(f.sort_values("dice").gt_label.iloc[0])
        z = int(np.argmax((gt == objetivo).sum((1, 2))))
        win = apply_bone_window(hu[z], 400, 1800)
        fig, ax = plt.subplots(1, 3, figsize=(13, 4.8))
        for a, (lab, t) in zip(ax, ((None, "CT (ventana ósea)"), (gt[z], f"referencia: {len(f)} fragmentos"),
                                    (pred[z], f"predicción: {n_pred} fragmentos"))):
            a.imshow(win, cmap="gray")
            if lab is not None:
                a.imshow(colorear(lab))
            a.contour(gt[z] == objetivo, levels=[0.5], colors="red", linewidths=1.0)
            a.set_title(t, fontsize=10)
            a.axis("off")
        txt = "; ".join(f"{p['fragmento']} {p['volumen_cm3']} cm³, {'en contacto' if p['en_contacto'] else 'separado'} → {p['que_paso']}"
                        for p in perdidos) or "sin secundarios perdidos: el error está en los principales"
        fig.suptitle(f"Paciente {cid} · Dice frag {salida[cid]['dice_fragmento']:.3f} · corte {z} (rojo: fragmento analizado)\n{txt}",
                     fontsize=9.5)
        fig.tight_layout()
        fig.savefig(fig_dir / f"peor_{cid}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(cid, json.dumps(salida[cid], ensure_ascii=False), flush=True)
    (REPO / "reports" / "eval" / "peores_casos.json").write_text(json.dumps(salida, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
