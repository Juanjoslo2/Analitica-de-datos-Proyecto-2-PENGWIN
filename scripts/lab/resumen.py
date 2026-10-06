"""resumen.py.

Tabla compacta del cribado y4xul: una fila por corrida con las métricas de la última época y el
mejor posproceso de cada método sobre sus predicciones fuera de fold.

Uso:
    python scripts/lab/resumen.py            # todas las corridas de reports/tuning/y4xul
    python scripts/lab/resumen.py _f0_e12    # solo las que contienen ese texto
Escribe además reports/tuning/y4xul/RESUMEN.csv (versionable).
"""

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
TUN = REPO / "reports" / "tuning" / "y4xul"
PARAMS = {"edt": ["seed_depth_mm", "edge_threshold"], "role": ["role_seed_depth_mm", "role_threshold", "role_smooth_mm"]}


def mejor(csv: Path, metodo: str) -> dict:
    """Mejor combinación por Dice de fragmento (desempate: IoU), promediando los folds presentes."""
    d = pd.read_csv(csv)
    g = d.groupby("combo").mean(numeric_only=True).sort_values(["dice_fragmento", "iou_fragmento"], ascending=False)
    b = g.iloc[0]
    return {f"{metodo}_combo": "/".join(f"{b[p]:g}" for p in PARAMS[metodo]),
            f"{metodo}_dice": b["dice_fragmento"], f"{metodo}_iou": b["iou_fragmento"],
            f"{metodo}_princ": b["dice_principal"], f"{metodo}_sec": b["dice_secundario"],
            f"{metodo}_rec": b["recuperados_secundarios_%"], f"{metodo}_mm": b["mediana_error_mm"]}


def main() -> None:
    filtro = sys.argv[1] if len(sys.argv) > 1 else ""
    filas = []
    for hist in sorted((REPO / "reports" / "train").glob("y*_f*_e*_history.csv")):
        name = hist.name.replace("_history.csv", "")
        if filtro not in name:
            continue
        h = pd.read_csv(hist)
        u = h.iloc[-1]
        fila = {"corrida": name, "ep": len(h), "s/ep": h.seconds.mean(), "edge_dice": 1 - u.get("val_edge_dice", float("nan")),
                "papel_sec": u.get("role_dice_secundario", float("nan")), "mAP": u["mAP@[.50:.95]"],
                "IoUcaja": u["IoU_promedio"], "DiceReg": u["dice_hueso"], "F1": u["cls_f1_macro"], "AUC": u["cls_auc_macro"]}
        for metodo in PARAMS:
            csv = TUN / f"{name}_{metodo}.csv"
            if csv.exists():
                fila.update(mejor(csv, metodo))
        filas.append(fila)
    if not filas:
        print("sin corridas terminadas")
        return
    df = pd.DataFrame(filas)
    df.to_csv(TUN / "RESUMEN.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    a = [c for c in ["corrida", "ep", "s/ep", "edge_dice", "papel_sec", "mAP", "IoUcaja", "DiceReg", "F1", "AUC"] if c in df]
    print(df[a].round(4).to_string(index=False))
    for metodo in PARAMS:
        cols = [c for c in df.columns if c.startswith(metodo + "_")]
        if cols:
            print(f"\n-- posproceso {metodo} (mejor combinación de cada corrida)")
            print(df[df[f"{metodo}_dice"].notna()][["corrida"] + cols].round(4).to_string(index=False))
    orac = TUN / "oraculo_role.json"
    if orac.exists():
        r = pd.DataFrame(json.loads(orac.read_text(encoding="utf-8"))["resumen"])
        print("\n-- oráculo role (GT):")
        print(r[["drop", "role_smooth_mm", "role_seed_depth_mm", "dice_fragmento", "iou_fragmento", "dice_principal", "dice_secundario",
                 "recuperados_secundarios_%"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
