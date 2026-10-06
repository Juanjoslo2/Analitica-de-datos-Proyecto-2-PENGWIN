"""progreso.py.

Avance de las corridas y4xul en curso: una línea por corrida con la última época de
``runs/<nombre>/history.csv`` y, con ``--curva``, el Dice de la cabeza de borde época a época
(la señal directa de si el salto a resolución completa mejora la cobertura del borde).

Uso:
    python scripts/lab/progreso.py [--curva] [filtro]
"""

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--curva"]
    filtro = args[0] if args else ""
    curva = "--curva" in sys.argv
    filas, curvas = [], {}
    for hist in sorted((REPO / "runs").glob("y*_f*_e*/history.csv")):
        name = hist.parent.name
        if filtro not in name:
            continue
        h = pd.read_csv(hist)
        u = h.iloc[-1]
        filas.append({"corrida": name, "ep": len(h), "min/ep": h.seconds.mean() / 60, "loss": u["train_total"],
                      "val": u["val_total"], "borde": 1 - u["val_edge_dice"],
                      "papel_sec": u.get("role_dice_secundario", float("nan")), "mAP": u["mAP@[.50:.95]"],
                      "IoUcaja": u["IoU_promedio"], "DiceReg": u["dice_hueso"], "F1": u["cls_f1_macro"]})
        curvas[name] = (1 - h["val_edge_dice"]).round(3).tolist()
        if "role_dice_secundario" in h:
            curvas[name + " papel"] = h["role_dice_secundario"].round(3).tolist()
    if not filas:
        print("sin épocas terminadas")
        return
    pd.set_option("display.width", 220)
    print(pd.DataFrame(filas).round(4).to_string(index=False))
    if curva:
        print()
        for k, v in curvas.items():
            print(f"{k:28s} {' '.join(f'{x:.3f}' for x in v)}")


if __name__ == "__main__":
    main()
