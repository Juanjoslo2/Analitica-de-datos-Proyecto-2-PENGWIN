"""plot_training_curves.py.

Curvas de entrenamiento de varios modelos en paralelo (ablación): métricas de val y
pérdidas por época, leídas de runs/<nombre>/history.csv.

Uso:
    python scripts/plot_training_curves.py                         # base, sin_cbam, sin_tl
    python scripts/plot_training_curves.py --runs base sin_cbam --out reports/figures/x.png

Paleta categórica validada (azul, naranja, aqua: ΔE CVD ≥ 9,2 entre todos los pares). Además
cada modelo tiene su propio estilo de línea, para que se distinga en blanco y negro (informe
IEEE impreso). El aqua queda bajo 3:1 de contraste sobre blanco: la identidad la dan también
el estilo de línea y la leyenda con texto.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

NAMES = {"base": "Base (CBAM + TL)", "sin_cbam": "Sin CBAM", "sin_tl": "Sin TL (desde cero)"}
STYLE = [("#2a78d6", "-"), ("#eb6834", "--"), ("#1baf7a", ":")]
PANELS = [
    ("mAP@[.50:.95]", "mAP@[.50:.95] (val)", "max"),
    ("mAP@0.50", "mAP@0.5 (val)", "max"),
    ("dice_hueso", "Dice por región (val)", "max"),
    ("cls_f1_macro", "F1 clasificación (val)", "max"),
    ("train_total", "Pérdida de entrenamiento", "min"),
    ("val_total", "Pérdida de validación", "min"),
]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["base", "sin_cbam", "sin_tl"])
    ap.add_argument("--out", type=Path, default=REPO / "reports" / "figures" / "semana9" / "curvas_entrenamiento.png")
    args = ap.parse_args()
    hist = {r: pd.read_csv(REPO / "runs" / r / "history.csv") for r in args.runs}

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK})
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6))
    for ax, (col, title, best) in zip(axes.flat, PANELS):
        for (run, h), (color, ls) in zip(hist.items(), STYLE):
            ax.plot(h["epoch"], h[col], color=color, ls=ls, lw=2, label=NAMES.get(run, run))
            i = h[col].idxmax() if best == "max" else h[col].idxmin()
            ax.plot(h["epoch"][i], h[col][i], "o", ms=6, color=color, mec="white", mew=1.5, zorder=3)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("época")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.005, "Punto = mejor época de cada modelo en esa métrica. Val = 15 pacientes de validación.",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(args.out)


if __name__ == "__main__":
    main()
