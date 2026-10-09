"""plot_informe_semana10.py.

Figuras de ``reports/informe_semana10.md``. Los datos están aquí escritos porque los crudos viven en
el laboratorio: son las tablas de ``reports/tuning/y4xul/RESUMEN.md`` y ``REFINAMIENTO.md``
(validación cruzada de 5 folds, 85 pacientes) y de ``reports/tuning/fase2/RESUMEN_FASE2.md``.

Uso:
    python scripts/plot_informe_semana10.py      # -> reports/figures/informe_s10/*.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "reports" / "figures" / "informe_s10"
SUPERFICIE, TINTA, TINTA2, MUDO, REJILLA = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e6e5e1"
AZUL, NARANJA, AZUL_CLARO = "#2a78d6", "#eb6834", "#9ec5f4"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": REJILLA,
                     "axes.labelcolor": TINTA2, "xtick.color": MUDO, "ytick.color": TINTA2,
                     "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE})


def _limpiar(ax, eje_x=True):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_visible(eje_x)
    ax.tick_params(length=0)


def evolucion():
    """Dice por fragmento en validación cruzada, etapa a etapa, contra el objetivo."""
    etapas = [("Separación solo por borde", 0.498), ("Semillas por distancia (edt)", 0.745),
              ("Sin transfer learning (control)", 0.752), ("+ resolución completa en el decodificador", 0.765),
              ("+ salida principal / secundario", 0.787), ("+ segunda pasada por hueso", 0.789),
              ("+ aumentación corregida, 40 épocas (final)", 0.809)]
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    ys = range(len(etapas))[::-1]
    for y, (nombre, v) in zip(ys, etapas):
        ax.barh(y, v, height=0.56, color=AZUL if "final" in nombre else AZUL_CLARO, edgecolor=SUPERFICIE, linewidth=2)
        ax.text(v + 0.008, y, f"{v:.3f}".replace(".", ","), va="center", color=TINTA, fontsize=10)
    ax.axvline(0.85, color=TINTA2, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(0.853, len(etapas) - 0.45, "objetivo 0,85", color=TINTA2, fontsize=9, va="bottom")
    ax.set_yticks(list(ys), [e[0] for e in etapas])
    ax.set_xlim(0.4, 0.92)
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9], ["0,4", "0,5", "0,6", "0,7", "0,8", "0,9"])
    ax.xaxis.grid(True, color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)
    _limpiar(ax, eje_x=False)
    ax.set_title("Dice por fragmento, etapa a etapa", loc="left", color=TINTA, fontsize=12, pad=10)
    ax.text(0, -0.2, "Validación cruzada de 5 folds, 85 pacientes. El eje empieza en 0,4.", transform=ax.transAxes,
            color=MUDO, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "01_evolucion_dice_fragmento.png", dpi=170)
    plt.close(fig)


def por_fold():
    """Control contra el modelo final (dos pasadas selectiva, role), fold a fold."""
    control = [0.6805, 0.7777, 0.7871, 0.7510, 0.7637]
    dos = [0.7778, 0.7992, 0.8112, 0.8334, 0.8209]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    xs = range(5)
    for x, a, b in zip(xs, control, dos):
        ax.plot([x, x], [a, b], color=REJILLA, linewidth=3, solid_capstyle="round", zorder=1)
    ax.scatter(xs, control, s=70, color=NARANJA, edgecolor=SUPERFICIE, linewidth=2, zorder=3, label="Control")
    ax.scatter(xs, dos, s=70, color=AZUL, edgecolor=SUPERFICIE, linewidth=2, zorder=3, label="Modelo final")
    for x, a, b in zip(xs, control, dos):
        ax.text(x + 0.11, b, f"{b:.3f}".replace(".", ","), va="center", color=TINTA, fontsize=9)
        ax.text(x + 0.11, a - (0.012 if b - a < 0.012 else 0), f"{a:.3f}".replace(".", ","), va="center", color=TINTA2, fontsize=9)
    ax.axhline(0.85, color=TINTA2, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(4.45, 0.853, "objetivo 0,85", color=TINTA2, fontsize=9, ha="right", va="bottom")
    ax.set_xticks(list(xs), [f"fold {k}" for k in xs])
    ax.set_xlim(-0.4, 4.6)
    ax.set_ylim(0.65, 0.88)
    ax.set_yticks([0.65, 0.70, 0.75, 0.80, 0.85], ["0,65", "0,70", "0,75", "0,80", "0,85"])
    ax.yaxis.grid(True, color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)
    _limpiar(ax, eje_x=False)
    ax.legend(loc="lower right", frameon=False, labelcolor=TINTA2, handletextpad=0.3, borderaxespad=0.2)
    ax.set_title("Dice por fragmento en cada fold", loc="left", color=TINTA, fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(OUT / "02_dice_por_fold.png", dpi=170)
    plt.close(fig)


def objetivos():
    """Las siete métricas del §5 frente a su objetivo (test, modelo final)."""
    filas = [("F1 clasificación", 0.994, 0.85), ("AUC clasificación", 0.999, 0.85), ("IoU de caja", 0.911, 0.65),
             ("mAP@0.50", 0.979, 0.65), ("mAP@[.50:.95]", 0.843, 0.40), ("Dice por fragmento", 0.833, 0.85),
             ("IoU por fragmento", 0.756, 0.70)]
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    ys = range(len(filas))[::-1]
    for y, (nombre, v, obj) in zip(ys, filas):
        cumple = v >= obj
        ax.barh(y, v, height=0.5, color=AZUL if cumple else NARANJA, edgecolor=SUPERFICIE, linewidth=2)
        ax.plot([obj, obj], [y - 0.36, y + 0.36], color=TINTA, linewidth=2)
        ax.text(1.03, y, f"{v:.3f}".replace(".", ",") + ("" if cumple else f"   no cumple (objetivo {obj:.2f})".replace(".", ",")),
                va="center", color=TINTA, fontsize=9.5)
    ax.set_yticks(list(ys), [f[0] for f in filas])
    ax.set_xlim(0, 1.42)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0,25", "0,50", "0,75", "1,00"])
    ax.xaxis.grid(True, color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)
    _limpiar(ax, eje_x=False)
    ax.set_title("Métricas del §5 frente a su objetivo", loc="left", color=TINTA, fontsize=12, pad=10)
    ax.text(0, -0.17, "Barra: valor medido en test (15 pacientes, modelo final). Marca negra: objetivo del enunciado.",
            transform=ax.transAxes, color=MUDO, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "03_metricas_vs_objetivo.png", dpi=170)
    plt.close(fig)


def parametros():
    """Reparto de los 2,93 M de parámetros entre los componentes."""
    comp = [("Backbone, bloque 4", 1484387), ("Detección", 608143), ("Backbone, bloque 3", 371555), ("Cuello", 197121),
            ("Segmentación", 157192), ("Backbone, bloque 2", 92544), ("Backbone, bloque 1", 19776), ("Clasificación", 771)]
    total = sum(v for _, v in comp)
    fig, ax = plt.subplots(figsize=(8.0, 3.7))
    ys = range(len(comp))[::-1]
    for y, (nombre, v) in zip(ys, comp):
        ax.barh(y, v / 1e6, height=0.56, color=AZUL, edgecolor=SUPERFICIE, linewidth=2)
        ax.text(v / 1e6 + 0.02, y, f"{v:,}".replace(",", " ") + f"  ({100 * v / total:.1f} %)".replace(".", ","),
                va="center", color=TINTA, fontsize=9)
    ax.set_yticks(list(ys), [c[0] for c in comp])
    ax.set_xlim(0, 2.05)
    ax.set_xticks([0, 0.5, 1.0, 1.5], ["0", "0,5 M", "1,0 M", "1,5 M"])
    ax.xaxis.grid(True, color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)
    _limpiar(ax, eje_x=False)
    ax.set_title(f"Parámetros por componente (total {total:,})".replace(",", " "), loc="left", color=TINTA, fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(OUT / "04_parametros.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    evolucion()
    por_fold()
    objetivos()
    parametros()
    print("figuras en", OUT)
