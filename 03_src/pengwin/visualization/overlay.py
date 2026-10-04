"""overlay.py.

Superposición 2D de cajas y máscaras sobre un corte (base del visualizador 2 del dashboard).
Un color por macro-hueso [enunciado §3.2]; la misma paleta se usará en la reconstrucción 3D.
Devuelve figuras de matplotlib sin llamar a ``plt.show()`` (reutilizable en Streamlit).
"""

from __future__ import annotations

from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle

from pengwin.data.targets import REGION_NAMES

# Misma paleta que el EDA y el notebook de sustentación (validada para daltonismo):
# SA azul, coxal izq. naranja, coxal der. aqua. Un hueso tiene el mismo color en todo el proyecto.
REGION_COLORS = {name: to_rgb(h) for name, h in {"SA": "#2a78d6", "LI": "#eb6834", "RI": "#1baf7a"}.items()}


def colorize(semantic: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """(H, W) 0..3 -> RGBA con transparencia en el fondo."""
    rgba = np.zeros(semantic.shape + (4,), np.float32)
    for k, name in enumerate(REGION_NAMES, start=1):
        rgba[semantic == k] = (*REGION_COLORS[name], alpha)
    return rgba


def draw_slice(ax, image: np.ndarray, semantic: np.ndarray | None = None, boxes: Dict | None = None,
               gt_boxes: np.ndarray | None = None, gt_present: np.ndarray | None = None, title: str = "") -> None:
    """``image`` (H, W) en [0, 1]; ``boxes`` = salida de ``grid.decode`` en numpy; GT punteada."""
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    if semantic is not None:
        ax.imshow(colorize(semantic))
    if gt_boxes is not None:
        for k, name in enumerate(REGION_NAMES):
            if gt_present is not None and not gt_present[k]:
                continue
            x0, y0, x1, y1 = gt_boxes[k]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ls="--", lw=1, ec="white"))
    if boxes is not None:
        for b, s, lab in zip(boxes["boxes"], boxes["scores"], boxes["labels"]):
            name = REGION_NAMES[int(lab)]
            ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, lw=1.6, ec=REGION_COLORS[name]))
            ax.text(b[0], b[1] - 2, f"{name} {s:.2f}", color=REGION_COLORS[name], fontsize=7, va="bottom")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def prediction_grid(samples: Sequence[Dict], ncols: int = 4):
    """Cada muestra: image, pred_semantic, pred_boxes, gt_semantic, gt_boxes, gt_present, title.

    Fila superior de cada par: GT; inferior: predicción, para comparar a simple vista.
    """
    n = len(samples)
    nrows = 2 * int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    for i, s in enumerate(samples):
        r, c = 2 * (i // ncols), i % ncols
        draw_slice(axes[r, c], s["image"], s["gt_semantic"], None, s["gt_boxes"], s["gt_present"], f"GT · {s['title']}")
        draw_slice(axes[r + 1, c], s["image"], s["pred_semantic"], s["pred_boxes"], s["gt_boxes"], s["gt_present"],
                   "predicción (GT punteada)")
    fig.tight_layout()
    return fig
