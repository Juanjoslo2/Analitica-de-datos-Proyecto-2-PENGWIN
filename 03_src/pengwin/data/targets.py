"""targets.py.

Objetivos por corte (2D) a partir de la etiqueta PENGWIN ya redimensionada a 256×256.
Solo NumPy/SciPy: se prueban con phantoms en CI sin torch y los usa ``dataset.py``
DESPUÉS de la aumentación, para que cajas, bordes y máscara salgan de la misma etiqueta.

Convenciones [DD §2-3]:
    región   0 fondo, 1 SA (1-10), 2 coxal izq. (11-20), 3 coxal der. (21-30)
    caja     una por región y corte = envolvente de TODAS sus islas, en coordenadas
             continuas de píxel (x0, y0, x1, y1) con x1 = última columna + 1
    ignorar  cajas con lado < ``min_box_px``: no son positivo ni negativo
    borde    píxeles de hueso con un 4-vecino de OTRO fragmento de la MISMA región
             (superficie de fractura), dilatado ``dilation_px`` dentro del hueso
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import ndimage as ndi

NUM_REGIONS = 3
REGION_NAMES = ("SA", "LI", "RI")


def region_of(label: np.ndarray) -> np.ndarray:
    """0 = fondo, 1 = SA, 2 = LI, 3 = RI (vectorizado, cualquier forma)."""
    return np.where(label > 0, (label.astype(np.int16) - 1) // 10 + 1, 0).astype(np.uint8)


def _shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Desplaza un arreglo 2D rellenando con 0 (``np.roll`` daría la vuelta por el borde)."""
    out = np.zeros_like(a)
    h, w = a.shape
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = a[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return out


def fracture_edge_2d(label: np.ndarray, dilation_px: int = 2) -> np.ndarray:
    """Mapa binario de la superficie de fractura en un corte [EDA §7-8, DD §3].

    Solo cuenta el contacto entre fragmentos de la misma región: la interfaz entre
    regiones (articulación sacroilíaca) no es fractura y ya la separa la semántica.
    """
    lab = label.astype(np.int16)
    reg = region_of(label)
    edge = np.zeros(lab.shape, bool)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ln, rn = _shift(lab, dy, dx), _shift(reg, dy, dx)
        edge |= (lab > 0) & (ln > 0) & (lab != ln) & (rn == reg)
    if dilation_px > 0 and edge.any():
        edge = ndi.binary_dilation(edge, iterations=dilation_px) & (lab > 0)
    return edge


def _shift3(a: np.ndarray, axis: int, step: int) -> np.ndarray:
    out = np.zeros_like(a)
    src = [slice(None)] * 3
    dst = [slice(None)] * 3
    src[axis] = slice(max(-step, 0), a.shape[axis] - max(step, 0))
    dst[axis] = slice(max(step, 0), a.shape[axis] - max(-step, 0))
    out[tuple(dst)] = a[tuple(src)]
    return out


def fracture_edge_3d(label: np.ndarray, dilation: int = 2) -> np.ndarray:
    """Superficie de fractura en 3D (Z, H, W): vóxeles de hueso con un 6-vecino de OTRO
    fragmento de la misma región, incluidos los vecinos de los cortes de arriba y abajo.

    Semana 10: el borde 2D (``fracture_edge_2d``) no ve los contactos ENTRE cortes. Con el
    borde real (oráculo) en test, el borde 2D deja separar solo el 35 % de los fragmentos
    secundarios; el 3D, el 71 %. Se dilata ``dilation`` veces con 6-vecindad, dentro del hueso.
    """
    lab = label.astype(np.int16)
    reg = region_of(label)
    edge = np.zeros(lab.shape, bool)
    for axis in range(3):
        for step in (1, -1):
            ln, rn = _shift3(lab, axis, step), _shift3(reg, axis, step)
            edge |= (lab > 0) & (ln > 0) & (lab != ln) & (rn == reg)
    if dilation > 0 and edge.any():
        edge = ndi.binary_dilation(edge, structure=ndi.generate_binary_structure(3, 1), iterations=dilation) & (lab > 0)
    return edge


def region_boxes_2d(label: np.ndarray, min_box_px: float = 4.0) -> Dict[str, np.ndarray]:
    """Caja envolvente por región.

    Devuelve:
        boxes   float32 (3, 4)  x0, y0, x1, y1 (ceros si la región no está)
        present bool    (3,)    la región aparece en el corte
        ignore  bool    (3,)    aparece pero su caja mide < ``min_box_px`` de lado
    """
    reg = region_of(label)
    boxes = np.zeros((NUM_REGIONS, 4), np.float32)
    present = np.zeros(NUM_REGIONS, bool)
    ignore = np.zeros(NUM_REGIONS, bool)
    for k in range(NUM_REGIONS):
        ys, xs = np.nonzero(reg == k + 1)
        if ys.size == 0:
            continue
        boxes[k] = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        present[k] = True
        ignore[k] = min(boxes[k, 2] - boxes[k, 0], boxes[k, 3] - boxes[k, 1]) < min_box_px
    return {"boxes": boxes, "present": present, "ignore": ignore}


def slice_targets(label: np.ndarray, min_box_px: float = 4.0, edge_dilation_px: int = 2,
                  edge: np.ndarray | None = None) -> Dict[str, np.ndarray]:
    """Todos los objetivos de un corte, listos para convertir a tensores.

    ``present`` es también el objetivo multi-etiqueta de la cabeza de clasificación:
    qué regiones anatómicas aparecen en el corte (los cortes vacíos dan [0, 0, 0]).
    ``edge``: corte del borde 3D precalculado en el caché (``edge.npy``); si no hay, borde 2D.
    """
    out = region_boxes_2d(label, min_box_px)
    out["semantic"] = region_of(label).astype(np.int64)
    out["edge"] = (edge > 0).astype(np.float32) if edge is not None else fracture_edge_2d(label, edge_dilation_px).astype(np.float32)
    return out
