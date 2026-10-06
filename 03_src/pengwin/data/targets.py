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
    núcleo   hueso que NO es borde: el interior del fragmento (representación
             núcleo/borde del ganador de PENGWIN 2024, ver ``core_edge_target``)
    dist     distancia en mm de cada vóxel de hueso a la superficie de fractura más cercana,
             recortada a ``dist_max_mm``; 0 fuera del hueso (ver ``fracture_distance_3d``) [F2B2]
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import ndimage as ndi

NUM_REGIONS = 3
REGION_NAMES = ("SA", "LI", "RI")
CORE3_NAMES = ("fondo", "nucleo", "borde")      # clases 0/1/2 de ``core_edge_target``

# [F2B2] Mapa de distancia a la superficie de fractura.
DIST_MAX_MM = 8.0          # recorte por defecto (``loss.dist_max_mm``): más lejos no aporta nada
DIST_MM_PER_LEVEL = 0.1    # cuantización uint8 del caché y del .npz: 1 nivel = 0,1 mm (0..25,5 mm)
DIST_CACHE_MAX_MM = 255 * DIST_MM_PER_LEVEL


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


def core_edge_target(label: np.ndarray, edge: np.ndarray | None = None, dilation_px: int = 2) -> np.ndarray:
    """Mapa de 3 clases **núcleo/borde**: 0 fondo, 1 núcleo, 2 borde de fractura [F2B1].

    El borde es la superficie de contacto entre fragmentos distintos del MISMO hueso
    (``fracture_edge_2d`` / ``fracture_edge_3d``, dilatada ``dilation_px``); el núcleo es el
    resto del hueso. La interfaz entre REGIONES (articulación sacroilíaca) no es borde: ya la
    separa la semántica de 4 clases.

    ``edge``: borde ya calculado (p. ej. el corte de ``edge.npy``, borde 3D del caché, cuya
    dilatación se elige con ``data.edge_file``); si no se pasa, se calcula del propio ``label``
    con ``dilation_px``. Acepta un corte (H, W) o un volumen (Z, H, W).
    """
    bone = label > 0
    if edge is None:
        edge = fracture_edge_3d(label, dilation_px) if label.ndim == 3 else fracture_edge_2d(label, dilation_px)
    e = (edge > 0) & bone                       # la dilatación del caché ya está dentro del hueso
    return np.where(e, 2, np.where(bone, 1, 0)).astype(np.int64)


def role_target(label: np.ndarray) -> np.ndarray:
    """Papel de cada vóxel dentro de su hueso: 0 fondo, 1 fragmento principal, 2 secundario [y4xul].

    En PENGWIN el principal de cada región es la etiqueta 1 / 11 / 21 (coincide con el de mayor
    volumen en 300/300 regiones, EDA §1); el resto (2-10, 12-20, 22-30) son secundarios. No
    necesita el caché de borde ni los cortes vecinos: es una función del propio ``label``.
    """
    lab = label.astype(np.int16)
    return np.where(lab == 0, 0, np.where((lab - 1) % 10 == 0, 1, 2)).astype(np.int64)


def fracture_distance_3d(label: np.ndarray, spacing_zyx, max_mm: float = DIST_CACHE_MAX_MM,
                         surface: np.ndarray | None = None) -> np.ndarray:
    """Distancia en mm de cada vóxel de hueso a la superficie de fractura, recortada a ``max_mm`` [F2B2].

    Objetivo **denso**: el borde binario dilatado es 1 vóxel de cada 83 del hueso, y por eso la
    cabeza de borde solo cubre el 19 % de la superficie de fractura (recall 0,193). Aquí cada
    vóxel de hueso tiene valor, así que no hay desbalance que compensar con ``pos_weight``, y en
    inferencia el núcleo sale de la operación que el oráculo demuestra que funciona:
    ``núcleo = distancia > umbral`` (con el GT: Dice por fragmento 0,9952 a 1,5 mm de erosión).

    - La superficie es la de ``fracture_edge_3d`` **sin dilatar**: contacto entre fragmentos
      distintos del MISMO hueso. La interfaz entre regiones (articulación sacroilíaca) no es
      fractura. Sin dilatación el 0 queda exactamente en el contacto, así que ``seed_depth_mm``
      del posproceso se lee directamente en mm desde la fractura.
    - **Una EDT por región**: cada vóxel mide la distancia a la fractura de SU PROPIO hueso. Con
      una sola EDT global, la fractura del sacro acerca a 0 los vóxeles del coxal que están a
      pocos mm al otro lado de la articulación (la EDT es euclídea, no geodésica) y metería un
      valle falso en un hueso sano. El posproceso ya trabaja región por región.
    - ``spacing_zyx`` en mm (dz, dy, dx): se mide con el spacing, como pide el enunciado §3.3.
    - Fuera del hueso vale 0. Una región sin fractura queda entera en ``max_mm``
      (``distance_transform_edt`` de un arreglo sin ceros devuelve basura: hay que cortocircuitar).
    - Solo 3D: la distancia dentro de un corte no ve las fracturas casi axiales, que se cierran
      entre cortes (el mismo motivo por el que el borde pasó de 2D a 3D en la semana 10, §10.3).
    """
    if label.ndim != 3:
        raise ValueError("fracture_distance_3d necesita un volumen (Z, H, W): la distancia es 3D")
    if surface is None:
        surface = fracture_edge_3d(label, dilation=0)
    s = (surface > 0) & (label > 0)
    reg = region_of(label)
    out = np.zeros(label.shape, np.float32)
    for r in range(1, NUM_REGIONS + 1):
        m = reg == r
        if not m.any():
            continue
        sr = s & m
        if not sr.any():                            # región sin fractura: toda saturada
            out[m] = max_mm
            continue
        d = ndi.distance_transform_edt(~sr, sampling=spacing_zyx)
        out[m] = np.minimum(d[m], max_mm)
    return out


def dist_target(dist_mm: np.ndarray, label: np.ndarray, dist_max_mm: float = DIST_MAX_MM) -> np.ndarray:
    """Objetivo de regresión: la distancia en mm normalizada a [0, 1] y enmascarada al hueso [F2B2].

    Se normaliza dividiendo por ``dist_max_mm`` porque la salida es ``sigmoid(dist_logits)`` (acotada
    a [0, 1] por construcción) y porque el término ``seg`` suma CE + Dice, ambos O(1): una distancia
    en mm (0..8) dominaría esa suma y los λ, que balancean ENTRE términos, no podrían corregirlo.
    """
    d = np.clip(np.asarray(dist_mm, np.float32) / float(dist_max_mm), 0.0, 1.0)
    return (d * (label > 0)).astype(np.float32)


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
                  edge: np.ndarray | None = None, dist_mm: np.ndarray | None = None,
                  dist_max_mm: float = DIST_MAX_MM) -> Dict[str, np.ndarray]:
    """Todos los objetivos de un corte, listos para convertir a tensores.

    ``present`` es también el objetivo multi-etiqueta de la cabeza de clasificación:
    qué regiones anatómicas aparecen en el corte (los cortes vacíos dan [0, 0, 0]).
    ``edge``: corte del borde 3D precalculado en el caché (``edge.npy``); si no hay, borde 2D.
    ``core3`` sale del MISMO borde (no se cachea aparte: es una función de ``label`` y ``edge``,
    los dos ya disponibles y ya aumentados).
    ``dist_mm``: corte de la distancia 3D a la fractura en mm (``dist.npy`` del caché, ya
    aumentado); solo si se pasa aparece la clave ``dist`` (normalizada a [0, 1]) [F2B2]. No se
    deriva aquí porque la distancia es 3D y este corte no ve los cortes vecinos.
    """
    out = region_boxes_2d(label, min_box_px)
    out["semantic"] = region_of(label).astype(np.int64)
    e = (edge > 0) if edge is not None else fracture_edge_2d(label, edge_dilation_px)
    out["edge"] = e.astype(np.float32)
    out["core3"] = core_edge_target(label, e)
    out["role3"] = role_target(label)
    if dist_mm is not None:
        out["dist"] = dist_target(dist_mm, label, dist_max_mm)
    return out
