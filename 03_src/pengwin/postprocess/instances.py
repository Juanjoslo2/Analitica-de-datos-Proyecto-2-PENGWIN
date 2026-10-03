"""instances.py.

Segunda etapa: de "región por píxel + borde de fractura" a fragmentos individuales [DD §3].

Para cada región (SA, LI, RI) por separado:
    1. núcleo = región sin los píxeles de borde (probabilidad ≥ ``edge_threshold``)
    2. componentes conexas 3D del núcleo (26-vecindad): cada una es un candidato a fragmento
    3. las componentes < ``min_fragment_cm3`` no son semilla (ruido o puentes)
    4. watershed sobre el mapa de borde: cada píxel de la región (bordes y piezas pequeñas
       incluidos) va a la semilla a la que llega sin cruzar un borde; lo que quede aislado
       se asigna a la semilla más cercana
    5. etiquetas PENGWIN ordenadas por volumen: la más grande es el principal (1 / 11 / 21)

Se trabaja en la grilla del modelo (Z, 256, 256) con su spacing (dz, mm/px, mm/px), que es
~6 veces más liviana que la nativa; el resultado se lleva a la grilla nativa con ``to_native``.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed

STRUCT_26 = np.ones((3, 3, 3), bool)


def _bbox(mask: np.ndarray, pad: int = 1) -> Tuple[slice, ...]:
    obj = ndi.find_objects(mask.astype(np.uint8))[0]
    return tuple(slice(max(s.start - pad, 0), min(s.stop + pad, n)) for s, n in zip(obj, mask.shape))


def separate_region(mask: np.ndarray, edge: np.ndarray, spacing_zyx: Sequence[float], edge_threshold: float = 0.5,
                    min_fragment_cm3: float = 0.1, max_fragments: int = 10) -> np.ndarray:
    """Fragmentos de UNA región. Devuelve int (mismo tamaño que ``mask``): 0 fondo, 1 principal, 2.. resto."""
    out = np.zeros(mask.shape, np.int32)
    if not mask.any():
        return out
    sl = _bbox(mask)
    m, e = mask[sl], edge[sl].astype(np.float32)
    vox_cm3 = float(np.prod(spacing_zyx)) / 1000.0
    core = m & (e < edge_threshold)
    cc, n = ndi.label(core, structure=STRUCT_26)
    if n == 0:                                          # todo es borde: la región es un solo fragmento
        out[sl][m] = 1
        return out
    sizes = np.bincount(cc.ravel())[1:] * vox_cm3
    order = np.argsort(-sizes)
    keep = [i + 1 for i in order if sizes[i] >= min_fragment_cm3][:max_fragments]
    if not keep:
        keep = [int(order[0]) + 1]                      # al menos el principal
    markers = np.zeros(cc.shape, np.int32)
    for new_id, old_id in enumerate(keep, start=1):
        markers[cc == old_id] = new_id                  # ya en orden de volumen: 1 = el más grande
    lab = watershed(e, markers=markers, mask=m)
    lost = m & (lab == 0)                               # islas sin semilla conectada
    if lost.any():
        idx = ndi.distance_transform_edt(lab == 0, sampling=spacing_zyx, return_distances=False, return_indices=True)
        lab[lost] = lab[tuple(i[lost] for i in idx)]
    # reordenar por volumen final (el watershed puede cambiar el tamaño relativo)
    final = np.bincount(lab.ravel())[1:]
    rank = np.zeros(final.size + 1, np.int32)
    rank[1:][np.argsort(-final)] = np.arange(1, final.size + 1)
    out[sl] = np.where(m, rank[lab], 0)
    return out


def separate_region_edt(mask: np.ndarray, edge: np.ndarray, spacing_zyx: Sequence[float], edge_threshold: float = 0.2,
                        seed_depth_mm: float = 4.0, seed_min_cm3: float = 0.02, max_fragments: int = 10) -> np.ndarray:
    """Variante por distancia (semana 10): las semillas son las zonas "profundas" del núcleo.

    1. núcleo = región sin borde (P(borde) ≥ ``edge_threshold``)
    2. d = distance_transform_edt(núcleo, spacing) en mm
    3. semillas = componentes 3D de d > ``seed_depth_mm``: las grietas y los cuellos finos entre
       fragmentos nunca son profundos, así que separan aunque la segmentación los haya rellenado
    4. watershed sobre −d + 5·borde dentro de la región
    En val (v2), recupera el 61 % de los secundarios frente al 2 % de la variante solo-borde.
    """
    out = np.zeros(mask.shape, np.int32)
    if not mask.any():
        return out
    sl = _bbox(mask)
    m, e = mask[sl], edge[sl].astype(np.float32)
    core = m & (e < edge_threshold)
    d = ndi.distance_transform_edt(core, sampling=spacing_zyx)
    cc, n = ndi.label(d > seed_depth_mm, structure=STRUCT_26)
    if n == 0:
        cc, n = ndi.label(core if core.any() else m, structure=STRUCT_26)
    vox_cm3 = float(np.prod(spacing_zyx)) / 1000.0
    sizes = np.bincount(cc.ravel())[1:] * vox_cm3
    order = np.argsort(-sizes)
    keep = [i + 1 for i in order if sizes[i] >= seed_min_cm3][:max_fragments] or [int(order[0]) + 1]
    markers = np.zeros(cc.shape, np.int32)
    for new_id, old_id in enumerate(keep, start=1):
        markers[cc == old_id] = new_id
    lab = watershed(-d + 5.0 * e, markers=markers, mask=m)
    lost = m & (lab == 0)
    if lost.any():
        idx = ndi.distance_transform_edt(lab == 0, sampling=spacing_zyx, return_distances=False, return_indices=True)
        lab[lost] = lab[tuple(i[lost] for i in idx)]
    final = np.bincount(lab.ravel())[1:]
    rank = np.zeros(final.size + 1, np.int32)
    rank[1:][np.argsort(-final)] = np.arange(1, final.size + 1)
    out[sl] = np.where(m, rank[lab], 0)
    return out


def separate_instances(semantic: np.ndarray, edge: np.ndarray, spacing_zyx: Sequence[float], edge_threshold: float = 0.5,
                       min_fragment_cm3: float = 0.1, max_fragments: int = 10, method: str = "edge",
                       seed_depth_mm: float = 4.0) -> np.ndarray:
    """Volumen de etiquetas PENGWIN (0, 1-10 SA, 11-20 LI, 21-30 RI) a partir de semántica + borde.

    ``method``: "edge" (semillas = componentes del núcleo sin borde) o "edt" (semillas = zonas a
    más de ``seed_depth_mm`` del borde del núcleo; ver ``separate_region_edt``).
    """
    labels = np.zeros(semantic.shape, np.uint8)
    for r in (1, 2, 3):
        if method == "edt":
            frag = separate_region_edt(semantic == r, edge, spacing_zyx, edge_threshold, seed_depth_mm,
                                       max_fragments=max_fragments)
        else:
            frag = separate_region(semantic == r, edge, spacing_zyx, edge_threshold, min_fragment_cm3, max_fragments)
        labels[frag > 0] = (r - 1) * 10 + frag[frag > 0]
    return labels
