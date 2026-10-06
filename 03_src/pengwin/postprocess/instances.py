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

Métodos disponibles (``postprocess.instance_method``):
    edge    semillas = componentes del núcleo derivado (región sin borde)        [semana 9]
    edt     semillas = zonas a más de ``seed_depth_mm`` del borde del núcleo     [semana 10]
    core3   semillas = componentes del núcleo PREDICHO como clase propia         [F2B1]
    dist    semillas = zonas con DISTANCIA PREDICHA a la fractura > umbral       [F2B2]
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed

STRUCT_26 = np.ones((3, 3, 3), bool)

# Valores por defecto del bloque ``postprocess`` de la config. Son los de la semana 9 (método
# "edge"); ``configs/base.yaml`` los sobrescribe. La lista fija también sirve de filtro: solo
# estas claves llegan a ``separate_instances``.
PP_DEFAULTS = {
    "instance_method": "edge",
    "edge_threshold": 0.5,
    "min_fragment_cm3": 0.1,
    "max_fragments": 10,        # la taxonomía PENGWIN admite hasta 10 fragmentos por región
    "seed_depth_mm": 4.0,
    "seed_min_cm3": 0.02,
    "edge_weight": 5.0,
    "core_threshold": 0.5,      # method="core3": P(núcleo) > umbral (0,5 = argmax entre núcleo y borde)
    "core_seed_depth_mm": 0.0,  # method="core3": erosión extra por distancia del núcleo (0 = ninguna)
    "hmax_h_mm": 1.5,          # method="hmax": dinámica de las h-máximas de la distancia [F2C1]
    "dist_max_mm": 8.0,         # method="dist": recorte del mapa predicho (= loss.dist_max_mm) [F2B2]
    "role_threshold": 0.5,      # method="role": P(secundario | hueso) ≥ umbral [y4xul]
    "role_seed_depth_mm": 1.5,  # method="role": erosión que separa secundarios que se tocan entre sí
    "role_smooth_mm": 0.0,      # method="role": σ en mm del suavizado de P(secundario) a lo largo de z
}


def resolve_postprocess(*fuentes: dict | None) -> dict:
    """Combina bloques ``postprocess`` sobre ``PP_DEFAULTS``, de menos a más prioritario.

    Las claves con valor ``None`` se ignoran, así se pueden pasar los argumentos de una CLI tal
    cual (los que el usuario no dio valen ``None`` y no pisan la config).
    """
    out = dict(PP_DEFAULTS)
    for f in fuentes:
        out.update({k: v for k, v in (f or {}).items() if k in PP_DEFAULTS and v is not None})
    return out


def instance_kwargs(pp: dict) -> dict:
    """Del bloque ``postprocess`` resuelto a los argumentos con nombre de ``separate_instances``."""
    kw = {k: pp[k] for k in ("edge_threshold", "min_fragment_cm3", "max_fragments", "seed_depth_mm",
                             "seed_min_cm3", "edge_weight", "core_threshold", "core_seed_depth_mm",
                             "dist_max_mm", "hmax_h_mm", "role_threshold", "role_seed_depth_mm",
                             "role_smooth_mm")}
    kw["method"] = pp["instance_method"]
    return kw



def _bbox(mask: np.ndarray, pad: int = 1) -> Tuple[slice, ...]:
    obj = ndi.find_objects(mask.astype(np.uint8))[0]
    return tuple(slice(max(s.start - pad, 0), min(s.stop + pad, n)) for s, n in zip(obj, mask.shape))


def _label_from_seeds(seeds: np.ndarray, m: np.ndarray, landscape: np.ndarray, spacing_zyx: Sequence[float],
                      fallback: np.ndarray, seed_min_cm3: float, max_fragments: int) -> np.ndarray:
    """Maquinaria común a los métodos con semillas: componentes 3D → watershed → ids por volumen.

    1. componentes conexas 3D (26-vecindad) de ``seeds``; si no hay ninguna, de ``fallback``
    2. se descartan las menores de ``seed_min_cm3`` y se recorta a ``max_fragments`` (taxonomía
       PENGWIN: hasta 10 fragmentos por región); siempre queda al menos la mayor
    3. watershed de ``landscape`` dentro de ``m``; las islas sin semilla conectada van a la
       etiqueta más cercana en mm
    4. se reordena por volumen final: 1 = principal
    """
    cc, n = ndi.label(seeds, structure=STRUCT_26)
    if n == 0:
        cc, n = ndi.label(fallback, structure=STRUCT_26)
    vox_cm3 = float(np.prod(spacing_zyx)) / 1000.0
    sizes = np.bincount(cc.ravel())[1:] * vox_cm3
    order = np.argsort(-sizes)
    keep = [i + 1 for i in order if sizes[i] >= seed_min_cm3][:max_fragments] or [int(order[0]) + 1]
    markers = np.zeros(cc.shape, np.int32)
    for new_id, old_id in enumerate(keep, start=1):
        markers[cc == old_id] = new_id
    lab = watershed(landscape, markers=markers, mask=m)
    lost = m & (lab == 0)
    if lost.any():
        idx = ndi.distance_transform_edt(lab == 0, sampling=spacing_zyx, return_distances=False, return_indices=True)
        lab[lost] = lab[tuple(i[lost] for i in idx)]
    final = np.bincount(lab.ravel())[1:]
    rank = np.zeros(final.size + 1, np.int32)
    rank[1:][np.argsort(-final)] = np.arange(1, final.size + 1)
    return np.where(m, rank[lab], 0)


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
                        seed_depth_mm: float = 4.0, seed_min_cm3: float = 0.02, max_fragments: int = 10,
                        edge_weight: float = 5.0) -> np.ndarray:
    """Variante por distancia (semana 10): las semillas son las zonas "profundas" del núcleo.

    1. núcleo = región sin borde (P(borde) ≥ ``edge_threshold``)
    2. d = distance_transform_edt(núcleo, spacing) en mm
    3. semillas = componentes 3D de d > ``seed_depth_mm``: las grietas y los cuellos finos entre
       fragmentos nunca son profundos, así que separan aunque la segmentación los haya rellenado
    4. watershed sobre −d + ``edge_weight``·borde dentro de la región
    En val (v2), recupera el 61 % de los secundarios frente al 2 % de la variante solo-borde.
    """
    out = np.zeros(mask.shape, np.int32)
    if not mask.any():
        return out
    sl = _bbox(mask)
    m, e = mask[sl], edge[sl].astype(np.float32)
    core = m & (e < edge_threshold)
    d = ndi.distance_transform_edt(core, sampling=spacing_zyx)
    out[sl] = _label_from_seeds(d > seed_depth_mm, m, -d + edge_weight * e, spacing_zyx,
                                core if core.any() else m, seed_min_cm3, max_fragments)
    return out


def separate_region_core3(mask: np.ndarray, edge: np.ndarray, spacing_zyx: Sequence[float],
                          core: np.ndarray | None = None, core_threshold: float = 0.5,
                          edge_threshold: float = 0.5, core_seed_depth_mm: float = 0.0,
                          seed_min_cm3: float = 0.02, max_fragments: int = 10,
                          edge_weight: float = 5.0) -> np.ndarray:
    """Variante núcleo/borde [F2B1]: las semillas son el NÚCLEO PREDICHO, no un núcleo derivado.

    Hasta ahora el núcleo se derivaba del borde (``región & P(borde) < umbral``), así que la
    separación dependía del recall de la cabeza de borde (medido en 0,193 en la Fase 1; con el
    borde real el Dice por fragmento sube de 0,743 a 0,902). Aquí el núcleo es una clase propia
    de la salida de 3 clases (``model.seg_outputs: [semantic4, core3]``).

    1. semillas = región & (``core`` > ``core_threshold``); con ``core`` = P(núcleo) de la salida
       de 3 clases, el umbral 0,5 equivale al argmax entre núcleo y borde
    2. si no se pasa ``core``, o si el núcleo predicho sale vacío, se cae al núcleo derivado
       ``región & (P(borde) < edge_threshold)``: así el método funciona también con predicciones
       guardadas sin el canal de núcleo y nunca degrada a "componentes crudas de la región"
    3. ``core_seed_depth_mm`` > 0 añade la erosión por distancia del método "edt" (0 = ninguna,
       que es la hipótesis pura: el núcleo ya viene separado)
    4. watershed sobre −d + ``edge_weight``·borde dentro de la región, con d = distancia dentro
       del núcleo (misma maquinaria que "edt")
    """
    out = np.zeros(mask.shape, np.int32)
    if not mask.any():
        return out
    sl = _bbox(mask)
    m, e = mask[sl], edge[sl].astype(np.float32)
    nucleo = m & (core[sl] > core_threshold) if core is not None else m & (e < edge_threshold)
    if not nucleo.any():        # núcleo vacío (umbral alto o región diminuta): respaldo = el derivado
        nucleo = m & (e < edge_threshold)
    d = ndi.distance_transform_edt(nucleo, sampling=spacing_zyx)
    seeds = (d > core_seed_depth_mm) if core_seed_depth_mm > 0 else nucleo
    out[sl] = _label_from_seeds(seeds, m, -d + edge_weight * e, spacing_zyx,
                                nucleo if nucleo.any() else m, seed_min_cm3, max_fragments)
    return out


def separate_region_dist(mask: np.ndarray, edge: np.ndarray, spacing_zyx: Sequence[float],
                         dist: np.ndarray | None = None, seed_depth_mm: float = 4.0,
                         edge_threshold: float = 0.2, seed_min_cm3: float = 0.02,
                         max_fragments: int = 10, edge_weight: float = 5.0,
                         dist_max_mm: float = 8.0) -> np.ndarray:
    """Variante por distancia PREDICHA a la fractura [F2B2]: semillas = ``dist > seed_depth_mm``.

    Diferencia con "edt": allí la distancia se calcula del núcleo derivado del borde, así que la
    profundidad la limita también la superficie EXTERNA del hueso; un fragmento delgado nunca
    llega a 5 mm de profundidad y se queda sin semilla (medido: los de < 5 cm³ dan Dice 0). Aquí
    la distancia es solo a la superficie de FRACTURA, que es lo que separa: un fragmento plano
    conserva semilla mientras esté a más de ``seed_depth_mm`` de la fractura.

    1. d = distancia predicha en mm (``dist``), recortada a ``dist_max_mm``
    2. semillas = componentes 3D de ``m & (d > seed_depth_mm)``
    3. watershed sobre −d + ``edge_weight``·borde dentro de la región (misma maquinaria que "edt";
       sin cabeza de borde binaria, P(borde) = 1 − d/dist_max es monótona en d y no distorsiona)
    4. si no se pasa ``dist`` (predicciones guardadas sin el canal), se cae a la distancia dentro
       del núcleo derivado del borde: el método se vuelve exactamente "edt" y nunca degrada a
       etiquetar las componentes crudas de la región
    """
    out = np.zeros(mask.shape, np.int32)
    if not mask.any():
        return out
    sl = _bbox(mask)
    m, e = mask[sl], edge[sl].astype(np.float32)
    if dist is not None:
        d = np.minimum(dist[sl].astype(np.float32), dist_max_mm) * m
        fallback = m
    else:                                               # respaldo = método "edt"
        core = m & (e < edge_threshold)
        d = ndi.distance_transform_edt(core, sampling=spacing_zyx)
        fallback = core if core.any() else m
    out[sl] = _label_from_seeds(m & (d > seed_depth_mm), m, -d + edge_weight * e, spacing_zyx,
                                fallback, seed_min_cm3, max_fragments)
    return out


def separate_region_hmax(mask: np.ndarray, edge: np.ndarray, spacing_zyx: Sequence[float],
                         edge_threshold: float = 0.2, hmax_h_mm: float = 1.5,
                         seed_min_cm3: float = 0.02, max_fragments: int = 10,
                         edge_weight: float = 5.0) -> np.ndarray:
    """Semillas por **h-máximas** de la transformada de distancia, no por umbral global [F2C1].

    ``separate_region_edt`` usa las componentes de ``d > seed_depth_mm``, un umbral **global**, y eso
    obliga a un solo valor a servir dos casos incompatibles: dos fragmentos grandes fusionados solo
    se separan erosionando profundo, pero un fragmento de pocos cm³ **desaparece** a esa profundidad
    (en test, Dice 0,000 en el estrato < 5 cm³).

    Las h-máximas dan **una semilla por montículo** de ``d`` cuya altura relativa supere
    ``hmax_h_mm``, sea cual sea su profundidad absoluta: el fragmento pequeño conserva la suya y los
    grandes siguen separándose. El watershed es el mismo (−d + ``edge_weight``·borde).
    """
    from skimage.morphology import h_maxima

    out = np.zeros(mask.shape, np.int32)
    if not mask.any():
        return out
    sl = _bbox(mask)
    m, e = mask[sl], edge[sl].astype(np.float32)
    core = m & (e < edge_threshold)
    if not core.any():
        core = m
    d = ndi.distance_transform_edt(core, sampling=spacing_zyx)
    picos = h_maxima(d, h=max(hmax_h_mm, 1e-6))        # fuera del núcleo d = 0: no genera máximas
    cc, n = ndi.label(picos, structure=STRUCT_26)
    if n == 0:                                          # sin montículos: se cae al método por umbral
        return separate_region_edt(mask, edge, spacing_zyx, edge_threshold, hmax_h_mm,
                                   seed_min_cm3, max_fragments, edge_weight)
    vox_cm3 = float(np.prod(spacing_zyx)) / 1000.0
    sizes = np.bincount(cc.ravel())[1:]
    keep = [int(i) + 1 for i in np.argsort(-sizes)][:max_fragments]
    markers = np.zeros(cc.shape, np.int32)
    for new_id, old_id in enumerate(keep, start=1):
        markers[cc == old_id] = new_id
    lab = watershed(-d + edge_weight * e, markers=markers, mask=m)
    lost = m & (lab == 0)
    if lost.any():
        idx = ndi.distance_transform_edt(lab == 0, sampling=spacing_zyx, return_distances=False, return_indices=True)
        lab[lost] = lab[tuple(i[lost] for i in idx)]
    # los fragmentos por debajo de seed_min_cm3 se absorben en el vecino más cercano
    cuenta = np.bincount(lab.ravel())
    for k in range(1, cuenta.size):
        if 0 < cuenta[k] * vox_cm3 < seed_min_cm3:
            lab[lab == k] = 0
    lost = m & (lab == 0)
    if lost.any() and (lab > 0).any():
        idx = ndi.distance_transform_edt(lab == 0, sampling=spacing_zyx, return_distances=False, return_indices=True)
        lab[lost] = lab[tuple(i[lost] for i in idx)]
    final = np.bincount(lab.ravel())[1:]
    rank = np.zeros(final.size + 1, np.int32)
    rank[1:][np.argsort(-final)] = np.arange(1, final.size + 1)
    out[sl] = np.where(m, rank[lab], 0)
    return out


def separate_region_edt_hmax(mask: np.ndarray, edge: np.ndarray, spacing_zyx: Sequence[float],
                             edge_threshold: float = 0.2, seed_depth_mm: float = 5.0,
                             hmax_h_mm: float = 1.5, seed_min_cm3: float = 0.02,
                             max_fragments: int = 10, edge_weight: float = 5.0) -> np.ndarray:
    """Híbrido: semillas de ``edt`` + **rescate** por h-máxima donde ``edt`` no encontró ninguna [F2C2].

    Los dos métodos fallan en sitios complementarios (medido en F2C1):
    - ``edt`` (umbral profundo) respeta el fragmento principal pero deja **sin semilla** a los
      fragmentos de pocos mm de profundidad;
    - ``hmax`` los recupera (+7,1 pp de secundarios) pero **parte el principal** (−0,043 de Dice
      principal en los 5 folds) al meter varias semillas en un hueso grande con relieve interno.

    Aquí las semillas profundas mandan y las h-máximas solo actúan en las componentes del núcleo
    que ninguna semilla profunda ocupa.
    """
    from skimage.morphology import h_maxima

    out = np.zeros(mask.shape, np.int32)
    if not mask.any():
        return out
    sl = _bbox(mask)
    m, e = mask[sl], edge[sl].astype(np.float32)
    core = m & (e < edge_threshold)
    if not core.any():
        core = m
    d = ndi.distance_transform_edt(core, sampling=spacing_zyx)
    vox_cm3 = float(np.prod(spacing_zyx)) / 1000.0

    zona_profunda = d > seed_depth_mm
    # Las h-máximas que caen DENTRO de la zona profunda son las del propio fragmento grande: ya
    # tienen semilla y añadirlas lo partiría (es el fallo medido de F2C1). Solo se rescatan las
    # máximas que están FUERA de toda zona profunda, es decir los fragmentos demasiado delgados
    # para que el umbral les deje semilla. No sirve filtrar por componente conexa del núcleo: el
    # fragmento pequeño suele estar unido al grande por un cuello, así que comparten componente.
    picos = h_maxima(d, h=max(hmax_h_mm, 1e-6)) > 0
    cc_pic, n_pic = ndi.label(picos, structure=STRUCT_26)
    rescatadas = np.zeros(core.shape, bool)
    for k in range(1, n_pic + 1):
        pico_k = cc_pic == k
        if not zona_profunda[pico_k].any():        # ninguna parte del pico está en zona profunda
            rescatadas |= pico_k
    semillas = zona_profunda | rescatadas
    cc, n = ndi.label(semillas, structure=STRUCT_26)
    if n == 0:
        cc, n = ndi.label(core, structure=STRUCT_26)
    sizes = np.bincount(cc.ravel())[1:]
    keep = [int(i) + 1 for i in np.argsort(-sizes)][:max_fragments]
    markers = np.zeros(cc.shape, np.int32)
    for new_id, old_id in enumerate(keep, start=1):
        markers[cc == old_id] = new_id
    lab = watershed(-d + edge_weight * e, markers=markers, mask=m)
    for _ in range(2):
        lost = m & (lab == 0)
        if not (lost.any() and (lab > 0).any()):
            break
        idx = ndi.distance_transform_edt(lab == 0, sampling=spacing_zyx, return_distances=False, return_indices=True)
        lab[lost] = lab[tuple(i[lost] for i in idx)]
        cuenta = np.bincount(lab.ravel())
        for k in range(1, cuenta.size):
            if 0 < cuenta[k] * vox_cm3 < seed_min_cm3:
                lab[lab == k] = 0
    final = np.bincount(lab.ravel())[1:]
    rank = np.zeros(final.size + 1, np.int32)
    rank[1:][np.argsort(-final)] = np.arange(1, final.size + 1)
    out[sl] = np.where(m, rank[lab], 0)
    return out


def separate_region_role(mask: np.ndarray, edge: np.ndarray, spacing_zyx: Sequence[float],
                         role: np.ndarray | None, role_threshold: float = 0.5, role_seed_depth_mm: float = 1.5,
                         edge_threshold: float = 0.2, seed_min_cm3: float = 0.02, max_fragments: int = 10,
                         seed_depth_mm: float = 5.0, edge_weight: float = 5.0,
                         min_fragment_cm3: float = 0.1, role_smooth_mm: float = 0.0) -> np.ndarray:
    """Variante por PAPEL predicho (principal / secundario) [y4xul].

    ``role`` es P(secundario | hueso) en [0, 1]. La red ya dijo qué vóxeles son del fragmento
    principal y cuáles de un secundario, así que la frontera principal-secundario no depende de
    la cabeza de borde (recall 0,19) ni de erosionar 5 mm:

    0. ``role`` se suaviza a lo largo de z con una gaussiana de σ = ``role_smooth_mm``. El modelo
       decide corte a corte: si en unos cortes no marca un secundario, ese fragmento queda partido
       en rodajas y cada rodaja cuenta como fragmento aparte. El oráculo lo mide: con el papel
       borrado en el 20 % de los cortes el Dice por fragmento cae de 1,00 a 0,81
    1. secundario = región ∧ ``role`` ≥ ``role_threshold``; principal = el resto de la región
    2. semillas del principal: sus componentes 3D enteras, SIN erosión (no se parte el principal)
    3. semillas de secundarios: componentes de d_sec > ``role_seed_depth_mm``, con d_sec la EDT en
       mm del secundario sin borde. La erosión solo separa secundarios que se tocan ENTRE SÍ; el
       oráculo de la Fase 2 pone ese óptimo en 1,5 mm. Lo que queda fuera del alcance de esas
       semillas (más delgado que 2 × la erosión) y mide al menos ``min_fragment_cm3`` aporta su
       mitad más profunda: así los fragmentos finos no desaparecen, que es lo que le pasa a
       ``edt`` a 5 mm
    4. watershed sobre −d por separado en cada papel: cada semilla crece solo dentro de su clase
    5. ids por volumen: 1 = el mayor

    Sin ``role`` (modelo sin la salida ``role3``) cae a ``edt``.
    """
    if role is None:
        return separate_region_edt(mask, edge, spacing_zyx, edge_threshold, seed_depth_mm, seed_min_cm3,
                                   max_fragments, edge_weight)
    out = np.zeros(mask.shape, np.int32)
    if not mask.any():
        return out
    sl = _bbox(mask)
    m = mask[sl]
    r = role[sl].astype(np.float32)
    if role_smooth_mm > 0 and r.shape[0] > 1:
        r = ndi.gaussian_filter1d(r, sigma=role_smooth_mm / float(spacing_zyx[0]), axis=0, mode="nearest")
    sec = m & (r >= role_threshold)
    main = m & ~sec
    if not main.any():                                   # todo "secundario": no hay principal que respetar
        main, sec = sec, main
    vox_cm3 = float(np.prod(spacing_zyx)) / 1000.0
    d = np.zeros(m.shape, np.float32)
    d[main] = ndi.distance_transform_edt(main, sampling=spacing_zyx)[main]

    markers = np.zeros(m.shape, np.int32)
    cc, n = ndi.label(main, structure=STRUCT_26)
    sizes = np.bincount(cc.ravel())[1:] * vox_cm3
    grandes = [i + 1 for i in np.argsort(-sizes) if sizes[i] >= seed_min_cm3] or [int(np.argmax(sizes)) + 1]
    grandes = grandes[:max_fragments]
    for new_id, old_id in enumerate(grandes, start=1):
        markers[cc == old_id] = new_id
    next_id = len(grandes) + 1

    if sec.any():
        sec_core = sec & (edge[sl] < edge_threshold)
        if not sec_core.any():
            sec_core = sec
        d_sec = ndi.distance_transform_edt(sec_core, sampling=spacing_zyx)
        d[sec] = d_sec[sec]
        seeds = d_sec > role_seed_depth_mm
        # Rescate: lo que queda fuera del alcance de las semillas (más delgado que 2 × la erosión)
        # no ha desaparecido, es un fragmento fino. Cada resto de al menos ``min_fragment_cm3``
        # aporta su mitad más profunda como semilla. Se mide el alcance desde las semillas y no por
        # componente porque una lámina que toca a un fragmento grueso forma una sola componente con él.
        diag = float(np.sqrt(np.sum(np.square(spacing_zyx))))
        alcance = (ndi.distance_transform_edt(~seeds, sampling=spacing_zyx) <= role_seed_depth_mm + diag
                   if seeds.any() else np.zeros(m.shape, bool))
        comp, n_comp = ndi.label(sec_core & ~alcance, structure=STRUCT_26)
        if n_comp:
            idx = np.arange(1, n_comp + 1)
            vol_resto = np.bincount(comp.ravel(), minlength=n_comp + 1)[1:] * vox_cm3
            medio = np.full(n_comp + 1, np.inf, np.float32)             # inf = resto demasiado pequeño
            ok = vol_resto >= min_fragment_cm3
            medio[1:][ok] = 0.5 * np.asarray(ndi.maximum(d_sec, comp, idx), np.float32)[ok]
            seeds |= (comp > 0) & (d_sec >= medio[comp])
        cs, ns = ndi.label(seeds, structure=STRUCT_26)
        vol = np.bincount(cs.ravel(), minlength=ns + 1)[1:]
        for i in np.argsort(-vol):
            if next_id > max_fragments:
                break
            markers[cs == i + 1] = next_id
            next_id += 1

    # Un watershed por papel: cada semilla crece solo dentro de su clase, así la frontera
    # principal-secundario queda exactamente donde la puso la red.
    lab = np.zeros(m.shape, np.int32)
    for zona in (main, sec):
        mk = np.where(zona, markers, 0)
        if mk.any():
            lab[zona] = watershed(-d, markers=mk, mask=zona)[zona]
    lost = m & (lab == 0)                                 # p. ej. motas de secundario sin semilla
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
                       seed_depth_mm: float = 4.0, seed_min_cm3: float = 0.02, edge_weight: float = 5.0,
                       core: np.ndarray | None = None, core_threshold: float = 0.5,
                       core_seed_depth_mm: float = 0.0, hmax_h_mm: float = 1.5, dist: np.ndarray | None = None,
                       dist_max_mm: float = 8.0, role: np.ndarray | None = None, role_threshold: float = 0.5,
                       role_seed_depth_mm: float = 1.5, role_smooth_mm: float = 0.0) -> np.ndarray:
    """Volumen de etiquetas PENGWIN (0, 1-10 SA, 11-20 LI, 21-30 RI) a partir de semántica + borde.

    ``method``:
        "edge"   semillas = componentes del núcleo sin borde (``separate_region``)
        "edt"    semillas = zonas a más de ``seed_depth_mm`` del borde del núcleo (``separate_region_edt``)
        "core3"  semillas = núcleo predicho como clase propia (``separate_region_core3``); ``core``
                 es P(núcleo) **en [0, 1]** con la misma forma que ``semantic`` (si viene del .npz
                 de ``cv_predict.py`` hay que dividirlo por 255, igual que ``edge``). Si no se
                 pasa, el núcleo se deriva del borde y se pierde la ventaja de F2B1.
        "dist"   semillas = distancia PREDICHA a la fractura > ``seed_depth_mm``
                 (``separate_region_dist``); ``dist`` **en mm**, misma forma que ``semantic`` (del
                 .npz de ``cv_predict.py``: uint8 × ``DIST_MM_PER_LEVEL``). Si no se pasa, el
                 método equivale a "edt".
        "role"   principal y secundarios según el papel PREDICHO (``separate_region_role``);
                 ``role`` es P(secundario | hueso) **en [0, 1]** (del .npz: uint8 / 255). Si no se
                 pasa, el método equivale a "edt".
    La taxonomía se aplica aquí: los ids de cada región se desplazan a 1-10 / 11-20 / 21-30 y
    ``max_fragments`` acota cuántos salen por región.
    """
    labels = np.zeros(semantic.shape, np.uint8)
    for r in (1, 2, 3):
        if method == "role":
            frag = separate_region_role(semantic == r, edge, spacing_zyx, role, role_threshold, role_seed_depth_mm,
                                        edge_threshold, seed_min_cm3, max_fragments, seed_depth_mm, edge_weight,
                                        min_fragment_cm3, role_smooth_mm)
        elif method == "edt_hmax":
            frag = separate_region_edt_hmax(semantic == r, edge, spacing_zyx, edge_threshold, seed_depth_mm,
                                            hmax_h_mm, seed_min_cm3, max_fragments, edge_weight)
        elif method == "hmax":
            frag = separate_region_hmax(semantic == r, edge, spacing_zyx, edge_threshold, hmax_h_mm,
                                        seed_min_cm3, max_fragments, edge_weight)
        elif method == "dist":
            frag = separate_region_dist(semantic == r, edge, spacing_zyx, dist, seed_depth_mm, edge_threshold,
                                        seed_min_cm3, max_fragments, edge_weight, dist_max_mm)
        elif method == "core3":
            frag = separate_region_core3(semantic == r, edge, spacing_zyx, core, core_threshold, edge_threshold,
                                         core_seed_depth_mm, seed_min_cm3, max_fragments, edge_weight)
        elif method == "edt":
            frag = separate_region_edt(semantic == r, edge, spacing_zyx, edge_threshold, seed_depth_mm,
                                       seed_min_cm3, max_fragments, edge_weight)
        else:
            frag = separate_region(semantic == r, edge, spacing_zyx, edge_threshold, min_fragment_cm3, max_fragments)
        labels[frag > 0] = (r - 1) * 10 + frag[frag > 0]
    return labels
