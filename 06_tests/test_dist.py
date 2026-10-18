"""F2B2: regresión del mapa de distancia a la superficie de fractura, sobre phantoms.

El posproceso separa fragmentos umbralizando una geometría. Con el borde/núcleo del GT y 1,5 mm
de erosión el Dice por fragmento es 0,9952; sin erosión, 0,8464. Con el borde PREDICHO hace falta
5 mm y a esa profundidad mueren los fragmentos pequeños, porque la cabeza de borde solo cubre el
19 % de la superficie de fractura: el borde dilatado es 1 vóxel de cada 83 del hueso. La distancia
es un objetivo DENSO, así que elimina ese desbalance en vez de compensarlo con ``pos_weight``.

Estos tests fijan las cinco piezas nuevas:

    objetivo    ``fracture_distance_3d``: mm a la fractura, recortada, 0 fuera del hueso y
                medida con el spacing
    caché       ``dist.npy`` (0,1 mm por nivel) y su llegada al lote del ``Dataset``
    salida      ``model.seg_outputs: [semantic4, dist]``, sin tocar las otras dos cabezas
    pérdida     Huber enmascarada al hueso DENTRO del término ``seg`` (siguen siendo 3 términos:
                cls, det, seg [enunciado §4.1])
    posproceso  ``method="dist"``: semillas = componentes 3D de ``distancia > seed_depth_mm``,
                con respaldo exacto a "edt" cuando el .npz no trae el canal
"""

import importlib.util
import json

import numpy as np
import pytest

from pengwin.data.targets import (
    DIST_MM_PER_LEVEL, dist_target, fracture_distance_3d, slice_targets,
)
from pengwin.postprocess.instances import (
    PP_DEFAULTS, instance_kwargs, resolve_postprocess, separate_instances,
)

SP = (1.0, 1.0, 1.0)


# ------------------------------------------------------------------ objetivo: distancia en mm
def _hueso_partido():
    """Coxal izq. partido en dos (contacto en y = 16) y coxal der. pegado al izq. (x = 16).

    El contacto 11-12 es superficie de fractura; la interfaz 11-21 es la articulación
    sacroilíaca, que ya separa la semántica de 4 clases y NO es fractura.
    """
    lab = np.zeros((6, 32, 32), np.uint8)
    lab[1:5, 4:16, 4:16] = 11          # coxal izq., principal
    lab[1:5, 16:24, 4:16] = 12         # coxal izq., secundario pegado al principal
    lab[1:5, 4:16, 16:24] = 21         # coxal der.: otra REGIÓN
    return lab


def test_vale_0_fuera_del_hueso_y_crece_hacia_el_interior_de_cada_fragmento():
    lab = _hueso_partido()
    d = fracture_distance_3d(lab, SP, max_mm=8.0)
    assert d.shape == lab.shape and d.dtype == np.float32
    assert (d[lab == 0] == 0).all(), "fuera del hueso, 0"
    assert (d[lab > 0] >= 0).all() and d.max() <= 8.0
    # el contacto está entre y = 15 y y = 16: ahí la distancia es ~0 y crece alejándose
    perfil = [d[2, y, 10] for y in (15, 14, 13, 12, 11)]
    assert perfil[0] < 1.0 and all(a < b for a, b in zip(perfil, perfil[1:])), perfil
    perfil2 = [d[2, y, 10] for y in (16, 17, 18, 19)]
    assert perfil2[0] < 1.0 and all(a < b for a, b in zip(perfil2, perfil2[1:])), perfil2
    # el vóxel del contacto queda a 0 (la superficie SÍ es parte del hueso)
    assert d[2, 15, 10] == 0.0 and d[2, 16, 10] == 0.0


def test_la_distancia_es_a_la_fractura_del_PROPIO_hueso():
    """El coxal derecho (21) toca al izquierdo en x = 16 pero no está fracturado: todo su hueso
    queda en el máximo. Con una sola EDT global, la fractura del 11-12 (a 7 mm en línea recta,
    al otro lado de la articulación) le metería un valle falso: la EDT es euclídea, no geodésica.
    """
    lab = _hueso_partido()
    d = fracture_distance_3d(lab, SP, max_mm=8.0)
    assert (d[lab == 21] == 8.0).all(), "sin fractura propia, el coxal der. queda saturado"
    # control: con una EDT global ese hueso sano sí bajaría del máximo
    sup = np.zeros(lab.shape, bool)
    sup[1:5, 15:17, 4:16] = True
    global_edt = np.minimum(fracture_distance_3d(np.where(lab > 0, 11, 0).astype(np.uint8), SP,
                                                 max_mm=8.0, surface=sup), 8.0)
    assert global_edt[2, 10, 20] < 8.0


def test_sin_ninguna_fractura_todo_el_hueso_queda_en_el_maximo():
    """``distance_transform_edt`` de un arreglo sin ceros devuelve basura: hay que cortocircuitar."""
    lab = np.zeros((4, 16, 16), np.uint8)
    lab[1:3, 4:12, 4:12] = 1
    d = fracture_distance_3d(lab, SP, max_mm=8.0)
    assert (d[lab > 0] == 8.0).all() and (d[lab == 0] == 0).all()


def test_la_distancia_se_mide_con_el_spacing():
    """Phantom anisótropo: la fractura es un plano z y los cortes miden 3 mm, el plano 1 mm.

    Con spacing isótropo (1, 1, 1) el vóxel a 2 cortes de la fractura estaría a 2 mm; con
    dz = 3 mm está a 6 mm. Es el spacing del header, como pide el enunciado §3.3.
    """
    lab = np.zeros((10, 16, 16), np.uint8)
    lab[1:5, 4:12, 4:12] = 1           # sacro, principal
    lab[5:9, 4:12, 4:12] = 2           # secundario pegado en z: la fractura es el plano z = 4|5
    iso = fracture_distance_3d(lab, (1.0, 1.0, 1.0), max_mm=25.0)
    aniso = fracture_distance_3d(lab, (3.0, 1.0, 1.0), max_mm=25.0)
    assert iso[2, 8, 8] == pytest.approx(2.0) and aniso[2, 8, 8] == pytest.approx(6.0)
    assert iso[1, 8, 8] == pytest.approx(3.0) and aniso[1, 8, 8] == pytest.approx(9.0)
    # en el plano, el vóxel pegado al contacto sigue a 0 en los dos
    assert iso[4, 8, 8] == 0.0 and aniso[4, 8, 8] == 0.0


def test_el_objetivo_se_normaliza_a_01_y_se_enmascara_al_hueso():
    lab = _hueso_partido()
    d = fracture_distance_3d(lab, SP, max_mm=25.0)
    t = dist_target(d, lab, dist_max_mm=8.0)
    assert t.dtype == np.float32 and t.min() == 0.0 and t.max() <= 1.0
    assert (t[lab == 0] == 0).all(), "el objetivo no supervisa el fondo"
    assert t[2, 11, 10] == pytest.approx(min(d[2, 11, 10] / 8.0, 1.0))
    assert (t[lab == 21] == 1.0).all(), "más allá de dist_max_mm, saturado a 1"


def test_slice_targets_devuelve_dist_solo_si_se_le_pasa():
    """La distancia es 3D: este corte no ve los cortes vecinos, así que viene del caché."""
    lab = _hueso_partido()
    d = fracture_distance_3d(lab, SP, max_mm=25.0)
    assert "dist" not in slice_targets(lab[2]), "sin dist_mm no aparece la clave (configs de siempre)"
    t = slice_targets(lab[2], dist_mm=d[2], dist_max_mm=8.0)
    assert t["dist"].shape == lab[2].shape
    assert np.allclose(t["dist"], dist_target(d[2], lab[2], 8.0))


# ------------------------------------------------------------------ caché y Dataset
def _cache_falso(tmp_path, lab, spacing=(3.0, 1.0, 1.0), case="900"):
    """Caso de caché escrito a mano (sin .mha): label, image y el meta mínimo que usa el Dataset."""
    d = tmp_path / "cache" / case
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "label.npy", lab.astype(np.uint8))
    np.save(d / "image.npy", np.where(lab > 0, 200, 20).astype(np.uint8))
    z, h, w = lab.shape
    meta = {"case_id": case, "native_shape_zyx": [z, h, w], "spacing_zyx": list(spacing),
            "image_size": w, "pixel_mm": spacing[2], "context_offset": 1,
            "crop": {"y0": 0, "y1": h, "x0": 0, "x1": w, "side_px": h},
            "bone_slices": np.nonzero(lab.reshape(z, -1).max(1) > 0)[0].tolist(),
            "labels_present": [int(v) for v in np.unique(lab) if v > 0]}
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return d


def test_el_cache_guarda_la_distancia_en_mm_cuantizada(tmp_path):
    """Se precalcula en el caché, no al vuelo: la distancia necesita el volumen entero (y su
    spacing) y un EDT 3D por corte en cada época sería inviable. Se guarda en mm (0,1 mm por
    nivel) y no normalizada, para que cambiar ``loss.dist_max_mm`` no obligue a reconstruirlo.
    También usa el spacing del meta (dz = 3 mm), no una grilla isótropa."""
    from pengwin.data.slice_cache import add_dist_cache, load_dist_cache

    lab = _hueso_partido()
    d = _cache_falso(tmp_path, lab, spacing=(3.0, 1.0, 1.0))
    maximo = add_dist_cache(d)
    u8 = load_dist_cache(d, mmap=False)
    assert u8.dtype == np.uint8 and u8.shape == lab.shape
    esperada = fracture_distance_3d(lab, (3.0, 1.0, 1.0))
    assert np.allclose(u8.astype(np.float32) * DIST_MM_PER_LEVEL, esperada, atol=0.05)
    assert maximo == pytest.approx(float(esperada.max()))
    assert add_dist_cache(d, skip_existing=True) == 0.0, "con --skip-existing no recalcula"


def test_el_dataset_entrega_la_distancia_normalizada(tmp_path):
    pytest.importorskip("torch")
    from conftest import REPO
    from pengwin.data.dataset import PengwinSlices
    from pengwin.data.slice_cache import add_dist_cache
    from pengwin.utils.config import load_config

    lab = _hueso_partido()
    d = _cache_falso(tmp_path, lab, spacing=(1.0, 1.0, 1.0))
    add_dist_cache(d)
    cfg = load_config(REPO / "configs" / "tuning" / "f2b2_dist.yaml")
    ds = PengwinSlices(tmp_path / "cache", ["900"], cfg, train=False)
    item = ds[ds.index.index(("900", 2))]
    assert item["dist"].shape == (1, 32, 32)
    esperado = dist_target(fracture_distance_3d(lab, (1.0, 1.0, 1.0))[2], lab[2], 8.0)
    assert np.allclose(item["dist"][0].numpy(), esperado, atol=0.01)


def test_sin_dist_npy_el_dataset_avisa_claro(tmp_path):
    """Si la config pide la salida ``dist`` y el caché no la tiene, falla con un mensaje útil
    en vez de entrenar con un objetivo vacío."""
    pytest.importorskip("torch")
    from conftest import REPO
    from pengwin.data.dataset import PengwinSlices
    from pengwin.utils.config import load_config

    _cache_falso(tmp_path, _hueso_partido())
    cfg = load_config(REPO / "configs" / "tuning" / "f2b2_dist.yaml")
    ds = PengwinSlices(tmp_path / "cache", ["900"], cfg, train=False)
    with pytest.raises(FileNotFoundError, match="dist"):
        ds[0]


def test_la_aumentacion_reescala_la_distancia_con_la_escala_de_la_afin(tmp_path):
    """La afín amplía la geometría por ``s``, así que las longitudes en mm del mapa transportado
    quedan multiplicadas por ``s``. Sin corregirlo el objetivo traería un sesgo sistemático de
    hasta el 10 % (scale 0,9-1,1) justo en la magnitud que el posproceso umbraliza."""
    pytest.importorskip("torch")
    from conftest import REPO
    from pengwin.data.dataset import PengwinSlices
    from pengwin.data.slice_cache import add_dist_cache
    from pengwin.utils.config import load_config

    lab = np.zeros((6, 64, 64), np.uint8)
    lab[1:5, 20:32, 20:44] = 1         # sacro, principal
    lab[1:5, 32:44, 20:44] = 2         # secundario pegado: fractura en el plano y = 31|32
    d = _cache_falso(tmp_path, lab, spacing=(1.0, 1.0, 1.0))
    add_dist_cache(d)
    cfg = load_config(REPO / "configs" / "tuning" / "f2b2_dist.yaml")
    cfg["loss"]["dist_max_mm"] = 40.0  # sin recorte: el factor de escala se vería truncado
    sin = PengwinSlices(tmp_path / "cache", ["900"], cfg, train=False)
    cfg_aug = {**cfg, "augment": {"scale": [1.5, 1.5], "rotate_deg": 0, "translate": 0.0,
                                  "gamma": [1.0, 1.0], "contrast": [1.0, 1.0], "brightness": 0.0}}
    con = PengwinSlices(tmp_path / "cache", ["900"], cfg_aug, train=True, augment=True)

    def media_mm(item):
        """Media en mm sobre el hueso (la media aguanta el vecino más cercano mejor que el máximo)."""
        hueso = item["semantic"] > 0
        return float(item["dist"][0][hueso].mean()) * 40.0

    a = media_mm(sin[sin.index.index(("900", 2))])
    b = media_mm(con[con.index.index(("900", 2))])
    assert b == pytest.approx(1.5 * a, rel=0.08), f"{a:.2f} -> {b:.2f} mm (escala 1,5)"


# ------------------------------------------------------------------ posproceso "dist"
def _bloques_pegados_con_distancia_predicha():
    """Coxal izq.: la región rellena la grieta y el borde predicho es 0 (la falla medida).

    La distancia predicha, en cambio, tiene su valle en la grieta (x ≈ 23) y crece hacia los dos
    extremos. Los bloques son gruesos (6 cortes, 12 px en y) para que haya dónde poner semillas.
    """
    sem = np.zeros((8, 20, 40), np.uint8)
    sem[1:7, 4:16, 4:36] = 2                 # región continua: no hay grieta que separar
    edge = np.zeros(sem.shape, np.float32)   # la cabeza de borde no predijo nada
    perfil = np.clip(np.abs(np.arange(40, dtype=np.float32) - 23.0) - 0.5, 0.0, 8.0)
    dist = perfil[None, None, :] * (sem > 0)
    return sem, edge, dist.astype(np.float32)


def test_dist_separa_dos_bloques_pegados():
    sem, edge, dist = _bloques_pegados_con_distancia_predicha()
    lab = separate_instances(sem, edge, SP, method="dist", dist=dist, seed_depth_mm=4.0)
    assert sorted(np.unique(lab[lab > 0]).tolist()) == [11, 12], "dos fragmentos del coxal izquierdo"
    assert (lab[1:7, 4:16, 4:19] == 11).all(), "el bloque mayor es el principal"
    assert (lab[1:7, 4:16, 28:36] == 12).all()
    assert ((lab > 0) == (sem > 0)).all(), "el watershed reparte también el valle entre semillas"


def test_sin_canal_de_distancia_los_bloques_se_fusionan():
    """Cota inferior: con el borde en 0 y sin canal de distancia, ni "edt" ni el respaldo de
    "dist" separan nada. Es exactamente la dependencia que F2B2 quita."""
    sem, edge, _ = _bloques_pegados_con_distancia_predicha()
    edt = separate_instances(sem, edge, SP, method="edt", seed_depth_mm=4.0)
    respaldo = separate_instances(sem, edge, SP, method="dist", seed_depth_mm=4.0, edge_threshold=0.2)
    assert np.unique(edt[edt > 0]).tolist() == [11]
    assert np.unique(respaldo[respaldo > 0]).tolist() == [11]
    assert (respaldo == edt).all(), "sin dist, el método es exactamente 'edt'"


def test_dist_respeta_max_fragments():
    """5 franjas profundas dentro de una sola región del sacro; max_fragments deja 3."""
    sem = np.zeros((6, 12, 50), np.uint8)
    sem[1:5, 3:9, 2:48] = 1
    dist = np.zeros(sem.shape, np.float32)
    for k in range(5):
        dist[1:5, 3:9, 3 + 9 * k:10 + 9 * k] = 6.0     # 7 columnas a 6 mm, separadas por 2 a 0
    edge = np.zeros(sem.shape, np.float32)
    todos = separate_instances(sem, edge, SP, method="dist", dist=dist, seed_depth_mm=4.0)
    tres = separate_instances(sem, edge, SP, method="dist", dist=dist, seed_depth_mm=4.0, max_fragments=3)
    assert len(np.unique(todos[todos > 0])) == 5
    assert sorted(np.unique(tres[tres > 0]).tolist()) == [1, 2, 3], "las 3 semillas mayores"


def test_los_ids_siguen_la_taxonomia_pengwin():
    """SA 1-10, coxal izq. 11-20, coxal der. 21-30: el principal es el 1 / 11 / 21."""
    for region, esperado in ((1, [1, 2]), (2, [11, 12]), (3, [21, 22])):
        sem = np.zeros((6, 12, 24), np.uint8)
        sem[1:5, 3:9, 2:22] = region
        dist = np.zeros(sem.shape, np.float32)
        dist[1:5, 3:9, 2:13] = 6.0           # la semilla mayor -> principal
        dist[1:5, 3:9, 15:22] = 6.0
        lab = separate_instances(sem, np.zeros(sem.shape, np.float32), SP, method="dist",
                                 dist=dist, seed_depth_mm=4.0)
        assert sorted(np.unique(lab[lab > 0]).tolist()) == esperado


def test_un_fragmento_delgado_conserva_semilla_con_dist_y_la_pierde_con_edt():
    """La hipótesis de F2B2, en un phantom: "edt" mide la profundidad dentro del núcleo, así que
    la superficie EXTERNA del hueso también la limita y una lámina de 2 vóxeles nunca llega a los
    4 mm. La distancia a la FRACTURA no depende del grosor, solo de cuánto se aleje del contacto.
    """
    sem = np.zeros((12, 40, 30), np.uint8)
    sem[1:11, 4:26, 4:26] = 2                # bloque principal: 10 cortes de grosor
    sem[5:7, 26:36, 4:26] = 2                # lámina pegada: 2 cortes de grosor, 10 px de largo
    edge = np.zeros(sem.shape, np.float32)
    perfil = np.minimum(np.abs(np.arange(40, dtype=np.float32) - 25.5), 8.0)
    dist = (perfil[None, :, None] * (sem > 0)).astype(np.float32)
    edt = separate_instances(sem, edge, SP, method="edt", seed_depth_mm=4.0)
    con_dist = separate_instances(sem, edge, SP, method="dist", dist=dist, seed_depth_mm=4.0)
    assert np.unique(edt[edt > 0]).tolist() == [11], "con 'edt' la lámina se funde con el principal"
    assert sorted(np.unique(con_dist[con_dist > 0]).tolist()) == [11, 12]
    assert (con_dist[5:7, 31:36, 4:26] == 12).all(), "la lámina es un fragmento propio"


def test_los_parametros_nuevos_llegan_al_separador():
    import inspect

    assert "dist_max_mm" in PP_DEFAULTS
    kw = instance_kwargs(resolve_postprocess({"instance_method": "dist"},
                                             {"seed_depth_mm": 3.0, "dist_max_mm": 6.0}))
    assert kw["method"] == "dist" and kw["seed_depth_mm"] == 3.0 and kw["dist_max_mm"] == 6.0
    assert set(kw) <= set(inspect.signature(separate_instances).parameters), "algún argumento no existe"


@pytest.mark.parametrize("profundidad, n_esperado", [(4.0, 2), (7.5, 1)])
def test_seed_depth_mm_llega_y_decide_cuantos_fragmentos_salen(profundidad, n_esperado):
    """El bloque secundario llega a 6,5 mm de la fractura: con 7,5 mm se queda sin semilla."""
    sem, edge, dist = _bloques_pegados_con_distancia_predicha()
    dist = np.minimum(dist, 6.5)
    dist[1:7, 4:16, 4:12] = 8.0                        # el principal sí pasa de 7,5 mm
    pp = resolve_postprocess({"instance_method": "dist"}, {"seed_depth_mm": profundidad})
    lab = separate_instances(sem, edge, SP, dist=dist, **instance_kwargs(pp))
    assert len(np.unique(lab[lab > 0])) == n_esperado


def test_dist_max_mm_recorta_el_mapa_predicho():
    """Red de seguridad: el .npz admite hasta 25,5 mm; con dist_max_mm = 3 nada pasa de 3 mm y
    ninguna semilla sobrevive al umbral de 4 mm (queda un solo fragmento por región)."""
    sem, edge, dist = _bloques_pegados_con_distancia_predicha()
    lab = separate_instances(sem, edge, SP, method="dist", dist=dist, seed_depth_mm=4.0, dist_max_mm=3.0)
    assert np.unique(lab[lab > 0]).tolist() == [11]


# ------------------------------------------------------------------ .npz sin el canal nuevo
def test_los_npz_sin_dist_siguen_funcionando_igual(tmp_path):
    """``tune_postprocess.run_case`` tiene que dar lo MISMO sobre un .npz con y sin ``dist``
    cuando el método no es "dist": el cambio es puramente aditivo y la cola está usando el
    script ahora mismo."""
    pytest.importorskip("pandas")
    from conftest import REPO

    spec = importlib.util.spec_from_file_location("tune_pp", REPO / "scripts" / "tune_postprocess.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    lab = _hueso_partido()
    for case in ("900", "901"):                     # el mismo caso con dos nombres de .npz
        _cache_falso(tmp_path, lab, spacing=(1.0, 1.0, 1.0), case=case)
    sem = np.where(lab > 0, (lab.astype(np.int16) - 1) // 10 + 1, 0).astype(np.uint8)
    edge = np.zeros(lab.shape, np.uint8)
    d_mm = fracture_distance_3d(lab, (1.0, 1.0, 1.0))
    oof = tmp_path / "oof"
    oof.mkdir()
    np.savez_compressed(oof / "900.npz", semantic=sem, edge=edge)
    np.savez_compressed(oof / "901.npz", semantic=sem, edge=edge,
                        dist=np.round(d_mm / DIST_MM_PER_LEVEL).astype(np.uint8))

    combos = [{"method": "edt", "seed_depth_mm": 2.0, "edge_threshold": 0.2},
              {"method": "dist", "seed_depth_mm": 2.0, "edge_threshold": 0.2}]
    cache = str(tmp_path / "cache")
    sin = mod.run_case(("900", 0, str(oof), cache, combos))
    con = mod.run_case(("901", 0, str(oof), cache, combos))
    dices = lambda filas: [round(float(f["dice"]), 6) for f in filas]
    # el .npz viejo: "edt" y "dist" (que cae al respaldo) dan exactamente lo mismo
    assert dices(sin[0][2]) == dices(sin[1][2]), "sin el canal, 'dist' es 'edt'"
    # y "edt" no cambia porque el .npz nuevo traiga el canal
    assert dices(sin[0][2]) == dices(con[0][2]), "el canal dist no altera los métodos de siempre"
    # con el canal, "dist" separa el secundario que "edt" no veía
    assert np.mean(dices(con[1][2])) > np.mean(dices(con[0][2]))


# ------------------------------------------------------------------ modelo y pérdida (con torch)
def _volumen_phantom(n: int = 32):
    """Volumen con sacro y coxal izq. partido, para construir objetivos reales."""
    lab = np.zeros((5, n, n), np.uint8)
    q = n // 8
    lab[1:4, 2 * q:5 * q, q:3 * q] = 11
    lab[1:4, 5 * q:6 * q, q:3 * q] = 12      # fragmento pegado al principal
    lab[1:4, 2 * q:5 * q, 4 * q:6 * q] = 1   # sacro
    return lab


def _lote_de_un_corte(n: int = 32, dist_max_mm: float = 8.0):
    """Lote de B = 1 con los objetivos REALES de ``slice_targets`` (incluido ``dist``)."""
    import torch

    vol = _volumen_phantom(n)
    d = fracture_distance_3d(vol, (1.0, 1.0, 1.0), max_mm=25.0)
    lab = vol[2]
    t = slice_targets(lab, dist_mm=d[2], dist_max_mm=dist_max_mm)
    return {
        "semantic": torch.from_numpy(t["semantic"])[None],
        "edge": torch.from_numpy(t["edge"])[None, None],
        "dist": torch.from_numpy(t["dist"])[None, None],
        "boxes": torch.from_numpy(t["boxes"])[None],
        "present": torch.from_numpy(t["present"].astype(np.float32))[None],
        "ignore": torch.from_numpy(t["ignore"])[None],
    }


def _salida_falsa(lote, dist_logits):
    """Salida mínima del modelo: solo ``dist_logits`` tiene gradiente; el resto es constante."""
    import torch

    b, h, w = lote["semantic"].shape
    g = h // 8
    return {"cls_logits": torch.zeros(b, 3), "det_scores": torch.zeros(b, 3, g, g),
            "det_ltrb": torch.full((b, 3, 4, g, g), 8.0), "seg_logits": torch.zeros(b, 4, h, w),
            "dist_logits": dist_logits}


CFG_DIST = {"seed": 42,
            "model": {"det_stride": 8, "seg_outputs": ["semantic4", "dist"]},
            "loss": {"seg_ce_weights": [1.0, 10.8, 8.7, 8.7], "dist_max_mm": 8.0, "dist_huber_beta": 0.1}}


def test_la_distancia_entra_dentro_del_termino_seg_y_no_como_cuarto_termino():
    """El enunciado §4.1 fija TRES términos (cls, det, seg): la regresión va dentro de ``seg``."""
    torch = pytest.importorskip("torch")
    from pengwin.losses.losses import TERMS, MultiTaskLoss

    lote = _lote_de_un_corte()
    parts = MultiTaskLoss(CFG_DIST)(_salida_falsa(lote, torch.zeros(1, 1, 32, 32)), lote)
    assert TERMS == ("cls", "det", "seg") and "dist" not in TERMS
    assert float(parts["seg"]) == pytest.approx(float(parts["seg_ce"] + parts["seg_dice"]
                                                      + parts["dist_reg"]), rel=1e-5)
    assert float(parts["total"]) == pytest.approx(float(parts["cls"] + parts["det"] + parts["seg"]), rel=1e-5)
    assert "edge_bce" not in parts, "sin cabeza de borde binaria no hay término de borde binario"


def test_la_perdida_de_distancia_esta_enmascarada_al_hueso():
    """Fuera del hueso la distancia no existe: cambiar la predicción ahí no debe mover nada."""
    torch = pytest.importorskip("torch")
    from pengwin.losses.losses import MultiTaskLoss

    lote = _lote_de_un_corte()
    hueso = (lote["semantic"] > 0)[:, None]
    base = torch.zeros(1, 1, 32, 32)
    ruido = base.clone()
    ruido[~hueso] = 7.0                                   # disparate solo en el fondo
    loss = MultiTaskLoss(CFG_DIST)
    a = float(loss(_salida_falsa(lote, base), lote)["dist_reg"])
    b = float(loss(_salida_falsa(lote, ruido), lote)["dist_reg"])
    assert a == pytest.approx(b, abs=1e-7)
    # y sí cambia si el disparate cae dentro del hueso
    dentro = base.clone()
    dentro[hueso] = 7.0
    assert float(loss(_salida_falsa(lote, dentro), lote)["dist_reg"]) != pytest.approx(a, abs=1e-4)


def test_la_perdida_distingue_el_objetivo_correcto():
    torch = pytest.importorskip("torch")
    from pengwin.losses.losses import MultiTaskLoss

    lote = _lote_de_un_corte()
    # logit exacto del objetivo (sigmoid^-1), acotado para no dar inf
    p = lote["dist"].clamp(1e-3, 1 - 1e-3)
    perfecto = torch.log(p / (1 - p))
    loss = MultiTaskLoss(CFG_DIST)
    bien = float(loss(_salida_falsa(lote, perfecto), lote)["dist_reg"])
    mal = float(loss(_salida_falsa(lote, -perfecto), lote)["dist_reg"])     # distancia invertida
    assert bien < 1e-4 and mal > 100 * max(bien, 1e-6), f"{bien:.5f} vs {mal:.5f}"


def test_la_perdida_baja_optimizando_la_salida_de_distancia():
    """120 pasos de descenso sobre los logits bajan el término ``seg`` y la distancia predicha
    acaba a menos de 0,3 mm de la real dentro del hueso: la pérdida es optimizable y apunta
    al sitio correcto."""
    torch = pytest.importorskip("torch")
    from pengwin.losses.losses import MultiTaskLoss

    lote = _lote_de_un_corte()
    logits = torch.zeros(1, 1, 32, 32, requires_grad=True)
    loss = MultiTaskLoss(CFG_DIST)
    opt = torch.optim.Adam([logits], lr=0.3)
    sched = torch.optim.lr_scheduler.StepLR(opt, 40, 0.3)   # sin decaimiento Adam oscila en ±lr
    seg, reg = [], []
    for _ in range(120):
        parts = loss(_salida_falsa(lote, logits), lote)
        opt.zero_grad(set_to_none=True)
        parts["seg"].backward()
        opt.step()
        sched.step()
        seg.append(float(parts["seg"].detach()))
        reg.append(float(parts["dist_reg"].detach()))
    assert seg[-1] < seg[0], f"el término seg no baja: {seg[0]:.3f} -> {seg[-1]:.3f}"
    assert reg[-1] < 0.1 * reg[0], f"distancia: {reg[0]:.4f} -> {reg[-1]:.4f}"
    hueso = (lote["semantic"] > 0)[:, None]
    err_mm = ((torch.sigmoid(logits.detach()) - lote["dist"]).abs() * 8.0)[hueso]
    assert float(err_mm.mean()) < 0.3, f"error medio {float(err_mm.mean()):.3f} mm"


def test_el_modelo_anade_dist_sin_tocar_las_otras_cabezas():
    torch = pytest.importorskip("torch")

    from conftest import REPO
    from pengwin.models.cbam import CBAM
    from pengwin.models.pengwin_net import PengwinNet, count_parameters
    from pengwin.utils.config import load_config

    torch.manual_seed(0)                                   # determinista: solo se miran formas
    base = PengwinNet(load_config(REPO / "configs" / "base.yaml")["model"]).eval()
    red = PengwinNet(load_config(REPO / "configs" / "tuning" / "f2b2_dist.yaml")["model"]).eval()
    out = red(torch.rand(1, 3, 64, 64))
    assert out["dist_logits"].shape == (1, 1, 64, 64), "un canal a resolución completa"
    assert out["seg_logits"].shape == (1, 4, 64, 64), "la semántica de 4 clases no cambia"
    assert "core_logits" not in out, "F2B2 no usa la representación de 3 clases de F2B1"
    # P(borde) se sigue publicando como 1 − distancia/dist_max: el posproceso edge/edt y
    # inference/volume.py funcionan igual sobre este checkpoint.
    assert torch.allclose(torch.sigmoid(out["edge_logits"]), 1 - torch.sigmoid(out["dist_logits"]), atol=1e-6)
    for nombre in ("backbone", "neck", "cls_head", "det_head"):
        n_base = sum(p.numel() for p in getattr(base, nombre).parameters())
        assert sum(p.numel() for p in getattr(red, nombre).parameters()) == n_base, nombre
    assert sum(isinstance(m, CBAM) for m in red.modules()) == 2, "el CBAM sigue en los bloques 3 y 4"
    assert count_parameters(red)["seg_head"] == count_parameters(base)["seg_head"], \
        "la conv 1×1 de borde binario se cambia por la de distancia: mismo tamaño"


def test_la_distancia_en_mm_sale_de_la_sigmoide():
    torch = pytest.importorskip("torch")
    from pengwin.models.heads import SegmentationHead

    z = torch.tensor([[[[-10.0, 0.0, 10.0]]]])
    mm = SegmentationHead.dist_mm_from_logits(z, 8.0)
    assert mm.min() >= 0.0 and mm.max() <= 8.0
    assert float(mm[0, 0, 0, 1]) == pytest.approx(4.0), "logit 0 = la mitad del rango"


def test_el_gradiente_llega_a_la_salida_nueva():
    """Cableado completo: objetivo -> lote -> canal de regresión -> pérdida -> gradiente."""
    torch = pytest.importorskip("torch")

    from conftest import REPO
    from pengwin.losses.losses import MultiTaskLoss
    from pengwin.models.pengwin_net import PengwinNet
    from pengwin.utils.config import load_config

    torch.manual_seed(0)
    cfg = load_config(REPO / "configs" / "tuning" / "f2b2_dist.yaml")
    model = PengwinNet(cfg["model"])
    lote = _lote_de_un_corte(64)
    lote["image"] = torch.rand(1, 3, 64, 64)
    MultiTaskLoss(cfg)(model(lote["image"]), lote)["total"].backward()
    for mod, nombre in ((model.seg_head.dist, "seg_head.dist"), (model.seg_head.semantic, "seg_head.semantic"),
                        (model.cls_head, "cls_head"), (model.det_head, "det_head")):
        assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in mod.parameters()), nombre
    assert model.seg_head.edge is None, "con dist no hay cabeza de borde binaria"
    assert model.seg_head.core is None, "ni la de 3 clases de F2B1"


def test_predict_case_publica_la_distancia_en_mm_y_el_npz_la_redondea(tmp_path):
    """Camino completo de inferencia: modelo -> volumen (mm) -> uint8 del .npz -> mm otra vez.

    El .npz guarda ``dist`` con 0,1 mm por nivel (0..25,5 mm), el mismo factor que el caché, así
    que ``tune_postprocess`` lo lee sin saber cuánto vale ``loss.dist_max_mm``.
    """
    torch = pytest.importorskip("torch")
    from conftest import REPO
    from pengwin.inference.volume import predict_case
    from pengwin.models.pengwin_net import PengwinNet
    from pengwin.utils.config import load_config

    torch.manual_seed(0)
    lab = _hueso_partido()
    _cache_falso(tmp_path, lab, spacing=(1.0, 1.0, 1.0))
    cfg = load_config(REPO / "configs" / "tuning" / "f2b2_dist.yaml")
    model = PengwinNet(cfg["model"]).eval()
    p = predict_case(model, tmp_path / "cache", "900", cfg, torch.device("cpu"), batch_size=3)
    assert p["dist"].shape == lab.shape and p["dist"].dtype == np.float16
    assert float(p["dist"].min()) >= 0.0 and float(p["dist"].max()) <= 8.0, "la sigmoide acota el rango"
    assert "core" not in p, "no hay salida core3 en F2B2"
    u8 = np.round(np.clip(p["dist"].astype(np.float32), 0, 255 * DIST_MM_PER_LEVEL)
                  / DIST_MM_PER_LEVEL).astype(np.uint8)
    assert np.allclose(u8.astype(np.float32) * DIST_MM_PER_LEVEL, p["dist"].astype(np.float32), atol=0.05)
    # P(borde) publicada = 1 − dist/dist_max: el posproceso "edge"/"edt" sigue corriendo
    assert np.allclose(p["edge"].astype(np.float32), 1 - p["dist"].astype(np.float32) / 8.0, atol=0.01)


def test_la_config_f2b2_solo_cambia_lo_necesario():
    pytest.importorskip("yaml")
    from conftest import REPO
    from pengwin.utils.config import load_config

    base = load_config(REPO / "configs" / "base.yaml")
    cfg = load_config(REPO / "configs" / "tuning" / "f2b2_dist.yaml")
    assert cfg["model"]["seg_outputs"] == ["semantic4", "dist"]
    assert cfg["postprocess"]["instance_method"] == "dist"
    assert cfg["loss"]["terms"] == ["cls", "det", "seg"], "la pérdida sigue siendo de 3 términos"
    assert cfg["loss"]["dist_max_mm"] == cfg["postprocess"]["dist_max_mm"], "mismo recorte en las dos"
    distintas = {k for k in base["model"] if base["model"][k] != cfg["model"][k]}
    assert distintas == {"seg_outputs"}, f"la config toca más de lo necesario: {distintas}"
