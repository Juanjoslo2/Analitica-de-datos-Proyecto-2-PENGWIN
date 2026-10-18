"""Semillas por h-máximas de la transformada de distancia (`method="hmax"`) [F2C1].

El método por umbral global (`edt`) no puede servir a la vez a dos fragmentos grandes fusionados
—que solo se separan erosionando profundo— y a un fragmento pequeño, que desaparece a esa misma
profundidad. Estos tests fijan ese comportamiento con phantoms: el mismo volumen donde `edt` pierde
el fragmento pequeño, `hmax` lo conserva.
"""

import numpy as np

from pengwin.postprocess.instances import (
    PP_DEFAULTS, instance_kwargs, resolve_postprocess, separate_instances,
)

SP = (1.0, 1.0, 1.0)


def _grande_y_pequeno_con_cuello():
    """Sacro: bloque grande y bloque pequeño unidos por un **cuello estrecho**.

    El grande tiene ~8 mm de profundidad interior y el pequeño ~2,5 mm. El cuello hace que la
    distancia baje entre los dos, así que cada bloque tiene su propia máxima local. Un umbral
    global de 5 mm deja semilla solo en el grande; las h-máximas encuentran las dos.
    """
    sem = np.zeros((20, 44, 30), np.uint8)
    sem[2:18, 4:20, 4:26] = 1            # grande
    sem[8:12, 20:24, 13:17] = 1          # cuello estrecho (4x4x4)
    sem[2:18, 24:29, 10:20] = 1          # pequeño
    return sem, np.zeros(sem.shape, np.float32)


def _pegados_por_cara_ancha():
    """El caso en que las h-máximas NO ayudan: unión por una cara ancha, sin cuello.

    Sin cuello la distancia crece de forma monótona del bloque pequeño al grande, así que el
    pequeño no tiene máxima local y ningún método basado en la distancia lo separa. Queda como
    límite documentado del método, no como un fallo.
    """
    sem = np.zeros((20, 40, 30), np.uint8)
    sem[2:18, 4:20, 4:26] = 1
    sem[2:18, 20:25, 10:20] = 1
    return sem, np.zeros(sem.shape, np.float32)


def test_hmax_conserva_el_fragmento_pequeno_donde_edt_lo_pierde():
    sem, edge = _grande_y_pequeno_con_cuello()
    kw = instance_kwargs(resolve_postprocess({"instance_method": "edt", "seed_depth_mm": 5.0}))
    lab_e = separate_instances(sem, edge, SP, **kw)
    kw = instance_kwargs(resolve_postprocess({"instance_method": "hmax", "hmax_h_mm": 1.0}))
    lab_h = separate_instances(sem, edge, SP, **kw)
    assert len(np.unique(lab_e[lab_e > 0])) == 1, "el umbral global de 5 mm deja sin semilla al pequeño"
    assert len(np.unique(lab_h[lab_h > 0])) == 2, "las h-máximas encuentran un montículo por bloque"


def test_limite_conocido_sin_cuello_ningun_metodo_separa():
    sem, edge = _pegados_por_cara_ancha()
    for met, extra in (("edt", {"seed_depth_mm": 5.0}), ("hmax", {"hmax_h_mm": 1.0})):
        kw = instance_kwargs(resolve_postprocess({"instance_method": met, **extra}))
        lab = separate_instances(sem, edge, SP, **kw)
        assert len(np.unique(lab[lab > 0])) == 1, f"{met}: sin cuello la distancia no tiene dos máximas"


def test_hmax_respeta_max_fragments():
    sem = np.zeros((12, 12, 60), np.uint8)
    for k in range(5):                       # 5 bloques separados por un hueco de 3 px
        sem[2:10, 2:10, 2 + 11 * k: 10 + 11 * k] = 1
    edge = np.zeros(sem.shape, np.float32)
    kw = instance_kwargs(resolve_postprocess({"instance_method": "hmax", "hmax_h_mm": 1.0,
                                              "max_fragments": 3}))
    lab = separate_instances(sem, edge, SP, **kw)
    assert len(np.unique(lab[lab > 0])) == 3, "max_fragments recorta a los 3 montículos mayores"


def test_hmax_respeta_la_taxonomia_por_region():
    sem = np.zeros((16, 30, 30), np.uint8)
    sem[2:14, 4:14, 4:26] = 2                # coxal izquierdo
    sem[2:14, 16:26, 4:26] = 3               # coxal derecho
    edge = np.zeros(sem.shape, np.float32)
    kw = instance_kwargs(resolve_postprocess({"instance_method": "hmax", "hmax_h_mm": 1.0}))
    lab = separate_instances(sem, edge, SP, **kw)
    ids = set(np.unique(lab[lab > 0]).tolist())
    assert all(11 <= i <= 20 for i in ids if i < 21), "el coxal izquierdo usa 11-20"
    assert any(21 <= i <= 30 for i in ids), "el coxal derecho usa 21-30"


def test_hmax_es_un_parametro_resuelto_y_tiene_valor_por_defecto():
    assert "hmax_h_mm" in PP_DEFAULTS
    pp = resolve_postprocess({"hmax_h_mm": 2.5})
    assert pp["hmax_h_mm"] == 2.5 and "hmax_h_mm" in instance_kwargs(pp)


def test_sin_monticulos_cae_al_metodo_por_umbral_sin_fallar():
    """Un volumen de 1 vóxel no tiene montículos: debe devolver algo válido, no reventar."""
    sem = np.zeros((6, 6, 6), np.uint8)
    sem[3, 3, 3] = 1
    kw = instance_kwargs(resolve_postprocess({"instance_method": "hmax"}))
    lab = separate_instances(sem, np.zeros(sem.shape, np.float32), SP, **kw)
    assert lab[3, 3, 3] == 1 and (lab > 0).sum() == 1


def test_hibrido_rescata_el_pequeno_y_no_parte_el_grande():
    """`edt_hmax`: las semillas profundas mandan y la h-máxima solo rescata lo que quedó huérfano.

    En el phantom con cuello, `edt` a 5 mm pierde el bloque pequeño y `hmax` puede partir el grande
    (tiene relieve interno). El híbrido debe dar exactamente 2 fragmentos y dejar el grande entero.
    """
    sem, edge = _grande_y_pequeno_con_cuello()
    kw = instance_kwargs(resolve_postprocess({"instance_method": "edt_hmax", "seed_depth_mm": 5.0,
                                              "hmax_h_mm": 1.0}))
    lab = separate_instances(sem, edge, SP, **kw)
    ids, cuentas = np.unique(lab[lab > 0], return_counts=True)
    assert len(ids) == 2, "el híbrido separa los dos bloques"
    # el grande debe seguir siendo una sola pieza y la mayor
    grande = (sem[2:18, 4:20, 4:26] > 0).sum()
    assert cuentas.max() >= grande * 0.9, "el bloque grande no se parte"


def test_hibrido_sin_huerfanas_coincide_con_edt():
    """Si todas las componentes del núcleo ya tienen semilla profunda, el híbrido == `edt`."""
    sem = np.zeros((20, 40, 30), np.uint8)
    sem[2:18, 4:18, 4:26] = 1
    sem[2:18, 22:36, 4:26] = 1          # dos bloques gruesos y SEPARADOS
    edge = np.zeros(sem.shape, np.float32)
    a = separate_instances(sem, edge, SP, **instance_kwargs(resolve_postprocess(
        {"instance_method": "edt", "seed_depth_mm": 5.0})))
    b = separate_instances(sem, edge, SP, **instance_kwargs(resolve_postprocess(
        {"instance_method": "edt_hmax", "seed_depth_mm": 5.0, "hmax_h_mm": 1.0})))
    assert (a == b).all(), "sin componentes huérfanas el híbrido no cambia nada"
