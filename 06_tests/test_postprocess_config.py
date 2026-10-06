"""Resolución del bloque ``postprocess`` de la config y su llegada a ``separate_instances``.

El posproceso se ajusta sin reentrenar, así que sus parámetros viajan por tres sitios: la config
del checkpoint, la config actual del repositorio y los argumentos de la CLI. Estos tests fijan
quién manda y comprueban, sobre phantoms, que cada parámetro llega de verdad al separador.
"""

import numpy as np
import pytest

from pengwin.postprocess.instances import (
    PP_DEFAULTS, instance_kwargs, resolve_postprocess, separate_instances,
)

SP = (1.0, 1.0, 1.0)


def test_sin_fuentes_devuelve_los_valores_por_defecto():
    assert resolve_postprocess() == PP_DEFAULTS


def test_la_fuente_mas_a_la_derecha_manda_y_los_none_no_pisan():
    pp = resolve_postprocess({"seed_depth_mm": 5.0, "edge_threshold": 0.2},   # config del checkpoint
                             {"seed_depth_mm": 3.0},                          # config del repositorio
                             {"seed_depth_mm": None, "edge_weight": 2.0})     # CLI (lo no dado es None)
    assert pp["seed_depth_mm"] == 3.0, "la CLI no dio valor: gana la config del repositorio"
    assert pp["edge_threshold"] == 0.2, "lo que solo está en el checkpoint se conserva"
    assert pp["edge_weight"] == 2.0, "la CLI manda cuando sí da valor"


def test_las_claves_desconocidas_no_pasan():
    pp = resolve_postprocess({"det_score_threshold": 0.3, "nms_iou": 0.5, "seed_min_cm3": 0.01})
    assert "det_score_threshold" not in pp and "nms_iou" not in pp, "solo las claves del separador"
    assert pp["seed_min_cm3"] == 0.01


def test_instance_kwargs_son_exactamente_los_argumentos_del_separador():
    import inspect

    kw = instance_kwargs(resolve_postprocess())
    acepta = set(inspect.signature(separate_instances).parameters) - {"semantic", "edge", "spacing_zyx"}
    assert set(kw) <= acepta, "algún argumento no existe en separate_instances"
    assert kw["method"] == PP_DEFAULTS["instance_method"], "instance_method se renombra a method"


def _bloques_con_cuello():
    """Coxal izq.: dos bloques gruesos unidos por un cuello de 4 px de ancho, sin borde predicho.

    Es el caso que motiva el método "edt": la región rellena la grieta, así que solo la erosión
    por distancia separa los dos fragmentos. Los bloques tienen que ser más gruesos que el doble
    de la profundidad que se vaya a probar, o no quedaría ninguna semilla.
    """
    sem = np.zeros((16, 40, 30), np.uint8)
    sem[1:15, 4:18, 4:26] = 2             # bloque principal (14 px en z e y, 22 en x)
    sem[1:15, 18:22, 13:17] = 2           # cuello estrecho: su distancia al fondo no pasa de ~2 mm
    sem[1:15, 22:36, 4:26] = 2            # bloque secundario
    return sem, np.zeros(sem.shape, np.float32)


@pytest.mark.parametrize("profundidad, n_esperado", [(3.0, 2), (12.0, 1)])
def test_seed_depth_mm_llega_y_decide_cuantos_fragmentos_salen(profundidad, n_esperado):
    sem, edge = _bloques_con_cuello()
    pp = resolve_postprocess({"instance_method": "edt"}, {"seed_depth_mm": profundidad})
    lab = separate_instances(sem, edge, SP, **instance_kwargs(pp))
    assert len(np.unique(lab[lab > 0])) == n_esperado


def test_seed_min_cm3_llega_y_descarta_las_semillas_pequenas():
    sem, edge = _bloques_con_cuello()
    base = {"instance_method": "edt", "seed_depth_mm": 3.0}
    pocas = separate_instances(sem, edge, SP, **instance_kwargs(resolve_postprocess(base, {"seed_min_cm3": 0.001})))
    una = separate_instances(sem, edge, SP, **instance_kwargs(resolve_postprocess(base, {"seed_min_cm3": 1000.0})))
    assert len(np.unique(pocas[pocas > 0])) == 2
    assert len(np.unique(una[una > 0])) == 1, "con el umbral altísimo solo sobrevive la semilla mayor"


def test_max_fragments_limita_las_instancias():
    sem = np.zeros((6, 10, 40), np.uint8)
    for k in range(5):                                  # 5 bloques separados por un hueco de 3 px
        sem[1:5, 2:8, 2 + 8 * k: 7 + 8 * k] = 1
    edge = np.zeros(sem.shape, np.float32)
    pp = resolve_postprocess({"instance_method": "edt", "seed_depth_mm": 1.5, "max_fragments": 3})
    lab = separate_instances(sem, edge, SP, **instance_kwargs(pp))
    assert len(np.unique(lab[lab > 0])) == 3, "max_fragments recorta a las 3 semillas mayores"
