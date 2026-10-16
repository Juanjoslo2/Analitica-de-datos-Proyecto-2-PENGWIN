"""Semana 11 §1.2: cierre de huecos de hueso delgado en el posproceso (``fill_thin_holes``).

Lo que se fija aquí:
    solo añade      ningún vóxel de un fragmento cambia ni desaparece; lo nuevo era fondo
    no pisa         un fragmento vecino queda idéntico
    regiones        con regions=(2, 3) el sacro (ids 1-10) no se toca
    cierra          un hueco pequeño en una lámina de 3 vóxeles queda tapado con 3 mm (la bola,
                    con un tapón en la capa del medio; el cubo, entero)
    apagado         fill_holes_mm = 0 (el valor por defecto de la config) no cambia nada
"""

import numpy as np

from pengwin.postprocess.instances import PP_DEFAULTS, fill_thin_holes, instance_kwargs, resolve_postprocess

SP = (1.0, 1.36, 1.36)


def _volumen():
    lab = np.zeros((24, 40, 40), np.uint8)
    lab[10:13, 5:35, 5:28] = 11                 # lámina de 3 vóxeles del coxal izq.
    lab[10:13, 18:20, 15:17] = 0                # hueco de 2 × 2 px (≈ 2,7 mm)
    lab[10:13, 5:35, 30:35] = 12                # otro fragmento del mismo hueso
    lab[18:21, 5:35, 5:35] = 1                  # sacro con un hueco igual
    lab[18:21, 18:20, 15:17] = 0
    return lab


def test_solo_anade_fondo_y_no_pisa_otro_fragmento():
    lab = _volumen()
    out = fill_thin_holes(lab, SP, 3.0)
    cambiados = out != lab
    assert cambiados.any()
    assert (lab[cambiados] == 0).all()
    assert ((out == 12) == (lab == 12)).all()


def test_cierra_el_hueco_y_respeta_regiones():
    lab = _volumen()
    out = fill_thin_holes(lab, SP, 3.0, regions=(2, 3))
    assert (out[11, 18:20, 15:17] == 11).all()             # la bola tapa el túnel en la capa del medio
    assert (fill_thin_holes(lab, SP, 3.0, regions=(2, 3), shape="caja")[10:13, 18:20, 15:17] == 11).all()
    assert (out[18:21, 18:20, 15:17] == 0).all()           # el sacro no se tocó
    assert ((out == 1) == (lab == 1)).all()


def test_apagado_por_defecto():
    lab = _volumen()
    assert PP_DEFAULTS["fill_holes_mm"] == 0.0
    assert instance_kwargs(resolve_postprocess(None))["fill_holes_mm"] == 0.0
    assert fill_thin_holes(lab, SP, 0.0) is lab
