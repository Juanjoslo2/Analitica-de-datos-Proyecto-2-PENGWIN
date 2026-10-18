"""Objetivos por corte (cajas por región, ignorar, borde de fractura) sobre phantoms 2D."""

import numpy as np

from pengwin.data.targets import fracture_edge_2d, region_boxes_2d, region_of, slice_targets


def _phantom():
    lab = np.zeros((64, 64), np.uint8)
    lab[10:30, 5:20] = 11          # coxal izq., principal
    lab[10:30, 20:25] = 12         # coxal izq., fragmento que TOCA al principal (x = 20)
    lab[40:50, 8:14] = 11          # isla separada del mismo coxal (misma región)
    lab[20:40, 30:40] = 1          # sacro
    lab[20:40, 40:44] = 21         # coxal der. pegado al sacro: interfaz entre REGIONES
    lab[60:62, 60:62] = 2          # fragmento de sacro diminuto, lejos
    return lab


def test_region_of_maps_ranges():
    assert region_of(np.array([0, 1, 10, 11, 20, 21, 30])).tolist() == [0, 1, 1, 2, 2, 3, 3]


def test_box_is_envelope_of_all_islands():
    b = region_boxes_2d(_phantom())
    assert b["present"].tolist() == [True, True, True]
    # coxal izq.: islas en y 10-29 y 40-49, x 5-24 -> una sola caja que las envuelve
    assert b["boxes"][1].tolist() == [5, 10, 25, 50]
    # sacro: el bloque grande + el fragmento diminuto lejano
    assert b["boxes"][0].tolist() == [30, 20, 62, 62]


def test_small_box_is_ignored():
    lab = np.zeros((32, 32), np.uint8)
    lab[5:8, 5:20] = 1             # 3 px de alto
    b = region_boxes_2d(lab, min_box_px=4)
    assert b["present"][0] and b["ignore"][0]


def test_edge_only_between_fragments_of_same_region():
    lab = _phantom()
    e = fracture_edge_2d(lab, dilation_px=0)
    assert e[15, 19] and e[15, 20], "contacto 11-12 debe ser borde de fractura"
    assert not e[25, 39] and not e[25, 40], "la interfaz sacro-coxal no es fractura"
    assert not e[45, 10], "una isla separada no tiene borde de contacto"


def test_edge_dilation_stays_in_bone():
    lab = _phantom()
    e = fracture_edge_2d(lab, dilation_px=2)
    assert not e[lab == 0].any()
    assert e[15, 17:23].all()


def test_empty_slice():
    t = slice_targets(np.zeros((16, 16), np.uint8))
    assert not t["present"].any() and t["edge"].sum() == 0 and t["semantic"].max() == 0


def test_edge_3d_sees_contact_between_slices():
    """Dos fragmentos apilados en z: el borde 2D no ve el contacto; el 3D sí."""
    from pengwin.data.targets import fracture_edge_3d

    lab = np.zeros((6, 16, 16), np.uint8)
    lab[0:3, 4:12, 4:12] = 11          # principal abajo
    lab[3:6, 4:12, 4:12] = 12          # fragmento encima, contacto entre z=2 y z=3
    e2d = np.stack([fracture_edge_2d(lab[z], 0) for z in range(6)])
    e3d = fracture_edge_3d(lab, dilation=0)
    assert not e2d.any()
    assert e3d[2, 4:12, 4:12].all() and e3d[3, 4:12, 4:12].all() and not e3d[0].any()
    assert not fracture_edge_3d(lab, dilation=2)[lab == 0].any(), "la dilatación no sale del hueso"
