"""Pruebas del preprocesado clásico (ventana HU y recorte del cuerpo)."""

import numpy as np

from pengwin.data.data_loader import apply_bone_window
from pengwin.data.preprocessing import apply_crop, body_mask_2d, compute_body_crop


def test_bone_window_range():
    v = np.array([-2000, -500, 400, 1300, 40000], np.float32)
    w = apply_bone_window(v, window_center=400, window_width=1800)
    assert w.min() == 0.0 and w.max() == 1.0
    assert np.isclose(w[2], 0.5)


def _phantom():
    vol = np.full((4, 64, 64), -1000, np.int16)
    yy, xx = np.mgrid[:64, :64]
    body = (yy - 28) ** 2 + (xx - 32) ** 2 < 18**2          # paciente
    vol[:, body] = 40
    vol[:, 56:59, 4:60] = 200                                # camilla separada del paciente
    return vol, body


def test_body_mask_removes_table():
    vol, body = _phantom()
    m = body_mask_2d(vol)
    assert not m[56:59].any(), "La camilla no debe formar parte del cuerpo"
    assert m[body].mean() > 0.95


def test_body_crop_is_square_and_contains_body():
    vol, body = _phantom()
    crop = compute_body_crop(vol, (1.0, 0.8, 0.8), margin_mm=0)
    out = apply_crop(vol, crop)
    assert out.shape[-1] == out.shape[-2] == crop.side_px
    assert (out > -500).sum() >= body.sum() * vol.shape[0] * 0.95


def _skeleton_phantom(split_pelvis: bool, with_hand: bool):
    """Volumen 20×96×96 con 'pelvis' ósea, opcionalmente partida en dos mitades y con una 'mano'."""
    vol = np.full((20, 96, 96), -1000, np.int16)
    yy, xx = np.mgrid[:96, :96]
    vol[:, (yy - 50) ** 2 / 40**2 + (xx - 48) ** 2 / 44**2 < 1] = 40      # cuerpo
    if split_pelvis:                                                     # dos mitades separadas 6 px
        vol[:, 40:60, 14:45] = 600
        vol[:, 40:60, 51:82] = 600
    else:
        vol[:, 40:60, 14:82] = 600
    if with_hand:
        vol[:, 16:20, 44:52] = 600                                       # hueso pequeño y lejano
    return vol


def test_bone_crop_keeps_both_pelvis_halves():
    from pengwin.data.preprocessing import compute_bone_crop

    vol = _skeleton_phantom(split_pelvis=True, with_hand=False)
    c = compute_bone_crop(vol, (1.0, 1.0, 1.0), margin_mm=2)
    assert c.x0 <= 14 and c.x1 >= 82, "Una mitad de la pelvis quedó fuera del recorte"


def test_bone_crop_ignores_distant_hand():
    from pengwin.data.preprocessing import compute_bone_crop

    with_hand = compute_bone_crop(_skeleton_phantom(False, True), (1.0, 1.0, 1.0), margin_mm=2)
    without = compute_bone_crop(_skeleton_phantom(False, False), (1.0, 1.0, 1.0), margin_mm=2)
    assert with_hand.side_px == without.side_px
