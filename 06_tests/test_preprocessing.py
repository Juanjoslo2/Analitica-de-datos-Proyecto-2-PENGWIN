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
