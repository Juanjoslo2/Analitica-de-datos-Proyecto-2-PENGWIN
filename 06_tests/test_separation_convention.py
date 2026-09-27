"""Convención de la distancia de separación (EDT con spacing físico) sobre phantoms.

Fija por test lo que el informe afirma: la EDT mide de centro a centro de vóxel,
así que dos fragmentos que se tocan dan 1 vóxel (no 0) y un hueco de k vóxeles
vacíos a lo largo de z da (k + 1) · dz mm. También cubre el error típico de pasar
el spacing en orden (x, y, z) en lugar de (z, y, x).
"""

import numpy as np
import pytest
from scipy import ndimage as ndi

from pengwin.eda.extract import _pairwise_min_distance, connectivity_stats

SPACING_ZYX = (2.0, 0.8, 0.8)          # anisotrópico a propósito: dz != dx


def _two_blocks(gap_z: int):
    vol = np.zeros((20, 12, 12), bool)
    main = vol.copy()
    frag = vol.copy()
    main[2:6, 3:9, 3:9] = True
    frag[6 + gap_z:9 + gap_z, 3:9, 3:9] = True
    return main, frag


def _edt_min(main, frag, sampling):
    return float(ndi.distance_transform_edt(~main, sampling=sampling)[frag].min())


def test_touching_fragments_measure_one_voxel():
    main, frag = _two_blocks(gap_z=0)
    assert _edt_min(main, frag, SPACING_ZYX) == pytest.approx(2.0)   # 1 vóxel en z


def test_gap_is_center_to_center():
    main, frag = _two_blocks(gap_z=2)                                  # 2 vóxeles vacíos
    assert _edt_min(main, frag, SPACING_ZYX) == pytest.approx(3 * 2.0)


def test_spacing_order_matters():
    main, frag = _two_blocks(gap_z=2)
    wrong = _edt_min(main, frag, SPACING_ZYX[::-1])                    # (x, y, z) por error
    assert wrong != pytest.approx(6.0)


def test_pairwise_helper_matches_direct_edt():
    main, frag = _two_blocks(gap_z=3)
    assert _pairwise_min_distance(main, frag, SPACING_ZYX) == pytest.approx(_edt_min(main, frag, SPACING_ZYX))


def test_connectivity_stats_detects_split_fragment():
    m = np.zeros((10, 10, 10), bool)
    m[1:5, 1:5, 1:5] = True           # pieza grande (64 vóxeles)
    m[7:9, 7:9, 7:9] = True           # pieza pequeña (8 vóxeles), separada
    n, frac, n_big = connectivity_stats(m, vox_mm3=10.0)               # 80 y 640 mm3
    assert n == 2
    assert frac == pytest.approx(64 / 72)
    assert n_big == 1                  # solo la pieza de 640 mm3 supera 100 mm3
