"""Pruebas de reorientación LPS y lateralidad.

1. Sintéticas (siempre corren): se construye un volumen con un "coxal izquierdo"
   conocido y se guarda con dirección LPS y RAS. Tras cargarlo, el objeto
   izquierdo debe quedar en x mayor en ambos casos.
2. Con datos reales (se omiten si no está 01_data): un caso LPS (001) y uno RAS (002).
"""

import numpy as np
import pytest
import SimpleITK as sitk

from pengwin.data.data_loader import (
    get_dataset_pairs,
    load_and_standardize_mha,
    verify_left_right_consistency,
)

RAS_DIRECTION = (-1.0, 0, 0, 0, -1.0, 0, 0, 0, 1.0)
LPS_DIRECTION = (1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0)


def _synthetic_label(direction, tmp_path, name):
    """Etiqueta 11 (coxal izq.) en el lado IZQUIERDO físico del paciente y 21 (der.) en el derecho.

    Se define en coordenadas físicas LPS (x+ = izquierda) y se escribe con la
    dirección pedida; para RAS el arreglo en disco queda espejado en x e y.
    """
    arr_lps = np.zeros((8, 16, 32), np.int16)          # (Z, Y, X) en LPS
    arr_lps[:, 4:12, 24:30] = 11                       # x grande = izquierda del paciente
    arr_lps[:, 4:12, 2:8] = 21
    arr_disk = arr_lps if direction == LPS_DIRECTION else arr_lps[:, ::-1, ::-1].copy()
    img = sitk.GetImageFromArray(arr_disk)
    img.SetDirection(direction)
    img.SetSpacing((0.8, 0.8, 1.0))
    path = tmp_path / f"{name}.mha"
    sitk.WriteImage(img, str(path))
    return path


@pytest.mark.parametrize("direction,name", [(LPS_DIRECTION, "lps"), (RAS_DIRECTION, "ras")])
def test_synthetic_left_is_high_x(direction, name, tmp_path):
    path = _synthetic_label(direction, tmp_path, name)
    mask, spacing, orig = load_and_standardize_mha(path, is_label=True)
    assert orig == name.upper()
    assert mask.dtype == np.uint8
    assert spacing == pytest.approx((1.0, 0.8, 0.8))    # (dz, dy, dx)
    ok, msg = verify_left_right_consistency(mask)
    assert ok, msg


def test_detects_unreoriented_ras(tmp_path):
    """Si NO se reorienta un RAS, el test de lateralidad debe fallar (antes pasaba siempre)."""
    path = _synthetic_label(RAS_DIRECTION, tmp_path, "ras_raw")
    raw = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    ok, _ = verify_left_right_consistency(raw)
    assert not ok


@pytest.mark.data
@pytest.mark.parametrize("case_id,expected_orig", [("001", "LPS"), ("002", "RAS")])
def test_real_cases_lateral_consistency(data_dir, case_id, expected_orig):
    pair = next(p for p in get_dataset_pairs(data_dir) if p["case_id"] == case_id)
    mask, spacing, orig = load_and_standardize_mha(pair["label_path"], is_label=True)
    assert orig == expected_orig
    ok, msg = verify_left_right_consistency(mask)
    assert ok, msg
