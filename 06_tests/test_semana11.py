"""Semana 11: volúmenes oblicuos (orientación por cosenos directores) y HD95/ASSD por fragmento.

Un volumen con gantry inclinado no queda alineado con DICOMOrient (solo permuta y voltea ejes):
``align_to_axes`` lo remuestrea a la grilla LPS sin mover el contenido en el espacio físico.
"""

import math

import numpy as np
import SimpleITK as sitk

from pengwin.data.data_loader import align_to_axes, load_and_standardize_mha, oblique_angle_deg


def _esfera(direccion, centro_idx=(30, 20, 25), shape_zyx=(40, 50, 60), spacing=(0.8, 0.8, 2.0)):
    arr = np.full(shape_zyx, -1024, np.int16)
    z, y, x = np.ogrid[:shape_zyx[0], :shape_zyx[1], :shape_zyx[2]]
    cx, cy, cz = centro_idx
    arr[((x - cx) * spacing[0]) ** 2 + ((y - cy) * spacing[1]) ** 2 + ((z - cz) * spacing[2]) ** 2 <= 36] = 1000
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    img.SetOrigin((-10.0, 5.0, 100.0))
    img.SetDirection(direccion)
    centro_fis = img.TransformContinuousIndexToPhysicalPoint([float(c) for c in centro_idx])
    return img, np.array(centro_fis)


def _rot_x(grados):
    a = math.radians(grados)
    return (1.0, 0.0, 0.0, 0.0, math.cos(a), -math.sin(a), 0.0, math.sin(a), math.cos(a))


def test_volumen_alineado_no_se_toca():
    img, _ = _esfera(np.eye(3).ravel().tolist())
    assert oblique_angle_deg(img) == 0.0
    assert align_to_axes(img) is img


def test_volumen_oblicuo_queda_alineado_y_en_su_sitio(tmp_path):
    img, centro = _esfera(_rot_x(12.0))
    assert abs(oblique_angle_deg(img) - 12.0) < 1e-6
    out = align_to_axes(img)
    assert oblique_angle_deg(out) == 0.0
    # el hueso sigue en el mismo punto físico: centroide de los vóxeles > 500 HU
    arr = sitk.GetArrayFromImage(out)
    zyx = np.argwhere(arr > 500).mean(0)
    fis = np.array(out.TransformContinuousIndexToPhysicalPoint(zyx[::-1].tolist()))
    assert np.linalg.norm(fis - centro) < 1.0                  # < 1 mm
    # y el cargador del proyecto lo aplica solo
    p = tmp_path / "oblicuo.mha"
    sitk.WriteImage(img, str(p))
    vol, spacing, orig = load_and_standardize_mha(p)
    assert vol.shape == arr.shape and np.allclose(spacing, out.GetSpacing()[::-1])


def test_etiqueta_oblicua_conserva_ids():
    img, _ = _esfera(_rot_x(20.0))
    lab = sitk.Cast(sitk.BinaryThreshold(img, 500, 5000, 21, 0), sitk.sitkUInt8)
    out = sitk.GetArrayFromImage(align_to_axes(lab, is_label=True))
    assert set(np.unique(out)) == {0, 21}                      # vecino más cercano: no inventa ids


def test_hd95_assd_de_superficie():
    from pengwin.evaluation.fragment_metrics import surface_distances
    a = np.zeros((30, 30, 30), bool)
    a[8:20, 8:20, 8:20] = True
    assert surface_distances(a, a, (1.0, 1.0, 1.0)) == {"hd95_mm": 0.0, "assd_mm": 0.0}
    b = np.roll(a, 3, axis=0)                                  # desplazado 3 mm en z
    d = surface_distances(a, b, (1.0, 1.0, 1.0))
    assert 0 < d["assd_mm"] < d["hd95_mm"] <= 3.0 + 1e-6
    d2 = surface_distances(a, b, (2.0, 1.0, 1.0))              # con spacing z de 2 mm el desplazamiento mide 6 mm
    assert abs(d2["hd95_mm"] - 2 * d["hd95_mm"]) < 1e-6
