"""Semana 10 sobre phantoms: separación de fragmentos 3D, distancia en mm, métricas por fragmento
y vuelta a la grilla nativa."""

import numpy as np
import pytest

from pengwin.evaluation.fragment_metrics import distance_comparison, match_fragments, summarize
from pengwin.measurement.separation import separation_table
from pengwin.postprocess.instances import separate_instances, separate_region

SP = (1.0, 1.0, 1.0)


def _two_touching_blocks():
    """Coxal izq.: bloque grande (principal) y bloque pequeño pegados en x = 20, con borde predicho."""
    sem = np.zeros((10, 40, 40), np.uint8)
    sem[2:8, 5:35, 5:20] = 2
    sem[2:8, 5:35, 20:28] = 2
    edge = np.zeros(sem.shape, np.float32)
    edge[2:8, 5:35, 19:21] = 0.9                        # superficie de fractura
    return sem, edge


def test_touching_fragments_are_split_by_edge():
    sem, edge = _two_touching_blocks()
    lab = separate_instances(sem, edge, SP)
    ids = sorted(np.unique(lab[lab > 0]).tolist())
    assert ids == [11, 12], "dos fragmentos del coxal izquierdo"
    assert (lab == 11).sum() > (lab == 12).sum(), "el principal es el más grande"
    assert ((lab > 0) == (sem > 0)).all(), "el watershed devuelve también los píxeles de borde"


def test_without_edge_they_merge():
    sem, _ = _two_touching_blocks()
    lab = separate_instances(sem, np.zeros(sem.shape, np.float32), SP)
    assert np.unique(lab[lab > 0]).tolist() == [11], "sin borde predicho se fusionan (modo de falla conocido)"


def test_small_component_is_not_a_fragment():
    m = np.zeros((10, 30, 30), bool)
    m[2:8, 2:20, 2:20] = True
    m[5, 25, 25] = True                                  # 1 vóxel aislado = 0,001 cm³
    frag = separate_region(m, np.zeros(m.shape, np.float32), SP, min_fragment_cm3=0.1)
    assert frag.max() == 1 and frag[5, 25, 25] == 1, "la pieza diminuta se reasigna, no se vuelve fragmento"


def test_separation_distance_in_mm():
    lab = np.zeros((20, 20, 20), np.uint8)
    lab[2:6, 5:15, 5:15] = 1                             # principal del sacro
    lab[9:12, 5:15, 5:15] = 2                            # 3 cortes vacíos de por medio
    lab[2:6, 5:15, 15:18] = 21                           # coxal der. sin fracturar (no aporta filas)
    rows = separation_table(lab, (2.0, 0.8, 0.8))
    assert len(rows) == 1 and rows[0]["region"] == "SA"
    assert rows[0]["dist_mm"] == pytest.approx(4 * 2.0)  # centro a centro: (3 + 1) · dz
    assert not rows[0]["in_contact"]


def test_fragment_matching_and_distance_error():
    gt = np.zeros((10, 40, 40), np.uint8)
    gt[2:8, 5:35, 5:20] = 11
    gt[2:8, 5:35, 22:28] = 12                            # separado 2 vóxeles del principal
    pred = gt.copy()
    pred[2:8, 5:35, 22:23] = 0                           # la predicción recorta una columna del secundario
    pred[pred == 11], pred[pred == 12] = 11, 13          # otra numeración: el emparejamiento no depende del id
    fr = match_fragments(gt, pred, SP)
    assert [r["recuperado"] for r in fr] == [True, True]
    assert fr[0]["dice"] == pytest.approx(1.0) and 0.8 < fr[1]["dice"] < 1.0
    dr = distance_comparison(gt, pred, SP, fr)
    assert dr[0]["dist_gt_mm"] == pytest.approx(3.0) and dr[0]["error_mm"] == pytest.approx(1.0)
    s = summarize(fr, dr)
    assert s["recuperados_%"] == pytest.approx(100.0) and s["mae_distancia_mm"] == pytest.approx(1.0)


def test_merged_fragment_counts_as_lost():
    gt = np.zeros((10, 40, 40), np.uint8)
    gt[2:8, 5:35, 5:20] = 11
    gt[2:8, 5:35, 20:28] = 12
    pred = np.where(gt > 0, 11, 0).astype(np.uint8)      # todo fusionado en un solo fragmento
    fr = match_fragments(gt, pred, SP)
    sec = [r for r in fr if not r["es_principal"]][0]
    assert sec["dice"] == 0.0 and not sec["recuperado"]


def test_to_native_inverts_the_cache(tmp_path):
    import SimpleITK as sitk

    from pengwin.data.slice_cache import build_case_cache, load_case_cache
    from pengwin.inference.volume import to_native

    img = np.full((6, 90, 120), -1000, np.int16)
    img[:, 10:80, 10:110] = 40
    lab = np.zeros(img.shape, np.uint8)
    lab[1:5, 30:60, 70:100] = 11
    lab[1:5, 30:60, 20:50] = 21
    img[lab > 0] = 700
    for arr, name in ((img, "i.mha"), (lab, "l.mha")):
        im = sitk.GetImageFromArray(arr)
        im.SetSpacing((0.8, 0.8, 1.0))
        sitk.WriteImage(im, str(tmp_path / name))
    build_case_cache("901", tmp_path / "i.mha", tmp_path / "l.mha", tmp_path / "c", image_size=64, crop_margin_mm=4)
    _, cached, meta = load_case_cache(tmp_path / "c" / "901")
    back = to_native(np.asarray(cached), meta, order=0)
    assert back.shape == lab.shape
    for k in (11, 21):
        inter = ((back == k) & (lab == k)).sum()
        dice = 2 * inter / ((back == k).sum() + (lab == k).sum())
        assert dice > 0.9, f"ida y vuelta de la etiqueta {k}: Dice {dice:.3f}"


def test_edt_seeds_split_fragments_joined_by_thin_bridge_without_edge():
    """La segmentación rellenó la grieta (puente de 2 vóxeles) y el borde no se predijo:
    la variante por distancia igual separa, porque el puente nunca es 'profundo'."""
    sem = np.zeros((20, 40, 60), np.uint8)
    sem[2:18, 5:35, 3:28] = 2                           # principal
    sem[4:16, 10:30, 32:52] = 2                          # secundario
    sem[9:11, 19:21, 28:32] = 2                          # puente fino entre ambos
    edge = np.zeros(sem.shape, np.float32)
    only_edge = separate_instances(sem, edge, SP, method="edge")
    assert np.unique(only_edge[only_edge > 0]).tolist() == [11], "solo-borde: se fusionan"
    lab = separate_instances(sem, edge, SP, method="edt", seed_depth_mm=4.0)
    assert np.unique(lab[lab > 0]).tolist() == [11, 12]
    assert (lab[2:18, 5:35, 3:28] == 11).mean() > 0.95 and (lab[4:16, 10:30, 32:52] == 12).mean() > 0.95
