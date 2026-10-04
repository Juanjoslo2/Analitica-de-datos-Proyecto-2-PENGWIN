"""Caché de cortes + Dataset 2.5D sobre un phantom .mha escrito en disco (sin dataset real).

Comprueba lo que más se rompe en silencio: que tras recortar y redimensionar la etiqueta siga
alineada con la imagen, que el phantom RAS quede con la lateralidad correcta y que el
contexto 2.5D use Δ = round(2 mm / dz).
"""

import json

import numpy as np
import pytest
import SimpleITK as sitk

from pengwin.data.slice_cache import build_case_cache, context_offset, load_case_cache, model_to_native_xy

RAS = (-1.0, 0, 0, 0, -1.0, 0, 0, 0, 1.0)


def _write_phantom(tmp_path, direction=RAS):
    """Cuerpo de agua con 'pelvis' ósea: coxal izq. (x grande en LPS) y der., sacro al centro."""
    img = np.full((24, 96, 128), -1000, np.int16)
    img[:, 10:90, 8:120] = 40                                # cuerpo
    lab = np.zeros(img.shape, np.uint8)
    lab[4:20, 40:60, 84:104] = 11                            # izquierda del paciente (LPS: x grande)
    lab[4:20, 40:60, 24:44] = 21
    lab[4:20, 40:60, 56:72] = 1
    lab[8:12, 40:60, 72:76] = 2                              # fragmento del sacro
    img[lab > 0] = 700
    if direction == RAS:                                     # en disco queda espejado en x e y
        img, lab = img[:, ::-1, ::-1].copy(), lab[:, ::-1, ::-1].copy()
    paths = []
    for arr, name in ((img, "img.mha"), (lab, "lab.mha")):
        im = sitk.GetImageFromArray(arr)
        im.SetSpacing((0.8, 0.8, 1.0))
        im.SetDirection(direction)
        sitk.WriteImage(im, str(tmp_path / name))
        paths.append(tmp_path / name)
    return paths


def test_context_offset():
    assert context_offset(0.8, 2.0) == 2 and context_offset(0.625, 2.0) == 3 and context_offset(2.5, 2.0) == 1


def test_cache_is_aligned_and_lps(tmp_path):
    img_p, lab_p = _write_phantom(tmp_path)
    meta = build_case_cache("900", img_p, lab_p, tmp_path / "cache", image_size=64, crop_margin_mm=4)
    image, label, meta2 = load_case_cache(tmp_path / "cache" / "900")
    assert meta == meta2 and meta["orientation_orig"] == "RAS"
    assert image.shape == label.shape == (24, 64, 64) and image.dtype == label.dtype == np.uint8
    assert meta["bone_slices"] == list(range(4, 20))
    assert set(meta["labels_present"]) == {1, 2, 11, 21}
    # alineación: el hueso etiquetado cae sobre píxeles brillantes (700 HU -> ~0,67 en la ventana)
    bone = label > 0
    assert image[bone].mean() > 150 and image[~bone & (image > 0)].mean() < 120
    # lateralidad LPS: coxal izq. (11) con x mayor que el der. (21)
    x = np.arange(64)
    xl = (label == 11).sum((0, 1)) @ x / (label == 11).sum()
    xr = (label == 21).sum((0, 1)) @ x / (label == 21).sum()
    assert xl > xr
    # vuelta al espacio nativo: el centro de la caja de 11 en la entrada cae en la columna nativa ~93,5
    ys, xs = np.nonzero(label[10] == 11)
    cx = (xs.min() + xs.max() + 1) / 2
    assert model_to_native_xy(np.array([cx, 0.0]), meta)[0] == pytest.approx(94.0, abs=1.5)


def test_dataset_items(tmp_path):
    torch = pytest.importorskip("torch")
    from conftest import REPO
    from pengwin.data.dataset import PengwinSlices, context_stack
    from pengwin.utils.config import load_config

    img_p, lab_p = _write_phantom(tmp_path)
    build_case_cache("900", img_p, lab_p, tmp_path / "cache", image_size=64, crop_margin_mm=4)
    cfg = load_config(REPO / "configs" / "base.yaml")
    ds = PengwinSlices(tmp_path / "cache", ["900"], cfg, train=False)
    n_bone = 16
    assert n_bone <= len(ds) <= n_bone + round(n_bone * 0.12 / 0.88) + 1
    item = ds[ds.index.index(("900", 10))]
    assert item["image"].shape == (3, 64, 64) and item["semantic"].shape == (64, 64)
    assert item["present"].tolist() == [1.0, 1.0, 1.0]
    assert item["edge"].sum() > 0, "el fragmento 2 toca al sacro 1: debe haber borde"
    # contexto 2.5D con Δ = round(2 / 1,0) = 2 cortes
    image, _, _ = load_case_cache(tmp_path / "cache" / "900")
    assert np.array_equal(context_stack(image, 0, 2)[0], image[0])          # borde repetido
    assert torch.equal(item["image"][2], torch.from_numpy(image[12].astype(np.float32) / 255))

    train = PengwinSlices(tmp_path / "cache", ["900"], cfg, train=True)
    a = train[0]
    assert a["image"].shape == (3, 64, 64) and json.dumps(a["case_id"])


def test_secondary_oversampling_and_edge_file(tmp_path):
    pytest.importorskip("torch")
    from conftest import REPO
    from pengwin.data.dataset import PengwinSlices
    from pengwin.data.slice_cache import add_edge_cache
    from pengwin.utils.config import load_config

    img_p, lab_p = _write_phantom(tmp_path)
    build_case_cache("900", img_p, lab_p, tmp_path / "cache", image_size=64, crop_margin_mm=4)
    cfg = load_config(REPO / "configs" / "base.yaml")
    base = PengwinSlices(tmp_path / "cache", ["900"], cfg, train=True)
    cfg3 = {**cfg, "data": {**cfg["data"], "secondary_oversample": 3}}
    over = PengwinSlices(tmp_path / "cache", ["900"], cfg3, train=True)
    assert len(over) - len(base) == 2 * 4, "los 4 cortes con el fragmento 2 (z 8-11) se repiten 2 veces más"
    assert len(PengwinSlices(tmp_path / "cache", ["900"], cfg3, train=False)) == len(
        PengwinSlices(tmp_path / "cache", ["900"], cfg, train=False)), "en val no se sobremuestrea"
    n = add_edge_cache(tmp_path / "cache" / "900", dilation=0, name="edge_d0.npy")
    cfg_e = {**cfg, "data": {**cfg["data"], "edge_file": "edge_d0.npy"}}
    ds = PengwinSlices(tmp_path / "cache", ["900"], cfg_e, train=False)
    assert ds[ds.index.index(("900", 10))]["edge"].sum() > 0 and n > 0


def test_cv_folds_are_disjoint_and_exclude_test():
    from conftest import REPO

    path = REPO / "reports" / "tuning" / "cv_folds.json"
    if not path.exists():
        pytest.skip("cv_folds.json no generado")
    cv = json.loads(path.read_text(encoding="utf-8"))
    flat = [c for f in cv["folds"] for c in f]
    assert len(flat) == len(set(flat)) == 85
    assert not set(flat) & set(cv["test_excluido"])
