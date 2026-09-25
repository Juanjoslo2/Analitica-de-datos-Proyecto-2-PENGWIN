"""Integridad del dataset (reemplaza a 03_src/pengwin/data/test.py)."""

import json

import pytest
import SimpleITK as sitk

from pengwin.data.data_loader import get_dataset_pairs


@pytest.mark.data
def test_all_images_have_labels(data_dir):
    pairs = get_dataset_pairs(data_dir)
    n_labels = len(list((data_dir / "PENGWIN_CT_train_labels").glob("*.mha")))
    assert len(pairs) == n_labels == 100


@pytest.mark.data
def test_headers_match(data_dir):
    """Imagen y etiqueta comparten tamaño, spacing, origen y dirección (sin leer los vóxeles)."""
    bad = []
    for p in get_dataset_pairs(data_dir):
        infos = []
        for path in (p["image_path"], p["label_path"]):
            r = sitk.ImageFileReader()
            r.SetFileName(str(path))
            r.ReadImageInformation()
            infos.append((r.GetSize(), r.GetSpacing(), r.GetOrigin(), r.GetDirection()))
        if infos[0] != infos[1]:
            bad.append(p["case_id"])
    assert not bad, f"Casos con headers distintos: {bad}"


def test_splits_are_disjoint_and_complete():
    from conftest import REPO

    path = REPO / "01_data" / "splits.json"
    if not path.exists():
        pytest.skip("splits.json aún no generado")
    s = json.loads(path.read_text(encoding="utf-8"))["splits"]
    tr, va, te = set(s["train"]), set(s["val"]), set(s["test"])
    assert not (tr & va) and not (tr & te) and not (va & te), "Fuga de pacientes entre particiones"
    assert len(tr | va | te) == 100
