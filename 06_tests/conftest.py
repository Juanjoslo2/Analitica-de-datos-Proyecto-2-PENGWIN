"""Fixtures compartidas por los tests."""

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "01_data"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    if not (DATA_DIR / "PENGWIN_CT_train_labels").is_dir():
        pytest.skip("Datos PENGWIN no disponibles en 01_data")
    return DATA_DIR
