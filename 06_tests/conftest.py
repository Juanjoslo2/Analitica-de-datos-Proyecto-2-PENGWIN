"""Fixtures compartidas por los tests.

Los datos se buscan en ``01_data``; la variable de entorno PENGWIN_DATA_DIR permite tenerlos
fuera del repositorio (p. ej. fuera de OneDrive).
"""

import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("PENGWIN_DATA_DIR", REPO / "01_data"))


@pytest.fixture(scope="session")
def data_dir() -> Path:
    if not (DATA_DIR / "PENGWIN_CT_train_labels").is_dir():
        pytest.skip("Datos PENGWIN no disponibles en 01_data")
    return DATA_DIR
