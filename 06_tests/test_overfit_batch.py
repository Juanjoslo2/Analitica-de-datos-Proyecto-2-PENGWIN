"""Prueba de overfit de 8 cortes reales [DD §5] (marca ``slow``: no corre en CI).

    python -m pytest -m slow 06_tests/test_overfit_batch.py
Necesita el caché de cortes: ``data.processed_dir`` de la config o la variable de
entorno PENGWIN_CACHE_DIR (p. ej. D:/PENGWIN/data_processed).
"""

import importlib.util
import os
from pathlib import Path

import pytest

from conftest import REPO

torch = pytest.importorskip("torch")


@pytest.mark.slow
@pytest.mark.data
def test_overfit_eight_slices():
    from pengwin.utils.config import load_config

    cfg = load_config(REPO / "configs" / "base.yaml")
    cache = Path(os.environ.get("PENGWIN_CACHE_DIR", REPO / cfg["data"]["processed_dir"]))
    if not (cache / "024" / "meta.json").exists():
        pytest.skip(f"Sin caché de cortes en {cache} (scripts/build_slice_cache.py)")
    spec = importlib.util.spec_from_file_location("overfit_batch", REPO / "scripts" / "overfit_batch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = mod.run(cfg, cache, "024", cfg["overfit_test"]["max_steps"], 1e-3, device, log=lambda *_: None)
    assert res["passed"], res["checks"]
