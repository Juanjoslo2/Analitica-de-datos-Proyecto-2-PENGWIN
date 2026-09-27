"""Configuración con herencia y semillas reproducibles."""

import random

import numpy as np

from conftest import REPO
from pengwin.utils.config import load_config
from pengwin.utils.seed import set_seed


def test_ablation_configs_inherit_base():
    base = load_config(REPO / "configs" / "base.yaml")
    no_cbam = load_config(REPO / "configs" / "ablation_no_cbam.yaml")
    scratch = load_config(REPO / "configs" / "ablation_scratch.yaml")
    assert base["model"]["cbam"] is True and no_cbam["model"]["cbam"] is False
    assert scratch["model"]["pretrained"] is False
    # la ablación solo cambia lo que estudia
    assert no_cbam["data"] == base["data"] and no_cbam["loss"] == base["loss"]
    assert base["loss"]["terms"] == ["cls", "det", "seg"]


def test_set_seed_repeats_random_streams():
    set_seed(123)
    a = (random.random(), np.random.rand(3))
    set_seed(123)
    b = (random.random(), np.random.rand(3))
    assert a[0] == b[0] and np.allclose(a[1], b[1])
