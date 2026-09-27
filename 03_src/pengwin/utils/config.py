"""config.py. Carga configs/*.yaml con herencia simple (clave ``base``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path) -> Dict[str, Any]:
    """Lee un YAML; si trae ``base: otro.yaml`` hereda de él y sobrescribe solo lo que define.

    Ejemplo de ablación (configs/ablation_no_cbam.yaml):
        base: base.yaml
        model: {cbam: false}
    """
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "base" in cfg:
        parent = load_config(path.parent / cfg.pop("base"))
        cfg = _merge(parent, cfg)
    return cfg
