"""seed.py.

Fijación estricta de semillas para reproducibilidad (entregable obligatorio).
Torch es opcional: el módulo funciona aunque torch no esté instalado (tests de CI).
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Fija las semillas de Python, NumPy y (si está) PyTorch en CPU y GPU.

    ``deterministic=True`` desactiva el autotuner de cuDNN y pide algoritmos
    deterministas: los resultados se repiten a costa de algo de velocidad.
    Úsenlo en ablaciones y en la prueba de overfit; pueden relajarlo en el
    entrenamiento final si documentan el cambio.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` para DataLoader: cada worker deriva su semilla de la de torch."""
    try:
        import torch
    except ImportError:
        return
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
