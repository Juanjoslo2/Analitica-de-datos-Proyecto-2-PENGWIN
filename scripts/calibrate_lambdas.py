"""calibrate_lambdas.py.

Calibración de los λ de la pérdida multitarea por balance de gradientes [DD §4]:

    1. ~200 iteraciones de entrenamiento con λ = 1, registrando en cada una
       G_k = ‖∇_θs L_k‖₂ sobre θs = parámetros del bloque 4 + su CBAM (lo último compartido).
    2. λ_k = media(G) / G_k, normalizado para que Σλ = 3.
    3. Tabla de G_k y λ_k en reports/lambdas.json (la lee el entrenamiento con ``lambdas: auto``).

Uso:
    python scripts/calibrate_lambdas.py --cache-dir D:/PENGWIN/data_processed
La sensibilidad (cada λ × 0,5 y × 2) se corre después con ``scripts/train.py --lambdas``.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "03_src"))

from pengwin.data.dataset import PengwinSlices, read_split  # noqa: E402
from pengwin.losses.losses import TERMS, MultiTaskLoss, grad_norms, lambdas_from_norms  # noqa: E402
from pengwin.models.pengwin_net import build_model  # noqa: E402
from pengwin.training.engine import autocast, to_device  # noqa: E402
from pengwin.utils.config import load_config  # noqa: E402
from pengwin.utils.seed import seed_worker, set_seed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "base.yaml")
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"], deterministic=False)
    cache_dir = args.cache_dir or REPO / cfg["data"]["processed_dir"]
    iters = args.iters or cfg["loss"]["lambda_calibration_iters"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = PengwinSlices(cache_dir, read_split(REPO / cfg["data"]["splits"], "train"), cfg, train=True)
    g = torch.Generator().manual_seed(cfg["seed"])
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True, generator=g,
                                         num_workers=cfg["train"]["num_workers"], worker_init_fn=seed_worker,
                                         drop_last=True, persistent_workers=cfg["train"]["num_workers"] > 0)
    model = build_model(cfg).to(device)
    loss_fn = MultiTaskLoss(cfg).to(device)                   # λ = 1
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    shared = model.backbone.shared_parameters()

    norms = {k: [] for k in TERMS}
    it = 0
    while it < iters:
        for batch in loader:
            if it >= iters:
                break
            batch = to_device(batch, device)
            model.train()
            # float32 sin AMP: en float16 y sin GradScaler los gradientes pequeños se irían a 0
            # y sesgarían G_k; son solo ``iters`` pasos, el costo extra es menor
            with autocast(device, False):
                out = model(batch["image"])
            parts = loss_fn(out, batch)
            for k, v in grad_norms(parts, shared).items():
                norms[k].append(v)
            opt.zero_grad(set_to_none=True)
            parts["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            opt.step()
            if it % 25 == 0:
                print(f"it {it:4d}  " + "  ".join(f"G_{k}={norms[k][-1]:.3g}" for k in TERMS), flush=True)
            it += 1

    lam = lambdas_from_norms(norms)
    result = {
        "metodo": "lambda_k = mean(G) / G_k, sum = 3; G_k = ||grad L_k|| sobre backbone.stages[3] (bloque 4 + CBAM)",
        "iters": iters, "config": str(args.config.name),
        "G_media": {k: float(np.mean(v)) for k, v in norms.items()},
        "G_mediana": {k: float(np.median(v)) for k, v in norms.items()},
        "lambdas": {k: round(v, 4) for k, v in lam.items()},
    }
    out = args.out or REPO / cfg["loss"]["lambdas_file"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
