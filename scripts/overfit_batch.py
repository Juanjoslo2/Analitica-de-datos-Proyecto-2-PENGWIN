"""overfit_batch.py.

Prueba de correctitud de la semana 9 [DD §5]: el modelo completo debe memorizar 8 cortes.
Si no lo logra, hay un error de cableado (formas, asignación del grid, decodificación,
IoU, NMS, alineación máscara↔imagen, pérdidas o gradiente que no llega a una cabeza).

Uso:
    python scripts/overfit_batch.py --cache-dir D:/PENGWIN/data_processed
    python scripts/overfit_batch.py --cache-dir ... --config configs/ablation_no_cbam.yaml

Selección de los 8 cortes (determinista): del caso de train ``--case`` (024 por defecto:
sacro fracturado y 2 secundarios separados del principal [EDA §7]), los cortes con al menos
dos fragmentos de una misma región, repartidos uniformemente en z.

Salida: reports/overfit/<nombre>.json (criterios y resultado) y <nombre>.png (curvas).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "03_src"))

from pengwin.data.dataset import PengwinSlices  # noqa: E402
from pengwin.data.slice_cache import load_case_cache  # noqa: E402
from pengwin.losses.losses import MultiTaskLoss  # noqa: E402
from pengwin.models.pengwin_net import build_model  # noqa: E402
from pengwin.training.engine import autocast, evaluate, to_device  # noqa: E402
from pengwin.utils.config import load_config  # noqa: E402
from pengwin.utils.seed import set_seed  # noqa: E402


def pick_slices(cache_dir: Path, case_id: str, n: int) -> list:
    _, label, meta = load_case_cache(cache_dir / case_id)
    good = []
    for z in meta["bone_slices"]:
        ids = np.unique(label[z])
        ids = ids[ids > 0]
        regions = (ids - 1) // 10
        if len(ids) > len(np.unique(regions)):          # alguna región con ≥ 2 fragmentos
            good.append(z)
    pool = good if len(good) >= n else meta["bone_slices"]
    idx = np.linspace(0, len(pool) - 1, n).round().astype(int)
    return [(case_id, int(pool[i])) for i in idx]


def run(cfg, cache_dir: Path, case_id: str, steps: int, lr: float, device: torch.device, log=print) -> dict:
    ov = cfg["overfit_test"]
    set_seed(cfg["seed"])
    slices = pick_slices(cache_dir, case_id, ov["n_slices"])
    ds = PengwinSlices(cache_dir, [case_id], cfg, train=False, augment=ov.get("augment", False), slices=slices)
    loader = torch.utils.data.DataLoader(ds, batch_size=len(ds), shuffle=False)
    batch = to_device(next(iter(loader)), device)

    model = build_model(cfg, log=log).to(device)
    loss_fn = MultiTaskLoss(cfg).to(device)               # λ = 1: la prueba no depende de la calibración
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    hist = {k: [] for k in ("total", "cls", "det", "seg", "det_pos_iou")}
    grads_ok = {}
    for step in range(steps):
        model.train()
        with autocast(device, True):
            out = model(batch["image"])
        parts = loss_fn(out, batch)
        opt.zero_grad(set_to_none=True)
        scaler.scale(parts["total"]).backward()
        if step == 0:                                      # el gradiente llega a las tres cabezas y al backbone
            for name, mod in (("backbone", model.backbone), ("cls_head", model.cls_head),
                              ("det_head", model.det_head), ("seg_head", model.seg_head)):
                grads_ok[name] = any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in mod.parameters())
        scaler.step(opt)
        scaler.update()
        for k in hist:
            hist[k].append(float(parts[k].detach()))
        if step % 50 == 0 or step == steps - 1:
            log(f"paso {step:4d}  total {hist['total'][-1]:.4f}  cls {hist['cls'][-1]:.4f}  "
                f"det {hist['det'][-1]:.4f}  seg {hist['seg'][-1]:.4f}  IoU+ {hist['det_pos_iou'][-1]:.3f}")

    metrics = evaluate(model, loader, loss_fn, device, cfg)
    drop = 1 - np.mean(hist["total"][-10:]) / hist["total"][0]
    acc = ov["accept"]
    checks = {
        "loss_drop": (float(drop), drop >= acc["loss_drop"]),
        "dice": (metrics["dice_hueso"], metrics["dice_hueso"] >= acc["dice"]),
        "map50": (metrics["mAP@0.50"], metrics["mAP@0.50"] >= acc["map50"]),
        "gradiente_en_todas_las_cabezas": (grads_ok, all(grads_ok.values())),
    }
    return {"slices": slices, "steps": steps, "lr": lr, "checks": checks,
            "passed": all(ok for _, ok in checks.values()), "metrics": metrics, "history": hist}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "base.yaml")
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--case", default="024")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    cache_dir = args.cache_dir or REPO / cfg["data"]["processed_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = args.name or f"overfit_{args.config.stem}"
    res = run(cfg, cache_dir, args.case, args.steps or cfg["overfit_test"]["max_steps"], args.lr, device)

    out_dir = REPO / "reports" / "overfit"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps({k: v for k, v in res.items() if k != "history"}, indent=1,
                                                     default=str), encoding="utf-8")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for k in ("total", "cls", "det", "seg"):
        ax[0].semilogy(res["history"][k], label=k)
    ax[0].set(xlabel="paso", ylabel="pérdida (log)", title="Overfit de 8 cortes")
    ax[0].legend()
    ax[1].plot(res["history"]["det_pos_iou"])
    ax[1].set(xlabel="paso", ylabel="IoU medio en celdas positivas", ylim=(0, 1), title="Cajas en positivos")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=120)
    print("\nCriterios [DD §5]:")
    for k, (v, ok) in res["checks"].items():
        print(f"  {'OK ' if ok else 'NO '} {k}: {v}")
    print("RESULTADO:", "APROBADA" if res["passed"] else "NO APROBADA")


if __name__ == "__main__":
    main()
