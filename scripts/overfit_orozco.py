"""overfit_orozco.py.

Prueba de correctitud de la semana 9 para el intento propio (detección con DFL):
el modelo debe memorizar 8 cortes fijos de train [DD §5, adaptado a solo detección].

Criterios de aceptación (config ``overfit_test.accept``):
    core_drop   ≥ 0.95   caída de la parte REGRESIVA (objectness + GIoU);
                         NO se mide sobre la total: el DFL tiene un piso de
                         entropía irreducible (ver configs/orozco_dfl.yaml)
    mAP@0.50    ≥ 0.95
    IoU_promedio ≥ 0.95   (GT presente, no ignorada; 0 si no hay predicción)

Además se verifica que el gradiente llegue al backbone y a la cabeza en el paso 0.

Uso:
    python scripts/overfit_orozco.py --raw-dir D:/PENGWIN/01_data
    python scripts/overfit_orozco.py --raw-dir ... --config configs/orozco_dfl.yaml --case 024

Salida: reports/overfit_orozco/<nombre>.json y <nombre>.png
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "03_src"))

from pengwin.data.data_loader import load_and_standardize_mha  # noqa: E402
from pengwin_orozco.dataset import DetSlices  # noqa: E402
from pengwin_orozco.engine import autocast, evaluate, to_device  # noqa: E402
from pengwin_orozco.losses import DetectionLoss  # noqa: E402
from pengwin_orozco.model import build_model  # noqa: E402
from pengwin.utils.config import load_config  # noqa: E402
from pengwin.utils.seed import set_seed  # noqa: E402


def pick_slices(raw_dir: Path, case_id: str, n: int) -> list:
    """8 cortes deterministas con ≥ 2 fragmentos de una misma región, repartidos en z."""
    lab_dir = raw_dir / "PENGWIN_CT_train_labels"
    mask, _, _ = load_and_standardize_mha(lab_dir / f"{case_id}.mha", is_label=True)
    good = []
    for z in range(mask.shape[0]):
        ids = np.unique(mask[z])
        ids = ids[ids > 0]
        regions = (ids - 1) // 10
        if len(ids) > len(np.unique(regions)):
            good.append(z)
    pool = good if len(good) >= n else list(np.where(mask.any(axis=(1, 2)))[0])
    idx = np.linspace(0, len(pool) - 1, n).round().astype(int)
    return [(case_id, int(pool[i])) for i in idx]


def run(cfg, raw_dir: Path, case_id: str, steps: int, lr: float, device: torch.device, log=print) -> dict:
    ov = cfg["overfit_test"]
    set_seed(cfg["seed"])
    slices = pick_slices(raw_dir, case_id, ov["n_slices"])
    ds = DetSlices(raw_dir, [case_id], cfg, train=False, slices=slices)
    loader = torch.utils.data.DataLoader(ds, batch_size=len(ds), shuffle=False)
    batch = to_device(next(iter(loader)), device)

    model = build_model(cfg).to(device)
    loss_fn = DetectionLoss(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    hist = {k: [] for k in ("total", "det_obj", "det_dfl", "det_giou", "det_pos_iou")}
    grads_ok = {}
    for step in range(steps):
        model.train()
        with autocast(device, True):
            out = model(batch["image"])
        parts = loss_fn(out, batch)
        opt.zero_grad(set_to_none=True)
        scaler.scale(parts["total"]).backward()
        if step == 0:
            grads_ok["backbone"] = any(p.grad is not None and float(p.grad.abs().sum()) > 0
                                       for p in model.backbone.parameters())
            grads_ok["det_head"] = any(p.grad is not None and float(p.grad.abs().sum()) > 0
                                       for p in model.det_head.parameters())
        scaler.step(opt)
        scaler.update()
        for k in hist:
            hist[k].append(float(parts[k].detach()))
        if step % 50 == 0 or step == steps - 1:
            log(f"paso {step:4d}  total {hist['total'][-1]:.4f}  obj {hist['det_obj'][-1]:.4f}  "
                f"dfl {hist['det_dfl'][-1]:.4f}  giou {hist['det_giou'][-1]:.4f}  IoU+ {hist['det_pos_iou'][-1]:.3f}",
                flush=True)

    metrics = evaluate(model, loader, loss_fn, device, cfg)
    # Caída de la parte REGRESIVA (objectness + GIoU): el DFL tiene un piso de
    # entropía, por eso el criterio no se mide sobre la total (ver config).
    core = np.asarray(hist["det_obj"]) + np.asarray(hist["det_giou"])
    core_drop = 1 - np.mean(core[-10:]) / core[0]
    total_drop = 1 - np.mean(hist["total"][-10:]) / hist["total"][0]
    acc = ov["accept"]
    checks = {
        "core_drop": (float(core_drop), core_drop >= acc["core_drop"]),
        "total_drop_referencia": (float(total_drop), None),
        "map50": (metrics["mAP@0.50"], metrics["mAP@0.50"] >= acc["map50"]),
        "iou": (metrics["IoU_promedio"], metrics["IoU_promedio"] >= acc["iou"]),
        "gradiente_en_backbone_y_cabeza": (grads_ok, all(grads_ok.values())),
    }
    return {"slices": slices, "steps": steps, "lr": lr, "checks": checks,
            "passed": all(ok for _, ok in checks.values() if ok is not None),
            "metrics": metrics, "history": hist}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "orozco_dfl.yaml")
    ap.add_argument("--raw-dir", type=Path, default=REPO / "01_data")
    ap.add_argument("--case", default="024")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = args.name or f"overfit_{args.config.stem}"
    res = run(cfg, args.raw_dir, args.case, args.steps or cfg["overfit_test"]["max_steps"], args.lr, device)

    out_dir = REPO / "reports" / "overfit_orozco"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps({k: v for k, v in res.items() if k != "history"},
                                                     indent=1, default=str), encoding="utf-8")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for k in ("total", "det_obj", "det_dfl", "det_giou"):
        ax[0].semilogy(res["history"][k], label=k)
    ax[0].set(xlabel="paso", ylabel="pérdida (log)", title=f"Overfit DFL de 8 cortes ({args.case})")
    ax[0].legend()
    ax[1].plot(res["history"]["det_pos_iou"])
    ax[1].set(xlabel="paso", ylabel="IoU medio en celdas positivas", ylim=(0, 1), title="Cajas en positivos")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=120)
    print("\nCriterios:")
    for k, (v, ok) in res["checks"].items():
        print(f"  {'OK ' if ok else 'NO '} {k}: {v}")
    print("RESULTADO:", "APROBADA" if res["passed"] else "NO APROBADA")


if __name__ == "__main__":
    main()
