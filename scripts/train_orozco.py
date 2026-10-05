"""train_orozco.py.

Entrenamiento del intento propio (detección con DFL) sobre los splits fijos.

Uso:
    python scripts/train_orozco.py --raw-dir D:/PENGWIN/01_data --name dfl
    python scripts/train_orozco.py --name rapido --epochs 2 --max-steps 200 --max-cases 20   # humo CPU

Salida:
    checkpoints/<name>/best.pth, last.pth
    runs/<name>/history.csv
    reports/train_orozco/<name>.json
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "03_src"))

from pengwin_orozco.dataset import DetSlices, read_split  # noqa: E402
from pengwin_orozco.engine import (  # noqa: E402
    evaluate, save_checkpoint, selection_score, train_epoch,
)
from pengwin_orozco.losses import DetectionLoss  # noqa: E402
from pengwin_orozco.model import build_model, count_parameters  # noqa: E402
from pengwin.utils.config import load_config  # noqa: E402
from pengwin.utils.seed import seed_worker, set_seed  # noqa: E402


class LimitedLoader:
    """Envuelve un DataLoader limitando los batches por iteración (smoke en CPU)."""

    def __init__(self, dl, max_steps: int | None):
        self.dl = dl
        self.max_steps = max_steps

    def __iter__(self):
        for i, batch in enumerate(self.dl):
            if self.max_steps is not None and i >= self.max_steps:
                break
            yield batch

    def __len__(self):
        return min(len(self.dl), self.max_steps) if self.max_steps is not None else len(self.dl)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "orozco_dfl.yaml")
    ap.add_argument("--raw-dir", type=Path, default=REPO / "01_data")
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--max-cases", type=int, default=None, help="limita los casos de train (smoke en CPU)")
    ap.add_argument("--val-max-steps", type=int, default=None, help="limita los batches de evaluación de val")
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"], deterministic=args.deterministic)
    epochs = args.epochs or cfg["train"]["epochs"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.deterministic:
        torch.backends.cudnn.benchmark = True

    splits = REPO / cfg["data"]["splits"]
    train_ds = DetSlices(args.raw_dir, read_split(splits, "train"), cfg, train=True, max_cases=args.max_cases)
    val_ds = DetSlices(args.raw_dir, read_split(splits, "val"), cfg, train=False)
    g = torch.Generator().manual_seed(cfg["seed"])
    t = cfg["train"]
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True, generator=g,
                                           num_workers=t["num_workers"], worker_init_fn=seed_worker,
                                           drop_last=True, persistent_workers=t["num_workers"] > 0)
    val_dl = LimitedLoader(torch.utils.data.DataLoader(val_ds, batch_size=t["batch_size"], shuffle=False),
                           args.val_max_steps)
    print(f"train: {len(train_ds)} cortes | val: {len(val_ds)} cortes | {device}", flush=True)

    model = build_model(cfg).to(device)
    print("parámetros:", count_parameters(model), flush=True)
    loss_fn = DetectionLoss(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=t["amp"] and device.type == "cuda")

    run_dir, ckpt_dir = REPO / "runs" / args.name, REPO / "checkpoints" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    best, best_metrics, rows = -1.0, {}, []
    for epoch in range(epochs):
        t0 = time.time()
        tr = train_epoch(model, train_dl, loss_fn, opt, scaler, device, t["amp"], t.get("grad_clip"), args.max_steps)
        va = evaluate(model, val_dl, loss_fn, device, cfg, t["amp"])
        sched.step()
        score = selection_score(va)
        row = {"epoch": epoch, "seconds": round(time.time() - t0, 1), "score": score,
               **{f"train_{k}": v for k, v in tr.items()}, **va}
        rows.append(row)
        with open(run_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[-1].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"época {epoch:3d} ({row['seconds']:.0f} s)  loss {tr['total']:.3f} | val loss {va['val_total']:.3f}  "
              f"mAP50 {va['mAP@0.50']:.3f}  mAP {va['mAP@[.50:.95]']:.3f}  IoU {va['IoU_promedio']:.3f}", flush=True)
        save_checkpoint(ckpt_dir / "last.pth", model, opt, epoch, va, cfg)
        if score > best:
            best, best_metrics = score, {"epoch": epoch, **va}
            save_checkpoint(ckpt_dir / "best.pth", model, None, epoch, va, cfg)

    out = REPO / "reports" / "train_orozco" / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": str(args.config.name), "epochs": epochs, "max_steps": args.max_steps,
                               "max_cases": args.max_cases, "parametros": count_parameters(model),
                               "mejor": best_metrics, "cfg": cfg}, indent=1, default=str), encoding="utf-8")
    print(f"Mejor época {best_metrics.get('epoch')}: score {best:.3f} -> {ckpt_dir / 'best.pth'}", flush=True)


if __name__ == "__main__":
    main()