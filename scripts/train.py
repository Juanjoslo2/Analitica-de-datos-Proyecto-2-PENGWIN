"""train.py.

Entrena el modelo multitarea sobre los splits fijos (train → selección en val).

Uso:
    python scripts/train.py --cache-dir D:/PENGWIN/data_processed --name base
    python scripts/train.py --config configs/ablation_no_cbam.yaml --name sin_cbam
    python scripts/train.py --name rapido --epochs 3 --max-steps 300         # humo / semana 9
    python scripts/train.py --name sens_det_x2 --lambdas '{"cls":1,"det":2,"seg":1}'

Salida:
    checkpoints/<name>/best.pth, last.pth       (no se versionan: GitHub Releases)
    runs/<name>/history.csv                     (una fila por época: pérdidas y métricas de val)
    reports/train/<name>.json                   (config efectiva, λ usados, mejores métricas; se versiona)
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

from pengwin.data.dataset import PengwinSlices, read_split  # noqa: E402
from pengwin.losses.losses import MultiTaskLoss, load_lambdas  # noqa: E402
from pengwin.models.pengwin_net import build_model, count_parameters  # noqa: E402
from pengwin.training.engine import (  # noqa: E402
    evaluate, param_groups, save_checkpoint, selection_score, train_one_epoch,
)
from pengwin.utils.config import load_config  # noqa: E402
from pengwin.utils.seed import seed_worker, set_seed  # noqa: E402


def loader(ds, cfg, shuffle: bool):
    g = torch.Generator().manual_seed(cfg["seed"])
    nw = cfg["train"]["num_workers"]
    return torch.utils.data.DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=shuffle, generator=g,
                                       num_workers=nw, worker_init_fn=seed_worker, pin_memory=True,
                                       drop_last=shuffle, persistent_workers=nw > 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "base.yaml")
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="pasos máximos por época (corridas cortas)")
    ap.add_argument("--lambdas", type=str, default=None, help='JSON, p. ej. \'{"cls":1,"det":2,"seg":1}\'')
    ap.add_argument("--deterministic", action="store_true", help="cuDNN determinista (más lento)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"], deterministic=args.deterministic)
    cache_dir = args.cache_dir or REPO / cfg["data"]["processed_dir"]
    epochs = args.epochs or cfg["train"]["epochs"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.deterministic:
        torch.backends.cudnn.benchmark = True

    splits = REPO / cfg["data"]["splits"]
    train_ds = PengwinSlices(cache_dir, read_split(splits, "train"), cfg, train=True)
    val_ds = PengwinSlices(cache_dir, read_split(splits, "val"), cfg, train=False)
    train_dl, val_dl = loader(train_ds, cfg, True), loader(val_ds, cfg, False)
    print(f"train: {len(train_ds)} cortes | val: {len(val_ds)} cortes | {device}", flush=True)

    lambdas = json.loads(args.lambdas) if args.lambdas else load_lambdas(cfg["loss"], REPO)
    print(f"λ = {lambdas or 'sin calibrar (1, 1, 1): corre scripts/calibrate_lambdas.py'}")
    model = build_model(cfg).to(device)
    print("parámetros:", count_parameters(model))
    loss_fn = MultiTaskLoss(cfg, lambdas).to(device)
    t = cfg["train"]
    opt = torch.optim.AdamW(param_groups(model, t["lr"], t.get("backbone_lr_factor", 1.0), t["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=t["amp"] and device.type == "cuda")

    run_dir, ckpt_dir = REPO / "runs" / args.name, REPO / "checkpoints" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    best, best_metrics, rows = -1.0, {}, []
    for epoch in range(epochs):
        t0 = time.time()
        train_ds.set_epoch(epoch)
        tr = train_one_epoch(model, train_dl, loss_fn, opt, scaler, device, t["amp"], t.get("grad_clip"), args.max_steps)
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
              f"mAP50 {va['mAP@0.50']:.3f}  mAP {va['mAP@[.50:.95]']:.3f}  IoU {va['IoU_promedio']:.3f}  "
              f"Dice {va['dice_hueso']:.3f}  F1 {va['cls_f1_macro']:.3f}  AUC {va['cls_auc_macro']:.3f}", flush=True)
        save_checkpoint(ckpt_dir / "last.pth", model, opt, epoch, va, cfg)
        if score > best:
            best, best_metrics = score, {"epoch": epoch, **va}
            save_checkpoint(ckpt_dir / "best.pth", model, None, epoch, va, cfg)

    out = REPO / "reports" / "train" / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": str(args.config.name), "epochs": epochs, "max_steps": args.max_steps,
                               "lambdas": lambdas, "parametros": count_parameters(model),
                               "mejor": best_metrics, "cfg": cfg}, indent=1, default=str), encoding="utf-8")
    print(f"Mejor época {best_metrics.get('epoch')}: score {best:.3f} -> {ckpt_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
