"""train.py.

Entrena el modelo multitarea sobre los splits fijos (train → selección en val).

Uso:
    python scripts/train.py --cache-dir D:/PENGWIN/data_processed --name base
    python scripts/train.py --config configs/ablation_no_cbam.yaml --name sin_cbam
    python scripts/train.py --name rapido --epochs 3 --max-steps 300         # humo / semana 9
    python scripts/train.py --name sens_det_x2 --lambdas '{"cls":1,"det":2,"seg":1}'
    python scripts/train.py --name cv_base_f0 --folds-file reports/tuning/cv_folds.json --fold 0   # validación cruzada

Salida:
    checkpoints/<name>/best.pth, last.pth       (no se versionan: GitHub Releases)
    runs/<name>/history.csv                     (una fila por época: pérdidas y métricas de val)
    reports/train/<name>.json                   (config efectiva, λ usados, mejores métricas; se versiona)
    reports/train/<name>_history.csv            (copia versionable del historial)
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
from pengwin.models.pengwin_net import build_model, count_parameters, load_expanding_input  # noqa: E402
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
    ap.add_argument("--folds-file", type=Path, default=None, help="reports/tuning/cv_folds.json (scripts/cv_folds.py)")
    ap.add_argument("--fold", type=int, default=None, help="con --folds-file: este fold es val y el resto train")
    ap.add_argument("--lambdas-file", type=Path, default=None, help="JSON de calibrate_lambdas.py (en vez del de la config)")
    ap.add_argument("--hi-cache-dir", type=Path, default=None,
                    help="caché de alta resolución; obligatorio con data.two_pass (segunda pasada por hueso)")
    ap.add_argument("--init-ckpt", type=Path, default=None,
                    help="parte de este checkpoint (los canales de entrada nuevos arrancan en cero)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"], deterministic=args.deterministic)
    cache_dir = args.cache_dir or REPO / cfg["data"]["processed_dir"]
    epochs = args.epochs or cfg["train"]["epochs"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.deterministic:
        torch.backends.cudnn.benchmark = True

    if args.folds_file is not None:
        folds = json.loads(args.folds_file.read_text(encoding="utf-8"))["folds"]
        val_cases = folds[args.fold]
        train_cases = sorted(c for k, f in enumerate(folds) if k != args.fold for c in f)
    else:
        splits = REPO / cfg["data"]["splits"]
        train_cases, val_cases = read_split(splits, "train"), read_split(splits, "val")
    if cfg["data"].get("two_pass"):
        # [y4xul] cortes completos + recortes por hueso a alta resolución con máscara previa
        from pengwin.data.two_pass import TwoPassSlices
        if args.hi_cache_dir is None:
            raise SystemExit("data.two_pass necesita --hi-cache-dir")
        train_ds = TwoPassSlices(cache_dir, args.hi_cache_dir, train_cases, cfg, train=True,
                                 roi_per_slice=float(cfg["data"].get("roi_per_slice", 1.0)))
    else:
        train_ds = PengwinSlices(cache_dir, train_cases, cfg, train=True)
    val_ds = PengwinSlices(cache_dir, val_cases, cfg, train=False)
    train_dl, val_dl = loader(train_ds, cfg, True), loader(val_ds, cfg, False)
    print(f"train: {len(train_ds)} cortes | val: {len(val_ds)} cortes | {device}", flush=True)

    if args.lambdas:
        lambdas = json.loads(args.lambdas)
    elif args.lambdas_file:
        lambdas = json.loads(args.lambdas_file.read_text(encoding="utf-8"))["lambdas"]
    else:
        lambdas = load_lambdas(cfg["loss"], REPO)
    print(f"λ = {lambdas or 'sin calibrar (1, 1, 1): corre scripts/calibrate_lambdas.py'}")
    model = build_model(cfg)
    if args.init_ckpt is not None:
        extra = load_expanding_input(model, torch.load(args.init_ckpt, map_location="cpu", weights_only=False)["model"])
        print(f"pesos iniciales de {args.init_ckpt} (+{extra} canales de entrada en cero)")
    model = model.to(device)
    print("parámetros:", count_parameters(model))
    loss_fn = MultiTaskLoss(cfg, lambdas).to(device)
    t = cfg["train"]
    opt = torch.optim.AdamW(param_groups(model, t["lr"], t.get("backbone_lr_factor", 1.0), t["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=t["amp"] and device.type == "cuda")

    # [y4xul] EMA de los pesos: se evalúa y se guarda la media móvil; el modelo sin promediar
    # queda en last_raw.pth para poder comparar.
    ema, ema_decay = None, float(t.get("ema_decay", 0) or 0)
    if ema_decay > 0:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay), use_buffers=True)
        print(f"EMA de pesos, decaimiento {ema_decay}")
    eval_model = ema.module if ema is not None else model
    on_step = (lambda: ema.update_parameters(model)) if ema is not None else None

    run_dir, ckpt_dir = REPO / "runs" / args.name, REPO / "checkpoints" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    best, best_metrics, rows = -1.0, {}, []
    for epoch in range(epochs):
        t0 = time.time()
        train_ds.set_epoch(epoch)
        tr = train_one_epoch(model, train_dl, loss_fn, opt, scaler, device, t["amp"], t.get("grad_clip"), args.max_steps,
                             on_step=on_step)
        va = evaluate(eval_model, val_dl, loss_fn, device, cfg, t["amp"])
        sched.step()
        score = selection_score(va, t.get("selection_metrics"))
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
        save_checkpoint(ckpt_dir / "last.pth", eval_model, opt, epoch, va, cfg)
        if ema is not None:
            save_checkpoint(ckpt_dir / "last_raw.pth", model, None, epoch, {}, cfg)
        if score > best:
            best, best_metrics = score, {"epoch": epoch, **va}
            save_checkpoint(ckpt_dir / "best.pth", eval_model, None, epoch, va, cfg)

    out = REPO / "reports" / "train" / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": str(args.config.name), "epochs": epochs, "max_steps": args.max_steps,
                               "lambdas": lambdas, "parametros": count_parameters(model),
                               "mejor": best_metrics, "cfg": cfg}, indent=1, default=str), encoding="utf-8")
    # copia versionable del historial (runs/ no se sube): la usan las curvas y el notebook de sustentación
    (out.parent / f"{args.name}_history.csv").write_bytes((run_dir / "history.csv").read_bytes())
    print(f"Mejor época {best_metrics.get('epoch')}: score {best:.3f} -> {ckpt_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
