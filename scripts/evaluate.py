"""evaluate.py.

Evalúa uno o varios checkpoints sobre un split (por defecto ``test``) con las métricas de
las tres cabezas (clasificación, detección y segmentación por región).

Uso:
    python scripts/evaluate.py --cache-dir D:/PENGWIN/data_processed --ckpt checkpoints/base/best.pth
    python scripts/evaluate.py --cache-dir ... --ckpt checkpoints/base/best.pth checkpoints/sin_cbam/best.pth checkpoints/sin_tl/best.pth

El split se muestrea igual que val (todos los cortes con hueso + ~12 % de vacíos, semilla fija),
para que las cifras de val y test sean comparables.
Salida: reports/eval/<nombre>_<split>.json (nombre = carpeta del checkpoint) y una tabla por consola.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "03_src"))

from pengwin.data.dataset import PengwinSlices, read_split  # noqa: E402
from pengwin.losses.losses import MultiTaskLoss  # noqa: E402
from pengwin.models.pengwin_net import PengwinNet  # noqa: E402
from pengwin.training.engine import evaluate  # noqa: E402

SHOW = ["mAP@0.50", "mAP@[.50:.95]", "IoU_promedio", "AP50_SA", "AP50_LI", "AP50_RI", "dice_hueso", "dice_SA",
        "dice_LI", "dice_RI", "iou_hueso", "cls_f1_macro", "cls_f1_SA", "cls_f1_LI", "cls_f1_RI", "cls_auc_macro"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, nargs="+", required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = REPO / "reports" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    table = {}
    for path in args.ckpt:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        model = PengwinNet(cfg["model"]).to(device)
        model.load_state_dict(ck["model"])
        cases = read_split(REPO / cfg["data"]["splits"], args.split)
        ds = PengwinSlices(args.cache_dir, cases, cfg, train=False)
        dl = torch.utils.data.DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=2)
        m = evaluate(model, dl, MultiTaskLoss(cfg).to(device), device, cfg, cfg["train"]["amp"])
        name = path.parent.name
        res = {"checkpoint": str(path), "epoch": ck.get("epoch"), "split": args.split, "casos": cases,
               "n_cortes": len(ds), "dispositivo": device.type, "metricas": m}
        (out_dir / f"{name}_{args.split}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        table[name] = {k: m[k] for k in SHOW}
        print(f"{name}: {len(ds)} cortes de {len(cases)} casos ({args.split})", flush=True)
    print(pd.DataFrame(table).round(4).to_string())


if __name__ == "__main__":
    main()
