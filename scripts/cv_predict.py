"""cv_predict.py.

Predicciones fuera de fold (OOF): el modelo de un fold predice los casos que NO vio y se guardan
en la grilla del modelo para ajustar el posproceso sin reentrenar.

Por caso se guarda ``<out-dir>/<case>.npz`` con
    semantic  uint8 (Z, 256, 256)   región por píxel
    edge      uint8 (Z, 256, 256)   P(borde) · 255
    core      uint8 (Z, 256, 256)   P(núcleo) · 255, solo con ``model.seg_outputs: [..., core3]``
    dist      uint8 (Z, 256, 256)   distancia a la fractura / DIST_MM_PER_LEVEL (0,1 mm por nivel,
                                    0..25,5 mm), solo con ``model.seg_outputs: [..., dist]`` [F2B2]
    role      uint8 (Z, 256, 256)   P(secundario | hueso) · 255, solo con ``role3`` [y4xul]
y en ``<out-dir>/metricas.json`` las métricas de clasificación y detección del fold (evaluate.py).

Uso:
    python scripts/cv_predict.py --ckpt checkpoints/cv_base_f0/best.pth --cache-dir ... \
        --folds-file reports/tuning/cv_folds.json --fold 0 --out-dir /data/oof/cv_base_f0
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
from pengwin.data.targets import DIST_MM_PER_LEVEL  # noqa: E402
from pengwin.inference.volume import TTA_DEFAULT, predict_case  # noqa: E402
from pengwin.losses.losses import MultiTaskLoss  # noqa: E402
from pengwin.models.pengwin_net import PengwinNet  # noqa: E402
from pengwin.training.engine import evaluate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--folds-file", type=Path, required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tta", action="store_true", help="promedia región y borde con TTA_DEFAULT (volume.py)")
    ap.add_argument("--skip-metrics", action="store_true", help="no recalcula metricas.json de det/cls (no cambian con TTA)")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = PengwinNet(cfg["model"]).to(device).eval()
    model.load_state_dict(ck["model"])
    cases = json.loads(args.folds_file.read_text(encoding="utf-8"))["folds"][args.fold]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for cid in cases:
        p = predict_case(model, args.cache_dir, cid, cfg, device, tta=TTA_DEFAULT if args.tta else None)
        arrays = {"semantic": p["semantic"], "edge": np.round(p["edge"].astype(np.float32) * 255).astype(np.uint8)}
        if "core" in p:          # modelos con la salida core3: P(núcleo)·255 [F2B1]
            arrays["core"] = np.round(p["core"].astype(np.float32) * 255).astype(np.uint8)
        if "dist" in p:          # modelos con la salida dist: mm / 0,1 mm por nivel [F2B2]
            arrays["dist"] = np.round(np.clip(p["dist"].astype(np.float32), 0, 255 * DIST_MM_PER_LEVEL)
                                      / DIST_MM_PER_LEVEL).astype(np.uint8)
        if "role" in p:          # modelos con la salida role3: P(secundario | hueso)·255 [y4xul]
            arrays["role"] = np.round(p["role"].astype(np.float32) * 255).astype(np.uint8)
        np.savez_compressed(args.out_dir / f"{cid}.npz", **arrays)
    if args.skip_metrics:
        (args.out_dir / "metricas.json").write_text(json.dumps({"checkpoint": str(args.ckpt), "fold": args.fold, "tta": args.tta,
                                                                "casos": cases}, indent=1), encoding="utf-8")
        print(f"fold {args.fold}: {len(cases)} casos (sin métricas de det/cls)")
        return
    ds = PengwinSlices(args.cache_dir, cases, cfg, train=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)
    m = evaluate(model, dl, MultiTaskLoss(cfg).to(device), device, cfg)
    (args.out_dir / "metricas.json").write_text(json.dumps({"checkpoint": str(args.ckpt), "fold": args.fold,
                                                            "epoch": ck.get("epoch"), "casos": cases, "metricas": m},
                                                           indent=1), encoding="utf-8")
    print(f"fold {args.fold}: {len(cases)} casos  mAP {m['mAP@[.50:.95]']:.3f}  Dice {m['dice_hueso']:.3f}  F1 {m['cls_f1_macro']:.3f}")


if __name__ == "__main__":
    main()
