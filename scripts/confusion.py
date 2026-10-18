"""confusion.py.

Matrices de confusión del modelo final en test (§7 del enunciado):

    1. Clasificación por corte: para cada hueso, ¿el modelo dice que está (P > 0,5) y está?
       Tres matrices 2×2 (SA, LI, RI) sobre TODOS los cortes del volumen.
    2. Segmentación semántica por píxel: 4×4 (fondo, SA, LI, RI) en la grilla del modelo
       (256 px), con la región de la inferencia final (TTA de configs/tuned.yaml).

Uso:
    python scripts/confusion.py --ckpt checkpoints/v2_last/best.pth --cache-dir D:/PENGWIN/data_processed
Salida: reports/eval/confusion_test.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "03_src"))

from pengwin.data.dataset import read_split  # noqa: E402
from pengwin.data.slice_cache import load_case_cache  # noqa: E402
from pengwin.inference.volume import TTA_SETS, predict_case  # noqa: E402
from pengwin.models.pengwin_net import PengwinNet  # noqa: E402
from pengwin.utils.config import load_config  # noqa: E402

CLASES = ["fondo", "SA", "LI", "RI"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "tuned.yaml")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = PengwinNet(cfg["model"]).to(dev).eval()
    model.load_state_dict(ck["model"])
    tta = (load_config(args.config).get("inference") or {}).get("tta")
    cases = read_split(REPO / cfg["data"]["splits"], args.split)

    cls_cm = np.zeros((3, 2, 2), np.int64)            # [hueso, real, predicho]
    pix_cm = np.zeros((4, 4), np.int64)               # [real, predicho]
    for cid in cases:
        p = predict_case(model, args.cache_dir, cid, cfg, dev, tta=TTA_SETS[str(tta)] if tta else None)
        _, label, _ = load_case_cache(args.cache_dir / cid)
        lab = np.asarray(label)
        reg = np.where(lab == 0, 0, (lab.astype(np.int64) - 1) // 10 + 1)
        pix_cm += np.bincount((reg * 4 + p["semantic"]).ravel(), minlength=16).reshape(4, 4)
        presente = np.stack([(reg == c).reshape(reg.shape[0], -1).any(1) for c in (1, 2, 3)], 1)
        pred = p["cls"] > 0.5
        for c in range(3):
            cls_cm[c] += np.bincount(presente[:, c].astype(int) * 2 + pred[:, c].astype(int), minlength=4).reshape(2, 2)
        print(cid, flush=True)

    res = {"checkpoint": str(args.ckpt), "split": args.split, "tta": tta, "casos": cases,
           "clasificacion_por_corte": {h: cls_cm[c].tolist() for c, h in enumerate(CLASES[1:])},
           "pixeles": {"clases": CLASES, "matriz": pix_cm.tolist()}}
    out = REPO / "reports" / "eval" / "confusion_test.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    norm = pix_cm / pix_cm.sum(1, keepdims=True)
    print(np.round(norm, 4))
    for c, h in enumerate(CLASES[1:]):
        print(h, cls_cm[c].tolist())


if __name__ == "__main__":
    main()
