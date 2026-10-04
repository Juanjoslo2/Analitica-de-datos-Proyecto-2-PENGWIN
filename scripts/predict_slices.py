"""predict_slices.py.

Galería de predicciones corte a corte de un checkpoint (entregable visual de la semana 9:
"primeras predicciones de bounding boxes visualmente razonables").

Uso:
    python scripts/predict_slices.py --ckpt checkpoints/base/best.pth --cache-dir D:/PENGWIN/data_processed
    python scripts/predict_slices.py --ckpt ... --case 004 --n 8

Por defecto toma el primer caso de val y 8 cortes con hueso repartidos en z.
Salida: reports/figures/semana9/pred_<case>_<ckpt>.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "03_src"))

from pengwin.data.dataset import PengwinSlices, read_split  # noqa: E402
from pengwin.data.slice_cache import load_case_cache  # noqa: E402
from pengwin.detection.grid import decode  # noqa: E402
from pengwin.models.pengwin_net import PengwinNet  # noqa: E402
from pengwin.training.engine import autocast  # noqa: E402
from pengwin.visualization.overlay import prediction_grid  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--case", default=None)
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = PengwinNet(cfg["model"]).to(device).eval()
    model.load_state_dict(ck["model"])

    case = args.case or read_split(REPO / cfg["data"]["splits"], "val")[0]
    _, _, meta = load_case_cache(args.cache_dir / case)
    bone = meta["bone_slices"]
    zs = [bone[i] for i in np.linspace(0, len(bone) - 1, args.n + 2).round().astype(int)[1:-1]]
    ds = PengwinSlices(args.cache_dir, [case], cfg, train=False, slices=[(case, z) for z in zs])
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=len(ds))))
    pp = cfg.get("postprocess", {})
    with torch.no_grad(), autocast(device, True):
        out = model(batch["image"].to(device))
    dets = decode(out["det_scores"], out["det_ltrb"], model.stride, batch["image"].shape[-1],
                  pp.get("det_score_threshold", 0.3), pp.get("nms_iou", 0.5), pp.get("max_boxes_per_class", 1))
    pred_sem = out["seg_logits"].argmax(1).cpu().numpy()
    samples = []
    for i, z in enumerate(zs):
        samples.append({
            "image": batch["image"][i, 1].numpy(), "title": f"caso {case} · z={z}",
            "gt_semantic": batch["semantic"][i].numpy(), "gt_boxes": batch["boxes"][i].numpy(),
            "gt_present": batch["present"][i].numpy() > 0.5,
            "pred_semantic": pred_sem[i], "pred_boxes": {k: v.cpu().numpy() for k, v in dets[i].items()},
        })
    fig = prediction_grid(samples)
    out_dir = REPO / "reports" / "figures" / "semana9"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"pred_{case}_{args.ckpt.parent.name}.png"
    fig.savefig(path, dpi=110)
    print(path)


if __name__ == "__main__":
    main()
