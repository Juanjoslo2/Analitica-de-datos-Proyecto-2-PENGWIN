"""sam_baseline.py.

SAM zero-shot como línea base [enunciado §4.2]: SOLO para comparar, nunca en el pipeline.

Para cada corte con hueso del split, SAM (ViT-B) recibe el mismo corte que ve el modelo
(canal central, 256 × 256) y, como prompt, la caja que predijo NUESTRO detector para cada
región. Se compara la máscara de SAM contra el GT de esa región, igual que nuestra
segmentación. Variante extra: SAM con la caja GT (cota superior de SAM, sin depender de
nuestro detector).

Dice e IoU por región se acumulan sobre todos los píxeles del split (como en evaluate.py).
Si el detector no propone caja para una región presente, SAM no tiene prompt y esa región
cuenta como no segmentada en ese corte.

Uso:
    python scripts/sam_baseline.py --ckpt checkpoints/base/best.pth --cache-dir D:/PENGWIN/data_processed \
        --sam-ckpt D:/PENGWIN/sam/sam_vit_b_01ec64.pth
Salida: reports/sam/<modelo>_<split>.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "03_src"))

from pengwin.data.dataset import read_split  # noqa: E402
from pengwin.data.targets import REGION_NAMES, region_boxes_2d, region_of  # noqa: E402
from pengwin.inference.volume import predict_case  # noqa: E402
from pengwin.data.slice_cache import load_case_cache  # noqa: E402
from pengwin.models.pengwin_net import PengwinNet  # noqa: E402


class Acc:
    """Intersección y tamaños por región, acumulados sobre todos los píxeles."""

    def __init__(self):
        self.i, self.p, self.g = np.zeros(3), np.zeros(3), np.zeros(3)

    def add(self, k, pred, gt):
        self.i[k] += np.logical_and(pred, gt).sum()
        self.p[k] += pred.sum()
        self.g[k] += gt.sum()

    def result(self):
        dice = 2 * self.i / np.maximum(self.p + self.g, 1)
        iou = self.i / np.maximum(self.p + self.g - self.i, 1)
        out = {f"dice_{n}": float(d) for n, d in zip(REGION_NAMES, dice)}
        out.update({f"iou_{n}": float(v) for n, v in zip(REGION_NAMES, iou)})
        out["dice"], out["iou"] = float(dice.mean()), float(iou.mean())
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--sam-ckpt", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--cases", nargs="*", default=None)
    args = ap.parse_args()
    from segment_anything import SamPredictor, sam_model_registry

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = PengwinNet(cfg["model"]).to(device).eval()
    model.load_state_dict(ck["model"])
    sam = SamPredictor(sam_model_registry["vit_b"](checkpoint=str(args.sam_ckpt)).to(device).eval())

    cases = args.cases or read_split(REPO / cfg["data"]["splits"], args.split)
    acc = {"propio": Acc(), "SAM + caja predicha": Acc(), "SAM + caja GT": Acc()}
    n_slices, t_sam = 0, 0.0
    for c, cid in enumerate(cases, 1):
        t0 = time.time()
        pred = predict_case(model, args.cache_dir, cid, cfg, device)
        image, label, meta = load_case_cache(args.cache_dir / cid)
        for z in meta["bone_slices"]:
            gt_reg = region_of(np.asarray(label[z]))
            gt_boxes = region_boxes_2d(np.asarray(label[z]))
            rgb = np.repeat(np.asarray(image[z])[..., None], 3, axis=2)
            ts = time.perf_counter()
            sam.set_image(rgb)
            det = pred["boxes"][z]
            for k in range(3):
                gt_k = gt_reg == k + 1
                acc["propio"].add(k, pred["semantic"][z] == k + 1, gt_k)
                sel = det["labels"] == k
                if sel.any():
                    box = det["boxes"][sel][np.argmax(det["scores"][sel])]
                    m, _, _ = sam.predict(box=box, multimask_output=False)
                    acc["SAM + caja predicha"].add(k, m[0], gt_k)
                else:
                    acc["SAM + caja predicha"].add(k, np.zeros_like(gt_k), gt_k)
                if gt_boxes["present"][k]:
                    m, _, _ = sam.predict(box=gt_boxes["boxes"][k], multimask_output=False)
                    acc["SAM + caja GT"].add(k, m[0], gt_k)
            t_sam += time.perf_counter() - ts
            n_slices += 1
        print(f"[{c}/{len(cases)}] {cid}  {len(meta['bone_slices'])} cortes  {time.time() - t0:.0f} s", flush=True)

    res = {k: a.result() for k, a in acc.items()}
    table = pd.DataFrame(res).T
    out = {"checkpoint": str(args.ckpt), "sam": "vit_b (sam_vit_b_01ec64.pth)", "split": args.split, "casos": cases,
           "cortes_con_hueso": n_slices, "ms_sam_por_corte": 1000 * t_sam / max(n_slices, 1), "resultados": res}
    out_dir = REPO / "reports" / "sam"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.ckpt.parent.name}_{args.split}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n{n_slices} cortes con hueso · SAM {out['ms_sam_por_corte']:.0f} ms/corte (embedding + 3-6 prompts)")
    print(table[["dice", "iou", "dice_SA", "dice_LI", "dice_RI"]].round(3).to_string())


if __name__ == "__main__":
    main()
