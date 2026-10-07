"""stress_detection.py.

Prueba de estrés de la detección: ¿el mAP@0.50 ≈ 0,98 y el IoU de caja ≈ 0,92 son reales o
están "enmascarados" por cómo se evalúa? (pregunta del profesor en el avance del 2026-10-02).

Recorre TODOS los cortes de cada volumen de test (no solo los que tienen hueso más un 12 % de
vacíos, como la evaluación de entrenamiento) y mide la detección en cuatro escenarios:

    A  oficial      cortes con hueso + 12 % de vacíos; cajas < 4 px ignoradas
    B  volumen      TODOS los cortes del volumen; cajas < 4 px ignoradas
    C  sin ignorar  TODOS los cortes; las cajas diminutas cuentan como objetos a detectar
    D  sin región   C, pero la predicción se compara contra cajas de OTRO paciente del mismo
                    corte relativo: línea base tonta que mide cuánto se gana solo por la
                    posición típica de la pelvis

Además desglosa el IoU de caja y el recall@0,5 por tamaño de la caja real y por posición
del corte dentro del hueso (extremos contra centro).

Uso:
    python scripts/stress_detection.py --ckpt checkpoints/v2_last/best.pth --cache-dir D:/PENGWIN/data_processed
Salida: reports/eval/stress_detection.json
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
from pengwin.data.targets import REGION_NAMES, region_boxes_2d  # noqa: E402
from pengwin.detection.boxes import box_iou  # noqa: E402
from pengwin.evaluation.det_metrics import DetectionEvaluator  # noqa: E402
from pengwin.inference.volume import predict_case  # noqa: E402
from pengwin.models.pengwin_net import PengwinNet  # noqa: E402


def _t(x):
    return torch.from_numpy(np.asarray(x))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = PengwinNet(cfg["model"]).to(dev).eval()
    model.load_state_dict(ck["model"])
    cases = read_split(REPO / cfg["data"]["splits"], args.split)
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    frac = float(cfg["data"].get("empty_slice_fraction", 0.12))

    ev = {k: DetectionEvaluator() for k in "ABCD"}
    filas = []                       # una por (corte, región presente): tamaño, posición, IoU
    gt_por_caso = {}
    for cid in cases:
        p = predict_case(model, args.cache_dir, cid, cfg, dev)
        _, label, meta = load_case_cache(args.cache_dir / cid)
        bone = sorted(meta["bone_slices"])
        z0, z1 = bone[0], bone[-1]
        empty = [z for z in range(label.shape[0]) if z not in set(bone)]
        n_emp = min(len(empty), int(round(len(bone) * frac / (1 - frac))))
        oficiales = set(bone) | set(rng.choice(empty, size=n_emp, replace=False).tolist() if n_emp else [])
        gts = [region_boxes_2d(np.asarray(label[z]), 4.0) for z in range(label.shape[0])]
        gt_por_caso[cid] = (gts, z0, z1)
        for z, (pred, g) in enumerate(zip(p["boxes"], gts)):
            pr = {k: _t(v) for k, v in pred.items()}
            sin_ign = np.zeros(3, bool)
            if z in oficiales:
                ev["A"].add(pr, _t(g["boxes"]), _t(g["present"]), _t(g["ignore"]))
            ev["B"].add(pr, _t(g["boxes"]), _t(g["present"]), _t(g["ignore"]))
            ev["C"].add(pr, _t(g["boxes"]), _t(g["present"]), _t(sin_ign))
            pos = (z - z0) / max(z1 - z0, 1)
            for c in range(3):
                if not g["present"][c]:
                    continue
                lado = float(min(g["boxes"][c, 2] - g["boxes"][c, 0], g["boxes"][c, 3] - g["boxes"][c, 1]))
                sel = pred["labels"] == c
                iou = 0.0
                if sel.any():
                    best = pred["boxes"][sel][np.argmax(pred["scores"][sel])]
                    iou = float(box_iou(_t(best[None]), _t(g["boxes"][c:c + 1]))[0, 0])
                filas.append({"caso": cid, "region": REGION_NAMES[c], "lado_px": lado, "pos": pos, "iou": iou})
        print(f"{cid}: {label.shape[0]} cortes ({len(bone)} con hueso)", flush=True)

    # D: cajas de otro paciente en la misma posición relativa (línea base "posición típica")
    ids = list(gt_por_caso)
    for i, cid in enumerate(ids):
        gts, z0, z1 = gt_por_caso[cid]
        otro, oz0, oz1 = gt_por_caso[ids[(i + 1) % len(ids)]]
        for z, g in enumerate(gts):
            rel = (z - z0) / max(z1 - z0, 1)
            oz = int(round(oz0 + rel * (oz1 - oz0)))
            if not (0 <= oz < len(otro)):
                continue
            o = otro[oz]
            k = np.nonzero(o["present"])[0]
            pr = {"boxes": _t(o["boxes"][k]), "scores": _t(np.ones(len(k), np.float32)), "labels": _t(k)}
            ev["D"].add(pr, _t(g["boxes"]), _t(g["present"]), _t(np.zeros(3, bool)))

    res = {"checkpoint": str(args.ckpt), "split": args.split, "casos": cases,
           "escenarios": {k: e.compute() for k, e in ev.items()}}
    import pandas as pd
    df = pd.DataFrame(filas)
    df["tamaño"] = pd.cut(df.lado_px, [0, 4, 16, 48, 1e9], right=False, labels=["< 4 px", "4-16 px", "16-48 px", "≥ 48 px"])
    df["posición"] = np.where((df.pos < 0.1) | (df.pos > 0.9), "extremos (10 % de cada punta)", "centro")
    agg = lambda g: {"n": int(len(g)), "IoU_medio": float(g.iou.mean()), "recall@0.5": float((g.iou >= 0.5).mean())}
    res["por_tamaño"] = {str(k): agg(g) for k, g in df.groupby("tamaño", observed=True)}
    res["por_posición"] = {str(k): agg(g) for k, g in df.groupby("posición")}
    res["por_región_y_tamaño"] = {f"{r} {t}": agg(g) for (r, t), g in df.groupby(["region", "tamaño"], observed=True)}
    out = REPO / "reports" / "eval" / "stress_detection.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: {m: round(v, 3) for m, v in e.items()} for k, e in res["escenarios"].items()}, indent=1))
    print(json.dumps({k: res[k] for k in ("por_tamaño", "por_posición")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
