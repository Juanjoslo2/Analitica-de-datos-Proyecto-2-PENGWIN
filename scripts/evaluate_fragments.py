"""evaluate_fragments.py.

Pipeline completo de la semana 10 sobre un split, caso por caso:

    caché → modelo corte a corte → volumen (Z, 256, 256) de región + borde
          → fragmentos 3D (postprocess/instances.py) → grilla nativa del .mha
          → métricas por fragmento contra el GT y distancia de separación en mm (pred y GT)

Uso:
    python scripts/evaluate_fragments.py --ckpt checkpoints/base/best.pth \
        --cache-dir D:/PENGWIN/data_processed --data-dir D:/PENGWIN/01_data
    ... --cases 001 002          # solo algunos casos
    ... --save-dir D:/PENGWIN/pred   # guarda la predicción de cada caso como .mha (visualizador 3)

Salida en reports/fragments/:
    <modelo>_<split>_fragmentos.csv   una fila por fragmento GT (dice, iou, recuperado, volumen)
    <modelo>_<split>_distancias.csv   una fila por secundario GT (distancia GT, predicha y error)
    <modelo>_<split>.json             resumen global y por caso
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "03_src"))

from pengwin.data.data_loader import get_dataset_pairs, load_and_standardize_mha  # noqa: E402
from pengwin.data.dataset import read_split  # noqa: E402
from pengwin.evaluation.fragment_metrics import distance_comparison, match_fragments, summarize  # noqa: E402
from pengwin.inference.volume import predict_case, to_native  # noqa: E402
from pengwin.models.pengwin_net import PengwinNet  # noqa: E402
from pengwin.postprocess.instances import separate_instances  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True, help="carpeta con los .mha (para el GT nativo)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--save-dir", type=Path, default=None)
    ap.add_argument("--method", default=None, choices=["edge", "edt"], help="por defecto, el de la config")
    ap.add_argument("--seed-depth-mm", type=float, default=None)
    ap.add_argument("--edge-threshold", type=float, default=None)
    ap.add_argument("--tag", default="", help="sufijo del nombre del reporte (p. ej. _edt)")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    # El posproceso sale de la config ACTUAL del repo (se ajusta sin reentrenar), no de la del checkpoint
    from pengwin.utils.config import load_config
    pp = {**cfg.get("postprocess", {}), **load_config(REPO / "configs" / "base.yaml").get("postprocess", {})}
    if args.edge_threshold is not None:
        pp["edge_threshold"] = args.edge_threshold
    method = args.method or pp.get("instance_method", "edge")
    depth = args.seed_depth_mm if args.seed_depth_mm is not None else pp.get("seed_depth_mm", 4.0)
    model = PengwinNet(cfg["model"]).to(device).eval()
    model.load_state_dict(ck["model"])
    # Solo se necesitan las etiquetas nativas (la imagen sale del caché): basta con PENGWIN_CT_train_labels
    labels_dir = args.data_dir / "PENGWIN_CT_train_labels"
    pairs = {f.stem: {"label_path": f} for f in labels_dir.glob("*.mha")} or {p["case_id"]: p for p in get_dataset_pairs(args.data_dir)}
    cases = args.cases or read_split(REPO / cfg["data"]["splits"], args.split)

    frag_rows, dist_rows, per_case = [], [], {}
    for k, cid in enumerate(cases, 1):
        t0 = time.time()
        pred = predict_case(model, args.cache_dir, cid, cfg, device)
        meta = pred["meta"]
        grid_spacing = (meta["spacing_zyx"][0], meta["pixel_mm"], meta["pixel_mm"])
        lab_grid = separate_instances(pred["semantic"], pred["edge"], grid_spacing, pp.get("edge_threshold", 0.5),
                                      pp.get("min_fragment_cm3", 0.1), method=method, seed_depth_mm=depth)
        lab_pred = to_native(lab_grid, meta, order=0)
        gt, spacing, _ = load_and_standardize_mha(pairs[cid]["label_path"], is_label=True)
        fr = match_fragments(gt, lab_pred, spacing)
        dr = distance_comparison(gt, lab_pred, spacing, fr)
        for row in fr + dr:
            row["case_id"] = cid
        frag_rows += fr
        dist_rows += dr
        per_case[cid] = summarize(fr, dr)
        if args.save_dir:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            img = sitk.GetImageFromArray(lab_pred)
            img.SetSpacing(tuple(float(s) for s in spacing[::-1]))
            sitk.WriteImage(img, str(args.save_dir / f"{cid}_pred.mha"), useCompression=True)
        s = per_case[cid]
        print(f"[{k}/{len(cases)}] {cid}  {time.time() - t0:.0f} s  Dice frag {s['dice_fragmento']:.3f}  "
              f"recuperados {s['recuperados_%']:.0f} %  fragmentos GT {int(s['n_fragmentos_gt'])} / pred "
              f"{len(np.unique(lab_pred)) - 1}", flush=True)

    name = f"{args.ckpt.parent.name}_{args.split}{args.tag}"
    out_dir = REPO / "reports" / "fragments"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(frag_rows).to_csv(out_dir / f"{name}_fragmentos.csv", index=False)
    pd.DataFrame(dist_rows).to_csv(out_dir / f"{name}_distancias.csv", index=False)
    total = summarize(frag_rows, dist_rows)
    (out_dir / f"{name}.json").write_text(json.dumps({"checkpoint": str(args.ckpt), "split": args.split,
                                                      "postproceso": {"metodo": method, "seed_depth_mm": depth, **{k: pp.get(k) for k in ("edge_threshold", "min_fragment_cm3")}},
                                                      "global": total, "por_caso": per_case}, indent=1), encoding="utf-8")
    print("\nResumen", name)
    print(pd.Series(total).round(3).to_string())


if __name__ == "__main__":
    main()
