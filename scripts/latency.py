"""latency.py.

Latencia por imagen (corte) en GPU y en CPU [enunciado §4.3].

Mide el modelo + decodificación de cajas + NMS, con batch 1 (un corte a la vez, como en el
slider del dashboard), después de un calentamiento. También mide un volumen completo
en GPU (todos los cortes, batch 16), que es lo que tarda el visualizador 3.

Uso:
    python scripts/latency.py --ckpt checkpoints/base/best.pth --cache-dir D:/PENGWIN/data_processed
Salida: reports/latency/<modelo>.json
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "03_src"))

from pengwin.data.dataset import context_stack, read_split  # noqa: E402
from pengwin.data.slice_cache import load_case_cache  # noqa: E402
from pengwin.detection.grid import decode  # noqa: E402
from pengwin.inference.volume import TTA_SETS, _tta_forward, predict_case  # noqa: E402
from pengwin.models.pengwin_net import PengwinNet  # noqa: E402


def time_slices(model, xs, device, amp: bool, warmup: int = 10, tta=None):
    times = []
    for i, x in enumerate(xs):
        x = x.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            if tta:
                out = _tta_forward(model, x, tta, amp, device)
                out["tta_sem"].argmax(1)
            else:
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    out = model(x)
                out["seg_logits"].argmax(1)
            decode(out["det_scores"], out["det_ltrb"], 8, 256)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if i >= warmup:
            times.append(1000 * (time.perf_counter() - t0))
    t = np.array(times)
    return {"media_ms": float(t.mean()), "p50_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95)),
            "n": int(t.size)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=110)
    ap.add_argument("--tta", nargs="?", const="5", default=None, choices=list(TTA_SETS),
                    help="mide con TTA (5 o 9 pasadas por corte); el reporte se guarda como <modelo>_tta.json")
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    case = read_split(REPO / cfg["data"]["splits"], "test")[0]
    image, _, meta = load_case_cache(args.cache_dir / case)
    zs = np.linspace(0, image.shape[0] - 1, args.n).astype(int)
    xs = [torch.from_numpy(context_stack(image, z, meta["context_offset"]).astype(np.float32)[None] / 255.0) for z in zs]

    tta = TTA_SETS[args.tta] if args.tta else None
    res = {"checkpoint": str(args.ckpt), "caso": case, "cpu": platform.processor(), "torch_threads": torch.get_num_threads(),
           "tta": args.tta}
    model = PengwinNet(cfg["model"]).eval()
    model.load_state_dict(ck["model"])
    res["cpu_fp32"] = time_slices(model, xs, torch.device("cpu"), amp=False, tta=tta)
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        model = model.to(dev)
        res["gpu"] = torch.cuda.get_device_name(0)
        res["gpu_amp"] = time_slices(model, xs, dev, amp=True, tta=tta)
        res["gpu_fp32"] = time_slices(model, xs, dev, amp=False, tta=tta)
        t0 = time.perf_counter()
        predict_case(model, args.cache_dir, case, cfg, dev, tta=tta)
        torch.cuda.synchronize()
        res["gpu_volumen_completo_s"] = time.perf_counter() - t0
        res["cortes_volumen"] = int(image.shape[0])
    out_dir = REPO / "reports" / "latency"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.ckpt.parent.name}{'_tta' if tta else ''}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
