"""latency_two_pass.py.

Latencia del modelo de dos pasadas sobre un volumen completo, en GPU y en CPU [enunciado §4.3].

Mide, para el mismo caso y el mismo checkpoint, el tiempo del modelo (sin el posproceso 3D) en tres
modos: solo la pasada 1, dos pasadas refinando todos los huesos, y dos pasadas con la segunda
selectiva (``--gate-px``). Reporta segundos por volumen y milisegundos por corte.

Uso:
    python scripts/latency_two_pass.py --ckpt checkpoints/y8_refine_eff_final/last.pth \
        --cache-dir /ruta/cache --hi-cache-dir /ruta/cache512 --case 001
Salida: reports/latency/<nombre del checkpoint>_dos_pasadas.json
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "03_src"))

from pengwin.inference.volume import predict_case, predict_case_two_pass  # noqa: E402
from pengwin.models.pengwin_net import PengwinNet  # noqa: E402


def medir(fn, device, repeticiones):
    tiempos = []
    for _ in range(repeticiones):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        tiempos.append(time.perf_counter() - t0)
    return min(tiempos), res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--hi-cache-dir", type=Path, required=True)
    ap.add_argument("--case", default="001")
    ap.add_argument("--gate-px", type=int, default=10)
    ap.add_argument("--repeticiones", type=int, default=2, help="se reporta la mejor de N (la primera calienta)")
    ap.add_argument("--solo-cpu", action="store_true")
    ap.add_argument("--hilos", type=int, default=8)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    torch.set_num_threads(args.hilos)
    out = {"checkpoint": str(args.ckpt), "caso": args.case, "gate_px": args.gate_px, "cpu": platform.processor() or platform.machine(),
           "torch_threads": args.hilos, "nota": "tiempo del modelo sobre el volumen completo; no incluye el posproceso 3D"}
    dispositivos = [torch.device("cpu")]
    if torch.cuda.is_available() and not args.solo_cpu:
        dispositivos.insert(0, torch.device("cuda"))
        out["gpu"] = torch.cuda.get_device_name(0)
    for device in dispositivos:
        model = PengwinNet(cfg["model"]).to(device).eval()
        model.load_state_dict(ck["model"])
        modos = {
            "pasada_1": lambda: predict_case(model, args.cache_dir, args.case, cfg, device),
            "dos_pasadas": lambda: predict_case_two_pass(model, args.cache_dir, args.hi_cache_dir, args.case, cfg, device),
            "dos_pasadas_selectiva": lambda: predict_case_two_pass(model, args.cache_dir, args.hi_cache_dir, args.case, cfg,
                                                                   device, gate_px=args.gate_px),
        }
        bloque = {}
        for nombre, fn in modos.items():
            seg, res = medir(fn, device, args.repeticiones if device.type == "cuda" else 1)
            cortes = int(res["semantic"].shape[0])
            bloque[nombre] = {"s_por_volumen": seg, "ms_por_corte": 1000 * seg / cortes, "cortes": cortes,
                              "recortes": int(res.get("n_roi", 0)), "recortes_posibles": int(res.get("n_roi_total", 0))}
            print(f"{device.type:4s} {nombre:22s} {seg:7.2f} s  {1000 * seg / cortes:7.1f} ms/corte  "
                  f"recortes {bloque[nombre]['recortes']}/{bloque[nombre]['recortes_posibles']}", flush=True)
        out[device.type] = bloque
    dst = REPO / "reports" / "latency" / f"{args.ckpt.parent.name}_dos_pasadas.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("->", dst)


if __name__ == "__main__":
    main()
