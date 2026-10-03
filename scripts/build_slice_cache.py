"""build_slice_cache.py.

Construye el caché de cortes 256×256 de todos los casos (reanudable) [DD §1].

Uso (desde la raíz del repositorio, con el entorno activo y `pip install -e .`):
    python scripts/build_slice_cache.py                         # rutas de configs/base.yaml
    python scripts/build_slice_cache.py --data-dir D:/PENGWIN/01_data --cache-dir D:/PENGWIN/data_processed
    python scripts/build_slice_cache.py --cases 001 002 --force

Salida: <cache-dir>/<case_id>/{image.npy, label.npy, meta.json} (~50 MB por caso, ~5 GB en total).
Si el repositorio está dentro de OneDrive, conviene poner datos y caché FUERA de la carpeta
sincronizada (--data-dir / --cache-dir): son ~32 GB + 5 GB que OneDrive intentaría subir.
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "03_src"))  # funciona aun sin `pip install -e .`

from pengwin.data.data_loader import get_dataset_pairs  # noqa: E402
from pengwin.data.slice_cache import build_case_cache  # noqa: E402
from pengwin.utils.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "base.yaml")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--edges-only", action="store_true",
                    help="solo añade edge.npy (borde de fractura 3D) a los casos ya cacheados; no relee los .mha")
    ap.add_argument("--shard", default="0/1", help="k/n: procesa solo los casos con índice %% n == k (paralelismo)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = cfg["data"]
    data_dir = args.data_dir or REPO / d["raw_dir"]
    cache_dir = args.cache_dir or REPO / d["processed_dir"]
    k_shard, n_shard = (int(v) for v in args.shard.split("/"))

    if args.edges_only:
        from pengwin.data.slice_cache import add_edge_cache
        dirs = sorted(p for p in cache_dir.iterdir() if (p / "label.npy").exists())
        if args.cases:
            dirs = [p for p in dirs if p.name in set(args.cases)]
        for k, d in enumerate(dirs[k_shard::n_shard], 1):
            n = add_edge_cache(d)
            print(f"[{k}] {d.name}: {n} vóxeles de borde", flush=True)
        return

    pairs = get_dataset_pairs(data_dir)
    if args.cases:
        pairs = [p for p in pairs if p["case_id"] in set(args.cases)]
    todo = [p for p in pairs if args.force or not (cache_dir / p["case_id"] / "meta.json").exists()]
    todo = todo[k_shard::n_shard]
    print(f"{len(pairs)} casos, {len(todo)} pendientes -> {cache_dir}", flush=True)
    for k, p in enumerate(todo, 1):
        t = time.time()
        try:
            meta = build_case_cache(p["case_id"], p["image_path"], p["label_path"], cache_dir,
                                    image_size=d["image_size"],
                                    window=(d["window"]["level"], d["window"]["width"]),
                                    context_mm=d["context_mm"], crop_margin_mm=d["crop_margin_mm"])
            print(f"[{k}/{len(todo)}] {p['case_id']} ok ({time.time() - t:.1f} s) "
                  f"{meta['pixel_mm']:.2f} mm/px, Δ={meta['context_offset']}, {len(meta['bone_slices'])} cortes con hueso",
                  flush=True)
        except Exception:  # un caso defectuoso no detiene el resto; queda registrado
            print(f"[{k}/{len(todo)}] {p['case_id']} ERROR\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
