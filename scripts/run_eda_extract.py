"""run_eda_extract.py.

Extrae las estadísticas del EDA para todos los casos (reanudable).

Uso (desde la raíz del repositorio, con el entorno activo y `pip install -e .`):
    python scripts/run_eda_extract.py                 # todos los casos pendientes
    python scripts/run_eda_extract.py --cases 001 002 # casos concretos
    python scripts/run_eda_extract.py --force         # recalcular todo

Salida: reports/eda/cache/<case_id>.json (uno por caso, escritura atómica).
Después, 04_notebook/01_eda_pengwin.ipynb consolida y analiza las tablas.
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "03_src"))  # funciona aun sin `pip install -e .`

from pengwin.data.data_loader import get_dataset_pairs  # noqa: E402
from pengwin.eda.extract import extract_case, save_case  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=REPO / "01_data")
    ap.add_argument("--out-dir", type=Path, default=REPO / "reports" / "eda" / "cache")
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--shard", default="0/1", help="k/n: procesa solo los casos con índice %% n == k (paralelismo)")
    ap.add_argument("--time-budget", type=float, default=None,
                    help="segundos máximos; no inicia un caso nuevo si quedan < 60 s")
    args = ap.parse_args()
    t_start = time.time()
    k_shard, n_shard = (int(v) for v in args.shard.split("/"))

    pairs = get_dataset_pairs(args.data_dir)
    if args.cases:
        pairs = [p for p in pairs if p["case_id"] in set(args.cases)]
    todo = [p for p in pairs if args.force or not (args.out_dir / f"{p['case_id']}.json").exists()]
    todo = todo[k_shard::n_shard]
    print(f"{len(pairs)} casos, {len(todo)} pendientes", flush=True)
    for k, p in enumerate(todo, 1):
        if args.time_budget and time.time() - t_start > args.time_budget - 60:
            print("Presupuesto de tiempo agotado; se reanuda en la siguiente ejecución.")
            break
        t = time.time()
        try:
            save_case(extract_case(p["case_id"], p["image_path"], p["label_path"]), args.out_dir)
            print(f"[{k}/{len(todo)}] {p['case_id']} ok ({time.time() - t:.1f} s)", flush=True)
        except Exception:  # un caso defectuoso no detiene el resto; queda registrado
            print(f"[{k}/{len(todo)}] {p['case_id']} ERROR\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
