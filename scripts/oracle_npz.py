"""oracle_npz.py.

Construye un "OOF oráculo": región y borde tomados del ground truth del caché, en el mismo formato
que las predicciones fuera de fold (un .npz por caso con ``semantic`` y ``edge``).

Sirve para medir el TECHO del posproceso: si con región y borde perfectos el Dice por fragmento ya
es bajo, el cuello de botella es el método de separación, no el modelo (`reports/tuning/RESUMEN.md` §2).

Uso:
    python scripts/oracle_npz.py --cache-dir /data/data_processed --out /data/oof/oracle_gt
    # después: scripts/tune_postprocess.py --oof /data/oof/oracle_gt (sin {fold})
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "03_src"))

from pengwin.data.slice_cache import load_case_cache, load_edge_cache  # noqa: E402
from pengwin.data.targets import region_of  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=Path("/data/data_processed"))
    ap.add_argument("--out", type=Path, default=Path("/data/oof/oracle_gt"))
    ap.add_argument("--folds-file", type=Path, default=REPO / "reports" / "tuning" / "cv_folds.json")
    args = ap.parse_args()
    folds = json.loads(args.folds_file.read_text(encoding="utf-8"))["folds"]
    cases = sorted(c for f in folds for c in f)
    args.out.mkdir(parents=True, exist_ok=True)
    for k, cid in enumerate(cases, 1):
        d = args.cache_dir / cid
        _, label, _ = load_case_cache(d)
        sem = region_of(np.asarray(label)).astype(np.uint8)
        edge = np.asarray(load_edge_cache(d))
        np.savez_compressed(args.out / f"{cid}.npz", semantic=sem, edge=(edge > 0).astype(np.uint8) * 255)
        if k % 20 == 0:
            print(f"[{k}/{len(cases)}]", flush=True)
    print("listo", len(cases))


if __name__ == "__main__":
    main()
