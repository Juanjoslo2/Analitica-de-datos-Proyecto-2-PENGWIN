"""cv_folds.py.

Folds de validación cruzada POR PACIENTE sobre train + val (85 casos). El test (15 casos) queda
fuera: no se usa para elegir nada y se mira una sola vez al final.

Los folds se estratifican por número de fragmentos secundarios del caso (0-1 / 2 / 3+), que es
lo que más mueve el Dice por fragmento; así ningún fold queda con todos los casos fáciles.

Uso:
    python scripts/cv_folds.py                 # k = 5, semilla 42 -> reports/tuning/cv_folds.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "03_src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=REPO / "reports" / "tuning" / "cv_folds.json")
    args = ap.parse_args()
    sp = json.loads((REPO / "01_data" / "splits.json").read_text(encoding="utf-8"))["splits"]
    cases = sorted(sp["train"] + sp["val"])
    frags = pd.read_csv(REPO / "reports" / "eda" / "eda_fragments.csv", dtype={"case_id": str})
    frags["case_id"] = frags.case_id.str.zfill(3)
    n_sec = frags[~frags.is_main].groupby("case_id").size().reindex(cases).fillna(0).astype(int)
    strata = np.clip(n_sec.to_numpy(), 1, 3)                      # 0-1 -> 1, 2 -> 2, 3+ -> 3
    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    folds = [sorted(np.array(cases)[val].tolist()) for _, val in skf.split(cases, strata)]
    assert sorted(c for f in folds for c in f) == cases and not set(cases) & set(sp["test"])
    resumen = [{"fold": k, "casos": len(f), "secundarios": int(n_sec[f].sum())} for k, f in enumerate(folds)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"k": args.k, "seed": args.seed, "estratos": "nº de secundarios 0-1/2/3+",
                                    "test_excluido": sp["test"], "resumen": resumen, "folds": folds}, indent=1),
                        encoding="utf-8")
    print(pd.DataFrame(resumen).to_string(index=False))


if __name__ == "__main__":
    main()
