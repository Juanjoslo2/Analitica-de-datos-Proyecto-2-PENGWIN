"""consolidate.py.

Une los JSON por caso de ``reports/eda/cache`` en tablas analizables:
    cases      : 1 fila por caso (geometría, HU, metal, recorte, conteos, split)
    fragments  : 1 fila por fragmento (volumen, forma, contacto, distancia al principal)
    slices     : 1 fila por corte axial con hueso (presencia, cajas, islas, bordes)
    hist_bone / hist_all : histogramas HU acumulados (bins de 16 HU entre -1024 y 3072)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from pengwin.eda.extract import HIST_BINS


def load_eda_tables(cache_dir: Path, splits_path: Path | None = None) -> Dict[str, object]:
    files = sorted(Path(cache_dir).glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No hay JSON en {cache_dir}. Ejecuta scripts/run_eda_extract.py")
    cases, frags, slices = [], [], []
    hist_bone = np.zeros(len(HIST_BINS) - 1)
    hist_all = np.zeros(len(HIST_BINS) - 1)
    for f in files:
        r = json.loads(f.read_text(encoding="utf-8"))
        cases.append(r["case"])
        frags.extend(r["fragments"])
        slices.extend(r["slices"])
        hist_bone += np.asarray(r["hist_bone"])
        hist_all += np.asarray(r["hist_all"])

    cases_df = pd.DataFrame(cases)
    frags_df = pd.DataFrame(frags)
    slices_df = pd.DataFrame(slices)
    for df in (cases_df, frags_df, slices_df):
        df["case_id"] = df["case_id"].astype(str).str.zfill(3)

    if splits_path is not None and Path(splits_path).exists():
        s = json.loads(Path(splits_path).read_text(encoding="utf-8"))["splits"]
        split_of = {c: k for k, ids in s.items() for c in ids}
        for df in (cases_df, frags_df, slices_df):
            df["split"] = df["case_id"].map(split_of).fillna("sin_split")

    return {"cases": cases_df, "fragments": frags_df, "slices": slices_df,
            "hist_bone": hist_bone, "hist_all": hist_all, "hist_edges": HIST_BINS}


def export_tables(tables: Dict[str, object], out_dir: Path) -> None:
    """Guarda tablas livianas y versionables (el caché JSON no se versiona)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables["cases"].to_csv(out_dir / "eda_cases.csv", index=False)
    tables["fragments"].to_csv(out_dir / "eda_fragments.csv", index=False)
    tables["slices"].to_csv(out_dir / "eda_slices.csv.gz", index=False, compression="gzip")
    np.savez(out_dir / "eda_hu_hist.npz", bone=tables["hist_bone"], all=tables["hist_all"],
             edges=tables["hist_edges"])
