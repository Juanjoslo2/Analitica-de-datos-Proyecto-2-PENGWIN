"""create_splits.py.

Ejecutar así en CMD/PowerShell desde la raíz del proyecto:
$env:PYTHONPATH="$PWD\03_src"
python -m pengwin.data.create_splits

Genera la partición fija del dataset (70% Train, 15% Val, 15% Test) por PACIENTE,
exporta 01_data/splits.json y calcula el resumen EDA obligatorio para la Semana 8.

v2 (tras el EDA, ver 04_notebook/01_eda_pengwin.ipynb §9): la v1 estratificaba solo
por "<= 6 / > 6 fragmentos" y dejaba el test con 67 % de sacros fracturados frente a
43 % en train. Ahora se buscan muchas particiones aleatorias (semilla fija) y se elige
la que minimiza la mayor diferencia estandarizada de medias (SMD) entre cada split y
el total en las variables que condicionan las métricas: fractura por región,
nº de fragmentos, fragmentos sin contacto, fragmentos pequeños, orientación y dz.
Si existe reports/eda/eda_cases.csv se usa (segundos); si no, se leen las máscaras.
"""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pengwin.data.data_loader import (
    decompose_labels_by_region,
    get_dataset_pairs,
    load_and_standardize_mha,
)
from sklearn.model_selection import train_test_split


def build_dataset_metadata(data_dir: Path) -> pd.DataFrame:
    """Inspecciona los 100 casos y extrae métricas de fragmentos para el EDA y

    estratificación.
    """
    pairs = get_dataset_pairs(data_dir)
    records = []

    print(f"Analizando {len(pairs)} casos para el EDA y estratificación...")

    for item in pairs:
        case_id = item["case_id"]
        # Cargamos únicamente la máscara (liviana y rápida)
        mask, spacing, orig_ori = load_and_standardize_mha(
            item["label_path"], is_label=True
        )
        regions_info = decompose_labels_by_region(mask)

        num_fragments_sa = regions_info.get("sacro", {}).get(
            "num_fragments", 0
        )
        num_fragments_li = regions_info.get("coxal_izquierdo", {}).get(
            "num_fragments", 0
        )
        num_fragments_ri = regions_info.get("coxal_derecho", {}).get(
            "num_fragments", 0
        )
        total_fragments = (
            num_fragments_sa + num_fragments_li + num_fragments_ri
        )

        has_comminuted = any(
            len(r["conminuted_ids"]) > 0 for r in regions_info.values()
        )

        # Categoría de severidad para estratificación:
        # 0: fractura con hasta 6 fragmentos
        # 1: fractura altamente fragmentada (> 6 fragmentos)
        # Se utilizan dos estratos porque solo existe un caso con
        # exactamente 3 fragmentos, lo que impide distribuir una
        # tercera categoría mediante estratificación en Train/Val/Test.
        if total_fragments <= 6:
            severity_stratum = 0
        else:
            severity_stratum = 1

        records.append(
            {
                "case_id": case_id,
                "slices_z": mask.shape[0],
                "height_y": mask.shape[1],
                "width_x": mask.shape[2],
                "spacing_z": spacing[0],
                "spacing_y": spacing[1],
                "spacing_x": spacing[2],
                "original_orientation": orig_ori,
                "fragments_sacro": num_fragments_sa,
                "fragments_coxal_izq": num_fragments_li,
                "fragments_coxal_der": num_fragments_ri,
                "total_fragments": total_fragments,
                "has_comminuted": has_comminuted,
                "stratum": severity_stratum,
            }
        )

    return pd.DataFrame(records)


BALANCE_COLUMNS = [
    "sacro_fracturado", "coxal_izq_fracturado", "coxal_der_fracturado",
    "total_fragments", "sec_sin_contacto", "sec_pequenos", "es_ras", "dz_grueso",
]


def metadata_from_eda(repo_root: Path) -> pd.DataFrame | None:
    """Variables de balance a partir de las tablas del EDA (si existen)."""
    cases_csv = repo_root / "reports" / "eda" / "eda_cases.csv"
    frags_csv = repo_root / "reports" / "eda" / "eda_fragments.csv"
    if not (cases_csv.exists() and frags_csv.exists()):
        return None
    c = pd.read_csv(cases_csv, dtype={"case_id": str})
    f = pd.read_csv(frags_csv, dtype={"case_id": str})
    sec = f[~f.is_main]
    df = pd.DataFrame({"case_id": c.case_id.str.zfill(3)})
    df["sacro_fracturado"] = (c.n_frag_SA > 1).astype(int)
    df["coxal_izq_fracturado"] = (c.n_frag_LI > 1).astype(int)
    df["coxal_der_fracturado"] = (c.n_frag_RI > 1).astype(int)
    df["total_fragments"] = c[["n_frag_SA", "n_frag_LI", "n_frag_RI"]].sum(axis=1)
    df["sec_sin_contacto"] = df.case_id.map(sec[~sec.contact_with_main.astype(bool)].groupby("case_id").size()).fillna(0)
    df["sec_pequenos"] = df.case_id.map(sec[sec.volume_cm3 < 5].groupby("case_id").size()).fillna(0)
    df["es_ras"] = (c.orientation_orig == "RAS").astype(int)
    df["dz_grueso"] = (c.dz > 0.9).astype(int)
    return df


def metadata_from_eda_cases(repo_root: Path) -> pd.DataFrame | None:
    """Mismas columnas que ``build_dataset_metadata`` pero leídas de reports/eda/eda_cases.csv."""
    cases_csv = repo_root / "reports" / "eda" / "eda_cases.csv"
    if not cases_csv.exists():
        return None
    c = pd.read_csv(cases_csv, dtype={"case_id": str})
    df = pd.DataFrame({
        "case_id": c.case_id.str.zfill(3), "slices_z": c.dim_z, "height_y": c.dim_y, "width_x": c.dim_x,
        "spacing_z": c.dz, "spacing_y": c.dy, "spacing_x": c.dx, "original_orientation": c.orientation_orig,
        "fragments_sacro": c.n_frag_SA, "fragments_coxal_izq": c.n_frag_LI, "fragments_coxal_der": c.n_frag_RI,
    })
    df["total_fragments"] = df[["fragments_sacro", "fragments_coxal_izq", "fragments_coxal_der"]].sum(axis=1)
    df["has_comminuted"] = df.total_fragments > 3
    df["stratum"] = (df.total_fragments > 6).astype(int)
    return df


def split_imbalance(df: pd.DataFrame, assign: np.ndarray, cols=BALANCE_COLUMNS) -> pd.DataFrame:
    """|media_split - media_total| / desviación_total por variable y split (SMD)."""
    X = df[cols].to_numpy(float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    rows = {name: np.abs(X[assign == k].mean(0) - mu) / sd for k, name in enumerate(["train", "val", "test"])}
    return pd.DataFrame(rows, index=cols)


def balanced_splits(df: pd.DataFrame, seed: int = 42, n_trials: int = 20000,
                    sizes=(0.70, 0.15, 0.15)) -> tuple[dict[str, list[str]], pd.DataFrame]:
    """Búsqueda aleatoria reproducible de la partición con menor SMD máximo."""
    rng = np.random.default_rng(seed)
    n = len(df)
    n_tr, n_va = int(round(sizes[0] * n)), int(round(sizes[1] * n))
    X = df[BALANCE_COLUMNS].to_numpy(float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    best, best_score = None, np.inf
    for _ in range(n_trials):
        perm = rng.permutation(n)
        parts = (perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:])
        score = max((np.abs(X[p].mean(0) - mu) / sd).max() for p in parts)
        if score < best_score:
            best, best_score = parts, score
    assign = np.empty(n, int)
    for k, p in enumerate(best):
        assign[p] = k
    ids = df.case_id.to_numpy()
    splits = {name: sorted(ids[assign == k].tolist()) for k, name in enumerate(["train", "val", "test"])}
    return splits, split_imbalance(df, assign)


def generate_stratified_splits(
    df: pd.DataFrame, seed: int = 42
) -> dict[str, list[str]]:
    """Divide en 70% Train, 15% Val, 15% Test manteniendo el balance del estrato."""
    # 1. Separar 70% Train y 30% Temporal (Val + Test)
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=seed, stratify=df["stratum"]
    )

    # 2. Dividir el 30% en dos mitades iguales: 15% Val y 15% Test
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=seed, stratify=temp_df["stratum"]
    )

    return {
        "train": sorted(train_df["case_id"].tolist()),
        "val": sorted(val_df["case_id"].tolist()),
        "test": sorted(test_df["case_id"].tolist()),
    }


def main():
    repo_root = Path(__file__).resolve().parents[3]
    data_dir = repo_root / "01_data"
    output_splits_path = data_dir / "splits.json"
    output_eda_csv = data_dir / "eda_metadata.csv"

    # Metadata: desde las tablas del EDA si existen (segundos); si no, leyendo las 100 máscaras
    df = metadata_from_eda_cases(repo_root)
    if df is None:
        df = build_dataset_metadata(data_dir)
    df.to_csv(output_eda_csv, index=False)
    print(f"Metadata EDA exportada a: {output_eda_csv}")

    # Partición balanceada (v2). Requiere las tablas del EDA; si no existen, se usa la v1.
    bal_df = metadata_from_eda(repo_root)
    if bal_df is not None:
        splits, smd = balanced_splits(bal_df, seed=42)
        method = "busqueda_aleatoria_min_max_SMD (20000 particiones, seed 42)"
        print("\nDiferencia estandarizada (SMD) de cada variable por split:")
        print(smd.round(3).to_string())
        print(f"SMD máximo: {smd.values.max():.3f}  (< 0.25 se considera balance aceptable)")
    else:
        splits = generate_stratified_splits(df, seed=42)
        method = "train_test_split estratificado por <=6 / >6 fragmentos (v1)"
        smd = None

    digest = hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()
    splits_data = {
        "_comentario": "Partición por caso (paciente). No modificar manualmente; regenerar con create_splits.py.",
        "version": 2 if smd is not None else 1,
        "metodo": method,
        "variables_balance": BALANCE_COLUMNS if smd is not None else ["stratum"],
        "smd_max": round(float(smd.values.max()), 4) if smd is not None else None,
        "sha256": digest,
        "seed": 42,
        "counts": {
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
            "total": len(df),
        },
        "splits": splits,
    }

    with open(output_splits_path, "w", encoding="utf-8") as f:
        json.dump(splits_data, f, indent=4)

    print(f"Splits guardados en: {output_splits_path}")

    # Reporte rápido del EDA para el informe
    print("\n" + "=" * 50)
    print("RESUMEN EDA OBLIGATORIO (Semana 8)")
    print("=" * 50)
    print(
        f"Total casos analizados: {len(df)} (Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])})"
    )
    print(
        f"Promedio de fragmentos por paciente: {df['total_fragments'].mean():.2f} (Rango: {df['total_fragments'].min()} - {df['total_fragments'].max()})"
    )
    print(
        f"Pacientes con fragmentos conminutos: {df['has_comminuted'].sum()} de {len(df)} ({df['has_comminuted'].mean()*100:.1f}%)"
    )
    print(
        f"Espaciado físico promedio Z (grosor de corte): {df['spacing_z'].mean():.2f} mm"
    )
    print(
        f"Espaciado físico promedio X/Y (píxel planar): {df['spacing_x'].mean():.2f} mm"
    )
    print(f"\nDistribución por macro-hueso (Total de fragmentos en el dataset):")
    print(f"  - Sacro: {df['fragments_sacro'].sum()} fragmentos")
    print(f"  - Coxal Izquierdo: {df['fragments_coxal_izq'].sum()} fragmentos")
    print(f"  - Coxal Derecho: {df['fragments_coxal_der'].sum()} fragmentos")
    print("=" * 50)


if __name__ == "__main__":
    main()