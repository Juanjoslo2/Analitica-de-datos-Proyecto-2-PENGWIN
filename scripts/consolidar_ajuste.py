"""consolidar_ajuste.py.

Junta los resultados crudos del ajuste de la semana 10 en tres archivos que sí se versionan.
Los crudos (uno por experimento y por fold, ~250 archivos) se quedan fuera de git (.gitignore)
y se pueden regenerar en ANTON; el notebook ``04_notebook/03_semana10_ajuste.ipynb`` solo lee
lo consolidado.

    reports/tuning/ajuste_posproceso.csv.gz   todos los barridos de posproceso: una fila por
                                              (experimento, combinación, fold). Sale de pp*.csv
    reports/tuning/ajuste_corridas.csv.gz     historial por época de cada entrenamiento de CV
                                              (reports/train/cv_*_history.csv) con su config,
                                              sus cambios respecto a base.yaml y sus λ
    reports/tuning/ajuste_borde.csv           precisión/recall de la cabeza de borde (borde_*.json)

Uso:
    python scripts/consolidar_ajuste.py
"""

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TUN = REPO / "reports" / "tuning"
TRAIN = REPO / "reports" / "train"


def posproceso() -> pd.DataFrame:
    partes = []
    for f in sorted(TUN.glob("pp*.csv")):
        df = pd.read_csv(f)
        df.insert(0, "experimento", f.stem)
        partes.append(df)
    return pd.concat(partes, ignore_index=True)


def cambios(config: str) -> str:
    """Lo que la config cambia respecto a base.yaml (el YAML sin la línea ``base:`` ni comentarios)."""
    for cand in (REPO / "configs" / config, REPO / "configs" / "tuning" / config):
        if cand.is_file():
            lineas = [ln for ln in cand.read_text(encoding="utf-8").splitlines()
                      if ln.strip() and not ln.lstrip().startswith(("#", "base:"))]
            return " | ".join(ln.strip() for ln in lineas)
    return ""


def corridas() -> pd.DataFrame:
    partes = []
    for h in sorted(TRAIN.glob("cv_*_history.csv")):
        nombre = h.name[: -len("_history.csv")]
        df = pd.read_csv(h)
        meta_f = TRAIN / f"{nombre}.json"
        meta = json.loads(meta_f.read_text(encoding="utf-8")) if meta_f.exists() else {}
        lam = meta.get("lambdas", {})
        cfg = Path(str(meta.get("config", ""))).name
        df.insert(0, "corrida", nombre)
        df.insert(1, "config", cfg)
        df.insert(2, "cambios", cambios(cfg) if cfg else "")
        for k in ("cls", "det", "seg"):
            df.insert(3, f"lambda_{k}", lam.get(k))
        partes.append(df)
    return pd.concat(partes, ignore_index=True)


def borde() -> pd.DataFrame:
    filas = []
    for f in sorted(TUN.glob("borde_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for ref, r in d["referencias"].items():
            for thr, pr in r["por_umbral"].items():
                filas.append({"corrida": d["corrida"], "fold": d.get("fold"), "referencia": ref, "umbral": float(thr),
                              "precision": pr["precision"], "recall": pr["recall"],
                              "P_media_en_borde_real": r["P_media_en_borde_real"]})
    return pd.DataFrame(filas)


def main() -> None:
    pp = posproceso()
    pp.to_csv(TUN / "ajuste_posproceso.csv.gz", index=False)
    co = corridas()
    co.to_csv(TUN / "ajuste_corridas.csv.gz", index=False)
    bo = borde()
    bo.to_csv(TUN / "ajuste_borde.csv", index=False)
    print(f"posproceso: {pp.experimento.nunique()} experimentos, {len(pp)} filas")
    print(f"corridas:   {co.corrida.nunique()} entrenamientos, {len(co)} épocas")
    print(f"borde:      {bo.corrida.nunique()} corridas")


if __name__ == "__main__":
    main()
