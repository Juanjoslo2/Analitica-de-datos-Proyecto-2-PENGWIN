"""confirmacion.py.

Análisis de la confirmación y4xul (5 folds, 20 épocas), emparejado por fold contra el control.

Para cada (config, método) se elige UNA combinación de posproceso: la de mayor Dice por fragmento
promediado en los 5 folds (desempate: IoU). Elegir la mejor de cada fold por separado sería
optimista. Después se compara fold a fold contra ``y0_sintl`` con ``edt``.

Entrada: reports/tuning/y4xul/conf_<config>_<método>.csv (``y4xul.sh confirmar``) y
reports/train/<config>_f<k>_e20_history.csv.
Salida: tabla por pantalla y reports/tuning/y4xul/CONFIRMACION.csv / CONFIRMACION_por_fold.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
TUN = REPO / "reports" / "tuning" / "y4xul"
PARAMS = {"edt": ["seed_depth_mm", "edge_threshold"], "role": ["role_seed_depth_mm", "role_threshold", "role_smooth_mm"]}
FRAG = ["dice_fragmento", "iou_fragmento", "dice_principal", "dice_secundario", "recuperados_secundarios_%",
        "mae_distancia_mm", "mediana_error_mm"]
DET = {"mAP@[.50:.95]": "mAP", "mAP@0.50": "mAP50", "IoU_promedio": "IoUcaja", "dice_hueso": "DiceReg",
       "cls_f1_macro": "F1", "cls_auc_macro": "AUC"}
T_CRIT_4GL = 2.776          # t de Student bilateral, α = 0,05, 4 grados de libertad
EPOCAS = 20


def main() -> None:
    filas, por_fold = [], {}
    for csv in sorted(TUN.glob("conf_*.csv")):
        cfg, metodo = csv.stem[len("conf_"):].rsplit("_", 1)
        d = pd.read_csv(csv)
        orden = d.groupby("combo")[["dice_fragmento", "iou_fragmento"]].mean().sort_values(
            ["dice_fragmento", "iou_fragmento"], ascending=False)
        b = d[d.combo == orden.index[0]].sort_values("fold").set_index("fold")
        clave = f"{cfg} [{metodo}]"
        por_fold[clave] = b[FRAG]
        fila = {"modelo": clave, "combo": "/".join(f"{b[p].iloc[0]:g}" for p in PARAMS[metodo]), "folds": len(b)}
        fila.update({m: b[m].mean() for m in FRAG})
        fila["dice_desv"] = b["dice_fragmento"].std()
        det = []
        base = cfg[:-2] if cfg.endswith("p1") else cfg       # "…p1" = el mismo modelo, solo la pasada 1
        for k in b.index:
            hs = sorted((REPO / "reports" / "train").glob(f"{base}_f{k}_e*_history.csv"))
            exacto = [h for h in hs if h.name == f"{base}_f{k}_e{EPOCAS}_history.csv"]
            if exacto or hs:
                det.append(pd.read_csv((exacto or hs)[-1]).iloc[-1][list(DET)])
        if det:
            fila.update({DET[c]: v for c, v in pd.DataFrame(det).astype(float).mean().items()})
        filas.append(fila)
    if not filas:
        print("no hay reports/tuning/y4xul/conf_*.csv: corre `bash scripts/lab/y4xul.sh confirmar`")
        return
    df = pd.DataFrame(filas)
    ref = next((k for k in por_fold if k.startswith("y0_") and k.endswith("[edt]")), None)
    for i, r in df.iterrows():
        if ref is None or r.modelo == ref:
            continue
        for m, tag in (("dice_fragmento", "dDice"), ("iou_fragmento", "dIoU"), ("dice_principal", "dPrinc")):
            delta = (por_fold[r.modelo][m] - por_fold[ref][m]).dropna()
            df.loc[i, tag] = delta.mean()
            if m == "dice_fragmento":
                se = delta.std() / np.sqrt(len(delta))
                df.loc[i, "t"] = delta.mean() / se if se > 0 else np.nan
                df.loc[i, "folds+"] = f"{int((delta > 0).sum())}/{len(delta)}"
    df.to_csv(TUN / "CONFIRMACION.csv", index=False)
    pd.concat(por_fold, names=["modelo"]).to_csv(TUN / "CONFIRMACION_por_fold.csv")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    corto = {"dice_fragmento": "Dice", "iou_fragmento": "IoU", "dice_principal": "princ", "dice_secundario": "sec",
             "recuperados_secundarios_%": "rec%", "mae_distancia_mm": "MAEmm", "mediana_error_mm": "medmm"}
    print(df.rename(columns=corto)[["modelo", "combo", "Dice", "dice_desv", "IoU", "princ", "sec", "rec%", "MAEmm",
                                    "medmm"]].round(4).to_string(index=False))
    extra = [c for c in ["dDice", "t", "folds+", "dIoU", "dPrinc", "mAP", "mAP50", "IoUcaja", "DiceReg", "F1", "AUC"] if c in df]
    print(f"\nemparejado contra {ref} (t crítico {T_CRIT_4GL}):")
    print(df[["modelo"] + extra].round(4).to_string(index=False))
    print("\nDice por fragmento, fold a fold:")
    print(pd.DataFrame({k: v["dice_fragmento"] for k, v in por_fold.items()}).T.round(4).to_string())


if __name__ == "__main__":
    main()
