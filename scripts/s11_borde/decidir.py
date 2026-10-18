"""Semana 11 · aplica las reglas de ``reports/tuning/s11_borde/PREREGISTRO.md`` sin intervención humana.

Corre en el host de ANTON (sin pandas ni numpy: solo biblioteca estándar).

    python3 scripts/s11_borde/decidir.py etapa1   ->  reports/tuning/s11_borde/etapa1.json   (cribado, fold 0, 12 épocas)
    python3 scripts/s11_borde/decidir.py etapa2   ->  reports/tuning/s11_borde/etapa2.json   (5 folds, 20 épocas, TTA + edt_hmax)

Imprime en la última línea los candidatos que pasan, separados por espacios (o nada).
"""
import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TUN = REPO / "reports" / "tuning"
S = TUN / "s11_borde"
REF = S / "referencia"
CANDIDATOS = ["c11_wd1e2", "c12_segw32", "c13_unetpp"]
T_CRIT_4GL = 2.776                       # t de Student, 4 g.l., α = 0,05 a dos colas


def mejor(path: Path):
    """Primera fila del ranking (mejor Dice por fragmento) de un pp_*.json de tune_postprocess."""
    r = json.loads(path.read_text(encoding="utf-8"))["ranking"][0]
    return {"dice": r["dice_fragmento_media"], "iou": r["iou_fragmento_media"],
            "principal": r["dice_principal_media"], "secundario": r["dice_secundario_media"],
            "rec_sec": r["recuperados_secundarios_%_media"],
            "seed_depth_mm": r.get("seed_depth_mm"), "edge_threshold": r.get("edge_threshold")}


def ultima_epoca(path: Path):
    with open(path, encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    out = {}
    for k, v in (filas[-1].items() if filas else []):
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            pass
    return out


def recall_borde(nombre: str):
    p = TUN / f"borde_{nombre}.json"
    if not p.exists():
        return None
    v = json.loads(p.read_text(encoding="utf-8"))["referencias"]["edge.npy"]["por_umbral"]["0.2"]
    return {"precision@0.2": v["precision"], "recall@0.2": v["recall"]}


def etapa1():
    refs = {}
    for n in ("cv_c00_base_e12_f0", "cv_c00b_seed1337_e12_f0", "cv_c00c_seed2024_e12_f0", "cv_c00c_seed7_e12_f0"):
        refs[n] = mejor(REF / f"pp_scr_{n}.json")
    nuevo = TUN / "pp_scr_s11_ref_e12_f0.json"
    if nuevo.exists():
        refs["s11_ref_e12_f0 (mismo código, repetición)"] = mejor(nuevo)
    dices = [r["dice"] for r in refs.values()]
    mu, sigma = st.mean(dices), st.stdev(dices)
    iou_ref = st.mean(r["iou"] for r in refs.values())
    umbral = mu + 2 * sigma
    filas = []
    for c in CANDIDATOS:
        n = f"s11_{c}_e12_f0"
        p = TUN / f"pp_scr_{n}.json"
        if not p.exists():
            filas.append({"candidato": c, "estado": "sin resultado (falló o no terminó)", "pasa": False})
            continue
        m = mejor(p)
        pasa = m["dice"] >= umbral and m["iou"] >= iou_ref
        filas.append({"candidato": c, **m, "delta_vs_media_ref": round(m["dice"] - mu, 4),
                      "z": round((m["dice"] - mu) / sigma, 2), "borde": recall_borde(n), "pasa": pasa})
    pasan = sorted([f for f in filas if f["pasa"]], key=lambda f: -f["dice"])[:2]
    res = {"regla": "pasa si Dice_frag ≥ media_ref + 2σ_ref y IoU_frag ≥ media IoU_ref; se confirman los 2 mejores",
           "referencias": refs, "media_ref": round(mu, 4), "sigma_ref": round(sigma, 4), "umbral": round(umbral, 4),
           "iou_ref": round(iou_ref, 4), "borde_ref": recall_borde("s11_ref_e12_f0"),
           "candidatos": filas, "a_confirmar": [f["candidato"] for f in pasan]}
    (S / "etapa1.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    return res["a_confirmar"]


def por_fold(path: Path):
    """{fold: fila} con el posproceso final fijo (edt_hmax, 5 mm, umbral 0,1, h 1,5)."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r["method"] == "edt_hmax" and float(r["seed_depth_mm"]) == 5.0 and float(r["edge_threshold"]) == 0.1
                    and float(r.get("hmax_h_mm") or 0) == 1.5):
                out[int(r["fold"])] = {k: float(r[k]) for k in ("dice_fragmento", "iou_fragmento", "dice_principal",
                                                                 "recuperados_secundarios_%")}
    return out


def etapa2():
    ref = por_fold(REF / "pp_tta_base_hmax.csv")
    map_ref = st.mean(ultima_epoca(REF / f"cv_c00_base_e20_f{k}_history.csv").get("mAP@[.50:.95]", float("nan"))
                      for k in range(5))
    a_conf = json.loads((S / "etapa1.json").read_text(encoding="utf-8"))["a_confirmar"]
    filas = []
    for c in a_conf:
        p = TUN / f"pp_s11_conf_{c}.csv"
        if not p.exists():
            filas.append({"candidato": c, "estado": "sin resultado (falló o no terminó)", "pasa": False})
            continue
        cand = por_fold(p)
        ks = sorted(set(ref) & set(cand))
        d = [cand[k]["dice_fragmento"] - ref[k]["dice_fragmento"] for k in ks]
        di = [cand[k]["iou_fragmento"] - ref[k]["iou_fragmento"] for k in ks]
        dp = [cand[k]["dice_principal"] - ref[k]["dice_principal"] for k in ks]
        t = st.mean(d) / (st.stdev(d) / math.sqrt(len(d))) if len(d) > 1 and st.stdev(d) > 0 else float("nan")
        hist = [REPO / "reports" / "train" / f"s11_conf_{c}_f{k}_history.csv" for k in ks]
        vals = [ultima_epoca(h).get("mAP@[.50:.95]", float("nan")) for h in hist if h.exists()]
        map_c = st.mean(vals) if vals else float("nan")
        crit = {"delta_dice>=0.026": st.mean(d) >= 0.026, "t>2.776": t > T_CRIT_4GL,
                "folds_a_favor>=4": sum(x > 0 for x in d) >= 4, "delta_iou>0": st.mean(di) > 0,
                "delta_principal>=-0.010": st.mean(dp) >= -0.010,
                "mAP_no_baja_mas_de_0.005": not (map_c < map_ref - 0.005)}
        filas.append({"candidato": c, "folds": ks, "dice": round(st.mean(cand[k]["dice_fragmento"] for k in ks), 4),
                      "dice_ref": round(st.mean(ref[k]["dice_fragmento"] for k in ks), 4),
                      "delta_dice": round(st.mean(d), 4), "t": round(t, 2), "folds_a_favor": f"{sum(x > 0 for x in d)}/{len(d)}",
                      "delta_iou": round(st.mean(di), 4), "delta_principal": round(st.mean(dp), 4),
                      "mAP@[.50:.95]": round(map_c, 4), "mAP_ref": round(map_ref, 4),
                      "delta_por_fold": [round(x, 4) for x in d], "criterios": crit, "pasa": all(crit.values())})
    ganadores = sorted([f for f in filas if f["pasa"]], key=lambda f: -f["delta_dice"])
    res = {"regla": "la de la semana 11 (S11_PREREGISTRO.md) + guardarraíl de mAP@[.50:.95]; referencia pp_tta_base_hmax",
           "candidatos": filas, "ganador": ganadores[0]["candidato"] if ganadores else None}
    (S / "etapa2.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    return [res["ganador"]] if res["ganador"] else []


if __name__ == "__main__":
    etapa = sys.argv[1]
    sale = etapa1() if etapa == "etapa1" else etapa2()
    print(" ".join(sale))
