"""Huecos que abre la grilla de 256 px en hueso delgado (semana 11).

En la reconstrucción 3D del caso 075 el ala ilíaca izquierda sale con un hueco que no es una fractura.
Este script mide de dónde viene:

1. Para los 100 casos, cuenta los túneles (agujeros pasantes) del fragmento principal de cada hueso en la
   etiqueta de los expertos a resolución nativa y en la misma etiqueta remuestreada a la grilla del modelo
   (el caché de 256 px, vecino más cercano). Túneles = objetos + cavidades − característica de Euler
   (``skimage.measure.euler_number``, conectividad 26).
2. Para un caso (por defecto el 075) corre el pipeline final y dibuja el principal del coxal izq. en
   vista anterior: referencia nativa, referencia en 256, predicción y predicción como la dibujaba el
   dashboard antes (gaussiano σ 0,8 + ``step_size`` 2).

No toma ninguna decisión de modelo: es un diagnóstico de la etiqueta y de la visualización.

    python scripts/huecos_remuestreo.py --cache-dir D:/PENGWIN/data_processed --data-dir D:/PENGWIN/01_data
Salida: reports/eval/huecos_remuestreo.json y reports/figures/semana11/hueco_075.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import euler_number

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "03_src"))

from pengwin.data.data_loader import load_and_standardize_mha  # noqa: E402
from pengwin.data.slice_cache import load_case_cache  # noqa: E402

PRINCIPALES = {1: "SA", 11: "LI", 21: "RI"}


def tuneles(m: np.ndarray) -> int:
    sl = ndi.find_objects(m.astype(np.uint8))[0]
    m = m[tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in sl)]
    objetos = ndi.label(m, structure=np.ones((3, 3, 3)))[1]
    cavidades = ndi.label(~m)[1] - 1
    return int(objetos + cavidades - euler_number(m, connectivity=3))


def vista_anterior(m: np.ndarray):
    """Primer vóxel ocupado a lo largo de y (de adelante hacia atrás) → imagen sombreada (z, x)."""
    hit = m.any(1)
    prof = np.where(hit, m.argmax(1), 0).astype(np.float32)
    gz, gx = np.gradient(ndi.gaussian_filter(prof, 0.7))
    luz = np.clip(0.75 - 0.35 * gx - 0.35 * gz, 0.15, 1.0)
    return np.where(hit, luz * (1 - 0.5 * prof / max(prof[hit].max(), 1)), np.nan), hit


def area_huecos_mm2(hit: np.ndarray, mm_z: float, mm_x: float) -> list:
    h = ndi.binary_fill_holes(hit) & ~hit
    lab, n = ndi.label(h)
    return sorted((float(a) * mm_z * mm_x for a in ndi.sum(h, lab, range(1, n + 1))), reverse=True)


def figura(caso: str, args, splits: dict) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from pengwin.inference.volume import TTA_SETS, predict_case
    from pengwin.models.pengwin_net import PengwinNet
    from pengwin.postprocess.instances import instance_kwargs, resolve_postprocess, separate_instances
    from pengwin.utils.config import load_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    net = PengwinNet(cfg["model"]).to(device).eval()
    net.load_state_dict(ck["model"])
    tuned = load_config(args.config)
    pp = resolve_postprocess(cfg.get("postprocess"), tuned.get("postprocess"))
    tta = (tuned.get("inference") or {}).get("tta")
    pred = predict_case(net, args.cache_dir, caso, cfg, device, batch_size=8, tta=TTA_SETS[str(tta)] if tta else None)
    meta = pred["meta"]
    sp = (meta["spacing_zyx"][0], meta["pixel_mm"], meta["pixel_mm"])
    lab = separate_instances(pred["semantic"], pred["edge"].astype(np.float32), sp, **instance_kwargs(pp))
    _, gt256, _ = load_case_cache(args.cache_dir / caso)
    nat, sp_n, _ = load_and_standardize_mha(args.data_dir / "PENGWIN_CT_train_labels" / f"{caso}.mha", is_label=True)

    m_nat, m_gt, m_pred = nat == 11, gt256 == 11, lab == 11
    antes = (ndi.gaussian_filter(m_pred.astype(np.float32), 0.8) > 0.5)[::2, ::2, ::2]
    ahora = (ndi.gaussian_filter(m_pred.astype(np.float32), 0.5) > 0.5)[::2, ::2, ::2]
    paneles = [("Referencia nativa (0,85 mm)", m_nat, (sp_n[0], sp_n[2])),
               ("Referencia en la grilla del modelo (1,36 mm)", m_gt, (sp[0], sp[2])),
               ("Predicción (máscara)", m_pred, (sp[0], sp[2])),
               ("Predicción como la dibujaba el dashboard\n(σ 0,8 + step 2)", antes, (2 * sp[0], 2 * sp[2])),
               ("Predicción con el dibujo corregido\n(σ 0,5 + step 2)", ahora, (2 * sp[0], 2 * sp[2]))]
    res = {"caso": caso, "particion": splits.get(caso), "huecos_vista_anterior_mm2": {},
           "tuneles": {"referencia_nativa": tuneles(m_nat), "referencia_256": tuneles(m_gt), "prediccion": tuneles(m_pred)}}
    fig, axes = plt.subplots(1, len(paneles), figsize=(19, 5.6), facecolor="white")
    for ax, (titulo, m, (mz, mx)) in zip(axes, paneles):
        sl = ndi.find_objects(m.astype(np.uint8))[0]
        img, hit = vista_anterior(m[sl])
        areas = area_huecos_mm2(hit, mz, mx)
        res["huecos_vista_anterior_mm2"][titulo.split("\n")[0]] = [round(a, 1) for a in areas[:3]]
        ax.imshow(img[::-1], cmap="gray", vmin=0, vmax=1, aspect=mz / mx, interpolation="nearest")
        txt = ", ".join(f"{a:.0f}" for a in areas[:2]) + " mm²" if areas else "ninguno"
        ax.set_title(f"{titulo}\nhuecos: {txt}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Caso {caso} · coxal izquierdo, fragmento principal · vista anterior", fontsize=11, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = REPO / "reports" / "figures" / "semana11" / f"hueco_{caso}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)

    # Grosor del ala donde la grilla de 256 abre el hueco, medido en la referencia nativa
    img_g, hit_g = vista_anterior(m_gt)
    h = ndi.binary_fill_holes(hit_g) & ~hit_g
    hl, hn = ndi.label(h)
    if hn:
        k = 1 + int(np.argmax(ndi.sum(h, hl, range(1, hn + 1))))
        zz, xx = np.nonzero(hl == k)
        crop = meta["crop"]
        z0 = int(round(zz.mean()))
        x0 = int(round(crop["x0"] + (xx.mean() + 0.5) * crop["side_px"] / meta["image_size"]))
        r = int(round(4 / sp_n[2]))
        g = (nat[z0 - 2:z0 + 3, :, x0 - r:x0 + r + 1] == 11).sum(1)
        res["grosor_ala_en_el_hueco_mm"] = {"mediana": round(float(np.median(g[g > 0])) * sp_n[1], 2),
                                            "minimo": round(float(g[g > 0].min()) * sp_n[1], 2)}
        hu, _, _ = load_and_standardize_mha(args.data_dir / ("PENGWIN_CT_train_images_part1" if int(caso) <= 50
                                                            else "PENGWIN_CT_train_images_part2") / f"{caso}.mha")
        en = nat[z0 - 2:z0 + 3, :, x0 - r:x0 + r + 1] == 11
        v = hu[z0 - 2:z0 + 3, :, x0 - r:x0 + r + 1][en]
        res["hu_del_ala_en_el_hueco"] = {"p10": round(float(np.percentile(v, 10))), "mediana": round(float(np.median(v))),
                                         "max": round(float(v.max()))}
    res["pixel_mm_modelo"] = round(meta["pixel_mm"], 3)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--caso", default="075")
    ap.add_argument("--ckpt", type=Path, default=REPO / "checkpoints" / "v2_last" / "best.pth")
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "tuned.yaml")
    ap.add_argument("--solo-figura", action="store_true", help="rehace solo el caso de ejemplo; conserva el conteo de los 100")
    args = ap.parse_args()
    splits = {c: p for p, cs in json.loads((REPO / "01_data" / "splits.json").read_text(encoding="utf-8"))["splits"].items()
              for c in cs}

    p = REPO / "reports" / "eval" / "huecos_remuestreo.json"
    if args.solo_figura and p.exists():
        out = json.loads(p.read_text(encoding="utf-8"))
        out["caso_ejemplo"] = figura(args.caso, args, splits)
        p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(out["caso_ejemplo"], indent=1, ensure_ascii=False))
        return

    filas = []
    for c in sorted(splits):
        nat, _, _ = load_and_standardize_mha(args.data_dir / "PENGWIN_CT_train_labels" / f"{c}.mha", is_label=True)
        _, g, _ = load_case_cache(args.cache_dir / c)
        for fid, reg in PRINCIPALES.items():
            if (nat == fid).any() and (g == fid).any():
                filas.append({"caso": c, "particion": splits[c], "hueso": reg,
                              "tuneles_nativo": tuneles(nat == fid), "tuneles_256": tuneles(g == fid)})
        print(c, " ".join(f"{f['hueso']} {f['tuneles_nativo']}->{f['tuneles_256']}" for f in filas if f["caso"] == c), flush=True)

    resumen = {}
    for hueso in ("SA", "LI", "RI", "todos"):
        fs = [f for f in filas if hueso == "todos" or f["hueso"] == hueso]
        mas = [f for f in fs if f["tuneles_256"] > f["tuneles_nativo"]]
        resumen[hueso] = {"principales": len(fs), "con_tuneles_de_mas_en_256": len(mas),
                          "tuneles_nativo": sum(f["tuneles_nativo"] for f in fs),
                          "tuneles_256": sum(f["tuneles_256"] for f in fs),
                          "tuneles_ganados": sum(max(f["tuneles_256"] - f["tuneles_nativo"], 0) for f in fs),
                          "tuneles_perdidos": sum(max(f["tuneles_nativo"] - f["tuneles_256"], 0) for f in fs)}
    out = {"que_mide": "túneles del fragmento principal: etiqueta nativa contra la misma etiqueta en el caché de 256 px",
           "resumen": resumen, "caso_ejemplo": figura(args.caso, args, splits), "por_fragmento": filas}
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"resumen": resumen, "caso_ejemplo": out["caso_ejemplo"]}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
