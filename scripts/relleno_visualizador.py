"""Cómo se ve el cierre de huecos en el visualizador 3 (03_semana11 §1.2).

Corre el pipeline final sobre un caso, separa fragmentos, aplica ``fill_thin_holes`` con cada radio
y arma la malla con la misma función del dashboard (``05_app/servicio._malla``: gaussiano σ 0,5 +
marching cubes). Dibuja la malla del fragmento principal de un hueso en vista anterior, tal como
llega al navegador, y mide el área de los huecos que se ven y el tiempo del cierre.

    python scripts/relleno_visualizador.py --cache-dir D:/PENGWIN/data_processed --caso 075 --radios 0,2,3
Salida: reports/figures/semana11/relleno_<caso>.png y reports/eval/relleno_visualizador_<caso>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "03_src"), str(REPO / "05_app"), str(REPO / "scripts")]

from huecos_remuestreo import area_huecos_mm2, tuneles, vista_anterior  # noqa: E402
from pengwin.postprocess.instances import fill_thin_holes  # noqa: E402

HUESOS = {1: "sacro", 11: "coxal izquierdo", 21: "coxal derecho"}


def huecos_de_la_malla(v: np.ndarray, f: np.ndarray, celda_mm: float = 0.5):
    """Rasteriza la malla en vista anterior (plano x-z) y mide los huecos cerrados que se ven."""
    x, z = v[:, 0], v[:, 2]
    x0, z0 = x.min() - 2, z.min() - 2
    W, H = int((x.max() - x0 + 2) / celda_mm) + 1, int((z.max() - z0 + 2) / celda_mm) + 1
    from matplotlib.path import Path as MPath
    img = np.zeros((H, W), bool)
    for tri in f:
        px = (x[tri] - x0) / celda_mm
        pz = (z[tri] - z0) / celda_mm
        i0, i1 = int(np.floor(pz.min())), int(np.ceil(pz.max())) + 1
        j0, j1 = int(np.floor(px.min())), int(np.ceil(px.max())) + 1
        if (i1 - i0) * (j1 - j0) <= 4:
            img[i0:i1, j0:j1] |= True
            continue
        jj, ii = np.meshgrid(np.arange(j0, j1) + 0.5, np.arange(i0, i1) + 0.5)
        dentro = MPath(np.c_[px, pz]).contains_points(np.c_[jj.ravel(), ii.ravel()], radius=0.5)
        img[i0:i1, j0:j1] |= dentro.reshape(jj.shape)
    return area_huecos_mm2(img, celda_mm, celda_mm)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--caso", default="075")
    ap.add_argument("--hueso", type=int, default=11, help="fragmento principal a dibujar: 1, 11 o 21")
    ap.add_argument("--radios", default="0,2,3")
    ap.add_argument("--regiones", default="todas", choices=["todas", "coxales"])
    ap.add_argument("--ckpt", type=Path, default=REPO / "checkpoints" / "v2_last" / "best.pth")
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "tuned.yaml")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    import servicio
    from pengwin.inference.volume import predict_case, to_native
    from pengwin.measurement.separation import separation_table
    from pengwin.postprocess.instances import instance_kwargs, separate_instances

    modelo = servicio.Modelo(args.ckpt, args.config)
    pred = predict_case(modelo.net, args.cache_dir, args.caso, modelo.cfg, modelo.device, tta=modelo.tta)
    meta = pred["meta"]
    sp = (meta["spacing_zyx"][0], meta["pixel_mm"], meta["pixel_mm"])
    kw = instance_kwargs(modelo.pp)
    kw.update(fill_holes_mm=0.0)
    lab0 = separate_instances(pred["semantic"], pred["edge"].astype(np.float32), sp, **kw)

    # Centro del hueco más grande de la máscara sin cierre (vista anterior), en mm de la grilla
    from scipy import ndimage as ndi
    _, hit = vista_anterior(lab0 == args.hueso)
    h = ndi.binary_fill_holes(hit) & ~hit
    hl, hn = ndi.label(h)
    zona = None
    if hn:
        kk = 1 + int(np.argmax(ndi.sum(h, hl, range(1, hn + 1))))
        zz, xx = np.nonzero(hl == kk)
        zona = (xx.mean() * sp[2], zz.mean() * sp[0])

    radios = [float(r) for r in args.radios.split(",")]
    res = {"caso": args.caso, "hueso": HUESOS[args.hueso], "regiones": args.regiones, "variantes": []}
    filas_fig = 2 if zona else 1
    fig = plt.figure(figsize=(4.6 * len(radios), 4.6 * filas_fig + 0.6), facecolor="white")
    for k, r in enumerate(radios):
        t0 = time.time()
        lab = fill_thin_holes(lab0, sp, r, (2, 3) if args.regiones == "coxales" else (1, 2, 3)) if r > 0 else lab0
        t_cierre = time.time() - t0
        filas = separation_table(to_native(lab, meta, order=0), meta["spacing_zyx"])
        fragmentos = servicio._tabla_fragmentos(lab, sp, filas)
        t0 = time.time()
        malla = servicio._malla(lab, sp, fragmentos)
        t_malla = time.time() - t0
        tr = next(t for t in malla["trazas"] if t["id"] == args.hueso)
        v = np.c_[tr["x"], tr["y"], tr["z"]]
        f = np.c_[tr["i"], tr["j"], tr["k"]]
        huecos = huecos_de_la_malla(v, f)
        peso = len(json.dumps(malla, separators=(",", ":"))) / 1e6
        res["variantes"].append({"radio_mm": r, "segundos_cierre": round(t_cierre, 2), "segundos_malla": round(t_malla, 2),
                                 "malla_mb": round(peso, 2), "vertices": len(v),
                                 "huecos_vistos_mm2": [round(a, 1) for a in huecos[:5]],
                                 "tuneles_principal": tuneles(lab == args.hueso),
                                 "vóxeles_añadidos": int((lab != lab0).sum())})
        tri = v[f]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
        luz = np.clip(0.35 + 0.65 * np.abs(n @ np.array([0.3, -0.85, 0.42])), 0, 1)
        col = np.array(matplotlib.colors.to_rgb(tr["color"]))[None] * luz[:, None]
        lo, hi = v.min(0), v.max(0)
        c, m = (lo + hi) / 2, (hi - lo).max() / 2
        vistas = [(c, m)] + ([(np.array([zona[0], c[1], zona[1]]), 18.0)] if zona else [])
        for fila, (cc, mm) in enumerate(vistas):
            ax = fig.add_subplot(filas_fig, len(radios), fila * len(radios) + k + 1, projection="3d")
            ax.add_collection3d(Poly3DCollection(tri, facecolors=col, edgecolor="none"))
            ax.set_xlim(cc[0] - mm, cc[0] + mm); ax.set_ylim(c[1] - m, c[1] + m); ax.set_zlim(cc[2] - mm, cc[2] + mm)
            ax.view_init(elev=0, azim=-90)
            ax.set_axis_off()
            if fila == 0:
                txt = ", ".join(f"{a:.0f}" for a in huecos[:3]) + " mm²" if huecos else "ninguno"
                ax.set_title(("sin cierre" if r == 0 else f"cierre r = {r:g} mm") + f"\nhuecos que se ven: {txt}", fontsize=10)
            else:
                ax.set_title("zoom ±18 mm sobre el hueco", fontsize=9)
    fig.suptitle(f"Caso {args.caso} · {HUESOS[args.hueso]}, fragmento principal · malla del visualizador 3, vista anterior",
                 fontsize=11)
    fig.tight_layout()
    out = REPO / "reports" / "figures" / "semana11" / f"relleno_{args.caso}.png"
    fig.savefig(out, dpi=110)
    p = REPO / "reports" / "eval" / f"relleno_visualizador_{args.caso}.json"
    p.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
