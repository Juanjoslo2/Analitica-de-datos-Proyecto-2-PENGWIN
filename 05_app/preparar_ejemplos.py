"""preparar_ejemplos.py.

Procesa casos de TEST como ejemplos fijos del dashboard (con su máscara de referencia, para
mostrar la comparación). No cambia ninguna métrica: es la misma predicción del modelo final.

    python 05_app/preparar_ejemplos.py --data-dir D:/PENGWIN/01_data --cases 075 097 069
"""

import argparse
import json
import os
from pathlib import Path

import servicio

AQUI = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--titulos", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("PENGWIN_EJEMPLOS", AQUI / "ejemplos")))
    ap.add_argument("--ckpt", type=Path, default=servicio.REPO / "checkpoints" / "v2_last" / "best.pth")
    ap.add_argument("--config", type=Path, default=servicio.REPO / "configs" / "tuned.yaml")
    args = ap.parse_args()
    modelo = servicio.Modelo(args.ckpt, args.config)
    for k, cid in enumerate(args.cases):
        img = next(args.data_dir.glob(f"PENGWIN_CT_train_images*/{cid}.mha"))
        gt = args.data_dir / "PENGWIN_CT_train_labels" / f"{cid}.mha"
        carpeta = args.out / cid
        r = servicio.procesar_volumen(modelo, img, carpeta, lambda m, f: print(f"  {f:4.0%} {m}", flush=True),
                                      ruta_gt=gt if gt.exists() else None)
        r["titulo"] = args.titulos[k] if args.titulos else f"Caso {cid} (test)"
        (carpeta / "resumen.json").write_text(json.dumps(r, default=float), encoding="utf-8")
        print(cid, "listo:", r["segundos_total"], "s;", len(r["fragmentos"]), "fragmentos;", r.get("metricas_gt"))


if __name__ == "__main__":
    main()
