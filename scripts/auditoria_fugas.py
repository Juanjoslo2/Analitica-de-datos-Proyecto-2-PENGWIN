"""auditoria_fugas.py.

Auditoría de fuga de datos e incertidumbre del test (semana 11).

    1. Pacientes repetidos: cabeceras idénticas (tamaño, spacing, origen) y huella de imagen
       (MIP axial del caché, reducido a 32×32 y estandarizado; correlación de Pearson entre casos).
    2. Intervalo de confianza del test por bootstrap de PACIENTES (5000 réplicas): los fragmentos de
       un mismo paciente no son independientes, así que se remuestrea el paciente completo.

Uso:
    python scripts/auditoria_fugas.py --data-dir D:/PENGWIN/01_data --cache-dir D:/PENGWIN/data_processed
Salida: reports/eval/auditoria.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()
    sp = json.loads((REPO / "01_data" / "splits.json").read_text(encoding="utf-8"))["splits"]
    part = {c: k for k, v in sp.items() for c in v}

    filas = []
    for f in sorted(args.data_dir.glob("PENGWIN_CT_train_images_part*/*.mha")):
        r = sitk.ImageFileReader()
        r.SetFileName(str(f))
        r.ReadImageInformation()
        c = f.stem
        im = np.load(args.cache_dir / c / "image.npy", mmap_mode="r")
        h = np.asarray(im).max(0)[::8, ::8].astype(np.float32).ravel()
        filas.append((c, (r.GetSize(), tuple(round(s, 3) for s in r.GetSpacing()), tuple(round(o, 1) for o in r.GetOrigin())),
                      (h - h.mean()) / (h.std() + 1e-6)))
    cab = {}
    for c, k, _ in filas:
        cab.setdefault(k, []).append(c)
    H = np.stack([h for *_, h in filas])
    C = H @ H.T / H.shape[1]
    np.fill_diagonal(C, -1)
    i, j = np.unravel_index(C.argmax(), C.shape)
    misma, distinta = [], []
    for a in range(len(filas)):
        for b in range(a + 1, len(filas)):
            (misma if part[filas[a][0]] == part[filas[b][0]] else distinta).append(C[a, b])

    f = pd.read_csv(REPO / "reports" / "fragments" / "v2_last_test_tuned_superficie_fragmentos.csv", dtype={"case_id": str})
    rng = np.random.default_rng(0)
    casos = f.case_id.unique()
    grupos = {c: f[f.case_id == c] for c in casos}
    bs = []
    for _ in range(args.n_boot):
        m = pd.concat([grupos[c] for c in rng.choice(casos, len(casos))])
        bs.append((m.dice.mean(), m.iou.mean()))
    bs = np.array(bs)

    res = {
        "casos": len(filas),
        "cabeceras_identicas": [v for v in cab.values() if len(v) > 1],
        "huella_par_mas_parecido": {"casos": [filas[i][0], filas[j][0]], "particiones": [part[filas[i][0]], part[filas[j][0]]],
                                    "correlacion": float(C[i, j])},
        "huella_pares_sobre_0_95": int((np.triu(C, 1) > 0.95).sum()),
        "huella_max_misma_particion": float(max(misma)), "huella_max_entre_particiones": float(max(distinta)),
        "bootstrap": {"n": args.n_boot, "pacientes": int(len(casos)), "fragmentos": int(len(f)),
                      "dice": float(f.dice.mean()), "dice_ic95": [float(np.percentile(bs[:, 0], 2.5)), float(np.percentile(bs[:, 0], 97.5))],
                      "iou": float(f.iou.mean()), "iou_ic95": [float(np.percentile(bs[:, 1], 2.5)), float(np.percentile(bs[:, 1], 97.5))],
                      "p_iou_mayor_igual_0_70": float((bs[:, 1] >= 0.70).mean()),
                      "p_dice_mayor_igual_0_85": float((bs[:, 0] >= 0.85).mean())},
    }
    out = REPO / "reports" / "eval" / "auditoria.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
