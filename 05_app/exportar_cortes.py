"""exportar_cortes.py.

Saca los cortes axiales de un volumen (.mha, .nii, .nrrd) como archivos sueltos, para probar la
reconstrucción desde cortes del dashboard (cortes.py) o para llevar un caso a otro programa.

    python 05_app/exportar_cortes.py D:/PENGWIN/01_data/PENGWIN_CT_train_images_part2/075.mha salida/075_png
    python 05_app/exportar_cortes.py 075.mha salida/075_dicom --formato dicom --zip
    python 05_app/exportar_cortes.py 075.mha salida/075_sin_orden --sin-metadatos --desordenar

Formatos:
    png     gris de 8 bits con la ventana del modelo (L400/W1800) y, salvo --sin-metadatos, el spacing
            (sCAL y PixelSpacing) y la posición del corte (SliceLocation) en el texto del PNG
    png16   PNG de 16 bits con HU + 1024 (sin pérdida)
    jpg     gris de 8 bits con la misma ventana; JPEG no lleva metadatos
    dicom   un DICOM CT por corte con ImagePositionPatient, ImageOrientationPatient, PixelSpacing y HU

El volumen se lleva a LPS antes de cortar: cada corte queda en la convención radiológica (anterior
arriba, izquierda del paciente a la derecha de la imagen) y el número del nombre crece hacia la cabeza
(con --al-reves, hacia los pies, como numeran muchos tomógrafos).
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image, PngImagePlugin

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "03_src"))

from pengwin.data.data_loader import apply_bone_window, load_and_standardize_mha  # noqa: E402

BS = "\\"


def exportar(entrada: Path, salida: Path, formato: str = "png", sin_metadatos: bool = False, desordenar: bool = False,
             al_reves: bool = False, cada: int = 1, semilla: int = 0) -> list:
    hu, (dz, dy, dx), _ = load_and_standardize_mha(entrada)
    zs = list(range(0, hu.shape[0], cada))
    if al_reves:
        zs = zs[::-1]
    salida.mkdir(parents=True, exist_ok=True)
    rng = random.Random(semilla)
    ext = {"png": ".png", "png16": ".png", "jpg": ".jpg", "dicom": ".dcm"}[formato]
    # nombres sin números si se desordena: así el orden solo puede salir del contenido
    nombres = ([f"corte_{k + 1:04d}{ext}" for k in range(len(zs))] if not desordenar else
               [f"{''.join(rng.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(10))}{ext}" for _ in zs])
    if desordenar:
        rng.shuffle(nombres)
    serie = f"2.25.{uuid.uuid4().int}"                           # UID derivado de un UUID (DICOM PS3.5 B.2)
    escritos = []
    for k, (z, nombre) in enumerate(zip(zs, nombres)):
        ruta = salida / nombre
        pos_mm = z * dz                                           # origen 0: el corte z está a z·dz mm
        if formato in ("png", "jpg"):
            gris = (np.clip(apply_bone_window(hu[z]), 0, 1) * 255).round().astype(np.uint8)
            img = Image.fromarray(gris)
            if formato == "jpg":
                img.save(ruta, quality=95)
            elif sin_metadatos:
                img.save(ruta)
            else:
                info = PngImagePlugin.PngInfo()
                info.add_text("SliceLocation", f"{pos_mm:.4f}")
                info.add_text("PixelSpacing", f"{dy:.6f}{BS}{dx:.6f}")
                info.add_text("WindowCenter", "400")
                info.add_text("WindowWidth", "1800")
                info.add(b"sCAL", b"\x01" + f"{dx:.6f}".encode() + b"\x00" + f"{dy:.6f}".encode())
                img.save(ruta, pnginfo=info)
        elif formato == "png16":
            Image.fromarray((np.clip(hu[z], -1024, 64511) + 1024).astype(np.uint16)).save(ruta)
        else:
            s = sitk.GetImageFromArray(np.clip(hu[z], -1024, 3071).astype(np.int16))
            s.SetSpacing((dx, dy))
            for tag, valor in (("0008|0060", "CT"), ("0008|0008", "ORIGINAL\\PRIMARY\\AXIAL"),
                               ("0020|000d", serie + ".0"), ("0020|000e", serie + ".1"),
                               ("0008|0018", f"{serie}.2.{k + 1}"), ("0020|0013", str(k + 1)),
                               ("0020|0032", BS.join(f"{v:.4f}" for v in (0.0, 0.0, pos_mm))),
                               ("0020|0037", BS.join(["1", "0", "0", "0", "1", "0"])),
                               ("0028|0030", f"{dy:.6f}{BS}{dx:.6f}"), ("0018|0050", f"{dz * cada:.4f}"),
                               ("0020|1041", f"{pos_mm:.4f}"), ("0010|0010", "PENGWIN^ANONIMO")):
                s.SetMetaData(tag, valor)
            w = sitk.ImageFileWriter()
            w.KeepOriginalImageUIDOn()
            w.SetFileName(str(ruta))
            w.Execute(s)
        escritos.append(ruta)
    return escritos


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("entrada", type=Path)
    ap.add_argument("salida", type=Path, help="carpeta donde quedan los cortes")
    ap.add_argument("--formato", choices=("png", "png16", "jpg", "dicom"), default="png")
    ap.add_argument("--sin-metadatos", action="store_true", help="PNG sin spacing ni posición (solo píxeles)")
    ap.add_argument("--desordenar", action="store_true", help="nombres al azar, sin números")
    ap.add_argument("--al-reves", action="store_true", help="numerar de la cabeza a los pies")
    ap.add_argument("--cada", type=int, default=1, help="exportar uno de cada N cortes")
    ap.add_argument("--zip", action="store_true", help="además, empaquetar la carpeta en un .zip")
    args = ap.parse_args()
    if args.salida.exists():
        shutil.rmtree(args.salida)
    t = time.time()
    rutas = exportar(args.entrada, args.salida, args.formato, args.sin_metadatos, args.desordenar, args.al_reves,
                     max(1, args.cada))
    print(f"{len(rutas)} cortes en {args.salida} ({time.time() - t:.1f} s)")
    if args.zip:
        z = args.salida.with_suffix(".zip")
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for r in rutas:
                zf.write(r, f"{args.salida.name}/{r.name}")
        print("zip:", z, f"{z.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
