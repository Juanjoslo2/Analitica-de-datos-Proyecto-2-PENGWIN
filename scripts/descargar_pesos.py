"""descargar_pesos.py.

Baja los pesos entregados (GitHub Release ``v1.0-entrega``) y deja cada uno en la ruta que
esperan los scripts, los notebooks y el dashboard. Comprueba el sha256 de cada archivo contra
``SHA256SUMS.txt`` del mismo release; si un archivo ya está y su suma coincide, no lo vuelve a bajar.
Solo usa la librería estándar (no hace falta ``gh`` ni iniciar sesión: el repositorio es público).

Uso (desde la raíz del repositorio):
    python scripts/descargar_pesos.py                      # los 7 archivos (~118 MB)
    python scripts/descargar_pesos.py --modelos v2_last    # solo el modelo final (35 MB)
    python scripts/descargar_pesos.py --verificar          # solo revisa las sumas de lo que ya está
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
URL = "https://github.com/Juanjoslo2/Analitica-de-datos-Proyecto-2-PENGWIN/releases/download/v1.0-entrega"

# nombre corto -> (archivo del release, ruta dentro del repositorio)
PESOS = {
    "v2_last": ("v2_last_best.pth", "checkpoints/v2_last/best.pth"),        # modelo final (dashboard)
    "v2": ("v2_best.pth", "checkpoints/v2/best.pth"),                       # v2, mejor época en val (22)
    "base": ("base_best.pth", "checkpoints/base/best.pth"),                 # semana 9, CBAM + TL
    "sin_cbam": ("sin_cbam_best.pth", "checkpoints/sin_cbam/best.pth"),     # ablación
    "sin_tl": ("sin_tl_best.pth", "checkpoints/sin_tl/best.pth"),           # ablación
    "semana9": ("semana9_prueba_last.pth", "checkpoints/semana9/last.pth"),  # corrida corta de 4 épocas
    "fundidora": ("fundidora_taller3.pth", "checkpoints/fundidora_taller3.pth"),  # entrada del transfer learning
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def sumas_del_release() -> dict:
    with urllib.request.urlopen(f"{URL}/SHA256SUMS.txt", timeout=60) as r:
        texto = r.read().decode("utf-8")
    # formato de sha256sum: "<suma> *<archivo>"
    return {linea.split()[1].lstrip("*"): linea.split()[0] for linea in texto.splitlines() if linea.strip()}


def bajar(archivo: str, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(destino.suffix + ".parcial")
    with urllib.request.urlopen(f"{URL}/{archivo}", timeout=60) as r, open(tmp, "wb") as f:
        total, hecho = int(r.headers.get("Content-Length", 0)), 0
        while bloque := r.read(1 << 20):
            f.write(bloque)
            hecho += len(bloque)
            if total:
                print(f"\r  {archivo}: {hecho / 1e6:5.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(destino)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--modelos", nargs="+", choices=list(PESOS), default=list(PESOS))
    ap.add_argument("--verificar", action="store_true", help="no baja nada: solo compara las sumas")
    args = ap.parse_args()

    sumas = sumas_del_release()
    fallas = 0
    for nombre in args.modelos:
        archivo, ruta = PESOS[nombre]
        destino = REPO / ruta
        esperada = sumas[archivo]
        if destino.exists() and sha256(destino) == esperada:
            print(f"ok      {ruta}")
            continue
        if args.verificar:
            print(f"FALTA   {ruta}" if not destino.exists() else f"DIFIERE {ruta}")
            fallas += 1
            continue
        bajar(archivo, destino)
        if sha256(destino) != esperada:
            print(f"ERROR   {ruta}: la suma sha256 no coincide con SHA256SUMS.txt")
            fallas += 1
        else:
            print(f"bajado  {ruta}")
    sys.exit(1 if fallas else 0)


if __name__ == "__main__":
    main()
