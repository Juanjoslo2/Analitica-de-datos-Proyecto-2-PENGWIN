"""cortes.py.

De un grupo de cortes sueltos a un volumen 3D que el dashboard procesa igual que un .mha.

Qué se acepta (se reconoce por el contenido, nunca por la extensión ni por el nombre):
    - DICOM, un archivo por corte: HU (GDCM aplica RescaleSlope/Intercept), posición y orientación
      del header.
    - PNG o JPEG en gris de 8 bits exportados con una ventana (por defecto L400/W1800, la del modelo),
      o PNG de 16 bits con las HU desplazadas (HU + 1024 es lo habitual).
Los cortes llegan juntos en un lote (el navegador concatena los archivos elegidos, ver ``leer_lote``)
o dentro de un .zip.

Cómo se decide el orden, de lo más a lo menos confiable:
    1. DICOM: posición del corte sobre la normal del plano (ImagePositionPatient · normal de
       ImageOrientationPatient). Da orden, sentido y separación sin suponer nada.
    2. PNG con ``SliceLocation`` en sus metadatos de texto (lo escribe ``exportar_cortes.py``).
    3. El número del nombre (corte_001.png, IMG-0042.jpg...): de los grupos de dígitos del nombre se
       usa el que cambia de un archivo a otro, y el contenido tiene que confirmarlo.
    4. El parecido entre cortes. Dos cortes vecinos de un CT casi no cambian, así que el orden es el
       camino que pasa por todos con el menor cambio total: orden espectral (vector de Fiedler del
       grafo de vecinos) refinado con 2-opt y reubicación de cortes sueltos.
Un PNG o JPEG no dice qué extremo es la cabeza. Eso lo decide ``servicio.sentido_anatomico`` con el
modelo: en los 100 casos de PENGWIN el centroide del sacro queda 17-56 mm por encima del de los coxales.

Los cortes repetidos se quitan. Si faltan cortes (saltos en la numeración o en la posición DICOM),
se interpolan para que la grilla en z quede uniforme. Todo queda anotado en el informe que ve el usuario.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import struct
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

MAGIA_LOTE = b"PWLOTE1\n"
MAX_CORTES = 2000
MAX_BYTES_CORTE = 64 * 1024 ** 2
MAX_BYTES_TOTAL = 1536 * 1024 ** 2          # descomprimido, sumando todos los cortes (zip incluido)
MAX_CABECERA = 4 * 1024 ** 2
MAX_BYTES_VOLUMEN = 1024 ** 3               # un .mha dentro de un .zip
EXT_VOLUMEN = (".mha", ".nii", ".nii.gz", ".nrrd")
MIN_CORTES_3D = 8
MAX_LADO = 2048
MAX_VOXELES = 400_000_000
VENTANA_MODELO = (400.0, 1800.0)
MM_PX_DEFECTO, MM_PX_DEFECTO_256, MM_CORTE_DEFECTO = 0.8, 1.27, 1.0   # medianas de PENGWIN (nativo y caché de 256 px)
DPI_DE_PANTALLA = {72, 96, 100, 144, 150, 200, 240, 300, 600}         # pHYs que pone el programa, no el escáner


class LoteInvalido(ValueError):
    """El mensaje va tal cual al usuario."""


# ---------------------------------------------------------------- lectura del lote

def tipo_de(datos: bytes) -> Optional[str]:
    if datos[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if datos[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if datos[128:132] == b"DICM":
        return "dicom"
    return None


def _limpiar_nombre(nombre: str) -> str:
    # solo se usa para ordenar y para mostrarlo: nunca como ruta
    return re.sub(r"[\x00-\x1f\x7f]", "", str(nombre))[-200:] or "sin-nombre"


def escribir_lote(archivos: List[Tuple[str, bytes]], ruta: Path | str) -> None:
    """Lo mismo que arma la página (static/app.js): MAGIA + u32 con el largo del índice JSON + índice +
    los archivos uno tras otro. Sirve para los tests y para subir un lote desde un script."""
    idx = json.dumps({"archivos": [{"nombre": n, "tam": len(d)} for n, d in archivos]}).encode("utf-8")
    with open(ruta, "wb") as f:
        f.write(MAGIA_LOTE + struct.pack("<I", len(idx)) + idx)
        for _, d in archivos:
            f.write(d)


def es_lote(ruta: Path | str) -> bool:
    with open(ruta, "rb") as f:
        return f.read(len(MAGIA_LOTE)) == MAGIA_LOTE


def _indice_lote(ruta: Path) -> Tuple[List[Tuple[str, int]], int]:
    tam = ruta.stat().st_size
    with open(ruta, "rb") as f:
        if f.read(len(MAGIA_LOTE)) != MAGIA_LOTE:
            raise LoteInvalido("el lote de cortes está dañado (falta la cabecera)")
        (n,) = struct.unpack("<I", f.read(4))
        if not 2 <= n <= MAX_CABECERA:
            raise LoteInvalido("el lote de cortes está dañado (índice inválido)")
        try:
            idx = json.loads(f.read(n).decode("utf-8"))["archivos"]
            archivos = [(_limpiar_nombre(a["nombre"]), int(a["tam"])) for a in idx]
        except (ValueError, KeyError, TypeError):
            raise LoteInvalido("el lote de cortes está dañado (índice ilegible)")
    inicio = len(MAGIA_LOTE) + 4 + n
    if not archivos or len(archivos) > MAX_CORTES:
        raise LoteInvalido(f"un lote lleva entre 1 y {MAX_CORTES} archivos")
    if any(not 0 <= t <= MAX_BYTES_CORTE for _, t in archivos):
        raise LoteInvalido("hay un archivo de más de 64 MB en el lote")
    if inicio + sum(t for _, t in archivos) != tam:
        raise LoteInvalido("el lote de cortes está incompleto")
    return archivos, inicio


def _indice_zip(ruta: Path) -> List[zipfile.ZipInfo]:
    try:
        zf = zipfile.ZipFile(ruta)
    except (zipfile.BadZipFile, OSError):
        raise LoteInvalido("el .zip está dañado")
    with zf:
        infos = [i for i in zf.infolist() if not i.is_dir() and not i.filename.startswith("__MACOSX/")
                 and not Path(i.filename).name.startswith(".")]
    if not infos:
        raise LoteInvalido("el .zip está vacío")
    if len(infos) > MAX_CORTES:
        raise LoteInvalido(f"el .zip tiene más de {MAX_CORTES} archivos")
    if any(i.flag_bits & 0x1 for i in infos):
        raise LoteInvalido("el .zip tiene contraseña")
    if any(i.file_size > MAX_BYTES_CORTE for i in infos) or sum(i.file_size for i in infos) > MAX_BYTES_TOTAL:
        raise LoteInvalido("el .zip descomprimido es demasiado grande")
    return infos


def _volumen_del_zip(zf: zipfile.ZipFile) -> Optional[zipfile.ZipInfo]:
    """Un .zip que trae un solo archivo y es un volumen (.mha, .nii...) se trata como ese volumen."""
    infos = [i for i in zf.infolist() if not i.is_dir() and not i.filename.startswith("__MACOSX/")
             and not Path(i.filename).name.startswith(".")]
    if len(infos) == 1 and infos[0].filename.lower().endswith(EXT_VOLUMEN):
        return infos[0]
    return None


def validar_lote(ruta: Path | str) -> Dict:
    """Revisión de la estructura antes de encolar (no decodifica ninguna imagen)."""
    ruta = Path(ruta)
    if es_lote(ruta):
        return {"formato": "lote", "archivos": len(_indice_lote(ruta)[0])}
    try:
        with zipfile.ZipFile(ruta) as zf:
            v = _volumen_del_zip(zf)
    except (zipfile.BadZipFile, OSError):
        raise LoteInvalido("el .zip está dañado")
    if v is not None:
        if v.flag_bits & 0x1:
            raise LoteInvalido("el .zip tiene contraseña")
        if v.file_size > MAX_BYTES_VOLUMEN:
            raise LoteInvalido("el volumen del .zip pesa más de 1 GB descomprimido")
        return {"formato": "zip con un volumen", "archivos": 1}
    return {"formato": "zip", "archivos": len(_indice_zip(ruta))}


def extraer_volumen(ruta: Path | str, destino: Path) -> Optional[Path]:
    """Si el .zip trae un solo volumen, lo saca a ``destino/entrada.<ext>`` y lo valida como cualquier
    volumen subido (validacion.validar_volumen). Devuelve None si el .zip trae cortes."""
    from validacion import ArchivoInvalido, extension, validar_volumen
    with zipfile.ZipFile(ruta) as zf:
        v = _volumen_del_zip(zf)
        if v is None:
            return None
        ext = extension(v.filename)
        out = destino / f"volumen{ext}"
        escrito = 0
        with zf.open(v) as src, open(out, "wb") as dst:
            while bloque := src.read(8 * 1024 ** 2):
                escrito += len(bloque)
                if escrito > MAX_BYTES_VOLUMEN:            # el tamaño declarado puede mentir
                    dst.close()
                    out.unlink(missing_ok=True)
                    raise LoteInvalido("el volumen del .zip pesa más de 1 GB descomprimido")
                dst.write(bloque)
    try:
        validar_volumen(out, out.name)
    except ArchivoInvalido as e:
        out.unlink(missing_ok=True)
        raise LoteInvalido(f"Archivo rechazado: {e}")
    return out


def leer_lote(ruta: Path | str) -> Iterator[Tuple[str, bytes]]:
    """(nombre, bytes) de cada archivo del lote del navegador o del .zip, sin cargarlos todos a la vez."""
    ruta = Path(ruta)
    if es_lote(ruta):
        archivos, pos = _indice_lote(ruta)
        with open(ruta, "rb") as f:
            f.seek(pos)
            for nombre, t in archivos:
                yield nombre, f.read(t)
        return
    total = 0
    with zipfile.ZipFile(ruta) as zf:
        for i in _indice_zip(ruta):
            with zf.open(i) as m:
                datos = m.read(MAX_BYTES_CORTE + 1)          # el tamaño declarado puede mentir: se acota al leer
            total += len(datos)
            if len(datos) > MAX_BYTES_CORTE or total > MAX_BYTES_TOTAL:
                raise LoteInvalido("el .zip descomprimido es demasiado grande")
            yield _limpiar_nombre(i.filename), datos


# ---------------------------------------------------------------- cortes en memoria

@dataclass
class Corte:
    nombre: str
    pixeles: np.ndarray                      # (Y, X): uint8/uint16 para imágenes, float32 en HU para DICOM
    posicion: Optional[float] = None         # mm sobre la normal (DICOM) o SliceLocation del PNG
    mm_px: Optional[Tuple[float, float]] = None   # (fila, columna) si el archivo lo trae
    color: bool = False
    extra: Dict = field(default_factory=dict)


@dataclass
class Reconstruccion:
    hu: np.ndarray                           # (Z, Y, X) float32, z crece en el sentido de la normal
    spacing_xyz: Tuple[float, float, float]
    origen: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    direccion: Tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    sentido_conocido: bool = False           # DICOM sí; PNG/JPEG lo decide el modelo
    nombres: List[str] = field(default_factory=list)    # nombre de cada corte ya ordenado (None = interpolado)
    informe: Dict = field(default_factory=dict)

    def invertir(self) -> None:
        self.hu = self.hu[::-1].copy()
        self.nombres = self.nombres[::-1]

    def imagen_sitk(self):
        import SimpleITK as sitk
        img = sitk.GetImageFromArray(np.clip(self.hu, -1024, 3071).astype(np.int16))
        img.SetSpacing([float(s) for s in self.spacing_xyz])
        img.SetOrigin([float(o) for o in self.origen])
        img.SetDirection([float(d) for d in self.direccion])
        return img


def _leer_imagen(nombre: str, datos: bytes) -> Corte:
    from validacion import ArchivoInvalido, validar_imagen
    try:
        img = validar_imagen(datos)
    except ArchivoInvalido as e:
        raise LoteInvalido(f"{nombre}: {e}")
    color = False
    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(img).astype(np.int32)
    else:
        rgb = np.asarray(img.convert("RGB"))
        color = not (np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2]))
        # en color se usa el canal verde: en la entrada 2.5D del modelo (z−Δ, z, z+Δ) es el corte central
        arr = rgb[..., 1].copy()
    c = Corte(nombre, arr, color=color)
    texto = getattr(img, "text", None) or {}
    try:
        if "SliceLocation" in texto:
            c.posicion = float(texto["SliceLocation"])
        if "PixelSpacing" in texto:
            fila, col = (float(v) for v in re.split(r"[\\, ]+", texto["PixelSpacing"].strip())[:2])
            c.mm_px = (fila, col)
    except ValueError:
        pass
    if c.mm_px is None:
        c.mm_px = _mm_px_png(datos) if datos[:4] == b"\x89PNG" else None
    if c.mm_px is None and img.info.get("dpi"):
        dpi = tuple(float(v) for v in img.info["dpi"])
        if all(round(v) not in DPI_DE_PANTALLA for v in dpi):
            c.mm_px = (25.4 / dpi[1], 25.4 / dpi[0])
    if c.mm_px is not None and not all(0.1 <= v <= 5.0 for v in c.mm_px):
        c.mm_px = None
    for k in ("RescaleIntercept", "WindowCenter", "WindowWidth"):
        if k in texto:
            try:
                c.extra[k] = float(texto[k])
            except ValueError:
                pass
    return c


def _mm_px_png(datos: bytes) -> Optional[Tuple[float, float]]:
    """Spacing del bloque sCAL (lo escriben ITK y SimpleITK: el número es el spacing en mm)."""
    i = 8
    while i + 8 <= len(datos):
        n = struct.unpack(">I", datos[i:i + 4])[0]
        t = datos[i + 4:i + 8]
        if t == b"sCAL" and n < 200:
            try:
                ancho, alto = datos[i + 9:i + 8 + n].split(b"\x00")[:2]
                return float(alto), float(ancho)
            except ValueError:
                return None
        if t == b"IDAT":
            return None
        i += 12 + n
    return None


def _leer_dicoms(rutas: List[Tuple[str, Path]]) -> Tuple[List[Corte], List[str]]:
    """Lee los DICOM de una serie: posición sobre la normal, spacing del plano y HU."""
    import SimpleITK as sitk
    cabeceras, ignorados = [], []
    for nombre, ruta in rutas:
        r = sitk.ImageFileReader()
        r.SetImageIO("GDCMImageIO")
        r.SetFileName(str(ruta))
        try:
            r.ReadImageInformation()
        except RuntimeError:
            ignorados.append(f"{nombre}: DICOM sin imagen (p. ej. DICOMDIR)")
            continue
        meta = {k: (r.GetMetaData(k).strip() if r.HasMetaDataKey(k) else "") for k in
                ("0020|000e", "0008|0008", "0008|0060", "0020|0013")}
        if "LOCALIZER" in meta["0008|0008"].upper():
            ignorados.append(f"{nombre}: imagen de localización (scout)")
            continue
        if r.GetNumberOfComponents() != 1:
            ignorados.append(f"{nombre}: DICOM en color")
            continue
        if r.GetDimension() != 3:
            ignorados.append(f"{nombre}: DICOM sin geometría 3D")
            continue
        size = r.GetSize()
        if len(size) == 3 and size[2] > 1:
            ignorados.append(f"{nombre}: DICOM multicorte (se espera un archivo por corte)")
            continue
        d = np.array(r.GetDirection(), float).reshape(3, 3)
        fila, col = d[:, 0], d[:, 1]
        normal = np.cross(fila, col)
        cabeceras.append({"nombre": nombre, "ruta": ruta, "serie": meta["0020|000e"], "modalidad": meta["0008|0060"],
                          "instancia": meta["0020|0013"], "size": tuple(size[:2]), "origen": np.array(r.GetOrigin()[:3], float),
                          "fila": fila, "col": col, "normal": normal / np.linalg.norm(normal),
                          "spacing": tuple(float(s) for s in r.GetSpacing()[:2])})
    if not cabeceras:
        return [], ignorados
    # varias series en la misma carpeta (reformateos, otra adquisición): se queda la más grande
    series = Counter(c["serie"] for c in cabeceras)
    serie, _ = series.most_common(1)[0]
    if len(series) > 1:
        ignorados.append(f"{sum(series.values()) - series[serie]} cortes de otras {len(series) - 1} series "
                         f"(se usó la serie más grande, de {series[serie]} cortes)")
    cab = [c for c in cabeceras if c["serie"] == serie]
    # misma orientación y tamaño que la mayoría
    clave = [(c["size"], tuple(np.round(c["normal"], 3))) for c in cab]
    ref = Counter(clave).most_common(1)[0][0]
    if any(k != ref for k in clave):
        ignorados.append(f"{sum(k != ref for k in clave)} cortes con otro tamaño u orientación")
    cab = [c for c, k in zip(cab, clave) if k == ref]
    cortes = []
    for c in cab:
        img = sitk.ReadImage(str(c["ruta"]), imageIO="GDCMImageIO")
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        arr = arr[0] if arr.ndim == 3 else arr
        cortes.append(Corte(c["nombre"], arr, posicion=float(c["origen"] @ c["normal"]),
                            mm_px=(c["spacing"][1], c["spacing"][0]),
                            extra={"origen": c["origen"], "fila": c["fila"], "col": c["col"], "normal": c["normal"],
                                   "modalidad": c["modalidad"], "instancia": c["instancia"]}))
    return cortes, ignorados


# ---------------------------------------------------------------- orden

_NUM = re.compile(r"\d+(?:\.\d+)?")


def numeros_del_nombre(nombres: List[str]) -> List[Dict]:
    """Candidatos a número de corte: cada posición de grupo de dígitos que cambia de un archivo a otro
    y no se repite. Devuelve [{"valores": [...], "decimal": bool, "desc": str}]."""
    tallos = [re.sub(r"\.(png|jpe?g|dcm|dicom|ima)$", "", Path(n.replace("\\", "/")).name, flags=re.I) for n in nombres]
    grupos = [_NUM.findall(t) for t in tallos]
    cand = []
    largos = {len(g) for g in grupos}
    posiciones = []
    if len(largos) == 1 and largos != {0}:
        posiciones = [(j, f"grupo {j + 1} de dígitos") for j in range(largos.pop())]
    elif all(grupos):
        posiciones = [(-1, "último grupo de dígitos")]
    for j, desc in posiciones:
        txt = [g[j] for g in grupos]
        vals = [float(v) for v in txt]
        if len(set(vals)) == len(vals):
            cand.append({"valores": vals, "decimal": any("." in v for v in txt), "desc": desc})
    return cand


def _miniaturas(hu: np.ndarray, lado: int = 64) -> np.ndarray:
    """Cada corte, en la ventana del modelo, reducido a lado × lado por promedio de bloques: el vector
    con que se comparan los cortes."""
    from PIL import Image
    out = np.empty((len(hu), lado * lado), np.float32)
    for k, h in enumerate(hu):
        v = np.clip(h, -500.0, 1300.0).astype(np.float32)
        out[k] = np.asarray(Image.fromarray(v).resize((lado, lado), Image.BOX), np.float32).ravel()
    return out


def distancias(vec: np.ndarray) -> np.ndarray:
    g = vec @ vec.T
    n = np.diag(g)
    return np.sqrt(np.maximum(n[:, None] + n[None, :] - 2 * g, 0.0))


def costo(D: np.ndarray, orden: np.ndarray) -> float:
    return float(D[orden[:-1], orden[1:]].sum())


def _fiedler(D: np.ndarray, k: int) -> np.ndarray:
    from scipy.sparse.csgraph import connected_components
    N = len(D)
    while True:
        vec = np.argsort(D, axis=1)[:, 1:k + 1]
        sigma = np.median(D[np.arange(N), vec[:, -1]]) + 1e-9
        W = np.zeros_like(D)
        filas = np.repeat(np.arange(N), k)
        W[filas, vec.ravel()] = np.exp(-(D[filas, vec.ravel()] / sigma) ** 2)
        W = np.maximum(W, W.T)
        if connected_components(W > 0, directed=False)[0] == 1 or k >= N - 1:
            break
        k = min(2 * k, N - 1)
    g = W.sum(1)
    Dm = 1.0 / np.sqrt(g + 1e-12)
    L = np.eye(N) - Dm[:, None] * W * Dm[None, :]
    _, v = np.linalg.eigh(L)
    return np.argsort(Dm * v[:, 1])


def _dos_opt(D: np.ndarray, orden: np.ndarray, pasadas: Optional[int] = None) -> np.ndarray:
    """Mejora el camino abierto: invierte tramos (2-opt) y mueve cortes sueltos de lugar.
    Cada pasada aplica el mejor movimiento; para cuando ninguno baja el costo."""
    p = orden.copy()
    N = len(p)
    for _ in range(pasadas or N):
        mejor = False
        # 2-opt: invertir p[i..j]. Aristas que cambian: (i-1, i) y (j, j+1).
        izq = np.r_[0.0, D[p[:-1], p[1:]]]                 # costo de la arista que entra a i (0 si i = 0)
        der = np.r_[D[p[:-1], p[1:]], 0.0]                 # costo de la arista que sale de j (0 si j = N-1)
        a = np.r_[p[0], p[:-1]]                            # nodo antes de i (sin uso si i = 0)
        b = np.r_[p[1:], p[-1]]                            # nodo después de j
        nueva_izq = D[a[:, None], p[None, :]]              # (i-1) con j
        nueva_der = D[p[:, None], b[None, :]]              # i con (j+1)
        nueva_izq[0, :] = 0.0
        nueva_der[:, -1] = 0.0
        delta = nueva_izq + nueva_der - izq[:, None] - der[None, :]
        delta[np.tril_indices(N)] = 0.0
        i, j = np.unravel_index(np.argmin(delta), delta.shape)
        if delta[i, j] < -1e-6:
            p[i:j + 1] = p[i:j + 1][::-1]
            mejor = True
        # reubicar un corte: sacarlo de su sitio y meterlo entre otros dos
        for _ in range(4):
            sale = np.zeros(N)
            sale[1:-1] = D[p[:-2], p[1:-1]] + D[p[1:-1], p[2:]] - D[p[:-2], p[2:]]
            sale[0], sale[-1] = D[p[0], p[1]], D[p[-2], p[-1]]
            m = int(np.argmax(sale))
            q = np.delete(p, m)
            entra = np.r_[D[p[m], q[0]], D[q[:-1], p[m]] + D[p[m], q[1:]] - D[q[:-1], q[1:]], D[q[-1], p[m]]]
            k = int(np.argmin(entra))
            if entra[k] < sale[m] - 1e-6:
                p = np.insert(q, k, p[m])
                mejor = True
            else:
                break
        if not mejor:
            break
    return p


def orden_por_contenido(D: np.ndarray) -> np.ndarray:
    return _dos_opt(D, _fiedler(D, k=min(8, len(D) - 1)))


# ---------------------------------------------------------------- de cortes a volumen

def _hu_de_imagenes(cortes: List[Corte], ventana: Tuple[float, float], avisos: List[str]) -> Tuple[np.ndarray, str]:
    """Gris -> HU. 8 bits: se deshace la ventana (nivel, ancho). 16 bits: HU desplazadas."""
    pix = np.stack([c.pixeles for c in cortes])
    muestra = pix[:, ::4, ::4]
    distintos = np.unique(muestra).size
    if muestra.min() >= 0 and muestra.max() <= 255 and distintos <= 64:
        raise LoteInvalido(f"Estos cortes parecen máscaras, no un CT: tienen solo {distintos} tonos distintos "
                           f"(de {muestra.min()} a {muestra.max()}). Una máscara marca los fragmentos; para "
                           "analizar hace falta la tomografía.")
    if pix.dtype != np.uint8 and pix.max() > 255:
        # 16 bits: se prueba qué desplazamiento deja el aire cerca de -1000 HU (los convenios habituales
        # son HU + 1024, como los DICOM sin RescaleIntercept, y HU + 32768)
        aire = float(np.percentile(muestra, 1))
        dado = next((c.extra["RescaleIntercept"] for c in cortes if "RescaleIntercept" in c.extra), None)
        if dado is not None:
            desp = -dado
        else:
            desp = min((1024.0, 32768.0, 0.0, 2048.0), key=lambda o: abs((aire - o) + 1000.0))
        hu = pix.astype(np.float32) - desp
        if not -1200 <= aire - desp <= -700:
            avisos.append(f"En los PNG de 16 bits no se reconoce el aire (el valor más bajo es {aire:.0f}); se "
                          f"supuso HU = valor − {desp:.0f}.")
        return hu, f"PNG de 16 bits, HU = valor − {desp:.0f}"
    nivel, ancho = ventana
    hu = pix.astype(np.float32) * (ancho / 255.0) + (nivel - ancho / 2.0)
    # 0 = «por debajo de la ventana», que en un CT es sobre todo aire: así la silueta del paciente y la
    # camilla se separan aunque la ventana empiece por encima de -500 HU
    hu[pix == 0] = -1024.0
    cuerpo = muestra > 20
    if cuerpo.any():
        saturado = float((muestra[cuerpo] >= 254).mean())
        if saturado > 0.03:
            avisos.append(f"El {100 * saturado:.0f} % del cuerpo está en blanco puro: los cortes parecen exportados con "
                          f"una ventana más estrecha que L{nivel:.0f}/W{ancho:.0f} y el hueso queda saturado. Si sabe "
                          "la ventana con que se exportaron, indíquela en las opciones.")
    if any(c.color for c in cortes):
        avisos.append("Algunos cortes vienen en color: se usó el canal verde (el corte central si es la entrada "
                      "2.5D del modelo). Si son capturas con dibujos encima, el resultado no sirve.")
    return hu, f"gris de 8 bits con la ventana L{nivel:.0f}/W{ancho:.0f}"


def _grilla_uniforme(hu: np.ndarray, pos: np.ndarray, nombres: List[str], avisos: List[str],
                     unidad: str) -> Tuple[np.ndarray, float, List[Optional[str]], int]:
    """Lleva los cortes (ordenados por ``pos``) a una grilla con paso constante en z.

    Si la separación es pareja (±10 %) no se toca nada. Si hay saltos (faltan cortes), se interpola
    linealmente entre los vecinos y se cuenta cuántos cortes se inventaron."""
    dif = np.diff(pos)
    paso = float(np.median(dif))
    if np.all(np.abs(dif - paso) <= 0.1 * paso):
        return hu, paso, list(nombres), 0
    n = int(round((pos[-1] - pos[0]) / paso)) + 1
    t = pos[0] + paso * np.arange(n)
    j = np.clip(np.searchsorted(pos, t, side="right") - 1, 0, len(pos) - 2)
    w = np.clip((t - pos[j]) / np.maximum(pos[j + 1] - pos[j], 1e-9), 0.0, 1.0)
    out = np.empty((n,) + hu.shape[1:], np.float32)
    nuevos: List[Optional[str]] = []
    exactos = 0
    for k in range(n):
        if w[k] < 1e-3 or w[k] > 1 - 1e-3:
            src = j[k] if w[k] < 0.5 else j[k] + 1
            out[k] = hu[src]
            nuevos.append(nombres[src])
            exactos += 1
        else:
            out[k] = (1 - w[k]) * hu[j[k]] + w[k] * hu[j[k] + 1]
            nuevos.append(None)
    inventados = n - exactos
    mayor = float(dif.max())
    if inventados:
        avisos.append(f"Faltan cortes: el mayor salto es de {mayor / paso:.1f} veces la separación normal "
                      f"({mayor:.2f} {unidad}). Se interpolaron {inventados} cortes para dejar el paso parejo.")
    return out, paso, nuevos, inventados


def reconstruir(ruta: Path | str, carpeta_tmp: Path, mm_px: Optional[float] = None, mm_corte: Optional[float] = None,
                ventana: Tuple[float, float] = VENTANA_MODELO) -> Reconstruccion:
    """Lee el lote, decide el orden y arma el volumen en HU. El sentido de PNG/JPEG queda pendiente."""
    carpeta_tmp.mkdir(parents=True, exist_ok=True)
    imagenes: List[Corte] = []
    dicoms: List[Tuple[str, Path]] = []
    ignorados: List[str] = []
    recibidos = 0
    for nombre, datos in leer_lote(ruta):
        recibidos += 1
        t = tipo_de(datos)
        if t is None:
            ignorados.append(f"{nombre}: no es PNG, JPEG ni DICOM")
        elif t == "dicom":
            p = carpeta_tmp / f"{len(dicoms):05d}.dcm"          # nombre propio: el del usuario nunca es ruta
            p.write_bytes(datos)
            dicoms.append((nombre, p))
        else:
            try:
                imagenes.append(_leer_imagen(nombre, datos))
            except LoteInvalido as e:                    # un archivo dañado no tumba el lote entero
                ignorados.append(str(e))
    avisos: List[str] = []
    if len(dicoms) >= 2:
        if imagenes:
            ignorados.append(f"{len(imagenes)} imágenes PNG/JPEG (el lote trae una serie DICOM y se usó esa)")
        cortes, ign = _leer_dicoms(dicoms)
        ignorados += ign
        formato = "DICOM"
    else:
        cortes = imagenes
        if dicoms:
            ignorados.append(f"{len(dicoms)} DICOM suelto(s) entre imágenes")
        formato = "PNG/JPEG"
    if len(cortes) < MIN_CORTES_3D:
        raise LoteInvalido(f"Para reconstruir en 3D hacen falta al menos {MIN_CORTES_3D} cortes del mismo estudio; "
                           f"llegaron {len(cortes)} útiles de {recibidos} archivos"
                           + (f" ({'; '.join(ignorados[:3])})" if ignorados else "")
                           + ". Para un corte solo use «Subir cortes» con una imagen.")
    # tamaño común
    tam = Counter(c.pixeles.shape for c in cortes)
    forma, n_forma = tam.most_common(1)[0]
    if len(tam) > 1:
        if n_forma < MIN_CORTES_3D:
            raise LoteInvalido(f"Los cortes tienen tamaños distintos ({', '.join(f'{a}×{b}' for b, a in list(tam)[:4])}); "
                               "todos los de un estudio miden lo mismo.")
        ignorados.append(f"{len(cortes) - n_forma} cortes de otro tamaño (se usaron los de {forma[1]}×{forma[0]} px)")
        cortes = [c for c in cortes if c.pixeles.shape == forma]
    if max(forma) > MAX_LADO or len(cortes) * forma[0] * forma[1] > MAX_VOXELES:
        raise LoteInvalido("el volumen que forman los cortes es demasiado grande")

    # el mismo archivo subido dos veces (p. ej. «corte_005 (copia).png»): fuera antes de ordenar, si no el
    # número repetido del nombre deja de servir para ordenar
    vistos, unicos = set(), []
    for c in cortes:
        h = hashlib.blake2b(c.pixeles.tobytes(), digest_size=16).digest()
        if h not in vistos:
            vistos.add(h)
            unicos.append(c)
    copias = len(cortes) - len(unicos)
    cortes = unicos
    nombres = [c.nombre for c in cortes]
    informe: Dict = {"formato": formato, "recibidos": recibidos, "ignorados": ignorados, "avisos": avisos}

    if formato == "DICOM":
        hu = np.stack([c.pixeles for c in cortes])
        pos = np.array([c.posicion for c in cortes])
        metodo = "posición de cada corte en el header DICOM (ImagePositionPatient sobre la normal del plano)"
        unidad = "mm"
        sentido_conocido = True
        inst = [c.extra["instancia"] for c in cortes]
        if np.ptp(pos) < 0.01 and all(i.isdigit() for i in inst) and len(set(inst)) == len(inst):
            # serie sin posiciones (p. ej. capturas secundarias): queda el número de instancia, que no
            # dice dónde está la cabeza
            pos = np.array([float(i) for i in inst])
            metodo, unidad, sentido_conocido = "InstanceNumber del DICOM (la serie no trae posiciones)", "números del nombre", False
            avisos.append("La serie DICOM no trae la posición de los cortes: se ordenó por InstanceNumber y la "
                          "separación entre cortes " + ("es la escrita." if mm_corte else f"se supuso de {MM_CORTE_DEFECTO:.2f} mm."))
        orden = np.argsort(pos, kind="stable")
        conversion = "HU del DICOM (RescaleSlope/Intercept)"
    else:
        hu, conversion = _hu_de_imagenes(cortes, ventana, avisos)
        D = distancias(_miniaturas(hu))
        contenido = orden_por_contenido(D)
        c_cont = costo(D, contenido)
        pos, metodo, unidad = None, "", "cortes"
        if all(c.posicion is not None for c in cortes) and len({c.posicion for c in cortes}) == len(cortes):
            pos = np.array([c.posicion for c in cortes])
            metodo, unidad = "SliceLocation guardado en cada PNG", "mm"
        else:
            cands = numeros_del_nombre(nombres)
            if cands:
                mejor = min(cands, key=lambda c: costo(D, np.argsort(c["valores"], kind="stable")))
                c_nom = costo(D, np.argsort(mejor["valores"], kind="stable"))
                informe["costo_nombre_vs_contenido"] = round(c_nom / max(c_cont, 1e-9), 3)
                if c_nom <= 1.25 * c_cont:
                    pos = np.array(mejor["valores"])
                    unidad = "mm" if mejor["decimal"] else "números del nombre"
                    metodo = (f"número del nombre ({mejor['desc']}), "
                              + ("confirmado por el parecido entre cortes" if c_nom <= 1.02 * c_cont
                                 else f"aunque el parecido entre cortes sugiere otro orden ({c_nom / c_cont:.2f} veces "
                                      "más cambio entre vecinos)"))
                else:
                    avisos.append(f"El número de los nombres no sigue la anatomía (con ese orden los cortes vecinos "
                                  f"cambian {c_nom / c_cont:.1f} veces más que en el mejor orden): se ordenaron por "
                                  "parecido entre cortes.")
        if pos is None:
            orden = contenido
            pos = np.empty(len(cortes))
            pos[orden] = np.arange(len(cortes), dtype=float)
            metodo = "parecido entre cortes vecinos (orden espectral + 2-opt)"
        else:
            orden = np.argsort(pos, kind="stable")
        vecinos = D[orden[:-1], orden[1:]]
        salto = float(vecinos.max() / max(np.median(vecinos), 1e-9))
        informe["salto_maximo_relativo"] = round(salto, 2)
        if salto > 4 and unidad == "cortes":
            avisos.append(f"Entre dos cortes contiguos hay un cambio {salto:.0f} veces mayor que el habitual: puede "
                          "faltar un tramo o haber cortes de otro estudio mezclados.")
        sentido_conocido = False
    informe["conversion"] = conversion

    # repetidos: misma posición o el mismo contenido
    hu = hu[orden]
    pos = pos[orden]
    nombres = [nombres[k] for k in orden]
    dup = np.r_[False, np.diff(pos) <= 1e-6]
    if formato != "DICOM":
        mismo = np.r_[False, [np.array_equal(hu[k], hu[k - 1]) for k in range(1, len(hu))]]
        dup |= mismo
    repetidos = copias + int(dup.sum())
    if repetidos:
        avisos.append(f"Se quitaron {repetidos} cortes repetidos.")
    if dup.any():
        hu, pos = hu[~dup], pos[~dup]
        nombres = [n for n, d in zip(nombres, dup) if not d]
    if len(hu) < MIN_CORTES_3D:
        raise LoteInvalido(f"Sin repetidos quedan {len(hu)} cortes: hacen falta al menos {MIN_CORTES_3D}.")
    hu, paso, nombres_g, inventados = _grilla_uniforme(hu, pos, nombres, avisos, unidad)

    # spacing
    if formato == "DICOM":
        ref = cortes[0]
        sy, sx = ref.mm_px
        fuente_xy = "header DICOM (PixelSpacing)"
        if unidad == "mm":
            dz, fuente_z = paso, "distancia entre posiciones DICOM"
            if mm_corte:
                avisos.append("Se ignoró el «mm entre cortes» escrito: un DICOM trae su propia geometría.")
        else:
            dz, fuente_z = (float(mm_corte), "escrito por el usuario") if mm_corte else (MM_CORTE_DEFECTO, "supuesto")
            paso = dz
        o = cortes[int(np.argmin([c.posicion for c in cortes]))].extra
        fila, col, normal = o["fila"], o["col"], o["normal"]
        if unidad != "mm" and normal[2] < 0:
            normal = -normal                     # sin posiciones el sentido lo pone la anatomía: z del arreglo = cabeza
        origen = tuple(float(v) for v in o["origen"])
        direccion = tuple(float(v) for v in np.stack([fila, col, normal], axis=1).ravel())
        if paso <= 0.05:
            raise LoteInvalido("Las posiciones DICOM de los cortes no avanzan: no forman un volumen.")
    else:
        origen = (0.0, 0.0, 0.0)
        direccion = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        del_archivo = Counter(c.mm_px for c in cortes if c.mm_px is not None)
        if mm_px:
            sy = sx = float(mm_px)
            fuente_xy = "escrito por el usuario"
        elif del_archivo:
            (sy, sx), _ = del_archivo.most_common(1)[0]
            fuente_xy = "metadatos del PNG"
        else:
            sy = sx = MM_PX_DEFECTO_256 if max(forma) <= 300 else MM_PX_DEFECTO
            fuente_xy = "supuesto"
            avisos.append(f"Los cortes no dicen cuántos mm mide un píxel: se supuso {sx:.2f} mm/px "
                          f"({'imagen de ≤ 300 px, como el recorte del modelo' if max(forma) <= 300 else 'mediana de PENGWIN'}). "
                          "Las distancias en mm dependen de ese valor; si lo conoce, escríbalo.")
        if mm_corte:
            dz, fuente_z = float(mm_corte), "escrito por el usuario"
        elif unidad == "mm":
            dz, fuente_z = paso, "posiciones en mm de los cortes"
        else:
            dz, fuente_z = MM_CORTE_DEFECTO, "supuesto"
            avisos.append(f"Los cortes no dicen a cuántos mm están uno de otro: se supuso {dz:.2f} mm. Si lo conoce, "
                          "escríbalo en «mm entre cortes».")
        if not 0.1 <= dz <= 20:
            raise LoteInvalido(f"La separación entre cortes ({dz:.3f} mm) está fuera de rango.")

    usados = sum(n is not None for n in nombres_g)
    informe.update({
        "orden": metodo, "cortes": int(hu.shape[0]), "usados": usados, "interpolados": inventados,
        "repetidos": repetidos, "forma_px": [int(forma[0]), int(forma[1])],
        "mm_px": [round(float(sy), 4), round(float(sx), 4)], "mm_corte": round(float(dz), 4),
        "fuente_mm_px": fuente_xy, "fuente_mm_corte": fuente_z,
    })
    return Reconstruccion(hu=hu.astype(np.float32, copy=False), spacing_xyz=(float(sx), float(sy), float(dz)),
                          origen=origen, direccion=direccion, sentido_conocido=sentido_conocido,
                          nombres=nombres_g, informe=informe)


def corte_suelto(datos: bytes, carpeta_tmp: Path, mm_px: Optional[float] = None,
                 ventana: Tuple[float, float] = VENTANA_MODELO) -> Tuple[np.ndarray, float, str, List[str]]:
    """Un solo corte (PNG, JPEG o DICOM) -> (canales en HU (C, Y, X), mm/px, de dónde salió el mm/px, avisos).

    C = 3 si un PNG/JPEG en color trae la entrada 2.5D del modelo (z−Δ, z, z+Δ) en sus canales."""
    t = tipo_de(datos)
    avisos: List[str] = []
    if t is None:
        raise LoteInvalido("el corte debe ser PNG, JPEG o DICOM")
    if t == "dicom":
        carpeta_tmp.mkdir(parents=True, exist_ok=True)
        p = carpeta_tmp / "corte.dcm"
        p.write_bytes(datos)
        try:
            cortes, ign = _leer_dicoms([("corte", p)])
        finally:
            p.unlink(missing_ok=True)
        if not cortes:
            raise LoteInvalido("el DICOM no trae una imagen que se pueda usar" + (f" ({ign[0]})" if ign else ""))
        c = cortes[0]
        canales, del_archivo, conv = c.pixeles[None], c.mm_px, "HU del DICOM"
    else:
        c = _leer_imagen("corte", datos)
        if c.color:
            from PIL import Image
            rgb = np.asarray(Image.open(io.BytesIO(datos)).convert("RGB"))
            planos = [Corte("corte", rgb[..., k].copy()) for k in range(3)]
        else:
            planos = [c]
        canales, conv = _hu_de_imagenes(planos, ventana, avisos)
        del_archivo = c.mm_px
        if c.color:
            avisos.append("La imagen viene en color: cada canal se tomó como un corte de la entrada 2.5D (z−Δ, z, z+Δ).")
    if mm_px:
        mm, fuente = float(mm_px), "escrito por el usuario"
    elif del_archivo:
        mm, fuente = float(np.mean(del_archivo)), "del archivo"
    else:
        mm = MM_PX_DEFECTO_256 if max(canales.shape[1:]) <= 300 else MM_PX_DEFECTO
        fuente = "supuesto"
        avisos.append(f"La imagen no dice cuántos mm mide un píxel: se supuso {mm:.2f} mm/px. Si lo sabe, escríbalo en "
                      "las opciones; las distancias dependen de ese valor.")
    return canales.astype(np.float32), mm, f"{conv}; {mm:.3f} mm/px ({fuente})", avisos


def primero_y_ultimo(rec: Reconstruccion) -> Tuple[Optional[str], Optional[str]]:
    reales = [n for n in rec.nombres if n is not None]
    return (reales[0], reales[-1]) if reales else (None, None)
