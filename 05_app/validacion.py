"""validacion.py.

Qué se deja pasar hacia el modelo: solo volúmenes médicos en un único archivo (.mha, .nii, .nii.gz,
.nrrd) y cortes en PNG o JPEG. Se valida el CONTENIDO (firma y cabecera), no la extensión.

Riesgos que se cierran:
- **Lectura de archivos del servidor.** La cabecera de un .mha (``ElementDataFile``) y la de un
  .nrrd (``data file``) pueden apuntar a OTRO archivo del disco, y SimpleITK lo leería. Solo se acepta
  el formato de un solo archivo: ``ElementDataFile = LOCAL`` y NRRD sin ``data file``. NIfTI de dos
  archivos (``ni1``, .hdr + .img) tampoco se acepta.
- **Bombas de descompresión.** Las dimensiones se leen de la cabecera ANTES de descomprimir nada y se
  acotan (lado ≤ 2048, ≤ 400 millones de vóxeles). Las imágenes, a ≤ 25 megapíxeles.
- **Archivos disfrazados.** Un ejecutable o un script renombrado a .mha no tiene la cabecera que se
  exige y se rechaza. Al final, SimpleITK lee solo la información (sin los vóxeles) y debe coincidir.
"""

from __future__ import annotations

import gzip
import io
import re
import struct
from pathlib import Path

MAX_LADO = 2048
MAX_VOXELES = 400_000_000
MAX_PIXELES_IMG = 25_000_000
EXTENSIONES = (".mha", ".nii", ".nii.gz", ".nrrd")
TIPOS_MHA = {"MET_CHAR", "MET_UCHAR", "MET_SHORT", "MET_USHORT", "MET_INT", "MET_UINT", "MET_FLOAT", "MET_DOUBLE"}
TIPOS_NII = {2, 4, 8, 16, 64, 256, 512, 768}          # uint8, int16, int32, float32, float64, int8, uint16, uint32
TIPOS_NRRD = {"char", "signed char", "int8", "int8_t", "uchar", "unsigned char", "uint8", "uint8_t", "short", "short int",
              "signed short", "signed short int", "int16", "int16_t", "ushort", "unsigned short", "unsigned short int",
              "uint16", "uint16_t", "int", "signed int", "int32", "int32_t", "uint", "unsigned int", "uint32", "uint32_t",
              "float", "double"}


class ArchivoInvalido(ValueError):
    pass


def extension(nombre: str) -> str:
    n = (nombre or "").lower()
    for e in (".nii.gz",) + EXTENSIONES:
        if n.endswith(e):
            return e
    raise ArchivoInvalido(f"formato no admitido; use {', '.join(EXTENSIONES)}")


def _dims_ok(dims) -> None:
    if len(dims) != 3 or any(d < 1 or d > MAX_LADO for d in dims):
        raise ArchivoInvalido(f"se espera un volumen 3D con lados entre 1 y {MAX_LADO}; llegó {list(dims)}")
    if dims[0] * dims[1] * dims[2] > MAX_VOXELES:
        raise ArchivoInvalido("el volumen es demasiado grande")
    if min(dims) < 8:
        raise ArchivoInvalido("el volumen tiene muy pocos cortes o píxeles para ser un CT")


def _mha(ruta: Path) -> list:
    with open(ruta, "rb") as f:
        cab = f.read(65536)
    if b"ElementDataFile" not in cab:
        raise ArchivoInvalido("no es un MetaImage (.mha) válido: falta la cabecera")
    texto = cab[:cab.index(b"ElementDataFile")]
    linea_dato = cab[cab.index(b"ElementDataFile"):].split(b"\n", 1)[0]
    try:
        texto = texto.decode("ascii")
        linea_dato = linea_dato.decode("ascii").strip()
    except UnicodeDecodeError:
        raise ArchivoInvalido("la cabecera del .mha no es texto")
    campos = {}
    for l in texto.splitlines():
        if "=" in l:
            k, v = l.split("=", 1)
            campos[k.strip()] = v.strip()
    if campos.get("ObjectType") != "Image":
        raise ArchivoInvalido("el .mha no es una imagen (ObjectType)")
    if re.sub(r"\s+", "", linea_dato) != "ElementDataFile=LOCAL":
        raise ArchivoInvalido("el .mha debe traer los datos dentro del mismo archivo (ElementDataFile = LOCAL)")
    if campos.get("NDims") != "3":
        raise ArchivoInvalido("el .mha debe ser 3D (NDims = 3)")
    if campos.get("ElementType") not in TIPOS_MHA:
        raise ArchivoInvalido("tipo de vóxel no admitido en el .mha")
    if int(campos.get("ElementNumberOfChannels", "1")) != 1:
        raise ArchivoInvalido("el .mha debe tener un solo canal")
    try:
        dims = [int(x) for x in campos["DimSize"].split()]
        esp = [float(x) for x in campos.get("ElementSpacing", campos.get("ElementSize", "1 1 1")).split()]
    except (KeyError, ValueError):
        raise ArchivoInvalido("DimSize o ElementSpacing inválidos en el .mha")
    if len(esp) != 3 or any(not (0.01 <= s <= 50) for s in esp):
        raise ArchivoInvalido("spacing fuera de rango en el .mha")
    return dims


def _cabecera_nifti(b: bytes) -> list:
    if len(b) >= 348 and struct.unpack("<i", b[:4])[0] in (348, 540):
        e = "<"
    elif len(b) >= 348 and struct.unpack(">i", b[:4])[0] in (348, 540):
        e = ">"
    else:
        raise ArchivoInvalido("no es un NIfTI válido")
    if struct.unpack(e + "i", b[:4])[0] == 348:                      # NIfTI-1
        magic = b[344:348]
        if magic != b"n+1\x00":
            raise ArchivoInvalido("solo se admite NIfTI de un archivo (n+1); el par .hdr/.img no")
        dim = struct.unpack(e + "8h", b[40:56])
        tipo = struct.unpack(e + "h", b[70:72])[0]
        pix = struct.unpack(e + "8f", b[76:108])
    else:                                                             # NIfTI-2
        if b[4:8] != b"n+2\x00":
            raise ArchivoInvalido("solo se admite NIfTI de un archivo (n+2)")
        tipo = struct.unpack(e + "h", b[12:14])[0]
        dim = struct.unpack(e + "8q", b[16:80])
        pix = struct.unpack(e + "8d", b[104:168])
    if not (dim[0] == 3 or (dim[0] == 4 and dim[4] == 1)):
        raise ArchivoInvalido("el NIfTI debe ser un volumen 3D")
    if tipo not in TIPOS_NII:
        raise ArchivoInvalido("tipo de vóxel no admitido en el NIfTI")
    if any(not (0.01 <= abs(p) <= 50) for p in pix[1:4]):
        raise ArchivoInvalido("spacing fuera de rango en el NIfTI")
    return [int(dim[1]), int(dim[2]), int(dim[3])]


def _nii(ruta: Path, comprimido: bool) -> list:
    with open(ruta, "rb") as f:
        inicio = f.read(4)
    if comprimido:
        if inicio[:2] != b"\x1f\x8b":
            raise ArchivoInvalido("el .nii.gz no está comprimido con gzip")
        try:
            with gzip.open(ruta, "rb") as g:                          # solo la cabecera: no descomprime el volumen
                cab = g.read(540)
        except OSError:
            raise ArchivoInvalido("el .nii.gz está dañado")
    else:
        with open(ruta, "rb") as f:
            cab = f.read(540)
    return _cabecera_nifti(cab)


def _nrrd(ruta: Path) -> list:
    with open(ruta, "rb") as f:
        cab = f.read(65536)
    if not cab.startswith(b"NRRD000"):
        raise ArchivoInvalido("no es un NRRD válido")
    fin = cab.find(b"\n\n")
    if fin < 0:
        raise ArchivoInvalido("cabecera NRRD incompleta")
    try:
        lineas = cab[:fin].decode("ascii").splitlines()[1:]
    except UnicodeDecodeError:
        raise ArchivoInvalido("la cabecera del NRRD no es texto")
    campos = {}
    for l in lineas:
        if l.startswith("#") or ":" not in l:
            continue
        k, v = l.split(":", 1)
        campos[k.strip().lower()] = v.lstrip("=").strip()
    if "data file" in campos or "datafile" in campos:
        raise ArchivoInvalido("el NRRD debe traer los datos en el mismo archivo (sin 'data file')")
    if campos.get("dimension") != "3":
        raise ArchivoInvalido("el NRRD debe ser 3D")
    if campos.get("type", "").lower() not in TIPOS_NRRD:
        raise ArchivoInvalido("tipo de vóxel no admitido en el NRRD")
    if campos.get("encoding", "raw").lower() not in ("raw", "gzip", "gz"):
        raise ArchivoInvalido("codificación del NRRD no admitida (use raw o gzip)")
    try:
        return [int(x) for x in campos["sizes"].split()]
    except (KeyError, ValueError):
        raise ArchivoInvalido("'sizes' inválido en el NRRD")


def validar_volumen(ruta: Path | str, nombre: str) -> dict:
    """Valida un volumen subido. Lanza ArchivoInvalido con un mensaje para el usuario."""
    ruta = Path(ruta)
    ext = extension(nombre)
    dims = {".mha": _mha, ".nrrd": _nrrd}.get(ext, lambda r: _nii(r, ext == ".nii.gz"))(ruta)
    _dims_ok(dims)
    # Segunda opinión: SimpleITK lee SOLO la cabecera (sin vóxeles) y debe coincidir
    import SimpleITK as sitk
    r = sitk.ImageFileReader()
    r.SetFileName(str(ruta))
    try:
        r.ReadImageInformation()
    except RuntimeError:
        raise ArchivoInvalido("SimpleITK no reconoce el archivo")
    if r.GetDimension() != 3 or sorted(r.GetSize()) != sorted(dims) or r.GetNumberOfComponents() != 1:
        raise ArchivoInvalido("la cabecera no coincide con lo que lee SimpleITK")
    return {"formato": ext, "dims_xyz": list(r.GetSize()), "spacing": list(r.GetSpacing())}


def validar_imagen(datos: bytes):
    """PNG o JPEG reales, ≤ 25 megapíxeles. Devuelve la imagen PIL ya cargada."""
    from PIL import Image
    if not (datos.startswith(b"\x89PNG\r\n\x1a\n") or datos.startswith(b"\xff\xd8\xff")):
        raise ArchivoInvalido("la imagen debe ser PNG o JPEG")
    Image.MAX_IMAGE_PIXELS = MAX_PIXELES_IMG
    try:
        img = Image.open(io.BytesIO(datos))
        if img.format not in ("PNG", "JPEG"):
            raise ArchivoInvalido("la imagen debe ser PNG o JPEG")
        if img.width * img.height > MAX_PIXELES_IMG or min(img.size) < 32:
            raise ArchivoInvalido("tamaño de imagen fuera de rango")
        img.verify()                                   # integridad sin decodificar todo
        img = Image.open(io.BytesIO(datos))
        img.load()
    except ArchivoInvalido:
        raise
    except Exception:                                  # noqa: BLE001 - cualquier fallo del decodificador
        raise ArchivoInvalido("la imagen está dañada o no es válida")
    return img
