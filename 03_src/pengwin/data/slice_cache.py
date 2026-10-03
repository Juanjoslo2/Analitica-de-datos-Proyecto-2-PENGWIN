"""slice_cache.py.

Caché de cortes listos para el modelo [EDA §2-4, DD §1]. Cada caso se lee UNA vez
(el .mha completo pesa 40-430 MB) y queda en ``<processed_dir>/<case_id>/``:

    image.npy  uint8 (Z, 256, 256)  ventana L400/W1800 cuantizada (≈ 7 HU por nivel)
    label.npy  uint8 (Z, 256, 256)  ids PENGWIN 0..30, vecino más cercano (no inventa ids)
    edge.npy   uint8 (Z, 256, 256)  borde de fractura 3D dilatado (objetivo de la salida de borde)
    meta.json  spacing nativo, recorte óseo, mm/px efectivo, Δ de contexto, cortes con hueso

Los .npy van sin comprimir para abrirlos con ``mmap_mode="r"``: el ``Dataset`` lee solo
los cortes que pide y 100 casos no tienen que caber en RAM.

Geometría (necesaria para volver al espacio nativo y medir en mm en la semana 10):
el corte nativo (Y, X) se recorta al cuadrado ``bone_crop`` (puede salirse del lienzo; se
rellena con aire / fondo) y se redimensiona a 256 con ``grid_mode=True``, es decir,
convención de área de píxel:  nativo = recorte0 + (u + 0,5) · lado / 256 − 0,5.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy import ndimage as ndi

from pengwin.data.data_loader import apply_bone_window, load_and_standardize_mha
from pengwin.data.preprocessing import HU_AIR, BodyCrop, apply_crop, compute_bone_crop
from pengwin.data.targets import fracture_edge_3d

CACHE_VERSION = 1
EDGE_DILATION = 2     # iteraciones de dilatación 3D del borde (oráculo en test: 71 % de secundarios separables)
CHUNK_Z = 48          # cortes por bloque al redimensionar (acota la RAM en casos de 400+ cortes)


def _resize_stack(stack: np.ndarray, out_size: int, order: int) -> np.ndarray:
    """(Z, S, S) -> (Z, out, out) solo en el plano axial; ``order`` 1 imagen, 0 etiqueta."""
    s = stack.shape[-1]
    zoom = (1.0, out_size / s, out_size / s)
    out = np.empty((stack.shape[0], out_size, out_size), stack.dtype)
    for z0 in range(0, stack.shape[0], CHUNK_Z):
        out[z0:z0 + CHUNK_Z] = ndi.zoom(stack[z0:z0 + CHUNK_Z], zoom, order=order, grid_mode=True,
                                        mode="nearest", prefilter=False)
    return out


def context_offset(dz_mm: float, context_mm: float) -> int:
    """Δ en cortes para la entrada 2.5D (z−Δ, z, z+Δ): Δ = round(context_mm / dz), mínimo 1."""
    return max(1, int(round(context_mm / dz_mm)))


def build_case_cache(
    case_id: str,
    image_path: Path | str,
    label_path: Path | str | None,
    out_dir: Path | str,
    image_size: int = 256,
    window: Tuple[float, float] = (400.0, 1800.0),
    context_mm: float = 2.0,
    crop_margin_mm: float = 15.0,
) -> Dict:
    """Preprocesa un caso y escribe ``image.npy``, ``label.npy`` y ``meta.json``.

    El recorte óseo se calcula SOLO con la imagen (``compute_bone_crop``), igual que en
    inferencia; la etiqueta se recorta con el mismo cuadrado.
    """
    out = Path(out_dir) / case_id
    out.mkdir(parents=True, exist_ok=True)

    hu, spacing, orig = load_and_standardize_mha(image_path)            # float32 (Z, Y, X), LPS
    crop = compute_bone_crop(hu, spacing, margin_mm=crop_margin_mm)
    img = apply_bone_window(apply_crop(hu, crop, fill=HU_AIR), window[0], window[1])
    native_shape = hu.shape
    del hu
    img = _resize_stack(img, image_size, order=1)
    np.save(out / "image.npy", np.round(np.clip(img, 0.0, 1.0) * 255).astype(np.uint8))
    del img

    meta: Dict = {
        "cache_version": CACHE_VERSION, "case_id": case_id, "orientation_orig": orig,
        "native_shape_zyx": list(native_shape), "spacing_zyx": [float(s) for s in spacing],
        "crop": {"y0": crop.y0, "y1": crop.y1, "x0": crop.x0, "x1": crop.x1, "side_px": crop.side_px},
        "image_size": image_size, "pixel_mm": crop.pixel_mm_at(image_size),
        "window": list(window), "context_offset": context_offset(spacing[0], context_mm),
    }
    if label_path is not None:
        lab, _, _ = load_and_standardize_mha(label_path, is_label=True)
        if lab.shape != native_shape:
            raise ValueError(f"{case_id}: imagen {native_shape} y etiqueta {lab.shape} no coinciden")
        lab = _resize_stack(apply_crop(lab, crop, fill=0), image_size, order=0)
        np.save(out / "label.npy", lab)
        np.save(out / "edge.npy", fracture_edge_3d(lab, EDGE_DILATION).astype(np.uint8))
        meta["edge_dilation"] = EDGE_DILATION
        meta["bone_slices"] = np.nonzero(lab.reshape(lab.shape[0], -1).max(1) > 0)[0].tolist()
        meta["labels_present"] = [int(v) for v in np.unique(lab) if v > 0]

    tmp = out / "meta.tmp"
    tmp.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    tmp.replace(out / "meta.json")      # meta.json se escribe al final: su existencia marca el caso como hecho
    return meta


def load_case_cache(case_dir: Path | str, mmap: bool = True) -> Tuple[np.ndarray, np.ndarray | None, Dict]:
    """Abre un caso del caché: (image uint8, label uint8 o None, meta)."""
    case_dir = Path(case_dir)
    meta = json.loads((case_dir / "meta.json").read_text(encoding="utf-8"))
    mode = "r" if mmap else None
    image = np.load(case_dir / "image.npy", mmap_mode=mode)
    label = np.load(case_dir / "label.npy", mmap_mode=mode) if (case_dir / "label.npy").exists() else None
    return image, label, meta


def add_edge_cache(case_dir: Path | str, dilation: int = 2) -> int:
    """Añade ``edge.npy`` (borde de fractura 3D) a un caso ya cacheado, sin releer el .mha.

    Devuelve el número de vóxeles de borde. Los cachés de la semana 9 no lo tienen.
    """
    case_dir = Path(case_dir)
    label = np.load(case_dir / "label.npy")
    edge = fracture_edge_3d(label, dilation).astype(np.uint8)
    np.save(case_dir / "edge.npy", edge)
    meta = json.loads((case_dir / "meta.json").read_text(encoding="utf-8"))
    meta["edge_dilation"] = dilation
    (case_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return int(edge.sum())


def load_edge_cache(case_dir: Path | str, mmap: bool = True) -> np.ndarray | None:
    path = Path(case_dir) / "edge.npy"
    return np.load(path, mmap_mode="r" if mmap else None) if path.exists() else None


def crop_from_meta(meta: Dict) -> BodyCrop:
    c = meta["crop"]
    sp = meta["spacing_zyx"]
    return BodyCrop(c["y0"], c["y1"], c["x0"], c["x1"], c["side_px"], (sp[1], sp[2]))


def model_to_native_xy(xy: np.ndarray, meta: Dict) -> np.ndarray:
    """Coordenadas continuas (x, y) en la entrada 256 -> índices (x, y) del corte nativo LPS.

    Usa la misma convención de área de píxel que ``_resize_stack``; sirve para llevar
    cajas y contornos al volumen nativo, donde se mide en mm.
    """
    c = meta["crop"]
    scale = c["side_px"] / meta["image_size"]
    xy = np.asarray(xy, np.float64)
    return np.stack([c["x0"] + xy[..., 0] * scale, c["y0"] + xy[..., 1] * scale], axis=-1)
