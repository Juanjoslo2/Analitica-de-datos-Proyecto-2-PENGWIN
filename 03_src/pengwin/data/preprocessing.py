"""preprocessing.py.

Operaciones clásicas de preprocesado que NO dependen del modelo y que se
comparten entre EDA, caché de cortes, visualizador 1 (MIP) e inferencia.

Convención de ejes: todos los arreglos vienen de ``load_case`` / ``load_and_standardize_mha``,
es decir (Z, Y, X) en orientación LPS:
    x crece hacia el lado IZQUIERDO del paciente,
    y crece hacia POSTERIOR,
    z crece hacia SUPERIOR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import ndimage as ndi

# Umbrales en Unidades Hounsfield (HU) usados en todo el proyecto.
HU_AIR = -1024.0          # valor mínimo físico; lo que está por debajo es relleno del escáner
HU_BODY = -500.0          # separa aire (< -500) de tejido del paciente
HU_BONE_MIP = 250.0       # umbral óseo del visualizador 1 (validado en el EDA)
HU_METAL = 3000.0         # por encima: implantes / material quirúrgico (criterio volumétrico del EDA)


@dataclass(frozen=True)
class BodyCrop:
    """Recorte cuadrado en el plano axial que contiene al paciente sin la camilla.

    Los índices están en píxeles del volumen nativo reorientado (Y, X).
    ``side_px`` es el lado del cuadrado; ``pixel_mm_at(256)`` da el tamaño
    efectivo del píxel tras redimensionar ese cuadrado a 256×256.
    """

    y0: int
    y1: int
    x0: int
    x1: int
    side_px: int
    spacing_yx_mm: Tuple[float, float]

    def pixel_mm_at(self, out_size: int) -> float:
        return self.side_px * max(self.spacing_yx_mm) / out_size


def body_mask_2d(volume_hu: np.ndarray, threshold_hu: float = HU_BODY) -> np.ndarray:
    """Silueta axial del paciente (proyección en Z), sin camilla ni objetos sueltos.

    Se proyecta ``HU > threshold`` a lo largo de Z y se conserva la componente
    conexa 2D más grande. La camilla y los cables quedan como componentes
    separadas y se descartan. Se calcula SOLO con la imagen (nunca con la etiqueta),
    por lo que es válido también en inferencia.
    """
    proj = (volume_hu > threshold_hu).any(axis=0)
    # Apertura morfológica: corta puentes finos (sábanas, bordes de camilla)
    proj = ndi.binary_opening(proj, structure=np.ones((5, 5), bool))
    lab, n = ndi.label(proj)
    if n == 0:
        return proj
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return ndi.binary_fill_holes(lab == sizes.argmax())


def body_mask_3d(volume_hu: np.ndarray, threshold_hu: float = HU_BODY) -> np.ndarray:
    """Máscara 3D del cuerpo = (HU > umbral) ∩ silueta 2D del paciente."""
    return (volume_hu > threshold_hu) & body_mask_2d(volume_hu, threshold_hu)[None]


def compute_body_crop(
    volume_hu: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
    margin_mm: float = 10.0,
) -> BodyCrop:
    """Cuadrado centrado en la silueta del paciente (+ margen), recortado al lienzo."""
    mask = body_mask_2d(volume_hu)
    _, h, w = volume_hu.shape
    ys, xs = np.where(mask)
    if ys.size == 0:
        return BodyCrop(0, h, 0, w, max(h, w), (spacing_zyx[1], spacing_zyx[2]))
    margin = int(round(margin_mm / min(spacing_zyx[1], spacing_zyx[2])))
    y0, y1 = ys.min() - margin, ys.max() + 1 + margin
    x0, x1 = xs.min() - margin, xs.max() + 1 + margin
    side = max(y1 - y0, x1 - x0)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    y0, x0 = cy - side // 2, cx - side // 2
    # El cuadrado puede salirse del lienzo: se guarda tal cual y se rellena con aire al recortar
    return BodyCrop(int(y0), int(y0 + side), int(x0), int(x0 + side), int(side),
                    (float(spacing_zyx[1]), float(spacing_zyx[2])))


def apply_crop(slice_or_volume: np.ndarray, crop: BodyCrop, fill: float = HU_AIR) -> np.ndarray:
    """Aplica un ``BodyCrop`` a un corte (Y, X) o volumen (Z, Y, X), rellenando fuera del lienzo."""
    arr = slice_or_volume
    h, w = arr.shape[-2:]
    out_shape = arr.shape[:-2] + (crop.side_px, crop.side_px)
    out = np.full(out_shape, fill, dtype=arr.dtype)
    sy0, sy1 = max(crop.y0, 0), min(crop.y1, h)
    sx0, sx1 = max(crop.x0, 0), min(crop.x1, w)
    out[..., sy0 - crop.y0:sy1 - crop.y0, sx0 - crop.x0:sx1 - crop.x0] = arr[..., sy0:sy1, sx0:sx1]
    return out


def compute_bone_crop(
    volume_hu: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
    margin_mm: float = 15.0,
    bone_threshold_hu: float = HU_BONE_MIP,
    min_thickness_mm: float = 10.0,
    attach_mm: float = 30.0,
) -> BodyCrop:
    """Cuadrado alrededor del ESQUELETO (no de la piel), calculado solo con la imagen.

    Se cuenta, por cada píxel (y, x), cuántos mm de hueso (HU ≥ umbral) atraviesa la
    columna en Z; se conservan los píxeles con ≥ ``min_thickness_mm`` dentro de la
    silueta del paciente. Así se ignoran calcificaciones y contraste aislados.
    Da mejor resolución efectiva que ``compute_body_crop`` porque la pelvis ocupa
    ~70 % del ancho del cuerpo. Solo se conservan las componentes 2D grandes
    (esqueleto axial) y lo que esté a <= 30 mm de él, para que las manos sobre el
    abdomen no agranden el recorte sin cortar fragmentos pélvicos alejados.
    """
    body = body_mask_2d(volume_hu)
    thick = (volume_hu >= bone_threshold_hu).sum(axis=0) * spacing_zyx[0]
    mask = (thick >= min_thickness_mm) & body
    # Manos/antebrazos apoyados sobre el abdomen agrandarían el recorte. Se conservan:
    #   (1) las componentes >= 25 % de la mayor (la pelvis puede proyectarse en 2 mitades
    #       si hay una fractura o una articulación sacroilíaca abierta), y
    #   (2) cualquier componente a <= ``attach_mm`` de ellas (ramas púbicas, cóccix).
    lab, n = ndi.label(mask)
    if n > 1:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        core = np.isin(lab, np.nonzero(sizes >= 0.25 * sizes.max())[0])
        px = min(spacing_zyx[1], spacing_zyx[2])
        near = ndi.binary_dilation(core, iterations=max(1, int(round(attach_mm / px))))
        keep = np.unique(lab[near & (lab > 0)])
        mask = np.isin(lab, keep)
    _, h, w = volume_hu.shape
    ys, xs = np.where(mask)
    if ys.size == 0:
        return compute_body_crop(volume_hu, spacing_zyx)
    margin = int(round(margin_mm / min(spacing_zyx[1], spacing_zyx[2])))
    y0, y1 = ys.min() - margin, ys.max() + 1 + margin
    x0, x1 = xs.min() - margin, xs.max() + 1 + margin
    side = max(y1 - y0, x1 - x0)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    y0, x0 = cy - side // 2, cx - side // 2
    return BodyCrop(int(y0), int(y0 + side), int(x0), int(x0 + side), int(side),
                    (float(spacing_zyx[1]), float(spacing_zyx[2])))
