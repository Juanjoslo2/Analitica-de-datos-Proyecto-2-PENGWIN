"""two_pass.py.

Segunda pasada por hueso a resolución nativa [y4xul, refinamiento por región].

El modelo es UNO y se ejecuta dos veces sobre el mismo backbone y las mismas tres cabezas:

    pasada 1   corte completo a 256 px (1,2-1,6 mm/px); el 4.º canal de entrada va en cero.
               Da clasificación, cajas y la región de cada hueso.
    pasada 2   por cada hueso presente, un recorte de 256 px tomado del caché de alta resolución
               (512 px, ~0,7 mm/px) centrado en ese hueso; el 4.º canal lleva la máscara que la
               pasada 1 predijo para ESE hueso. Da el papel (principal / secundario) y el borde
               con el doble de resolución, que es donde una grieta de 1-2 px deja de ser invisible.

Fuga de datos: en entrenamiento la máscara previa NO es el ground truth. Sale de predicciones
fuera de fold (``scripts/build_prior.py``): cada caso lo predijo un modelo que no lo vio. Así la
segunda pasada aprende con máscaras tan imperfectas como las que recibirá en inferencia.

Geometría. El caché de alta resolución es el mismo recorte óseo remuestreado a ``hi`` px, así que
un píxel (x, y) de la grilla de 256 es el bloque [s·x, s·x + s) de la de ``hi``, con s = hi / 256.
Las ventanas se fuerzan a coordenadas múltiplos de s para que pegar de vuelta sea exacto.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from pengwin.data.dataset import PengwinSlices, _random_affine, _random_intensity, context_stack
from pengwin.data.slice_cache import load_case_cache
from pengwin.data.targets import NUM_REGIONS, slice_targets

PRIOR_SEM = "prior_sem.npy"          # (Z, 256, 256) uint8: región predicha fuera de fold
PRIOR_BOXES = "prior_boxes.npy"      # (Z, 3, 4) int16: caja de cada región en la grilla de 256
ROI_MARGIN = 8                       # px de la grilla de 256 alrededor de la caja
ROI_MIN_PX = 20                      # una región con menos píxeles en el corte no se refina


def prior_boxes(sem: np.ndarray, min_px: int = ROI_MIN_PX) -> np.ndarray:
    """Caja (x0, y0, x1, y1) de cada región en cada corte de ``sem`` (Z, H, W); ceros si no está."""
    out = np.zeros((sem.shape[0], NUM_REGIONS, 4), np.int16)
    for r in range(1, NUM_REGIONS + 1):
        m = sem == r
        zs = np.nonzero(m.reshape(len(m), -1).sum(1) >= min_px)[0]
        for z in zs:
            ys, xs = np.nonzero(m[z])
            out[z, r - 1] = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    return out


def roi_window(box: Sequence[float], scale: int, out: int = 256, margin: int = ROI_MARGIN,
               shift: Tuple[float, float] = (0.0, 0.0), zoom: float = 1.0) -> Tuple[int, int, int]:
    """Ventana cuadrada (x0, y0, lado) en la grilla de alta resolución que contiene ``box``.

    ``box`` está en la grilla de 256. El lado es al menos ``out`` (recorte 1:1, resolución nativa) y
    crece si el hueso no cabe; ``shift`` (px de 256) y ``zoom`` son la aumentación. Todo se redondea
    a múltiplos de ``scale`` para que cada píxel de 256 caiga entero dentro o fuera.
    """
    x0, y0, x1, y1 = (float(v) for v in box)
    lado = max(out / scale, (max(x1 - x0, y1 - y0) + 2 * margin)) * zoom
    lado = int(np.ceil(lado))
    cx, cy = (x0 + x1) / 2 + shift[0], (y0 + y1) / 2 + shift[1]
    return int(round(cx - lado / 2)) * scale, int(round(cy - lado / 2)) * scale, lado * scale


def crop_square(a: np.ndarray, x0: int, y0: int, side: int) -> np.ndarray:
    """Recorte [y0:y0+side, x0:x0+side] de ``a`` (..., H, W) con relleno de ceros fuera del lienzo."""
    H, W = a.shape[-2:]
    out = np.zeros(a.shape[:-2] + (side, side), a.dtype)
    sy0, sy1, sx0, sx1 = max(y0, 0), min(y0 + side, H), max(x0, 0), min(x0 + side, W)
    if sy1 > sy0 and sx1 > sx0:
        out[..., sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = a[..., sy0:sy1, sx0:sx1]
    return out


def resize(t: torch.Tensor, out: int, nearest: bool) -> torch.Tensor:
    """(C, h, w) -> (C, out, out). Bilineal para imagen y probabilidades, vecino para mapas enteros."""
    if t.shape[-1] == out and t.shape[-2] == out:
        return t
    x = t[None].float()
    y = F.interpolate(x, size=(out, out), mode="nearest") if nearest else \
        F.interpolate(x, size=(out, out), mode="bilinear", align_corners=False, antialias=out < t.shape[-1])
    return y[0].to(t.dtype) if nearest else y[0]


def paste_window(dst: np.ndarray, patch: torch.Tensor, x0: int, y0: int, side: int, scale: int,
                 where: np.ndarray | None = None) -> None:
    """Devuelve ``patch`` (h, w), que cubre la ventana (x0, y0, side) de alta resolución, a la grilla
    de 256 de ``dst`` (H, W): promedio por bloque y pegado donde ``where`` es verdadero."""
    lo = side // scale
    p = F.adaptive_avg_pool2d(patch[None, None].float(), (lo, lo))[0, 0].cpu().numpy()
    X0, Y0 = x0 // scale, y0 // scale
    H, W = dst.shape
    sy0, sy1, sx0, sx1 = max(Y0, 0), min(Y0 + lo, H), max(X0, 0), min(X0 + lo, W)
    if sy1 <= sy0 or sx1 <= sx0:
        return
    src = p[sy0 - Y0:sy1 - Y0, sx0 - X0:sx1 - X0]
    if where is None:
        dst[sy0:sy1, sx0:sx1] = src
    else:
        w = where[sy0:sy1, sx0:sx1]
        dst[sy0:sy1, sx0:sx1][w] = src[w]


def roi_input(hi_image: np.ndarray, z: int, delta: int, prior_r: np.ndarray, win: Tuple[int, int, int],
              scale: int, out: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    """Entrada de la pasada 2: (3, out, out) float en [0, 1] y la máscara previa (out, out) uint8.

    ``prior_r`` es la máscara binaria del hueso en la grilla de 256. Se usa igual en
    entrenamiento y en inferencia, así las dos ven exactamente la misma construcción.
    """
    x0, y0, side = win
    img = torch.from_numpy(crop_square(context_stack(hi_image, z, delta), x0, y0, side).astype(np.float32) / 255.0)
    pri = torch.from_numpy(crop_square(prior_r.astype(np.uint8), x0 // scale, y0 // scale, side // scale))
    return resize(img, out, nearest=False), resize(pri[None], out, nearest=True)[0]


class TwoPassSlices(PengwinSlices):
    """Cortes completos (pasada 1) + recortes por hueso a alta resolución (pasada 2), mezclados.

    Cada época usa todos los cortes completos y el mismo número de recortes, elegidos al azar entre
    todas las ternas (caso, corte, hueso) con máscara previa. El modelo ve las dos tareas en cada
    lote y conserva la detección y la clasificación de la pasada 1.
    """

    def __init__(self, cache_dir: Path | str, hi_cache_dir: Path | str, case_ids: Sequence[str], cfg: Dict,
                 train: bool = True, augment: bool | None = None, roi_per_slice: float = 1.0):
        super().__init__(cache_dir, case_ids, cfg, train=train, augment=augment)
        self.hi_dir = Path(hi_cache_dir)
        self.out = int(cfg["data"]["image_size"])
        self.hi_meta, self.boxes, rois = {}, {}, []
        for cid in self.meta:
            self.hi_meta[cid] = json.loads((self.hi_dir / cid / "meta.json").read_text(encoding="utf-8"))
            f = self.cache_dir / cid / PRIOR_BOXES
            if not f.exists():
                raise FileNotFoundError(f"{cid}: falta {PRIOR_BOXES}; créalo con scripts/build_prior.py "
                                        "(predicciones fuera de fold, nunca el ground truth)")
            b = np.load(f)
            self.boxes[cid] = b
            zs, rs = np.nonzero((b[..., 2] > b[..., 0]))
            rois += [(cid, int(z), int(r) + 1) for z, r in zip(zs, rs)]
        self.scale = int(next(iter(self.hi_meta.values()))["image_size"]) // self.out
        self.rois = rois
        self.n_roi = min(len(rois), int(round(len(self.index) * roi_per_slice)))
        self._hi: Dict[str, tuple] = {}
        self._prior: Dict[str, np.ndarray] = {}
        self.set_epoch(0)

    def __getstate__(self):
        state = super().__getstate__()
        state["_hi"], state["_prior"] = {}, {}
        return state

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        rng = np.random.default_rng((int(self.cfg.get("seed", 42)), 7919, epoch))
        self.roi_sel = rng.choice(len(self.rois), size=self.n_roi, replace=False) if self.n_roi else np.zeros(0, int)

    def __len__(self) -> int:
        return len(self.index) + self.n_roi

    def _hi_case(self, cid: str):
        if cid not in self._hi:
            image, label, _ = load_case_cache(self.hi_dir / cid)
            self._hi[cid] = (image, label)
            self._prior[cid] = np.load(self.cache_dir / cid / PRIOR_SEM, mmap_mode="r")
        return self._hi[cid] + (self._prior[cid],)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        if i < len(self.index):
            return super().__getitem__(i)
        cid, z, r = self.rois[int(self.roi_sel[i - len(self.index)])]
        meta = self.meta[cid]
        hi_image, hi_label, prior = self._hi_case(cid)
        _, _, edge3d, _ = self._case(cid)
        s = self.scale
        shift, zoom = (0.0, 0.0), 1.0
        rng = np.random.default_rng((int(self.cfg.get("seed", 42)), self._epoch, i))
        if self.augment:
            shift = tuple(rng.uniform(-10, 10, size=2))
            zoom = float(rng.uniform(0.9, 1.15))
        win = roi_window(self.boxes[cid][z, r - 1], s, self.out, shift=shift, zoom=zoom)
        x0, y0, side = win
        img, pri = roi_input(hi_image, z, meta["context_offset"], np.asarray(prior[z]) == r, win, s, self.out)
        lab = resize(torch.from_numpy(crop_square(np.asarray(hi_label[z]), x0, y0, side))[None], self.out, True)[0]
        maps: List[torch.Tensor] = [lab, pri]
        if edge3d is not None:           # el borde 3D solo existe en la grilla de 256: se amplía con vecino
            e = torch.from_numpy(crop_square(np.asarray(edge3d[z]), x0 // s, y0 // s, side // s))
            maps.append(resize(e[None], self.out, True)[0])
        if self.augment:
            img, todos, _ = _random_affine(img, torch.stack(maps), rng, self.aug_cfg)
            maps = list(todos)
            img = _random_intensity(img, rng, self.aug_cfg)
        lab, pri = maps[0], maps[1]
        t = slice_targets(lab.numpy(), self.min_box_px, self.edge_dilation, maps[2].numpy() if len(maps) > 2 else None)
        return {
            "image": torch.cat([img, pri[None].float()], 0),
            "semantic": torch.from_numpy(t["semantic"]),
            "edge": torch.from_numpy(t["edge"])[None],
            "core3": torch.from_numpy(t["core3"]),
            "role3": torch.from_numpy(t["role3"]),
            "boxes": torch.from_numpy(t["boxes"]),
            "present": torch.from_numpy(t["present"].astype(np.float32)),
            "ignore": torch.from_numpy(t["ignore"]),
            "case_id": cid,
            "z": z,
            "pixel_mm": float(self.hi_meta[cid]["pixel_mm"]) * side / self.out,
        }
