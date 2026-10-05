"""dataset.py. Dataset de detección propio: carga los .mha directamente, sin caché.

Punto de vista distinto al pipeline compartido: en vez de pre-generar cortes
256×256 en disco (caché), cada caso se lee con SimpleITK una vez por época y se
recorta/redimensiona en memoria. Más simple y sin estado en disco; el coste es
lectura de disco por época (aceptable para la semana 9: overfit y entrenamientos
cortos).

Cada muestra (corte z del caso) devuelve:
    image    (3, 256, 256)  cortes (z−Δ, z, z+Δ) con ventana ósea en [0, 1]
    boxes    (3, 4)         envolvente por región en píxeles de la imagen 256
    present  (3,)           regiones presentes en el corte
    ignore   (3,)           cajas con lado < det_min_box_px (no evaluables)

El contexto Δ = round(2 mm / dz) [EDA §2]; en los bordes de la pila se repite el
corte extremo (a diferencia del diseño compartido, que lanzaba RuntimeError).
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy import ndimage as ndi

from pengwin.data.data_loader import load_and_standardize_mha
from pengwin.data.preprocessing import HU_AIR, HU_METAL, apply_crop, compute_bone_crop

REGION_IDS = (1, 2, 3)                                   # SA, coxal izq., coxal der.
REGION_NAMES = ("SA", "LI", "RI")


def read_split(path: Path, name: str) -> List[str]:
    """Lee ``splits.json`` (v2) y devuelve la lista de IDs del split pedido."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [str(x) for x in data["splits"][name]]


def region_boxes_2d(label: np.ndarray, min_box_px: float = 4.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Caja envolvente por región anatómica en un corte (Y, X) de etiquetas.

    Una región puede aparecer partida en islas (28–42 % de cortes [EDA §8]); la
    caja GT es la envolvente de TODAS sus islas. Devuelve:
        boxes   (3, 4) x0, y0, x1, y1  (x1 = última columna + 1, coordenadas continuas)
        present (3,)  región presente
        ignore  (3,)  lado < min_box_px → caja no evaluable [DD §2]
    """
    boxes = np.zeros((3, 4), float)
    present = np.zeros(3, bool)
    ignore = np.zeros(3, bool)
    for k, r in enumerate(REGION_IDS):
        ys, xs = np.nonzero(((label - 1) // 10 + 1 == r) & (label > 0))
        if ys.size == 0:
            continue
        present[k] = True
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        side = min(y1 - y0, x1 - x0)
        boxes[k] = (x0, y0, x1, y1)
        ignore[k] = side < min_box_px
    return boxes, present, ignore


class DetSlices(torch.utils.data.Dataset):
    """Cortes con hueso + fracción de cortes vacíos (``empty_slice_fraction``)."""

    def __init__(self, raw_dir: Path, case_ids: List[str], cfg: Dict, train: bool = True,
                 slices: List[Tuple[str, int]] | None = None, max_cases: int | None = None):
        self.raw_dir = Path(raw_dir)
        self.cfg = cfg
        self.train = train
        d = cfg["data"]
        self.image_size = int(d.get("image_size", 256))
        self.context_mm = float(d.get("context_mm", 2.0))
        self.min_box_px = float(cfg["model"].get("det_min_box_px", 4.0))
        self.empty_frac = float(d.get("empty_slice_fraction", 0.12))
        self._cache: "OrderedDict[str, Dict]" = OrderedDict()

        if slices is not None:
            self.index = list(slices)
            return
        if max_cases is not None:
            case_ids = list(case_ids)[:max_cases]
        # Índice: cortes con hueso por caso + cortes vacíos deterministas.
        # Se lee SOLO la máscara (rápido); la imagen se carga bajo demanda en
        # __getitem__ (cache LRU), que es donde hace falta el recorte óseo.
        self.index: List[Tuple[str, int]] = []
        rng = np.random.default_rng(int(cfg.get("seed", 42)))
        for cid in case_ids:
            _, mask, _ = self._load_mask_only(cid)
            bone = np.where(mask.any(axis=(1, 2)))[0]
            empty = np.where(~mask.any(axis=(1, 2)))[0]
            for z in bone:
                self.index.append((cid, int(z)))
            n_empty = min(len(empty), int(round(len(bone) * self.empty_frac / max(1 - self.empty_frac, 1e-6))))
            if n_empty:
                for z in rng.choice(empty, size=n_empty, replace=False):
                    self.index.append((cid, int(z)))
        self.index.sort()

    def _load_mask_only(self, case_id: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        lab_dir = self.raw_dir / "PENGWIN_CT_train_labels"
        mask, _, _ = load_and_standardize_mha(lab_dir / f"{case_id}.mha", is_label=True)
        return None, mask, {"dz": 0.0}

    def _load_case(self, case_id: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        if case_id in self._cache:
            return self._cache[case_id]["img"], self._cache[case_id]["mask"], self._cache[case_id]["meta"]
        img_dir = self.raw_dir / "PENGWIN_CT_train_images_part1"
        if not (img_dir / f"{case_id}.mha").exists():
            img_dir = self.raw_dir / "PENGWIN_CT_train_images_part2"
        img_path = img_dir / f"{case_id}.mha"
        lab_path = self.raw_dir / "PENGWIN_CT_train_labels" / f"{case_id}.mha"
        img, spacing, _ = load_and_standardize_mha(img_path, image_dtype=np.float32)
        mask, _, _ = load_and_standardize_mha(lab_path, is_label=True)
        crop = compute_bone_crop(img, spacing, margin_mm=float(self.cfg["data"].get("crop_margin_mm", 15)))
        img_c = apply_crop(img, crop, fill=HU_AIR).astype(np.float32)
        mask_c = apply_crop(mask, crop, fill=0).astype(np.uint8)
        dz = spacing[0]
        meta = {"crop": crop, "dz": dz, "context_offset": max(1, int(round(self.context_mm / dz)))}
        self._cache[case_id] = {"img": img_c, "mask": mask_c, "meta": meta}
        while len(self._cache) > 4:                          # caché en RAM pequeña (LRU)
            self._cache.popitem(last=False)
        return img_c, mask_c, meta

    def _slice(self, vol: np.ndarray, z: int, off: int) -> np.ndarray:
        z0, z1 = z - off, z + off
        z0, z1 = max(z0, 0), min(z1, vol.shape[0] - 1)      # bordes: repetir extremo
        return vol[z0], vol[z], vol[z1]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        case_id, z = self.index[idx]
        img_c, mask_c, meta = self._load_case(case_id)
        d = self.cfg["data"]
        level, width = float(d["window"]["level"]), float(d["window"]["width"])
        lo, hi = float(d["hu_clip"][0]), float(d["hu_clip"][1])
        scale = self.image_size / meta["crop"].side_px

        stack = [self._slice(img_c, z, meta["context_offset"])[i] for i in range(3)]
        out = np.empty((3, self.image_size, self.image_size), np.float32)
        for i, s in enumerate(stack):
            s = np.clip(s, lo, hi)
            s = (s - (level - width / 2)) / width
            out[i] = ndi.zoom(s, scale, order=1).clip(0.0, 1.0)
        mask256 = ndi.zoom(mask_c[z], scale, order=0).astype(np.uint8)
        boxes, present, ignore = region_boxes_2d(mask256, self.min_box_px)
        return {
            "image": torch.from_numpy(out),
            "boxes": torch.from_numpy(boxes),
            "present": torch.from_numpy(present.astype(np.float32)),
            "ignore": torch.from_numpy(ignore.astype(np.float32)),
            "case_id": case_id,
            "z": int(z),
        }