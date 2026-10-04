"""dataset.py.

``Dataset`` de PyTorch corte a corte con contexto 2.5D [EDA §2, §8; DD §1].

Cada muestra es un corte axial z del caché con sus vecinos (z−Δ, z, z+Δ) como 3 canales,
Δ = round(2 mm / dz) guardado por caso en ``meta.json``. Los 3 canales son además
compatibles con la primera capa de FundidoraPC (``in_channels=3``).

Muestreo: todos los cortes con hueso + una fracción fija de cortes vacíos
(``empty_slice_fraction`` ≈ 12 % del total) para que la cabeza de clasificación vea
negativos [EDA §8]. La selección de vacíos es determinista (semilla de la config).

Aumentación (solo entrenamiento): afín pequeña (escala, rotación, traslación) aplicada a
imagen y etiqueta a la vez + brillo/contraste/gamma. Nunca flip horizontal: intercambiaría
coxal izquierdo y derecho [DD §1]. Los objetivos se calculan DESPUÉS de aumentar.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from pengwin.data.slice_cache import load_case_cache, load_edge_cache
from pengwin.data.targets import slice_targets


def read_split(splits_path: Path | str, name: str) -> List[str]:
    return list(json.loads(Path(splits_path).read_text(encoding="utf-8"))["splits"][name])


def context_stack(image: np.ndarray, z: int, delta: int) -> np.ndarray:
    """Cortes (z−Δ, z, z+Δ) con el borde del volumen repetido -> (3, H, W)."""
    zmax = image.shape[0] - 1
    return np.stack([image[min(max(z + k * delta, 0), zmax)] for k in (-1, 0, 1)], axis=0)


def _random_affine(img: torch.Tensor, lab: torch.Tensor, rng: np.random.Generator, aug: Dict):
    """Afín aleatoria común a imagen (bilineal) y mapas enteros (vecino más cercano).

    ``lab`` puede ser (H, W) o (K, H, W): etiqueta y borde se transforman juntos.
    """
    s = rng.uniform(*aug.get("scale", (0.9, 1.1)))
    a = math.radians(rng.uniform(-aug.get("rotate_deg", 10), aug.get("rotate_deg", 10)))
    tx, ty = rng.uniform(-aug.get("translate", 0.05), aug.get("translate", 0.05), size=2) * 2
    theta = torch.tensor([[math.cos(a) / s, -math.sin(a) / s, tx],
                          [math.sin(a) / s, math.cos(a) / s, ty]], dtype=torch.float32)[None]
    grid = F.affine_grid(theta, (1, 1, *img.shape[-2:]), align_corners=False)
    img = F.grid_sample(img[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
    stack = lab[None] if lab.dim() == 2 else lab
    out = F.grid_sample(stack[None].float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)[0]
    return img, (out[0] if lab.dim() == 2 else out).to(torch.uint8)


def _random_intensity(img: torch.Tensor, rng: np.random.Generator, aug: Dict) -> torch.Tensor:
    g = rng.uniform(*aug.get("gamma", (0.8, 1.25)))
    c = rng.uniform(*aug.get("contrast", (0.9, 1.1)))
    b = rng.uniform(-aug.get("brightness", 0.05), aug.get("brightness", 0.05))
    return ((img.clamp(0, 1) ** g) * c + b).clamp(0, 1)


class PengwinSlices(Dataset):
    """Cortes 2.5D de los casos ``case_ids`` leídos del caché (``scripts/build_slice_cache.py``)."""

    def __init__(
        self,
        cache_dir: Path | str,
        case_ids: Sequence[str],
        cfg: Dict,
        train: bool = False,
        augment: bool | None = None,
        slices: Sequence[tuple] | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cfg = cfg
        self.train = train
        self.augment = train if augment is None else augment
        self.min_box_px = float(cfg["model"].get("det_min_box_px", 4))
        self.edge_dilation = int(cfg["loss"].get("edge_dilation_px", 2))
        self.aug_cfg = cfg.get("augment", {})
        self.edge_file = cfg["data"].get("edge_file", "edge.npy")
        # Solo se guardan los meta: los .npy se abren con mmap de forma perezosa en cada
        # proceso (en Windows los workers se crean con spawn y un memmap se copiaría entero).
        self.meta = {}
        for cid in case_ids:
            meta = json.loads((self.cache_dir / cid / "meta.json").read_text(encoding="utf-8"))
            if "bone_slices" not in meta:
                raise ValueError(f"{cid}: el caché no tiene etiqueta (¿caso de inferencia?)")
            self.meta[cid] = meta
        self._arrays: Dict[str, tuple] = {}

        if slices is not None:                     # lista explícita (prueba de overfit)
            self.index = [(str(c), int(z)) for c, z in slices]
        else:
            self.index = self._build_index(float(cfg["data"].get("empty_slice_fraction", 0.12)), int(cfg.get("seed", 42)))
            k = int(cfg["data"].get("secondary_oversample", 1))
            if train and k > 1:
                self.index = self._oversample_secondary(self.index, k)
        self._epoch = 0

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_arrays"] = {}
        return state

    def _case(self, cid: str):
        if cid not in self._arrays:
            image, label, _ = load_case_cache(self.cache_dir / cid)
            self._arrays[cid] = (image, label, load_edge_cache(self.cache_dir / cid, name=self.edge_file))
        return self._arrays[cid]

    def _build_index(self, empty_fraction: float, seed: int) -> List[tuple]:
        rng = np.random.default_rng(seed)
        index = []
        for cid, meta in self.meta.items():
            bone = set(meta["bone_slices"])
            empty = [z for z in range(meta["native_shape_zyx"][0]) if z not in bone]
            n_empty = min(len(empty), int(round(len(bone) * empty_fraction / (1 - empty_fraction))))
            chosen = rng.choice(empty, size=n_empty, replace=False).tolist() if n_empty else []
            index += [(cid, z) for z in sorted(bone)] + [(cid, int(z)) for z in sorted(chosen)]
        return index

    def _oversample_secondary(self, index: List[tuple], k: int) -> List[tuple]:
        """Repite ``k`` veces los cortes que contienen algún fragmento secundario (id 2-10 dentro de
        su región). Los secundarios ocupan poco del total de cortes y son los que más se pierden."""
        extra = []
        for cid in self.meta:
            label = load_case_cache(self.cache_dir / cid)[1]
            frag = (label.reshape(label.shape[0], -1).astype(np.int16) - 1) % 10
            has_sec = ((frag >= 1) & (label.reshape(label.shape[0], -1) > 0)).any(1)
            extra += [(cid, int(z)) for z in np.nonzero(has_sec)[0]] * (k - 1)
        return index + extra

    def set_epoch(self, epoch: int) -> None:
        """Cambia la semilla de la aumentación por época (reproducible y distinta en cada época)."""
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        cid, z = self.index[i]
        image, label, edge3d = self._case(cid)
        meta = self.meta[cid]
        img = torch.from_numpy(context_stack(image, z, meta["context_offset"]).astype(np.float32) / 255.0)
        lab = torch.from_numpy(np.array(label[z], dtype=np.uint8))
        edge = torch.from_numpy(np.array(edge3d[z], dtype=np.uint8)) if edge3d is not None else None

        if self.augment:
            rng = np.random.default_rng((int(self.cfg.get("seed", 42)), self._epoch, i))
            if edge is None:
                img, lab = _random_affine(img, lab, rng, self.aug_cfg)
            else:
                img, both = _random_affine(img, torch.stack([lab, edge]), rng, self.aug_cfg)
                lab, edge = both[0], both[1]
            img = _random_intensity(img, rng, self.aug_cfg)

        # borde 3D del caché si existe (semana 10); si no, el 2D calculado del corte (semana 9)
        t = slice_targets(lab.numpy(), self.min_box_px, self.edge_dilation, None if edge is None else edge.numpy())
        return {
            "image": img,
            "semantic": torch.from_numpy(t["semantic"]),
            "edge": torch.from_numpy(t["edge"])[None],
            "boxes": torch.from_numpy(t["boxes"]),
            "present": torch.from_numpy(t["present"].astype(np.float32)),
            "ignore": torch.from_numpy(t["ignore"]),
            "case_id": cid,
            "z": z,
            "pixel_mm": float(meta["pixel_mm"]),
        }
