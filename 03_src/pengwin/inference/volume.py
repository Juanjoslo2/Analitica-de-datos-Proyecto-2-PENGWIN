"""volume.py.

Inferencia de un caso completo: el modelo 2D recorre todos los cortes del caché y las
salidas se apilan en un volumen (Z, 256, 256). Después ``to_native`` deshace el
redimensionado y el recorte para volver a la grilla del .mha, donde se mide en mm.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from scipy import ndimage as ndi

from pengwin.data.dataset import context_stack
from pengwin.data.slice_cache import CHUNK_Z, load_case_cache
from pengwin.detection.grid import decode


@torch.no_grad()
def predict_case(model, cache_dir: Path | str, case_id: str, cfg: Dict, device: torch.device,
                 batch_size: int = 16) -> Dict:
    """Corre el modelo sobre todos los cortes de un caso.

    Devuelve en la grilla del modelo (Z, 256, 256):
        semantic  uint8    región por píxel (0 fondo, 1 SA, 2 LI, 3 RI)
        edge      float16  probabilidad de borde de fractura
        cls       float32  (Z, 3) probabilidad de presencia de cada región
        boxes     lista de Z dicts {boxes, scores, labels} (numpy)
    y el ``meta`` del caché.
    """
    image, _, meta = load_case_cache(Path(cache_dir) / case_id)
    model.eval()
    pp = cfg.get("postprocess", {})
    stride = int(cfg["model"].get("det_stride", 8))
    Z, H, W = image.shape
    semantic = np.zeros((Z, H, W), np.uint8)
    edge = np.zeros((Z, H, W), np.float16)
    cls = np.zeros((Z, 3), np.float32)
    boxes = []
    amp = device.type == "cuda"
    for z0 in range(0, Z, batch_size):
        zs = range(z0, min(z0 + batch_size, Z))
        x = np.stack([context_stack(image, z, meta["context_offset"]) for z in zs]).astype(np.float32) / 255.0
        x = torch.from_numpy(x).to(device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = model(x)
        semantic[z0:z0 + len(zs)] = out["seg_logits"].argmax(1).cpu().numpy().astype(np.uint8)
        edge[z0:z0 + len(zs)] = torch.sigmoid(out["edge_logits"].float())[:, 0].cpu().numpy().astype(np.float16)
        cls[z0:z0 + len(zs)] = torch.sigmoid(out["cls_logits"].float()).cpu().numpy()
        dets = decode(out["det_scores"], out["det_ltrb"], stride, W, pp.get("det_score_threshold", 0.3),
                      pp.get("nms_iou", 0.5), pp.get("max_boxes_per_class", 1))
        boxes += [{k: v.cpu().numpy() for k, v in d.items()} for d in dets]
    return {"semantic": semantic, "edge": edge, "cls": cls, "boxes": boxes, "meta": meta}


def to_native(vol: np.ndarray, meta: Dict, order: int = 0) -> np.ndarray:
    """(Z, 256, 256) en la grilla del modelo -> (Z, Y, X) en la grilla nativa del .mha (LPS).

    Inverso exacto de ``slice_cache``: mismo ``grid_mode=True`` al redimensionar y el mismo
    cuadrado de recorte (que puede salirse del lienzo: esa parte se descarta).
    ``order=0`` para etiquetas, ``order=1`` para probabilidades.
    """
    Z, Y, X = meta["native_shape_zyx"]
    c = meta["crop"]
    side = c["side_px"]
    f = side / vol.shape[-1]
    out = np.zeros((Z, Y, X), vol.dtype)
    sy0, sy1 = max(c["y0"], 0), min(c["y1"], Y)
    sx0, sx1 = max(c["x0"], 0), min(c["x1"], X)
    for z0 in range(0, Z, CHUNK_Z):
        big = ndi.zoom(vol[z0:z0 + CHUNK_Z], (1, f, f), order=order, grid_mode=True, mode="nearest", prefilter=False)
        big = big[:, :side, :side]
        out[z0:z0 + CHUNK_Z, sy0:sy1, sx0:sx1] = big[:, sy0 - c["y0"]:sy1 - c["y0"], sx0 - c["x0"]:sx1 - c["x0"]]
    return out
