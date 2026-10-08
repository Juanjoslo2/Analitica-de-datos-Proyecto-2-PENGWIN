"""volume.py.

Inferencia de un caso completo: el modelo 2D recorre todos los cortes del caché y las
salidas se apilan en un volumen (Z, 256, 256). Después ``to_native`` deshace el
redimensionado y el recorte para volver a la grilla del .mha, donde se mide en mm.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

from pengwin.data.dataset import context_stack
from pengwin.data.slice_cache import CHUNK_Z, load_case_cache
from pengwin.data.targets import DIST_MAX_MM
from pengwin.detection.grid import decode
from pengwin.models.heads import SegmentationHead


# TTA (semana 10, 2026-10-06): transformaciones (escala, rotación en grados, traslación x, y en
# fracción del lado) dentro del rango de ``augment`` de base.yaml, así el modelo las vio al entrenar.
# Sin volteo: el modelo distingue coxal izquierdo y derecho [DD §1].
TTA_DEFAULT: Tuple[Tuple[float, float, float, float], ...] = (
    (1.0, 0.0, 0.0, 0.0), (0.92, 0.0, 0.0, 0.0), (1.08, 0.0, 0.0, 0.0), (1.0, 8.0, 0.0, 0.0), (1.0, -8.0, 0.0, 0.0),
)


def _theta(scale: float, rot_deg: float, tx: float, ty: float) -> torch.Tensor:
    """Matriz 3×3 en coordenadas normalizadas [-1, 1] de ``affine_grid`` (salida -> entrada).

    Muestrear la imagen con esta matriz la agranda ``scale`` veces, la gira ``rot_deg`` y la
    desplaza (tx, ty) fracciones del lado.
    """
    a = math.radians(rot_deg)
    c, s = math.cos(a) / scale, math.sin(a) / scale
    return torch.tensor([[c, -s, -2 * tx], [s, c, -2 * ty], [0.0, 0.0, 1.0]], dtype=torch.float32)


def _warp(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    grid = F.affine_grid(theta[:2][None].expand(x.shape[0], 2, 3).to(x), list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def _tta_forward(model, x: torch.Tensor, tta: Sequence[Tuple[float, float, float, float]], amp: bool, device) -> Dict:
    """Promedio de P(región) y P(borde) sobre las transformaciones de ``tta``.

    Cada pasada transforma la entrada, predice y devuelve las probabilidades a la posición
    original con la transformación inversa. Los píxeles que una pasada deja fuera del lienzo no
    cuentan en su promedio (peso = cobertura). Cajas y clasificación salen de la primera pasada,
    que debe ser la identidad: la detección no cambia con TTA.
    """
    sem_acc = edge_acc = peso = None
    first = None
    for k, t in enumerate(tta):
        th = _theta(*t)
        xi = x if k == 0 and t == (1.0, 0.0, 0.0, 0.0) else _warp(x, th)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = model(xi)
        if first is None:
            first = out
        probs = torch.cat([out["seg_logits"].float().softmax(1), torch.sigmoid(out["edge_logits"].float())], 1)
        cov = torch.ones_like(probs[:, :1])
        if xi is not x:
            inv = torch.linalg.inv(th)
            probs, cov = _warp(probs, inv), _warp(cov, inv)
        sem_acc = probs[:, :-1] * cov if sem_acc is None else sem_acc + probs[:, :-1] * cov
        edge_acc = probs[:, -1:] * cov if edge_acc is None else edge_acc + probs[:, -1:] * cov
        peso = cov if peso is None else peso + cov
    peso = peso.clamp_min(1e-6)
    return {**first, "tta_sem": sem_acc / peso, "tta_edge": edge_acc / peso}


@torch.no_grad()
def predict_case(model, cache_dir: Path | str, case_id: str, cfg: Dict, device: torch.device,
                 batch_size: int = 16, tta: Sequence[Tuple[float, float, float, float]] | None = None) -> Dict:
    """Corre el modelo sobre todos los cortes de un caso.

    Devuelve en la grilla del modelo (Z, 256, 256):
        semantic  uint8    región por píxel (0 fondo, 1 SA, 2 LI, 3 RI)
        edge      float16  probabilidad de borde de fractura
        core      float16  probabilidad de núcleo, SOLO si el modelo tiene la salida ``core3``
        dist      float16  distancia a la superficie de fractura en mm (0..``loss.dist_max_mm``),
                           SOLO si el modelo tiene la salida ``dist`` [F2B2]
        role      float16  P(secundario | hueso) = p_sec / (p_principal + p_sec), SOLO si el modelo
                           tiene la salida ``role3`` [y4xul]
        cls       float32  (Z, 3) probabilidad de presencia de cada región
        boxes     lista de Z dicts {boxes, scores, labels} (numpy)
    y el ``meta`` del caché.

    ``tta``: lista de transformaciones (escala, rotación °, tx, ty) para promediar ``semantic`` y
    ``edge`` (``TTA_DEFAULT`` es la de 5 pasadas). ``None`` = una sola pasada, como siempre.
    Con TTA no se calculan ``core``, ``dist`` ni ``role``.
    """
    image, _, meta = load_case_cache(Path(cache_dir) / case_id)
    model.eval()
    pp = cfg.get("postprocess", {})
    stride = int(cfg["model"].get("det_stride", 8))
    in_ch = int(cfg["model"].get("in_channels", 3))
    dist_max_mm = float(cfg.get("loss", {}).get("dist_max_mm", DIST_MAX_MM))
    Z, H, W = image.shape
    semantic = np.zeros((Z, H, W), np.uint8)
    edge = np.zeros((Z, H, W), np.float16)
    core: np.ndarray | None = None          # se crea solo si el modelo tiene la salida core3
    dist: np.ndarray | None = None          # ídem con la salida dist [F2B2]
    role: np.ndarray | None = None          # ídem con la salida role3 [y4xul]
    cls = np.zeros((Z, 3), np.float32)
    boxes = []
    amp = device.type == "cuda"
    for z0 in range(0, Z, batch_size):
        zs = range(z0, min(z0 + batch_size, Z))
        x = np.stack([context_stack(image, z, meta["context_offset"]) for z in zs]).astype(np.float32) / 255.0
        x = torch.from_numpy(x).to(device)
        if in_ch > x.shape[1]:          # [y4xul] 4.º canal (máscara previa) en cero: es la pasada 1
            x = torch.cat([x, x.new_zeros(x.shape[0], in_ch - x.shape[1], *x.shape[2:])], 1)
        if tta:
            out = _tta_forward(model, x, tta, amp, device)
            semantic[z0:z0 + len(zs)] = out["tta_sem"].argmax(1).cpu().numpy().astype(np.uint8)
            edge[z0:z0 + len(zs)] = out["tta_edge"][:, 0].cpu().numpy().astype(np.float16)
            out = {k: v for k, v in out.items() if k not in ("core_logits", "dist_logits", "role_logits")}
        else:
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                out = model(x)
            semantic[z0:z0 + len(zs)] = out["seg_logits"].argmax(1).cpu().numpy().astype(np.uint8)
            edge[z0:z0 + len(zs)] = torch.sigmoid(out["edge_logits"].float())[:, 0].cpu().numpy().astype(np.float16)
        if "core_logits" in out:                        # P(núcleo) de la salida de 3 clases [F2B1]
            if core is None:
                core = np.zeros((Z, H, W), np.float16)
            core[z0:z0 + len(zs)] = out["core_logits"].float().softmax(1)[:, 1].cpu().numpy().astype(np.float16)
        if "dist_logits" in out:                        # distancia a la fractura en mm [F2B2]
            if dist is None:
                dist = np.zeros((Z, H, W), np.float16)
            mm = SegmentationHead.dist_mm_from_logits(out["dist_logits"], dist_max_mm)
            dist[z0:z0 + len(zs)] = mm[:, 0].cpu().numpy().astype(np.float16)
        if "role_logits" in out:                        # P(secundario | hueso) [y4xul]
            if role is None:
                role = np.zeros((Z, H, W), np.float16)
            # Se normaliza entre principal y secundario: QUÉ es hueso lo decide la semántica de 4
            # clases (etapa 1); esta salida solo reparte el hueso entre los dos papeles.
            pr = out["role_logits"].float()[:, 1:].softmax(1)[:, 1]
            role[z0:z0 + len(zs)] = pr.cpu().numpy().astype(np.float16)
        cls[z0:z0 + len(zs)] = torch.sigmoid(out["cls_logits"].float()).cpu().numpy()
        dets = decode(out["det_scores"], out["det_ltrb"], stride, W, pp.get("det_score_threshold", 0.3),
                      pp.get("nms_iou", 0.5), pp.get("max_boxes_per_class", 1))
        boxes += [{k: v.cpu().numpy() for k, v in d.items()} for d in dets]
    res = {"semantic": semantic, "edge": edge, "cls": cls, "boxes": boxes, "meta": meta}
    if core is not None:
        res["core"] = core
    if dist is not None:
        res["dist"] = dist
    if role is not None:
        res["role"] = role
    return res


@torch.no_grad()
def predict_case_two_pass(model, cache_dir: Path | str, hi_cache_dir: Path | str, case_id: str, cfg: Dict,
                          device: torch.device, batch_size: int = 16, roi_batch: int = 32,
                          gate_px: int = 0, gate_z: int = 3) -> Dict:
    """Las dos pasadas del MISMO modelo sobre un caso [y4xul, refinamiento por región].

    Pasada 1 = ``predict_case`` (corte completo, máscara previa en cero): región, cajas,
    clasificación y una primera versión del papel y del borde.
    Pasada 2: por cada corte y cada hueso que la pasada 1 encontró, un recorte del caché de alta
    resolución con la máscara de ESE hueso en el 4.º canal. Su papel y su borde se promedian por
    bloque para volver a la grilla de 256 y sustituyen a los de la pasada 1 solo dentro del hueso.

    ``gate_px`` > 0 = segunda pasada SELECTIVA: un hueso solo se refina en los cortes donde la
    pasada 1 sospecha fractura (al menos ``gate_px`` píxeles con P(secundario) ≥ 0,3 o P(borde) ≥
    0,2) o a menos de ``gate_z`` cortes de uno así. En tres de cada cuatro recortes el hueso es una
    sola pieza y refinarlo no cambia nada; ``n_roi_total`` dice cuántos habría sin filtrar.

    Devuelve lo mismo que ``predict_case`` con ``edge`` y ``role`` refinados, más ``edge1`` y
    ``role1`` (los de la pasada 1, para medir cuánto aporta la segunda) y ``n_roi``.
    La máscara previa es la que predice el propio modelo: nunca se usa el ground truth.
    """
    from pengwin.data.two_pass import ROI_MIN_PX, paste_window, prior_boxes, roi_input, roi_window

    res = predict_case(model, cache_dir, case_id, cfg, device, batch_size)
    if "role" not in res:
        raise ValueError("la segunda pasada necesita la salida role3 (model.seg_outputs)")
    hi_image, _, hi_meta = load_case_cache(Path(hi_cache_dir) / case_id)
    meta, sem = res["meta"], res["semantic"]
    out_px = sem.shape[-1]
    scale = int(hi_meta["image_size"]) // out_px
    res["edge1"], res["role1"] = res["edge"], res["role"]
    edge = res["edge"].astype(np.float32)
    role = res["role"].astype(np.float32)
    boxes = prior_boxes(sem, ROI_MIN_PX)
    presente = boxes[..., 2] > boxes[..., 0]
    res["n_roi_total"] = int(presente.sum())
    if gate_px > 0:
        sospecha = (res["role1"].astype(np.float32) >= 0.3) | (res["edge1"].astype(np.float32) >= 0.2)
        cuenta = np.stack([(sospecha & (sem == r)).reshape(len(sem), -1).sum(1) for r in (1, 2, 3)], 1)
        activo = ndi.maximum_filter1d((cuenta >= gate_px).astype(np.uint8), size=2 * gate_z + 1, axis=0, mode="constant") > 0
        presente &= activo
    zs, rs = np.nonzero(presente)
    amp = device.type == "cuda"
    for i0 in range(0, len(zs), roi_batch):
        lote = [(int(z), int(r) + 1) for z, r in zip(zs[i0:i0 + roi_batch], rs[i0:i0 + roi_batch])]
        wins, xs = [], []
        for z, r in lote:
            win = roi_window(boxes[z, r - 1], scale, out_px)
            img, pri = roi_input(hi_image, z, meta["context_offset"], sem[z] == r, win, scale, out_px)
            wins.append(win)
            xs.append(torch.cat([img, pri[None].float()], 0))
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = model(torch.stack(xs).to(device))
        p_role = out["role_logits"].float()[:, 1:].softmax(1)[:, 1]
        p_edge = torch.sigmoid(out["edge_logits"].float())[:, 0]
        for k, ((z, r), (x0, y0, side)) in enumerate(zip(lote, wins)):
            where = sem[z] == r
            # el recorte se amplió o redujo a out_px: se devuelve a su lado real antes de promediar
            pr = F.interpolate(p_role[k][None, None], size=(side, side), mode="bilinear", align_corners=False)[0, 0]
            pe = F.interpolate(p_edge[k][None, None], size=(side, side), mode="bilinear", align_corners=False)[0, 0]
            paste_window(role[z], pr, x0, y0, side, scale, where)
            paste_window(edge[z], pe, x0, y0, side, scale, where)
    res["edge"], res["role"] = edge.astype(np.float16), role.astype(np.float16)
    res["n_roi"] = int(len(zs))
    return res


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
