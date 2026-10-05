"""engine.py. Bucle de entrenamiento y evaluación del intento propio (detección).

Igual disciplina que el resto del proyecto: precisión mixta con la API actual
``torch.amp``, ``clip_grad_norm_``, semillas fijas, y latencia por corte.
"""

from __future__ import annotations

import time
from typing import Dict

import numpy as np
import torch

from pengwin_orozco.detector import decode
from pengwin_orozco.metrics import DetectionEvaluator

TENSOR_KEYS = ("image", "boxes", "present", "ignore")


def to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: (v.to(device, non_blocking=True) if k in TENSOR_KEYS else v) for k, v in batch.items()}


def autocast(device: torch.device, enabled: bool):
    return torch.amp.autocast(device_type=device.type, dtype=torch.float16,
                              enabled=enabled and device.type == "cuda")


def train_epoch(model, loader, loss_fn, optimizer, scaler, device, amp: bool = True,
                grad_clip: float | None = 10.0, max_steps: int | None = None) -> Dict[str, float]:
    model.train()
    sums: Dict[str, float] = {}
    n = 0
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        batch = to_device(batch, device)
        with autocast(device, amp):
            out = model(batch["image"])
        parts = loss_fn(out, batch)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(parts["total"]).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + float(v.detach())
        n += 1
    return {k: v / max(n, 1) for k, v in sums.items()}


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, cfg: Dict, amp: bool = True) -> Dict[str, float]:
    model.eval()
    mc = cfg["model"]
    stride = int(mc.get("det_stride", 8))
    reg_max = int(mc.get("reg_max", 32))
    pp = cfg.get("postprocess", {})
    det_eval = DetectionEvaluator(int(mc.get("num_classes", 3)))
    sums: Dict[str, float] = {}
    n, t_model = 0, 0.0
    for batch in loader:
        batch = to_device(batch, device)
        t0 = time.perf_counter()
        with autocast(device, amp):
            out = model(batch["image"])
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_model += time.perf_counter() - t0
        parts = loss_fn(out, batch)
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + float(v)
        n += 1
        dets = decode(out["det_scores"], out["det_dist"], stride, batch["image"].shape[-1], reg_max,
                      pp.get("det_score_threshold", 0.3), pp.get("nms_iou", 0.5),
                      pp.get("max_boxes_per_class", 1))
        for i, d in enumerate(dets):
            det_eval.add({"boxes": d["boxes"].cpu().numpy(), "scores": d["scores"].cpu().numpy(),
                          "labels": d["labels"].cpu().numpy()},
                         batch["boxes"][i].cpu().numpy(), batch["present"][i].cpu().numpy(),
                         batch["ignore"][i].cpu().numpy())
    res = {f"val_{k}": v / max(n, 1) for k, v in sums.items()}
    res.update(det_eval.compute())
    res["ms_por_corte_modelo"] = 1000 * t_model / max(n, 1)
    return res


def save_checkpoint(path, model, optimizer=None, epoch: int = 0, metrics: Dict | None = None, cfg: Dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict() if optimizer else None,
                "epoch": epoch, "metrics": metrics or {}, "cfg": cfg}, path)


def selection_score(metrics: Dict[str, float]) -> float:
    """Criterio del mejor checkpoint: mAP@0.5 (+ IoU promedio si hay datos)."""
    vals = [metrics.get("mAP@0.50")]
    iou = metrics.get("IoU_promedio")
    if iou is not None and not np.isnan(iou):
        vals.append(iou)
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    return float(np.mean(vals)) if vals else 0.0