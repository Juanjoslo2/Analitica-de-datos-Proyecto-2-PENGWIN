"""engine.py.

Bucle de entrenamiento y evaluación compartido por ``scripts/train.py``,
``scripts/overfit_batch.py`` y ``scripts/calibrate_lambdas.py``.

Precisión mixta obligatoria [enunciado §4.3]: ``torch.amp.autocast("cuda")`` +
``torch.amp.GradScaler``, que es la API actual de ``torch.cuda.amp`` (los nombres viejos
siguen existiendo en torch 2.10 pero emiten aviso de obsolescencia). En CPU el autocast
se desactiva y todo corre en float32, así el mismo código sirve para medir latencia en CPU.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

from pengwin.data.targets import REGION_NAMES
from pengwin.detection.grid import decode
from pengwin.evaluation.det_metrics import DetectionEvaluator, SegmentationEvaluator

TENSOR_KEYS = ("image", "semantic", "edge", "boxes", "present", "ignore")


def to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: (v.to(device, non_blocking=True) if k in TENSOR_KEYS else v) for k, v in batch.items()}


def autocast(device: torch.device, enabled: bool):
    return torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")


def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device, amp: bool = True,
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
        parts = loss_fn(out, batch)                 # pérdidas en float32 (las salidas se convierten dentro)
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
    """Pérdidas + métricas de las tres cabezas sobre un ``DataLoader`` completo."""
    model.eval()
    pp = cfg.get("postprocess", {})
    stride = int(cfg["model"].get("det_stride", 8))
    det_eval, seg_eval = DetectionEvaluator(), SegmentationEvaluator()
    sums: Dict[str, float] = {}
    probs, targets = [], []
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
        dets = decode(out["det_scores"], out["det_ltrb"], stride, batch["image"].shape[-1],
                      pp.get("det_score_threshold", 0.3), pp.get("nms_iou", 0.5), pp.get("max_boxes_per_class", 1))
        for i, d in enumerate(dets):
            det_eval.add(d, batch["boxes"][i], batch["present"][i], batch["ignore"][i])
        seg_eval.add(out["seg_logits"].argmax(1), batch["semantic"])
        probs.append(torch.sigmoid(out["cls_logits"].float()).cpu())
        targets.append(batch["present"].cpu())

    res = {f"val_{k}": v / max(n, 1) for k, v in sums.items()}
    res.update(det_eval.compute())
    res.update(seg_eval.compute())
    p, t = torch.cat(probs).numpy(), torch.cat(targets).numpy().astype(int)
    res["cls_f1_macro"] = float(f1_score(t, p > 0.5, average="macro", zero_division=0))
    try:
        res["cls_auc_macro"] = float(roc_auc_score(t, p, average="macro"))
    except ValueError:                                  # una clase sin positivos o sin negativos
        res["cls_auc_macro"] = float("nan")
    for c, name in enumerate(REGION_NAMES):
        res[f"cls_f1_{name}"] = float(f1_score(t[:, c], p[:, c] > 0.5, zero_division=0))
    res["ms_por_corte_modelo"] = 1000 * t_model / max(len(p), 1)
    return res


def save_checkpoint(path: Path, model, optimizer=None, epoch: int = 0, metrics: Dict | None = None, cfg: Dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict() if optimizer else None,
                "epoch": epoch, "metrics": metrics or {}, "cfg": cfg}, path)


def selection_score(metrics: Dict[str, float]) -> float:
    """Criterio para guardar el mejor checkpoint: media de mAP@[.50:.95], Dice de hueso y F1.

    Semana 10: antes usaba mAP@0.5, que se satura en ~0,98 desde la época 15 y hacía elegir
    épocas tempranas (v2: época 22 contra 39, con 0,016 menos de mAP@[.50:.95] en test).
    """
    vals = [metrics.get("mAP@[.50:.95]"), metrics.get("dice_hueso"), metrics.get("cls_f1_macro")]
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    return float(np.mean(vals)) if vals else 0.0


def param_groups(model, lr: float, backbone_lr_factor: float = 1.0, weight_decay: float = 1e-4) -> Iterable[Dict]:
    """Backbone y cabezas en grupos separados: permite un lr menor en el backbone preentrenado."""
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    heads = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    return [{"params": bb, "lr": lr * backbone_lr_factor, "weight_decay": weight_decay},
            {"params": heads, "lr": lr, "weight_decay": weight_decay}]
