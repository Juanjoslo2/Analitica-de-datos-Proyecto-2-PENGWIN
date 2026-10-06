"""losses.py.

Pérdida multitarea de tres términos [DD §4, enunciado §4.1]:

    L = λ_cls · L_cls + λ_det · L_det + λ_seg · L_seg

    L_cls  BCE multi-etiqueta sobre la presencia de SA / coxal izq. / coxal der. en el corte
    L_det  focal sigmoide sobre el grid (normalizada por nº de positivos) + (1 − GIoU) en positivos
    L_seg  CE con pesos ∝ 1/√frecuencia + Dice (semántica) + lo que pida ``model.seg_outputs``:
           ``fracture_edge`` -> BCE con pos_weight + Dice del borde binario (semanas 9-10)
           ``core3``         -> CE con pesos ∝ 1/√frecuencia + Dice sobre fondo/núcleo/borde [F2B1]
           ``dist``          -> Huber (smooth L1) enmascarada al hueso sobre la distancia a la
                               superficie de fractura, normalizada a [0, 1] [F2B2]

**Siguen siendo TRES términos** (enunciado §4.1): núcleo/borde y la distancia a la fractura
entran DENTRO de ``seg``, no como un cuarto término. Cambiar la composición de ``seg`` obliga a
recalibrar los λ.

La distancia de separación NO es un término: es una medición posterior sobre las máscaras.
Los λ salen de ``calibrate_lambdas`` (balance de normas de gradiente sobre la última capa
compartida) y se guardan en ``reports/lambdas.json``; el archivo es lo que lee el
entrenamiento cuando la config dice ``lambdas: auto``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from pengwin.detection.boxes import paired_iou_giou
from pengwin.detection.grid import assign_targets, ltrb_to_boxes
from pengwin.models.heads import DEFAULT_SEG_OUTPUTS

TERMS = ("cls", "det", "seg")

# 1/√frecuencia de fondo / núcleo / borde, normalizado a fondo = 1. Medido sobre los cortes con
# hueso de 12 casos del caché (fracciones 0,96642 / 0,03326 / 0,00032; hueso:borde = 1:104).
CORE3_CE_WEIGHTS = (1.0, 5.4, 54.8)

# [F2B2] Umbral de la Huber sobre la distancia NORMALIZADA (0,1 · 8 mm = 0,8 mm). Justo por encima
# del error propio de la etiqueta: se remuestreó a 256² con vecino más cercano, así que la posición
# de la superficie de fractura trae ±0,5 vóxel ≈ ±0,6 mm. Por debajo de ese ruido la pérdida es
# cuadrática (el gradiente se apaga en vez de pelear con el ruido) y por encima es L1 (robusta).
DIST_HUBER_BETA = 0.1
# [y4xul] role3 (fondo, principal, secundario): misma regla que ``seg_ce_weights``, peso ∝
# 1/√frecuencia relativa al fondo. Hueso/fondo = 0,035 (de 1 / 10,8 / 8,7 / 8,7) y los secundarios
# son el 10,5 % del volumen de hueso en train+val (reports/eda/eda_fragments.csv, sin test).
ROLE3_CE_WEIGHTS = (1.0, 5.6, 16.5)


def sigmoid_focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Focal loss (Lin et al., 2017) elemento a elemento, sin reducir."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    a_t = alpha * targets + (1 - alpha) * (1 - targets)
    return a_t * (1 - p_t) ** gamma * ce


def soft_dice_loss(probs: torch.Tensor, onehot: torch.Tensor, dims=(0, 2, 3), eps: float = 1.0) -> torch.Tensor:
    """1 − Dice suave por canal, promediado. ``probs`` y ``onehot`` (B, C, H, W)."""
    inter = (probs * onehot).sum(dims)
    denom = probs.sum(dims) + onehot.sum(dims)
    return 1 - ((2 * inter + eps) / (denom + eps)).mean()


def detection_loss(scores: torch.Tensor, ltrb: torch.Tensor, batch: Dict, stride: int, focal: Dict) -> Dict[str, torch.Tensor]:
    g = scores.shape[-1]
    t = assign_targets(batch["boxes"], batch["present"], batch["ignore"], g, stride, focal.get("center_radius", 1.5))
    pos, valid = t["pos"], t["valid"]
    n_pos = pos.sum().clamp(min=1).float()
    l_obj = (sigmoid_focal_loss(scores.float(), pos.float(), focal.get("alpha", 0.25), focal.get("gamma", 2.0)) * valid).sum() / n_pos
    if pos.any():
        pred_boxes = ltrb_to_boxes(ltrb, stride)[pos]                     # (P, 4)
        gt = batch["boxes"][:, :, None, None, :].expand(*pos.shape, 4)[pos]
        iou, giou = paired_iou_giou(pred_boxes, gt)
        l_box = (1 - giou).mean()
        mean_iou = iou.detach().mean()
    else:
        l_box = ltrb.sum() * 0.0
        mean_iou = torch.zeros((), device=scores.device)
    return {"det_obj": l_obj, "det_box": l_box, "det": l_obj + l_box, "det_pos_iou": mean_iou}


class MultiTaskLoss(nn.Module):
    def __init__(self, cfg: Dict, lambdas: Dict[str, float] | None = None):
        super().__init__()
        lc = cfg["loss"]
        self.stride = int(cfg["model"].get("det_stride", 8))
        self.register_buffer("ce_weights", torch.tensor(lc.get("seg_ce_weights", [1.0, 1.0, 1.0, 1.0]), dtype=torch.float32))
        self.register_buffer("core3_weights", torch.tensor(lc.get("core3_ce_weights", CORE3_CE_WEIGHTS), dtype=torch.float32))
        self.register_buffer("role3_weights", torch.tensor(lc.get("role3_ce_weights", ROLE3_CE_WEIGHTS), dtype=torch.float32))
        self.seg_outputs = tuple(cfg["model"].get("seg_outputs", DEFAULT_SEG_OUTPUTS) or DEFAULT_SEG_OUTPUTS)
        self.edge_pos_weight = float(lc.get("edge_pos_weight", 21.0))
        self.dist_huber_beta = float(lc.get("dist_huber_beta", DIST_HUBER_BETA))   # 0 = L1 pura [F2B2]
        self.focal = {"alpha": lc.get("focal_alpha", 0.25), "gamma": lc.get("focal_gamma", 2.0),
                      "center_radius": lc.get("center_radius", 1.5)}
        self.terms = tuple(lc.get("terms", TERMS))
        self.lambdas = {k: 1.0 for k in TERMS}
        if lambdas:
            self.lambdas.update({k: float(v) for k, v in lambdas.items() if k in TERMS})

    def forward(self, out: Dict[str, torch.Tensor], batch: Dict) -> Dict[str, torch.Tensor]:
        parts: Dict[str, torch.Tensor] = {}
        # Clasificación multi-etiqueta (cortes vacíos incluidos)
        parts["cls"] = F.binary_cross_entropy_with_logits(out["cls_logits"].float(), batch["present"])

        # Detección
        parts.update(detection_loss(out["det_scores"], out["det_ltrb"], batch, self.stride, self.focal))

        # Segmentación: semántica de 4 clases (etapa 1) + la representación de fragmentos (etapa 2)
        seg = out["seg_logits"].float()
        sem = batch["semantic"]
        probs = seg.softmax(1)
        onehot = F.one_hot(sem, seg.shape[1]).permute(0, 3, 1, 2).float()
        parts["seg_ce"] = F.cross_entropy(seg, sem, weight=self.ce_weights)
        parts["seg_dice"] = soft_dice_loss(probs[:, 1:], onehot[:, 1:])          # Dice sin el fondo
        parts["seg"] = parts["seg_ce"] + parts["seg_dice"]
        if "fracture_edge" in self.seg_outputs:
            edge = out["edge_logits"].float()
            pw = torch.tensor(self.edge_pos_weight, device=edge.device)
            parts["edge_bce"] = F.binary_cross_entropy_with_logits(edge, batch["edge"], pos_weight=pw)
            parts["edge_dice"] = soft_dice_loss(torch.sigmoid(edge), batch["edge"])
            parts["seg"] = parts["seg"] + parts["edge_bce"] + parts["edge_dice"]
        if "core3" in self.seg_outputs:
            # Núcleo como clase propia: el núcleo y el borde compiten en el mismo softmax, así la
            # red tiene que dejar hueco entre fragmentos en vez de marcar una superficie fina.
            core = out["core_logits"].float()
            tgt = batch["core3"]
            hot3 = F.one_hot(tgt, core.shape[1]).permute(0, 3, 1, 2).float()
            parts["core3_ce"] = F.cross_entropy(core, tgt, weight=self.core3_weights)
            parts["core3_dice"] = soft_dice_loss(core.softmax(1)[:, 1:], hot3[:, 1:])   # núcleo y borde
            parts["seg"] = parts["seg"] + parts["core3_ce"] + parts["core3_dice"]
        if "role3" in self.seg_outputs:
            # Principal / secundario como clases densas. Va dentro del término ``seg`` (la pérdida
            # sigue siendo de 3 términos) con la misma forma que la semántica: CE ponderada + Dice.
            role = out["role_logits"].float()
            tgt = batch["role3"]
            hot = F.one_hot(tgt, role.shape[1]).permute(0, 3, 1, 2).float()
            parts["role3_ce"] = F.cross_entropy(role, tgt, weight=self.role3_weights)
            parts["role3_dice"] = soft_dice_loss(role.softmax(1)[:, 1:], hot[:, 1:])     # principal y secundario
            parts["seg"] = parts["seg"] + parts["role3_ce"] + parts["role3_dice"]
        if "dist" in self.seg_outputs:
            # Regresión densa de la distancia a la fractura, ENMASCARADA AL HUESO: fuera del hueso
            # la distancia no existe y supervisar ese 96 % de píxeles con un 0 constante volvería a
            # meter el desbalance que este objetivo elimina. El promedio es por vóxel de hueso, así
            # que la escala no depende de cuánto hueso traiga el corte.
            pred = torch.sigmoid(out["dist_logits"].float())          # [0, 1] por construcción
            tgt = batch["dist"].float()                               # ya normalizado por dist_max_mm
            hueso = (sem > 0)[:, None].float()
            b = self.dist_huber_beta
            err = F.smooth_l1_loss(pred, tgt, reduction="none", beta=b) if b > 0 else (pred - tgt).abs()
            parts["dist_reg"] = (err * hueso).sum() / hueso.sum().clamp(min=1.0)
            parts["seg"] = parts["seg"] + parts["dist_reg"]

        parts["total"] = sum(self.lambdas[k] * parts[k] for k in self.terms)
        return parts


# --------------------------------------------------------------------------- calibración de λ
def load_lambdas(loss_cfg: Dict, repo_root: Path) -> Dict[str, float] | None:
    """``lambdas: auto`` → lee ``reports/lambdas.json`` (si existe); un dict → se usa tal cual."""
    lam = loss_cfg.get("lambdas", "auto")
    if isinstance(lam, dict):
        return lam
    path = repo_root / loss_cfg.get("lambdas_file", "reports/lambdas.json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["lambdas"]
    return None


def grad_norms(loss_parts: Dict[str, torch.Tensor], shared: Iterable[nn.Parameter], terms=TERMS) -> Dict[str, float]:
    """G_k = ‖∇_θs L_k‖₂ para cada término sobre los parámetros compartidos θs."""
    shared = list(shared)
    out = {}
    for k in terms:
        grads = torch.autograd.grad(loss_parts[k], shared, retain_graph=True, allow_unused=True)
        out[k] = float(torch.sqrt(sum((g.float() ** 2).sum() for g in grads if g is not None)))
    return out


def lambdas_from_norms(norms: Dict[str, Iterable[float]], total: float = 3.0) -> Dict[str, float]:
    """λ_k = media(G) / G_k, normalizado para que Σλ = ``total`` [DD §4]."""
    g = {k: float(torch.tensor(list(v)).mean()) for k, v in norms.items()}
    mean_g = sum(g.values()) / len(g)
    raw = {k: mean_g / max(v, 1e-12) for k, v in g.items()}
    s = sum(raw.values())
    return {k: total * v / s for k, v in raw.items()}
