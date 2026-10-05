"""test_orozco_pipeline.py. Tests del intento propio (rama orozco) — solo phantoms sintéticos.

Cubren: cajas/NMS, targets del grid, DFL (esperanza y pérdida), decodificación,
formas del modelo, gradientes y targets de dataset. Sin dependencia de 01_data.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pengwin_orozco.boxes import batched_nms, box_iou, nms, paired_iou_giou
from pengwin_orozco.dataset import region_boxes_2d
from pengwin_orozco.detector import DFLGridHead, assign_dfl_targets, decode, dfl_expectation
from pengwin_orozco.losses import DetectionLoss, dfl_loss, sigmoid_focal_loss
from pengwin_orozco.metrics import DetectionEvaluator, average_precision
from pengwin_orozco.model import PengwinOrozcoNet, build_model, count_parameters


# --------------------------------------------------------------------------- cajas y NMS
def test_iou_giou_valores_conocidos():
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    assert box_iou(a, a)[0, 0] == pytest.approx(1.0)
    # Cajas en diagonal: intersección 5×5, unión 175, envolvente 15×15 → GIoU < IoU
    b = torch.tensor([[5.0, 5.0, 15.0, 15.0]])
    iou, giou = paired_iou_giou(a.expand(1, 4), b)
    assert iou[0] == pytest.approx(25 / 175)
    assert giou[0] == pytest.approx(iou[0] - (225 - 175) / 225)
    far = torch.tensor([[100.0, 100.0, 110.0, 110.0]])
    iou, giou = paired_iou_giou(a.expand(1, 4), far)
    assert iou[0] == pytest.approx(0.0)
    assert giou[0] < 0.0


def test_nms_orden_y_max_por_clase():
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60], [51, 51, 61, 61]])
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert set(keep.tolist()) == {0, 2}         # conserva la mejor de cada par
    keep = nms(boxes, scores, iou_threshold=0.5, max_keep=1)
    assert keep.tolist() == [0]
    labels = torch.tensor([0, 0, 1, 1])
    keep = batched_nms(boxes, scores, labels, iou_threshold=0.5, max_per_class=1)
    assert set(keep.tolist()) == {0, 2}
    empty = nms(torch.zeros(0, 4), torch.zeros(0))
    assert empty.numel() == 0


# --------------------------------------------------------------------------- targets del grid
def test_assign_targets_center_sampling_y_fallback():
    boxes = torch.tensor([[[40.0, 40.0, 80.0, 80.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])
    present = torch.tensor([[1.0, 0.0, 0.0]])
    ignore = torch.zeros_like(present)
    t = assign_dfl_targets(boxes, present, ignore, 32, 8, radius=1.5)
    pos = t["pos"][0, 0]
    assert pos.any()
    # la celda del centro (índice 7 ≈ 60 px) es positiva
    assert pos[7, 7]
    # todas las positivas están dentro de la caja y cerca del centro
    ys, xs = torch.nonzero(pos, as_tuple=True)
    assert ((xs.float() + 0.5) * 8 >= 40).all() and ((xs.float() + 0.5) * 8 <= 80).all()
    # caja diminuta: fallback a la celda de su centro
    tiny = torch.tensor([[[40.0, 40.0, 44.0, 44.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])
    t2 = assign_dfl_targets(tiny, present, ignore, 32, 8, radius=1.5)
    assert t2["pos"][0, 0].sum() >= 1
    # caja ignorada: sus celdas no son positivas ni válidas
    ig = torch.tensor([[1.0, 0.0, 0.0]])
    t3 = assign_dfl_targets(tiny, present, ig, 32, 8, radius=1.5)
    assert not t3["pos"][0, 0].any()


def test_region_boxes_2d_envolvente_e_ignore():
    label = np.zeros((64, 64), np.uint8)
    label[10:20, 10:20] = 1        # SA: isla principal
    label[30:35, 30:38] = 1        # SA: isla separada (misma región)
    label[5:7, 5:9] = 11           # coxal izq. diminuto
    boxes, present, ignore = region_boxes_2d(label, min_box_px=4.0)
    assert present[0] and not ignore[0]
    assert boxes[0].tolist() == [10, 10, 38, 35]     # envolvente de AMBAS islas
    assert present[1] and ignore[1]                  # lado = 2 < 4 → ignorar
    assert not present[2]


# --------------------------------------------------------------------------- DFL
def test_dfl_expectation_one_hot():
    dist = torch.zeros(1, 3, 4, 16, 32, 32)
    dist[0, 0, 0, 5] = 100.0        # pico en el bin 5 → esperanza 5 celdas
    ltrb = dfl_expectation(dist, stride=8)
    assert ltrb[0, 0, 0, 15, 15] == pytest.approx(5 * 8)
    # distribución plana → esperanza (15/2)·8
    dist_flat = torch.zeros(1, 1, 4, 16, 32, 32)
    e = dfl_expectation(dist_flat, stride=8)
    assert e[0, 0, 0, 0, 0] == pytest.approx(7.5 * 8)


def test_dfl_head_shapes_y_salidas_positivas():
    head = DFLGridHead(in_channels=256, num_classes=3, stride=8, reg_max=16)
    x = torch.randn(2, 256, 32, 32)
    scores, dist = head(x)
    assert scores.shape == (2, 3, 32, 32)
    assert dist.shape == (2, 3, 4, 16, 32, 32)
    # con el sesgo inicial, la esperanza arranca cerca de 2 celdas (16 px)
    ltrb = dfl_expectation(dist, stride=8)
    assert float(ltrb[0, 0, 0, 16, 16]) == pytest.approx(16.0, abs=1.0)


def test_dfl_roundtrip_decode():
    """Distribución one-hot en los targets → decode recupera las cajas con IoU ≈ 1.

    Las cajas se eligen alineadas a los centros de celda (bordes en 4 + 8·k) para
    que todas las distancias de las celdas positivas caigan en bins exactos, y
    los scores son altos solo en las celdas positivas: así el decode es exacto.
    """
    boxes = torch.tensor([[[36.0, 36.0, 100.0, 100.0], [92.0, 4.0, 156.0, 68.0], [0.0, 0.0, 0.0, 0.0]]])
    present = torch.tensor([[1.0, 1.0, 0.0]])
    ignore = torch.zeros_like(present)
    t = assign_dfl_targets(boxes, present, ignore, 32, 8, radius=1.5)
    pos, ltrb = t["pos"], t["ltrb_cells"]
    dist = torch.full((1, 3, 4, 16, 32, 32), -100.0)
    scores = torch.full((1, 3, 32, 32), -10.0)
    for ci in range(3):
        ii, jj = torch.nonzero(pos[0, ci], as_tuple=True)                  # celdas positivas
        vals = ltrb[0, ci, :, ii, jj].clamp(0, 15).round().long()          # (4, P) bines
        for bd in range(4):
            dist[0, ci, bd, vals[bd], ii, jj] = 100.0
        scores[0, ci, ii, jj] = 10.0                                        # solo positivas pasan el umbral
    dets = decode(scores, dist, 8, 256, 16, score_threshold=0.3, nms_iou=1.0,
                  max_per_class=None, pre_nms_top_k=1024)[0]
    ious = box_iou(dets["boxes"], boxes[0, :2])
    assert ious.max(dim=0).values.mean() > 0.99


def test_dfl_loss_perfecta_vs_mala():
    torch.manual_seed(0)
    pos = torch.zeros(1, 1, 32, 32, dtype=torch.bool)
    pos[0, 0, 16, 16] = True
    target = torch.zeros(1, 1, 4, 32, 32)
    target[0, 0, :, 16, 16] = 3.0                  # valor 3.0 celdas
    good = torch.full((1, 1, 4, 16, 32, 32), -100.0)
    good[0, 0, :, 3, 16, 16] = 0.0                 # pico exacto en el bin 3
    l_good = dfl_loss(good, target, pos)
    bad = torch.full((1, 1, 4, 16, 32, 32), -100.0)
    bad[0, 0, :, 15, 16, 16] = 0.0                 # pico en el bin 15
    l_bad = dfl_loss(bad, target, pos)
    assert float(l_good) < 1e-3
    assert float(l_bad) > 1.0
    assert torch.isfinite(l_good) and torch.isfinite(l_bad)


def test_focal_loss_pondera_menos_los_faciles():
    easy = torch.tensor([[10.0]])
    hard = torch.tensor([[0.1]])
    t = torch.ones(1, 1)
    le = sigmoid_focal_loss(easy, t, alpha=0.25, gamma=2.0).item()
    lh = sigmoid_focal_loss(hard, t, alpha=0.25, gamma=2.0).item()
    assert lh > le and le < 1.0


# --------------------------------------------------------------------------- modelo y gradientes
def test_model_formas_y_parametros(cfg_mini):
    net = PengwinOrozcoNet(cfg_mini["model"])
    x = torch.randn(2, 3, 256, 256)
    out = net(x)
    assert out["det_scores"].shape == (2, 3, 32, 32)
    assert out["det_dist"].shape == (2, 3, 4, 16, 32, 32)
    p = count_parameters(net)
    assert p["backbone"] > 0 and p["det_head"] > 0 and p["total"] == p["backbone"] + p["det_head"]


def test_gradientes_llegan_a_backbone_y_cabeza(cfg_mini):
    net = PengwinOrozcoNet(cfg_mini["model"])
    batch = {
        "boxes": torch.tensor([[[40.0, 40.0, 80.0, 80.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]),
        "present": torch.tensor([[1.0, 0.0, 0.0]]),
        "ignore": torch.zeros(1, 3),
    }
    loss = DetectionLoss(cfg_mini)(net(torch.randn(1, 3, 256, 256)), batch)
    loss["total"].backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in net.backbone.parameters())
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in net.det_head.parameters())


# --------------------------------------------------------------------------- métricas
def test_average_precision_curva_conocida():
    tp = np.array([1, 0, 1, 1])
    sc = np.array([0.9, 0.8, 0.7, 0.3])
    ap = average_precision(tp, sc, n_gt=3)
    assert 0.0 < ap <= 1.0
    assert np.isnan(average_precision(tp, sc, n_gt=0))
    assert average_precision(np.array([]), np.array([]), n_gt=2) == 0.0


def test_evaluator_perfecto_y_falla_una_clase():
    ev = DetectionEvaluator(3)
    gt = np.array([[10, 10, 50, 50], [60, 10, 120, 50], [10, 60, 50, 120]], float)
    # Corte 1: las 3 GT con predicciones perfectas
    ev.add({"boxes": gt, "scores": np.ones(3), "labels": np.arange(3)}, gt,
           np.ones(3), np.zeros(3))
    # Corte 2: solo la clase 0 tiene predicción (las clases 1-2 quedan sin cubrir)
    ev.add({"boxes": gt[:1], "scores": np.ones(1), "labels": np.array([0])}, gt,
           np.ones(3), np.zeros(3))
    res = ev.compute()
    # Clase 0: 2/2 GT cubiertas → AP 1. Clases 1-2: 1 GT cubierta de 2 → recall 0.5 → AP 0.5
    assert res["AP50_cls0"] == pytest.approx(1.0)
    assert res["mAP@0.50"] == pytest.approx(2 / 3)
    # IoU promedio: 4 GT cubiertas (IoU 1) + 2 GT sin predicción (0) → 4/6
    assert res["IoU_promedio"] == pytest.approx(2 / 3)


# --------------------------------------------------------------------------- fixtures
@pytest.fixture
def cfg_mini():
    return {"model": {"backbone": "fundidora", "widths": [32, 64, 128, 256], "cbam": True,
                      "cbam_blocks": [3, 4], "cbam_reduction": 16, "det_stride": 8,
                      "reg_max": 16, "center_radius": 1.5, "det_min_box_px": 4.0,
                      "num_classes": 3},
            "loss": {"focal_alpha": 0.25, "focal_gamma": 2.0},
            "postprocess": {"det_score_threshold": 0.3, "nms_iou": 0.5, "max_boxes_per_class": 1}}


def test_build_model_desde_config(cfg_mini):
    net = build_model(cfg_mini)
    assert isinstance(net, PengwinOrozcoNet)