"""Pérdidas, calibración de λ y métricas de detección/segmentación con resultado conocido."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pengwin.evaluation.det_metrics import DetectionEvaluator, SegmentationEvaluator, average_precision  # noqa: E402
from pengwin.losses.losses import lambdas_from_norms, sigmoid_focal_loss, soft_dice_loss  # noqa: E402


def test_focal_loss_down_weights_easy_examples():
    easy = sigmoid_focal_loss(torch.tensor([-6.0]), torch.tensor([0.0]))
    hard = sigmoid_focal_loss(torch.tensor([2.0]), torch.tensor([0.0]))
    assert float(easy) < 1e-5 < float(hard)


def test_dice_loss_perfect_and_disjoint():
    m = torch.zeros(1, 1, 8, 8)
    m[..., 2:6, 2:6] = 1
    assert float(soft_dice_loss(m, m, eps=1e-6)) == pytest.approx(0.0, abs=1e-5)
    assert float(soft_dice_loss(1 - m, m, eps=1e-6)) == pytest.approx(1.0, abs=1e-5)


def test_lambdas_inverse_to_gradient_norm_and_sum_three():
    lam = lambdas_from_norms({"cls": [4.0, 4.0], "det": [1.0], "seg": [2.0]})
    assert sum(lam.values()) == pytest.approx(3.0)
    assert lam["det"] / lam["cls"] == pytest.approx(4.0) and lam["seg"] / lam["cls"] == pytest.approx(2.0)


def test_average_precision_known_curve():
    # 2 GT; predicciones ordenadas: TP, FP, TP -> precisión 1, 1/2, 2/3 en recall 0,5, 0,5, 1
    ap = average_precision(np.array([True, False, True]), np.array([0.9, 0.8, 0.7]), n_gt=2)
    assert ap == pytest.approx(0.5 * 1.0 + 0.5 * (2 / 3))
    assert math.isnan(average_precision(np.array([]), np.array([]), 0))


def _pred(boxes, scores, labels):
    return {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "scores": torch.tensor(scores, dtype=torch.float32), "labels": torch.tensor(labels, dtype=torch.long)}


def test_detection_evaluator_perfect_and_missed():
    gt = torch.tensor([[10.0, 10, 50, 50], [60, 10, 100, 50], [0, 0, 0, 0]])
    present, ignore = torch.tensor([1.0, 1, 0]), torch.zeros(3, dtype=torch.bool)
    ev = DetectionEvaluator()
    ev.add(_pred(gt[:2].tolist(), [0.9, 0.8], [0, 1]), gt, present, ignore)
    m = ev.compute()
    assert m["mAP@0.50"] == pytest.approx(1.0) and m["mAP@[.50:.95]"] == pytest.approx(1.0)
    assert m["IoU_promedio"] == pytest.approx(1.0)

    ev = DetectionEvaluator()
    ev.add(_pred(gt[:1].tolist(), [0.9], [0]), gt, present, ignore)      # el coxal izq. no se detectó
    m = ev.compute()
    assert m["AP50_SA"] == pytest.approx(1.0) and m["AP50_LI"] == pytest.approx(0.0)
    assert m["IoU_promedio"] == pytest.approx(0.5)                      # (1 + 0) / 2


def test_detection_evaluator_ignores_tiny_gt_and_counts_fp():
    gt = torch.tensor([[10.0, 10, 12, 40], [0, 0, 0, 0], [0, 0, 0, 0]])
    ev = DetectionEvaluator()
    ev.add(_pred([[10, 10, 12, 40], [100, 100, 140, 140]], [0.9, 0.8], [0, 1]),
           gt, torch.tensor([1.0, 0, 0]), torch.tensor([True, False, False]))
    m = ev.compute()
    assert math.isnan(m["AP50_SA"])                                      # sin GT válida en SA
    assert m["AP50_LI"] == pytest.approx(0.0) or math.isnan(m["AP50_LI"])


def test_segmentation_evaluator():
    gt = torch.zeros(1, 10, 10, dtype=torch.long)
    gt[0, :5] = 1
    pred = gt.clone()
    pred[0, 4] = 0                                                     # se pierde una fila de 10 px de SA
    ev = SegmentationEvaluator()
    ev.add(pred, gt)
    m = ev.compute()
    assert m["dice_SA"] == pytest.approx(2 * 40 / (40 + 50))
    assert m["iou_SA"] == pytest.approx(40 / 50)
