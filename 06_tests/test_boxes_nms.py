"""IoU, GIoU y NMS propios con casos de control de resultado conocido."""

import pytest

torch = pytest.importorskip("torch")

from pengwin.detection.boxes import box_iou, cxcywh_to_xyxy, paired_iou_giou, xyxy_to_cxcywh  # noqa: E402
from pengwin.detection.nms import batched_nms, nms  # noqa: E402


def test_iou_known_values():
    a = torch.tensor([[0.0, 0, 10, 10]])
    b = torch.tensor([[0.0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30], [0, 0, 5, 5]])
    iou = box_iou(a, b)[0]
    assert iou.tolist() == pytest.approx([1.0, 50 / 150, 0.0, 25 / 100], abs=1e-6)


def test_giou_is_negative_for_disjoint_boxes():
    iou, giou = paired_iou_giou(torch.tensor([0.0, 0, 10, 10]), torch.tensor([20.0, 0, 30, 10]))
    assert float(iou) == 0.0
    assert float(giou) == pytest.approx(0 - (300 - 200) / 300)


def test_format_round_trip():
    b = torch.tensor([[3.0, 4, 11, 20]])
    assert torch.allclose(cxcywh_to_xyxy(xyxy_to_cxcywh(b)), b)


def test_nms_keeps_best_and_non_overlapping():
    boxes = torch.tensor([[0.0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60], [0, 0, 9, 10]])
    scores = torch.tensor([0.6, 0.9, 0.5, 0.3])
    keep = nms(boxes, scores, 0.5)
    assert keep.tolist() == [1, 2]                    # 0 y 3 los suprime la caja 1


def test_nms_threshold_is_strict():
    boxes = torch.tensor([[0.0, 0, 10, 10], [5, 0, 15, 10]])   # IoU = 1/3
    assert nms(boxes, torch.tensor([0.9, 0.8]), 0.34).tolist() == [0, 1]
    assert nms(boxes, torch.tensor([0.9, 0.8]), 0.33).tolist() == [0]


def test_batched_nms_does_not_cross_classes_and_caps():
    boxes = torch.tensor([[0.0, 0, 10, 10], [0, 0, 10, 10], [30, 30, 40, 40], [0, 0, 10, 10]])
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
    labels = torch.tensor([0, 1, 1, 0])
    keep = batched_nms(boxes, scores, labels, 0.5, max_per_class=1)
    assert sorted(keep.tolist()) == [0, 1]            # una por clase; la de clase 1 no la suprime la de clase 0


def test_nms_empty():
    assert nms(torch.zeros((0, 4)), torch.zeros(0)).numel() == 0
