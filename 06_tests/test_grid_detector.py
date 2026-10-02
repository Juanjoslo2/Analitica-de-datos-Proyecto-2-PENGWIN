"""Grid anchor-free de stride 8: asignación de celdas, ignorar y decodificación (ida y vuelta)."""

import pytest

torch = pytest.importorskip("torch")

from pengwin.detection.boxes import box_iou  # noqa: E402
from pengwin.detection.grid import assign_targets, cell_centers, decode  # noqa: E402

G, S = 32, 8


def _gt(boxes, present=None, ignore=None):
    b = torch.tensor([boxes], dtype=torch.float32)                       # (1, K, 4)
    k = b.shape[1]
    p = torch.tensor([present if present is not None else [True] * k])
    i = torch.tensor([ignore if ignore is not None else [False] * k])
    return b, p, i


def test_cell_centers():
    c = cell_centers(G, S)
    assert c[0, 0].tolist() == [4.0, 4.0] and c[0, 1].tolist() == [12.0, 4.0] and c[1, 0].tolist() == [4.0, 12.0]


def test_positives_are_inside_and_near_center():
    b, p, i = _gt([[40, 40, 120, 104], [0, 0, 0, 0], [0, 0, 0, 0]], [True, False, False])
    t = assign_targets(b, p, i, G, S, radius=1.5)
    ys, xs = torch.nonzero(t["pos"][0, 0], as_tuple=True)
    cx, cy = (xs + 0.5) * S, (ys + 0.5) * S
    assert len(xs) > 0 and (cx > 40).all() and (cx < 120).all() and (cy > 40).all() and (cy < 104).all()
    assert ((cx - 80).abs() < 12).all() and ((cy - 72).abs() < 12).all()
    assert not t["pos"][0, 1:].any(), "regiones ausentes no tienen positivos"


def test_tiny_box_gets_one_positive():
    b, p, i = _gt([[17, 17, 21, 21]])
    t = assign_targets(b, p, i, G, S)
    assert int(t["pos"].sum()) == 1 and bool(t["pos"][0, 0, 2, 2])     # celda que contiene el centro (19, 19)


def test_ignored_box_masks_only_the_cells_it_contains():
    b, p, i = _gt([[40, 40, 100, 100]], [True], [True])
    t = assign_targets(b, p, i, G, S)
    assert not t["pos"].any(), "una caja ignorada no aporta positivos"
    c = cell_centers(G, S)
    inside = (c[..., 0] > 40) & (c[..., 0] < 100) & (c[..., 1] > 40) & (c[..., 1] < 100)
    assert torch.equal(~t["valid"][0, 0], inside)                     # ni positivo ni negativo dentro; negativo fuera


def test_overlapping_classes_get_their_own_boxes():
    b, p, i = _gt([[60, 60, 140, 140], [40, 40, 160, 160], [0, 0, 0, 0]], [True, True, False])
    t = assign_targets(b, p, i, G, S)
    both = t["pos"][0, 0] & t["pos"][0, 1]
    assert both.any(), "una celda puede ser positiva para dos regiones"
    y, x = torch.nonzero(both)[0]
    l0, l1 = t["ltrb"][0, 0, 0, y, x], t["ltrb"][0, 1, 0, y, x]
    assert l1 - l0 == pytest.approx(20.0)                               # distancias por clase distintas


def test_decode_round_trip_recovers_gt():
    gt = [[40, 48, 120, 104], [130, 30, 210, 200], [5, 5, 30, 30]]
    b, p, i = _gt(gt)
    t = assign_targets(b, p, i, G, S)
    logits = torch.where(t["pos"], torch.tensor(5.0), torch.tensor(-5.0))
    dets = decode(logits, t["ltrb"], S, 256, score_threshold=0.5, nms_iou=0.5, max_per_class=1)[0]
    assert sorted(dets["labels"].tolist()) == [0, 1, 2]
    for box, lab in zip(dets["boxes"], dets["labels"]):
        assert float(box_iou(box[None], b[0, lab][None])) == pytest.approx(1.0, abs=1e-5)
