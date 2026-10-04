"""Backbone FundidoraPC extendida + CBAM + 3 cabezas: formas, ablación, transfer learning y gradiente."""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from conftest import REPO  # noqa: E402
from pengwin.models.backbone import FundidoraBackbone, load_fundidora_weights  # noqa: E402
from pengwin.models.cbam import CBAM  # noqa: E402
from pengwin.models.pengwin_net import PengwinNet, count_parameters  # noqa: E402
from pengwin.utils.config import load_config  # noqa: E402

BASE = load_config(REPO / "configs" / "base.yaml")


class CourseFundidoraPC(nn.Module):
    """Copia de la FundidoraPC de S6_Bloque1 (conv con sesgo + BN + ReLU + MaxPool, 4 bloques)."""

    def __init__(self):
        super().__init__()
        for k, (ci, co) in enumerate([(3, 32), (32, 64), (64, 128), (128, 256)], start=1):
            setattr(self, f"conv{k}", nn.Conv2d(ci, co, 3, padding=1))
            setattr(self, f"bn{k}", nn.BatchNorm2d(co))

    def forward(self, x):
        for k in range(1, 5):
            x = nn.functional.max_pool2d(torch.relu(getattr(self, f"bn{k}")(getattr(self, f"conv{k}")(x))), 2)
        return x


def test_output_shapes():
    model = PengwinNet(BASE["model"]).eval()
    out = model(torch.rand(2, 3, 256, 256))
    assert out["cls_logits"].shape == (2, 3)
    assert out["det_scores"].shape == (2, 3, 32, 32)               # stride 8 -> 32×32 celdas
    assert out["det_ltrb"].shape == (2, 3, 4, 32, 32) and (out["det_ltrb"] > 0).all()
    assert out["seg_logits"].shape == (2, 4, 256, 256)
    assert out["edge_logits"].shape == (2, 1, 256, 256)


def test_cbam_ablation_changes_only_cbam():
    with_cbam = PengwinNet(BASE["model"])
    without = PengwinNet({**BASE["model"], "cbam": False})
    assert sum(isinstance(m, CBAM) for m in with_cbam.modules()) == 2
    assert sum(isinstance(m, CBAM) for m in without.modules()) == 0
    cbam_params = sum(p.numel() for m in with_cbam.modules() if isinstance(m, CBAM) for p in m.parameters())
    assert count_parameters(with_cbam)["total"] - count_parameters(without)["total"] == cbam_params


def test_course_weights_load_and_reproduce_course_output(tmp_path):
    """Con residuales recién creadas (identidad) y sin CBAM, el backbone cargado debe dar
    EXACTAMENTE la salida de la FundidoraPC del curso, sesgo de la conv incluido."""
    torch.manual_seed(0)
    course = CourseFundidoraPC()
    with torch.no_grad():
        for k in range(1, 5):
            bn = getattr(course, f"bn{k}")
            bn.running_mean.uniform_(-0.5, 0.5)
            bn.running_var.uniform_(0.5, 2.0)
            bn.weight.uniform_(0.5, 1.5)
            bn.bias.uniform_(-0.2, 0.2)
    path = tmp_path / "fundidora_curso.pth"
    torch.save({"model": {f"backbone.{k}": v for k, v in course.state_dict().items()}}, path)

    bb = FundidoraBackbone(residual=True, cbam=False)
    report = load_fundidora_weights(bb, path)
    assert report["loaded"] == ["stage1", "stage2", "stage3", "stage4"] and not report["skipped"]
    x = torch.rand(2, 3, 64, 64)
    course.eval(), bb.eval()
    assert torch.allclose(bb(x)[-1], course(x), atol=1e-5)


def test_grayscale_first_conv_is_adapted(tmp_path):
    sd = {"blocks.0.0.weight": torch.randn(32, 1, 3, 3)}
    path = tmp_path / "gris.pth"
    torch.save(sd, path)
    bb = FundidoraBackbone()
    report = load_fundidora_weights(bb, path)
    assert report["loaded"] == ["stage1"]
    assert torch.allclose(bb.stages[0].conv.weight.sum(1, keepdim=True), sd["blocks.0.0.weight"], atol=1e-5)


def test_gradient_reaches_backbone_and_three_heads():
    from pengwin.losses.losses import MultiTaskLoss

    model = PengwinNet(BASE["model"])
    loss_fn = MultiTaskLoss(BASE)
    sem = torch.zeros(2, 64, 64, dtype=torch.long)
    sem[:, 20:40, 10:30] = 2
    batch = {"semantic": sem, "edge": torch.zeros(2, 1, 64, 64), "present": torch.tensor([[0.0, 1, 0]] * 2),
             "boxes": torch.tensor([[[0.0, 0, 0, 0], [10, 20, 30, 40], [0, 0, 0, 0]]] * 2),
             "ignore": torch.zeros(2, 3, dtype=torch.bool)}
    loss_fn(model(torch.rand(2, 3, 64, 64)), batch)["total"].backward()
    for name in ("backbone", "cls_head", "det_head", "seg_head", "neck"):
        mod = getattr(model, name)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in mod.parameters()), name
    assert all(p.grad is not None for p in model.backbone.shared_parameters())


def test_gamma_gates_start_as_identity_and_report():
    from pengwin.models.cbam import CBAM

    cbam = CBAM(16, gamma=True).eval()
    x = torch.rand(2, 16, 8, 8)
    assert torch.equal(cbam(x), x), "con γ = 0 el CBAM debe arrancar como identidad"
    model = PengwinNet({**BASE["model"], "cbam_gamma": True, "neck_gamma": True})
    g = model.gammas()
    assert g["CBAM b3"] == g["CBAM b4"] == 0.0 and g["cuello C4"] == 1.0
    assert all(g[f"residual b{k}"] == 0.0 for k in range(1, 5))


def test_spatial_dropout_only_deep_blocks_and_only_in_train():
    cfg = {**BASE["model"], "spatial_dropout": 0.5, "spatial_dropout_blocks": [3, 4], "seg_spatial_dropout": 0.5}
    model = PengwinNet(cfg)
    drops = [type(st.drop).__name__ for st in model.backbone.stages]
    assert drops == ["Identity", "Identity", "Dropout2d", "Dropout2d"]
    x = torch.rand(1, 3, 64, 64)
    model.eval()
    with torch.no_grad():
        a, b = model(x)["seg_logits"], model(x)["seg_logits"]
    assert torch.equal(a, b), "en eval el dropout no actúa"
    model.train()
    c4 = model.backbone(x)[-1]
    zeros = (c4.flatten(2).abs().sum(-1) == 0).float().mean()
    assert zeros > 0.2, "Dropout2d debe apagar canales completos"
