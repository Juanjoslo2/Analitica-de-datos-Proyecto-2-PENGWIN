"""Semana 12: decodificador U-Net++ (C0 a resolución completa + supervisión profunda) y ancho del decodificador.

Lo que se fija aquí:
    C0          ``forward_full`` da el mapa del bloque 1 antes del MaxPool y NO cambia C1-C4
    compatible  sin las claves nuevas el modelo es el de siempre (mismos nombres y formas de
                parámetros): ``v2_last`` y el dashboard siguen cargando
    U-Net++     en train devuelve dos salidas intermedias para la supervisión profunda; en eval
                solo la final, con la misma forma que el decodificador de siempre
    pérdida     L_seg se promedia sobre las tres salidas y siguen siendo 3 términos; el gradiente
                llega a C0 (bloque 1 del backbone) y a las cabezas intermedias
    ancho       ``seg_width`` achica solo el decodificador de segmentación
"""

import pytest

torch = pytest.importorskip("torch")

from conftest import REPO  # noqa: E402
from pengwin.losses.losses import MultiTaskLoss  # noqa: E402
from pengwin.models.pengwin_net import PengwinNet, count_parameters  # noqa: E402
from pengwin.utils.config import load_config  # noqa: E402

BASE = load_config(REPO / "configs" / "base.yaml")
UNETPP = {**BASE["model"], "seg_decoder": "unetpp", "seg_deep_supervision": True}


def _batch(b=2, h=64):
    torch.manual_seed(0)
    sem = torch.zeros(b, h, h, dtype=torch.long)
    sem[:, 10:30, 10:30] = 2
    sem[:, 34:50, 30:54] = 1
    edge = torch.zeros(b, 1, h, h)
    edge[:, 0, 19:21, 10:30] = 1
    boxes = torch.tensor([[34, 30, 54, 50], [10, 10, 30, 30], [0, 0, 0, 0]], dtype=torch.float32).repeat(b, 1, 1)
    return {"image": torch.rand(b, 3, h, h), "semantic": sem, "edge": edge, "boxes": boxes,
            "present": torch.tensor([1.0, 1.0, 0.0]).repeat(b, 1), "ignore": torch.zeros(b, 3, dtype=torch.bool)}


def test_forward_full_da_c0_y_no_cambia_c1_c4():
    net = PengwinNet(BASE["model"]).eval()
    x = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        feats = net.backbone(x)
        c0, feats_full = net.backbone.forward_full(x)
    assert c0.shape == (2, net.backbone.widths[0], 64, 64), "C0: bloque 1 antes del MaxPool, stride 1"
    for a, b in zip(feats, feats_full):
        assert torch.equal(a, b)


def test_sin_claves_nuevas_el_modelo_es_el_de_siempre():
    viejo = {k: tuple(v.shape) for k, v in PengwinNet(BASE["model"]).state_dict().items()}
    explicito = {k: tuple(v.shape) for k, v in
                 PengwinNet({**BASE["model"], "seg_decoder": "unet", "seg_width": 64}).state_dict().items()}
    assert viejo == explicito
    assert not any("nodes" in k for k in viejo)


def test_unetpp_salidas_en_train_y_en_eval():
    net = PengwinNet(UNETPP)
    x = torch.rand(2, 3, 64, 64)
    net.train()
    out = net(x)
    assert out["seg_logits"].shape == (2, 4, 64, 64) and out["edge_logits"].shape == (2, 1, 64, 64)
    assert len(out["seg_aux"]) == 2
    for a in out["seg_aux"]:
        assert a["seg_logits"].shape == (2, 4, 64, 64) and a["edge_logits"].shape == (2, 1, 64, 64)
    net.eval()
    with torch.no_grad():
        out = net(x)
    assert "seg_aux" not in out, "en inferencia solo sale X^{0,3}: TTA y posproceso no cambian"
    assert out["det_scores"].shape == (2, 3, 8, 8)


def test_unetpp_sin_supervision_profunda_no_devuelve_auxiliares():
    net = PengwinNet({**UNETPP, "seg_deep_supervision": False}).train()
    assert "seg_aux" not in net(torch.rand(1, 3, 64, 64))


def test_unetpp_solo_admite_semantica_y_borde():
    with pytest.raises(ValueError):
        PengwinNet({**UNETPP, "seg_outputs": ["semantic4", "core3"]})


def test_supervision_profunda_promedia_lseg_y_llega_a_c0():
    cfg = {**BASE, "model": UNETPP}
    net = PengwinNet(UNETPP).train()
    loss_fn = MultiTaskLoss(cfg)
    batch = _batch()
    parts = loss_fn(net(batch["image"]), batch)
    assert set(loss_fn.terms) == {"cls", "det", "seg"}, "siguen siendo tres términos"
    assert "seg_final" in parts and not torch.isclose(parts["seg"], parts["seg_final"])
    parts["total"].backward()
    g_c0 = net.backbone.stages[0].conv.weight.grad
    assert g_c0 is not None and g_c0.abs().sum() > 0
    for head in net.seg_head.semantic:
        assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0, "cada salida recibe gradiente"


def test_decodificador_de_siempre_no_trae_seg_final():
    net = PengwinNet(BASE["model"]).train()
    batch = _batch()
    parts = MultiTaskLoss(BASE)(net(batch["image"]), batch)
    assert "seg_final" not in parts


def test_seg_width_achica_solo_el_decodificador():
    ancho = count_parameters(PengwinNet(BASE["model"]))
    angosto = count_parameters(PengwinNet({**BASE["model"], "seg_width": 32}))
    assert angosto["seg_head"] < ancho["seg_head"] / 2
    assert {k: v for k, v in angosto.items() if k not in ("seg_head", "total")} == \
           {k: v for k, v in ancho.items() if k not in ("seg_head", "total")}


@pytest.mark.parametrize("nombre", ["c11_wd1e2", "c12_segw32", "c13_unetpp"])
def test_configs_de_la_semana_12_construyen_el_modelo(nombre):
    cfg = load_config(REPO / "configs" / "tuning" / f"{nombre}.yaml")
    net = PengwinNet(cfg["model"]).eval()
    with torch.no_grad():
        out = net(torch.rand(1, 3, 64, 64))
    assert out["seg_logits"].shape == (1, 4, 64, 64)
