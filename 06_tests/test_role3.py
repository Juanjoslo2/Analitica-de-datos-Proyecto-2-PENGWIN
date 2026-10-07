"""y4xul: salida principal/secundario (``role3``), salto a resolución completa y pooling, sobre phantoms.

Tres cambios que la Fase 2 dejó sin probar:

    role3         cada vóxel de hueso es del fragmento principal o de un secundario. Es una clase
                  densa: no depende de marcar una superficie de fractura de 1-2 px
    fullres skip  el decodificador recibe C0 (bloque 1 antes del pooling); sin él, de stride 2 a 1
                  solo hay una interpolación
    pool          ``max`` (FundidoraPC) o ``avg``

Los tres son opcionales: la configuración base no cambia ni un parámetro.
"""

import numpy as np
import pytest
import torch

from pengwin.data.targets import role_target, slice_targets
from pengwin.losses.losses import MultiTaskLoss
from pengwin.models.pengwin_net import PengwinNet, count_parameters
from pengwin.postprocess.instances import (
    PP_DEFAULTS, instance_kwargs, resolve_postprocess, separate_instances, separate_region_role,
)
from pengwin.utils.config import load_config

SP = (1.0, 1.0, 1.0)
PARAMETROS_BASE = 2_921_886          # el modelo de la semana 10; no debe moverse


def _cfg(**model):
    cfg = load_config("configs/base.yaml")
    cfg["model"] = {**cfg["model"], "pretrained": False, **model}
    return cfg


# --------------------------------------------------------------------- objetivo
def test_role_target_separa_principal_y_secundarios_por_region():
    lab = np.array([[0, 1, 2, 10], [11, 12, 20, 0], [21, 22, 30, 0]], np.uint8)
    assert role_target(lab).tolist() == [[0, 1, 2, 2], [1, 2, 2, 0], [1, 2, 2, 0]]
    assert role_target(lab).dtype == np.int64


def test_slice_targets_entrega_role3_del_mismo_hueso():
    lab = np.zeros((16, 16), np.uint8)
    lab[2:8, 2:8], lab[8:12, 2:8] = 11, 12
    t = slice_targets(lab)
    assert ((t["role3"] > 0) == (lab > 0)).all(), "principal + secundario = hueso"
    assert (t["role3"][lab == 11] == 1).all() and (t["role3"][lab == 12] == 2).all()


# --------------------------------------------------------------------- modelo
def test_la_config_base_no_cambia():
    net = PengwinNet(_cfg()["model"])
    assert count_parameters(net)["total"] == PARAMETROS_BASE
    assert "role_logits" not in net(torch.rand(1, 3, 64, 64))


def test_role3_y_el_salto_anaden_salidas_sin_tocar_las_otras_cabezas():
    base = PengwinNet(_cfg()["model"])
    net = PengwinNet(_cfg(seg_outputs=["semantic4", "fracture_edge", "role3"], seg_fullres_skip=True)["model"])
    out = net(torch.rand(2, 3, 64, 64))
    assert out["role_logits"].shape == (2, 3, 64, 64)
    assert out["seg_logits"].shape == (2, 4, 64, 64) and out["edge_logits"].shape == (2, 1, 64, 64)
    a, b = count_parameters(base), count_parameters(net)
    assert all(a[k] == b[k] for k in ("backbone", "neck", "cls_head", "det_head"))
    assert 0 < b["seg_head"] - a["seg_head"] < 12_000, "el salto y la salida nueva cuestan < 12 k parámetros"


def test_el_gradiente_de_role3_llega_por_el_salto_a_resolucion_completa():
    net = PengwinNet(_cfg(seg_outputs=["semantic4", "fracture_edge", "role3"], seg_fullres_skip=True)["model"])
    x = torch.rand(1, 3, 64, 64, requires_grad=True)
    net(x)["role_logits"].sum().backward()
    assert net.seg_head.role.weight.grad.abs().sum() > 0
    assert net.backbone.stages[0].conv.weight.grad.abs().sum() > 0
    # C0 entra a stride 1: la conv del último paso recibe width//2 + 32 canales
    assert net.seg_head.up1[0].in_channels == 32 + 32


def test_el_salto_usa_caracteristicas_antes_del_pooling():
    net = PengwinNet(_cfg(seg_fullres_skip=True)["model"]).eval()
    x = torch.rand(1, 3, 64, 64)
    c0, feats = net.backbone.forward_with_stem(x)
    assert c0.shape == (1, 32, 64, 64) and feats[0].shape == (1, 32, 32, 32)
    for a, b in zip(feats, net.backbone(x)):
        assert torch.equal(a, b), "forward_with_stem no cambia C1-C4"


def test_pool_avg_cambia_solo_la_operacion_y_rechaza_valores_raros():
    a, b = PengwinNet(_cfg()["model"]), PengwinNet(_cfg(pool="avg")["model"])
    assert count_parameters(a) == count_parameters(b), "el pooling no tiene parámetros"
    assert isinstance(b.backbone.stages[0].pool, torch.nn.AvgPool2d)
    assert isinstance(a.backbone.stages[0].pool, torch.nn.MaxPool2d)
    with pytest.raises(ValueError):
        PengwinNet(_cfg(pool="mediana")["model"])


# --------------------------------------------------------------------- pérdida
def test_la_perdida_suma_role3_dentro_de_seg_y_siguen_siendo_tres_terminos():
    cfg = _cfg(seg_outputs=["semantic4", "fracture_edge", "role3"])
    net, loss = PengwinNet(cfg["model"]), MultiTaskLoss(cfg)
    lab = np.zeros((64, 64), np.uint8)
    lab[8:40, 8:40], lab[40:56, 8:40] = 11, 12
    t = slice_targets(lab)
    batch = {"semantic": torch.from_numpy(t["semantic"])[None], "edge": torch.from_numpy(t["edge"])[None, None],
             "role3": torch.from_numpy(t["role3"])[None], "boxes": torch.from_numpy(t["boxes"])[None],
             "present": torch.from_numpy(t["present"].astype(np.float32))[None], "ignore": torch.from_numpy(t["ignore"])[None]}
    parts = loss(net(torch.rand(1, 3, 64, 64)), batch)
    assert {"role3_ce", "role3_dice"} <= set(parts)
    esperado = parts["seg_ce"] + parts["seg_dice"] + parts["edge_bce"] + parts["edge_dice"] + parts["role3_ce"] + parts["role3_dice"]
    assert torch.isclose(parts["seg"], esperado)
    assert loss.terms == ("cls", "det", "seg")


# --------------------------------------------------------------------- posproceso "role"
def _coxal_con_dos_secundarios():
    """Principal 20×20×8 con dos secundarios pegados: uno grueso y una lámina de 2 vóxeles.

    Todo se toca y el borde predicho es 0: es el caso que ``edge`` y ``edt`` no separan.
    """
    gt = np.zeros((10, 40, 40), np.uint8)
    gt[1:9, 4:24, 4:24] = 11           # principal
    gt[1:9, 24:34, 4:24] = 12          # secundario grueso, pegado por y = 24
    gt[1:9, 4:24, 24:26] = 13          # lámina de 2 vóxeles, pegada por x = 24
    sem = np.where(gt > 0, 2, 0).astype(np.uint8)
    role = (role_target(gt) == 2).astype(np.float32)
    return gt, sem, role, np.zeros(gt.shape, np.float32)


def test_role_separa_lo_que_edt_fusiona_y_no_parte_el_principal():
    gt, sem, role, edge = _coxal_con_dos_secundarios()
    lab = separate_instances(sem, edge, SP, method="role", role=role, role_seed_depth_mm=1.5, min_fragment_cm3=0.05)
    assert sorted(np.unique(lab).tolist()) == [0, 11, 12, 13]
    assert (lab[gt == 11] == 11).all(), "el principal queda entero y es el de mayor volumen"
    assert (lab[gt == 12] == 12).all()
    assert (lab[gt == 13] == 13).all(), "la lámina de 2 vóxeles conserva su semilla"
    fusionado = separate_instances(sem, edge, SP, method="edt", seed_depth_mm=5.0)
    assert len(np.unique(fusionado)) == 2, "sin borde, edt deja un solo fragmento"


def test_role_separa_dos_secundarios_que_se_tocan_por_un_cuello():
    sem = np.zeros((12, 40, 60), np.uint8)
    sem[1:11, 4:36, 4:56] = 2
    role = np.zeros(sem.shape, np.float32)
    role[1:11, 4:36, 30:56] = 1.0       # zona secundaria: dos bloques unidos por un cuello fino
    role[1:11, 18:22, 30:56] = 0.0
    role[4:8, 18:22, 42:44] = 1.0       # el cuello
    lab = separate_instances(sem, np.zeros(sem.shape, np.float32), SP, method="role", role=role,
                             role_seed_depth_mm=2.0, seed_min_cm3=0.01)
    assert lab[5, 10, 45] != lab[5, 30, 45], "los dos secundarios quedan separados"
    assert lab[5, 10, 10] == 11 and lab[5, 10, 45] != 11


def test_role_sin_mapa_equivale_a_edt_y_el_umbral_se_respeta():
    gt, sem, role, edge = _coxal_con_dos_secundarios()
    a = separate_instances(sem, edge, SP, method="role", role=None, seed_depth_mm=5.0)
    b = separate_instances(sem, edge, SP, method="edt", seed_depth_mm=5.0)
    assert (a == b).all()
    todo_principal = separate_region_role(sem == 2, edge, SP, role * 0.4, role_threshold=0.5)
    assert todo_principal.max() == 1, "por debajo del umbral nada es secundario"


def test_role_respeta_la_taxonomia_de_diez_fragmentos():
    sem = np.zeros((6, 12, 130), np.uint8)
    sem[1:5, 2:10, 2:128] = 1
    role = np.zeros(sem.shape, np.float32)
    for k in range(14):                 # 14 secundarios sueltos dentro del sacro
        role[1:5, 2:10, 12 + 8 * k:16 + 8 * k] = 1.0
    lab = separate_instances(sem, np.zeros(sem.shape, np.float32), SP, method="role", role=role,
                             role_seed_depth_mm=0.5, min_fragment_cm3=0.001)
    ids = np.unique(lab)
    assert ids.max() <= 10 and len(ids) - 1 <= 10


def test_el_suavizado_en_z_repara_los_cortes_donde_falta_el_papel():
    """La red decide corte a corte; si omite el secundario en unos cortes, sin suavizar queda en rodajas."""
    gt = np.zeros((24, 30, 30), np.uint8)
    gt[1:23, 4:26, 4:16], gt[1:23, 4:26, 16:26] = 11, 12
    sem = np.where(gt > 0, 2, 0).astype(np.uint8)
    role = (role_target(gt) == 2).astype(np.float32)
    role[[6, 12, 13, 18]] = 0.0          # cuatro cortes sin secundario
    edge = np.zeros(gt.shape, np.float32)
    roto = separate_instances(sem, edge, SP, method="role", role=role, min_fragment_cm3=0.05)
    suave = separate_instances(sem, edge, SP, method="role", role=role, min_fragment_cm3=0.05, role_smooth_mm=2.0)
    assert len(np.unique(roto)) - 1 >= 4, "sin suavizar, el secundario sale en rodajas"
    assert sorted(np.unique(suave).tolist()) == [0, 11, 12]
    assert (suave[gt == 12] == 12).mean() > 0.95 and (suave[gt == 11] == 11).mean() > 0.95


def test_la_config_de_posproceso_conoce_role():
    assert PP_DEFAULTS["role_threshold"] == 0.5 and PP_DEFAULTS["role_seed_depth_mm"] == 1.5
    assert PP_DEFAULTS["role_smooth_mm"] == 0.0
    kw = instance_kwargs(resolve_postprocess({"instance_method": "role", "role_seed_depth_mm": 2.0}))
    assert kw["method"] == "role" and kw["role_seed_depth_mm"] == 2.0 and kw["role_threshold"] == 0.5


def test_role3_viaja_al_dispositivo_con_el_resto_del_lote():
    """En CPU no se nota; en GPU la pérdida fallaba porque ``role3`` se quedaba en CPU."""
    from pengwin.training.engine import TENSOR_KEYS
    assert "role3" in TENSOR_KEYS


# --------------------------------------------------------------------- Dice ponderado y selección
def test_el_dice_ponderado_es_la_media_ponderada_de_los_dice_por_clase():
    from pengwin.losses.losses import soft_dice_loss
    g = torch.zeros(1, 2, 8, 8)
    g[0, 0, :4], g[0, 1, 4:] = 1, 1
    p = g.clone()
    p[0, 1] = 0.0                                   # el secundario no se predice: D_sec ≈ 0
    simple = soft_dice_loss(p, g, eps=1e-6)
    pesado = soft_dice_loss(p, g, eps=1e-6, weights=torch.tensor([1.0, 3.0]))
    assert torch.isclose(simple, torch.tensor(0.5), atol=1e-4)
    assert torch.isclose(pesado, torch.tensor(0.75), atol=1e-4), "fallar el secundario cuesta 3/4 con pesos (1, 3)"
    assert torch.isclose(soft_dice_loss(p, g, weights=torch.tensor([1.0, 1.0])), soft_dice_loss(p, g))


def test_las_configs_y5_y6_solo_cambian_el_peso_del_dice_y_la_seleccion():
    base, y5, y6 = (load_config(f"configs/tuning/{n}.yaml") for n in ("y4_fullres_role", "y5_role_dw3", "y6_role_dw6"))
    assert y5["model"] == base["model"] == y6["model"]
    assert y5["loss"]["role3_dice_weights"] == [1.0, 3.0] and y6["loss"]["role3_dice_weights"] == [1.0, 6.0]
    assert MultiTaskLoss(y6).role3_dice_weights.tolist() == [1.0, 6.0]
    assert MultiTaskLoss(base).role3_dice_weights.tolist() == [1.0, 1.0]
    assert "role_dice_secundario" in y5["train"]["selection_metrics"]


def test_el_criterio_de_seleccion_acepta_metricas_propias():
    from pengwin.training.engine import selection_score
    m = {"mAP@[.50:.95]": 0.8, "dice_hueso": 0.9, "cls_f1_macro": 1.0, "role_dice_secundario": 0.5}
    assert selection_score(m) == pytest.approx(0.9)
    assert selection_score(m, ["mAP@[.50:.95]", "dice_hueso", "cls_f1_macro", "role_dice_secundario"]) == pytest.approx(0.8)
