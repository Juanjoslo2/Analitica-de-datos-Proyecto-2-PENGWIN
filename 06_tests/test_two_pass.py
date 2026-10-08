"""y4xul: segunda pasada por hueso a alta resolución (data/two_pass.py), sobre phantoms.

Un solo modelo, dos pasadas: corte completo (máscara previa en cero) y recorte por hueso del caché
de alta resolución con la máscara que predijo la pasada 1. Aquí se fija la geometría (recortar y
pegar de vuelta es exacto), que ampliar los canales de entrada no cambia el modelo de origen, que
el dataset mezcla las dos pasadas sin usar el ground truth como máscara y que la EMA promedia.
"""

import json

import numpy as np
import pytest
import torch

from pengwin.data.two_pass import (
    PRIOR_BOXES, PRIOR_SEM, TwoPassSlices, crop_square, paste_window, prior_boxes, roi_input, roi_window,
)
from pengwin.models.pengwin_net import PengwinNet, load_expanding_input
from pengwin.utils.config import load_config


def _cfg(**model):
    cfg = load_config("configs/tuning/y7_refine.yaml")
    cfg["model"] = {**cfg["model"], **model}
    return cfg


# --------------------------------------------------------------------- geometría
def test_la_ventana_contiene_la_caja_y_cae_en_multiplos_de_la_escala():
    x0, y0, side = roi_window((100, 60, 180, 150), scale=2, out=256)
    assert side == 256 and x0 % 2 == 0 and y0 % 2 == 0, "un hueso que cabe se recorta 1:1 (resolución nativa)"
    assert x0 <= 200 and x0 + side >= 360 and y0 <= 120 and y0 + side >= 300
    assert roi_window((10, 10, 200, 230), scale=2, out=256)[2] == (220 + 16) * 2, "si no cabe, la ventana crece"


def test_recortar_fuera_del_lienzo_rellena_con_ceros():
    a = np.arange(16, dtype=np.uint8).reshape(4, 4) + 1
    c = crop_square(a, -1, 2, 4)
    assert c.shape == (4, 4) and (c[:, 0] == 0).all() and (c[2:] == 0).all()
    assert (c[:2, 1:] == a[2:, :3]).all()


def test_pegar_de_vuelta_promedia_por_bloque_y_respeta_la_mascara():
    hi = torch.zeros(8, 8)
    hi[:4, :4] = 1.0                              # bloque 4x4 de alta resolución = 2x2 en baja
    dst = np.full((6, 6), -1.0, np.float32)
    where = np.ones((6, 6), bool)
    where[1, 1] = False
    paste_window(dst, hi, x0=2, y0=2, side=8, scale=2, where=where)
    assert dst[1, 1] == -1.0, "fuera de la máscara no se toca"
    assert dst[1, 2] == 1.0 and dst[2, 1] == 1.0 and dst[2, 2] == 1.0 and dst[3, 3] == 0.0
    assert dst[0, 0] == -1.0 and dst[5, 5] == -1.0


def test_prior_boxes_ignora_las_regiones_diminutas():
    sem = np.zeros((2, 32, 32), np.uint8)
    sem[0, 4:14, 6:20] = 2
    sem[1, 0, :3] = 1                              # 3 píxeles: no merece un recorte
    b = prior_boxes(sem)
    assert b[0, 1].tolist() == [6, 4, 20, 14] and not b[1].any() and not b[0, 0].any()


def test_la_entrada_de_la_segunda_pasada_es_la_misma_en_entrenamiento_y_en_inferencia():
    hi = (np.random.default_rng(0).random((5, 64, 64)) * 255).astype(np.uint8)
    prior = np.zeros((32, 32), bool)
    prior[8:20, 8:20] = True
    win = roi_window((8, 8, 20, 20), scale=2, out=64)
    img, pri = roi_input(hi, 2, 1, prior, win, 2, 64)
    assert img.shape == (3, 64, 64) and pri.shape == (64, 64) and win[2] == 64
    x0, y0, side = win
    assert torch.allclose(img[1], torch.from_numpy(crop_square(hi[2], x0, y0, side)).float() / 255)
    assert pri.sum() == 4 * prior.sum(), "la máscara de 256 se amplía con vecino más cercano"


# --------------------------------------------------------------------- modelo
def test_ampliar_los_canales_de_entrada_no_cambia_el_modelo_de_origen():
    origen = PengwinNet(_cfg(in_channels=3)["model"]).eval()
    nuevo = PengwinNet(_cfg()["model"]).eval()
    assert load_expanding_input(nuevo, origen.state_dict()) == 1
    x = torch.rand(2, 3, 64, 64)
    a = origen(x)
    b = nuevo(torch.cat([x, torch.zeros(2, 1, 64, 64)], 1))
    c = nuevo(torch.cat([x, torch.ones(2, 1, 64, 64)], 1))
    for k in ("seg_logits", "role_logits", "det_scores", "cls_logits"):
        assert torch.allclose(a[k], b[k], atol=1e-5), k
        assert torch.allclose(a[k], c[k], atol=1e-5), "el canal nuevo arranca en cero: aún no influye"
    assert nuevo.backbone.stages[0].conv.weight.shape[1] == 4


def test_es_un_solo_modelo_con_las_tres_cabezas():
    net = PengwinNet(_cfg()["model"])
    out = net(torch.rand(1, 4, 64, 64))
    assert {"cls_logits", "det_scores", "det_ltrb", "seg_logits", "edge_logits", "role_logits"} <= set(out)


# --------------------------------------------------------------------- dataset
def _mini_cache(tmp_path, con_prior=True):
    lo, hi = tmp_path / "lo", tmp_path / "hi"
    Z = 4
    lab_hi = np.zeros((Z, 128, 128), np.uint8)
    lab_hi[:, 20:80, 20:60], lab_hi[:, 80:100, 20:60] = 11, 12
    lab_lo = lab_hi[:, ::2, ::2]
    for d, lab, size in ((lo, lab_lo, 64), (hi, lab_hi, 128)):
        c = d / "001"
        c.mkdir(parents=True)
        np.save(c / "image.npy", (lab > 0).astype(np.uint8) * 200)
        np.save(c / "label.npy", lab)
        (c / "meta.json").write_text(json.dumps({"image_size": size, "pixel_mm": 128 / size, "context_offset": 1,
                                                 "spacing_zyx": [1.0, 1.0, 1.0], "native_shape_zyx": [Z, 128, 128],
                                                 "bone_slices": list(range(Z))}), encoding="utf-8")
    np.save(lo / "001" / "edge.npy", np.zeros(lab_lo.shape, np.uint8))
    if con_prior:
        sem = np.where(lab_lo > 0, 2, 0).astype(np.uint8)
        sem[:, 10:14] = 0                          # la máscara previa NO es el ground truth: le falta una franja
        np.save(lo / "001" / PRIOR_SEM, sem)
        np.save(lo / "001" / PRIOR_BOXES, prior_boxes(sem))
    return lo, hi


def _cfg_mini():
    cfg = _cfg()
    cfg["data"] = {**cfg["data"], "image_size": 64, "empty_slice_fraction": 0.0}
    return cfg


def test_el_dataset_mezcla_cortes_completos_y_recortes_con_la_mascara_previa(tmp_path):
    lo, hi = _mini_cache(tmp_path)
    ds = TwoPassSlices(lo, hi, ["001"], _cfg_mini(), train=True, augment=False)
    assert len(ds) == 8 and ds.scale == 2, "4 cortes completos + 4 recortes"
    completo, recorte = ds[0], ds[5]
    assert completo["image"].shape == recorte["image"].shape == (4, 64, 64)
    assert completo["image"][3].abs().sum() == 0, "pasada 1: máscara previa en cero"
    assert recorte["image"][3].sum() > 0, "pasada 2: lleva la máscara previa"
    assert set(completo) == set(recorte), "las dos pasadas comparten claves: van en el mismo lote"
    assert set(np.unique(recorte["role3"].numpy()).tolist()) == {0, 1, 2}
    # la máscara previa es la predicha (con su franja de menos), no la región real del recorte
    assert (recorte["image"][3] > 0).sum() < (recorte["semantic"] > 0).sum()
    assert recorte["pixel_mm"] < completo["pixel_mm"], "el recorte tiene más resolución que el corte completo"


def test_sin_mascara_previa_el_dataset_no_arranca(tmp_path):
    lo, hi = _mini_cache(tmp_path, con_prior=False)
    with pytest.raises(FileNotFoundError, match="build_prior"):
        TwoPassSlices(lo, hi, ["001"], _cfg_mini(), train=True)


def test_cada_epoca_elige_recortes_reproducibles(tmp_path):
    lo, hi = _mini_cache(tmp_path)
    ds = TwoPassSlices(lo, hi, ["001"], _cfg_mini(), train=True, roi_per_slice=0.5)
    assert ds.n_roi == 2
    ds.set_epoch(3)
    a = ds.roi_sel.copy()
    ds.set_epoch(3)
    assert (ds.roi_sel == a).all()


# --------------------------------------------------------------------- inferencia y EMA
def test_la_segunda_pasada_solo_cambia_papel_y_borde_dentro_del_hueso(tmp_path):
    from pengwin.inference.volume import predict_case_two_pass
    lo, hi = _mini_cache(tmp_path)
    cfg = _cfg_mini()
    torch.manual_seed(0)
    net = PengwinNet(cfg["model"]).eval()
    p = predict_case_two_pass(net, lo, hi, "001", cfg, torch.device("cpu"))
    assert p["role"].shape == p["role1"].shape == p["semantic"].shape
    fuera = p["semantic"] == 0
    assert np.array_equal(p["role"][fuera], p["role1"][fuera]), "fuera del hueso predicho no se toca"
    assert np.array_equal(p["edge"][fuera], p["edge1"][fuera])


def test_la_ema_promedia_los_pesos():
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
    m = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        m.weight.fill_(0.0)
    ema = AveragedModel(m, multi_avg_fn=get_ema_multi_avg_fn(0.9), use_buffers=True)
    ema.update_parameters(m)
    with torch.no_grad():
        m.weight.fill_(1.0)
    ema.update_parameters(m)
    assert ema.module.weight.item() == pytest.approx(0.1)
