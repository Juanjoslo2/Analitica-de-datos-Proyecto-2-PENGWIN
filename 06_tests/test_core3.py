"""F2B1: representación núcleo/borde (3 clases) sobre phantoms.

El núcleo se derivaba del borde predicho (``región & P(borde) < umbral``), así que la separación
de fragmentos dependía del recall de la cabeza de borde, medido en 0,193. Aquí el núcleo es una
clase propia. Estos tests fijan las cuatro piezas nuevas:

    objetivo    ``core_edge_target``: 0 fondo, 1 núcleo, 2 borde, y el borde SOLO entre
                fragmentos del mismo hueso
    salida      ``model.seg_outputs: [semantic4, core3]``, sin tocar las otras dos cabezas
    pérdida     CE ponderada + Dice de 3 clases DENTRO del término ``seg`` (siguen siendo 3
                términos: cls, det, seg [enunciado §4.1])
    posproceso  ``method="core3"``: semillas = componentes 3D del núcleo predicho
"""

import numpy as np
import pytest

from pengwin.data.targets import core_edge_target, fracture_edge_2d, fracture_edge_3d, slice_targets
from pengwin.postprocess.instances import (
    PP_DEFAULTS, instance_kwargs, resolve_postprocess, separate_instances,
)

SP = (1.0, 1.0, 1.0)


# --------------------------------------------------------------------- objetivo de 3 clases
def _hueso_partido():
    """Coxal izq. partido en dos (contacto en y = 16) y coxal der. pegado al izq. (x = 16).

    El contacto 11-12 es superficie de fractura; la interfaz 11-21 es la articulación
    sacroilíaca, que ya separa la semántica de 4 clases y NO es borde.
    """
    lab = np.zeros((6, 32, 32), np.uint8)
    lab[1:5, 4:16, 4:16] = 11          # coxal izq., principal
    lab[1:5, 16:24, 4:16] = 12         # coxal izq., secundario pegado al principal
    lab[1:5, 4:16, 16:24] = 21         # coxal der.: otra REGIÓN
    return lab


def test_el_borde_es_solo_entre_fragmentos_del_mismo_hueso():
    c = core_edge_target(_hueso_partido(), dilation_px=0)
    assert c[2, 15, 10] == 2 and c[2, 16, 10] == 2, "el contacto 11-12 es borde de fractura"
    assert c[2, 10, 15] == 1 and c[2, 10, 16] == 1, "la interfaz entre huesos distintos es núcleo"
    assert c[0, 10, 10] == 0, "fuera del hueso, fondo"
    assert sorted(np.unique(c).tolist()) == [0, 1, 2]


def test_las_tres_clases_parten_el_hueso_y_la_dilatacion_no_se_sale():
    lab = _hueso_partido()
    c0 = core_edge_target(lab, dilation_px=0)
    c2 = core_edge_target(lab, dilation_px=2)
    assert ((c0 > 0) == (lab > 0)).all(), "núcleo + borde = hueso"
    assert (c2[lab == 0] == 0).all(), "la dilatación del borde se queda dentro del hueso"
    assert (c2 == 2).sum() > (c0 == 2).sum() and ((c2 > 0) == (lab > 0)).all()


def test_el_objetivo_usa_el_borde_del_cache_si_se_le_pasa():
    """En entrenamiento el borde viene de ``edge.npy`` (borde 3D, dilatación de ``data.edge_file``)."""
    lab = _hueso_partido()
    del_cache = core_edge_target(lab, fracture_edge_3d(lab, 2))
    assert (del_cache == core_edge_target(lab, dilation_px=2)).all()


def test_slice_targets_devuelve_core3_coherente_con_el_borde():
    lab = _hueso_partido()[2]
    t = slice_targets(lab, edge=fracture_edge_2d(lab, 2))
    assert t["core3"].shape == lab.shape and t["core3"].dtype == np.int64
    assert ((t["core3"] == 2) == (t["edge"] > 0)).all(), "la clase 2 es el mismo borde"
    assert ((t["core3"] > 0) == (lab > 0)).all()


# --------------------------------------------------------------------- posproceso "core3"
def _bloques_pegados_con_nucleo_partido():
    """Coxal izq.: la región rellena la grieta y el borde predicho es 0 (la falla medida).

    El núcleo predicho, en cambio, viene partido en dos: 18 px de ancho el principal y 12 el
    secundario, con un hueco de 2 px. Los bloques son gruesos (6 px en z, 12 en y) para que el
    método por distancia tenga dónde poner semillas si se le pide profundidad.
    """
    sem = np.zeros((8, 20, 40), np.uint8)
    sem[1:7, 4:16, 4:36] = 2                 # región continua: no hay grieta que separar
    core = np.zeros(sem.shape, np.float32)
    core[1:7, 4:16, 4:22] = 0.9              # núcleo del principal
    core[1:7, 4:16, 24:36] = 0.9             # núcleo del secundario
    edge = np.zeros(sem.shape, np.float32)   # la cabeza de borde no predijo nada
    return sem, edge, core


def test_core3_separa_dos_bloques_pegados_cuando_el_nucleo_viene_partido():
    sem, edge, core = _bloques_pegados_con_nucleo_partido()
    lab = separate_instances(sem, edge, SP, method="core3", core=core)
    assert sorted(np.unique(lab[lab > 0]).tolist()) == [11, 12], "dos fragmentos del coxal izquierdo"
    assert (lab[1:7, 4:16, 4:22] == 11).all() and (lab[1:7, 4:16, 24:36] == 12).all()
    assert ((lab > 0) == (sem > 0)).all(), "el watershed reparte también el hueco entre núcleos"


def test_sin_nucleo_predicho_los_bloques_se_fusionan():
    """Cota inferior: con el borde en 0 y sin canal de núcleo, ni "edt" ni el respaldo de
    "core3" separan nada. Es exactamente la dependencia que F2B1 quita."""
    sem, edge, _ = _bloques_pegados_con_nucleo_partido()
    edt = separate_instances(sem, edge, SP, method="edt", seed_depth_mm=5.0)
    respaldo = separate_instances(sem, edge, SP, method="core3", edge_threshold=0.5)
    assert np.unique(edt[edt > 0]).tolist() == [11]
    assert np.unique(respaldo[respaldo > 0]).tolist() == [11]


def test_core3_respeta_max_fragments():
    """5 núcleos separados dentro de una sola región del sacro; max_fragments deja 3."""
    sem = np.zeros((6, 12, 50), np.uint8)
    sem[1:5, 3:9, 2:48] = 1
    core = np.zeros(sem.shape, np.float32)
    for k in range(5):
        core[1:5, 3:9, 3 + 9 * k:10 + 9 * k] = 0.9
    edge = np.zeros(sem.shape, np.float32)
    todos = separate_instances(sem, edge, SP, method="core3", core=core)
    tres = separate_instances(sem, edge, SP, method="core3", core=core, max_fragments=3)
    assert len(np.unique(todos[todos > 0])) == 5
    assert sorted(np.unique(tres[tres > 0]).tolist()) == [1, 2, 3], "las 3 semillas mayores"


def test_los_ids_siguen_la_taxonomia_pengwin():
    """SA 1-10, coxal izq. 11-20, coxal der. 21-30: el principal es el 1 / 11 / 21."""
    for region, esperado in ((1, [1, 2]), (2, [11, 12]), (3, [21, 22])):
        sem = np.zeros((6, 12, 24), np.uint8)
        sem[1:5, 3:9, 2:22] = region
        core = np.zeros(sem.shape, np.float32)
        core[1:5, 3:9, 2:13] = 0.9           # el núcleo mayor -> principal
        core[1:5, 3:9, 15:22] = 0.9
        lab = separate_instances(sem, np.zeros(sem.shape, np.float32), SP, method="core3", core=core)
        assert sorted(np.unique(lab[lab > 0]).tolist()) == esperado


# --------------------------------------------------------------------- parámetros nuevos de la config
def test_los_parametros_nuevos_llegan_al_separador():
    import inspect

    assert {"core_threshold", "core_seed_depth_mm"} <= set(PP_DEFAULTS)
    kw = instance_kwargs(resolve_postprocess({"instance_method": "core3"},
                                             {"core_threshold": 0.8, "core_seed_depth_mm": 2.0}))
    assert kw["method"] == "core3" and kw["core_threshold"] == 0.8 and kw["core_seed_depth_mm"] == 2.0
    assert set(kw) <= set(inspect.signature(separate_instances).parameters), "algún argumento no existe"


@pytest.mark.parametrize("umbral, n_esperado", [(0.5, 2), (0.95, 1)])
def test_core_threshold_llega_y_decide_cuantos_fragmentos_salen(umbral, n_esperado):
    """El núcleo del secundario tiene P = 0,7: con umbral 0,95 desaparece y queda un fragmento."""
    sem, edge, core = _bloques_pegados_con_nucleo_partido()
    core[1:7, 4:16, 4:22] = 0.99
    core[1:7, 4:16, 24:36] = 0.70
    pp = resolve_postprocess({"instance_method": "core3"}, {"core_threshold": umbral})
    lab = separate_instances(sem, edge, SP, core=core, **instance_kwargs(pp))
    assert len(np.unique(lab[lab > 0])) == n_esperado


def test_si_el_nucleo_predicho_sale_vacio_se_cae_al_nucleo_derivado():
    """Con un umbral imposible el método no se queda sin semillas: usa el núcleo derivado del
    borde (el respaldo de "edt") en vez de etiquetar las componentes crudas de la región."""
    sem, edge, core = _bloques_pegados_con_nucleo_partido()
    edge[1:7, 4:16, 22:24] = 0.9             # aquí el borde SÍ está predicho, en la grieta
    lab = separate_instances(sem, edge, SP, method="core3", core=core,
                             core_threshold=1.5, edge_threshold=0.5)
    assert sorted(np.unique(lab[lab > 0]).tolist()) == [11, 12]


def test_core_seed_depth_mm_recupera_la_erosion_del_metodo_edt():
    """Si el núcleo predicho NO viene partido (dos bloques unidos por un cuello fino), la erosión
    por distancia lo separa. Con la misma profundidad, "core3" coincide con "edt"."""
    sem = np.zeros((16, 40, 30), np.uint8)
    sem[1:15, 4:18, 4:26] = 2                # bloque principal (14 px en z e y)
    sem[1:15, 18:22, 13:17] = 2              # cuello estrecho: su distancia al fondo no pasa de ~2
    sem[1:15, 22:36, 4:26] = 2               # bloque secundario
    edge = np.zeros(sem.shape, np.float32)
    core = (sem > 0).astype(np.float32)      # el núcleo predicho cubre toda la región
    sin_erosion = separate_instances(sem, edge, SP, method="core3", core=core, core_seed_depth_mm=0.0)
    con_erosion = separate_instances(sem, edge, SP, method="core3", core=core, core_seed_depth_mm=3.0)
    edt = separate_instances(sem, edge, SP, method="edt", seed_depth_mm=3.0, edge_threshold=0.5)
    assert len(np.unique(sin_erosion[sin_erosion > 0])) == 1
    assert len(np.unique(con_erosion[con_erosion > 0])) == 2
    assert (con_erosion == edt).all(), "misma maquinaria: core3 con erosión == edt"


# --------------------------------------------------------------------- modelo y pérdida (necesitan torch)
def _corte_phantom(n: int = 32):
    """Un corte con sacro y coxal izq. partido, para construir objetivos reales."""
    lab = np.zeros((n, n), np.uint8)
    q = n // 8
    lab[2 * q:5 * q, q:3 * q] = 11
    lab[5 * q:6 * q, q:3 * q] = 12           # fragmento pegado al principal
    lab[2 * q:5 * q, 4 * q:6 * q] = 1        # sacro
    return lab


def _lote_de_un_corte(lab):
    """Lote de B = 1 con los objetivos REALES de ``slice_targets`` (incluido ``core3``)."""
    import torch

    t = slice_targets(lab, edge=fracture_edge_2d(lab, 2))
    return {
        "semantic": torch.from_numpy(t["semantic"])[None],
        "core3": torch.from_numpy(t["core3"])[None],
        "edge": torch.from_numpy(t["edge"])[None, None],
        "boxes": torch.from_numpy(t["boxes"])[None],
        "present": torch.from_numpy(t["present"].astype(np.float32))[None],
        "ignore": torch.from_numpy(t["ignore"])[None],
    }


def _salida_falsa(lote, core_logits):
    """Salida mínima del modelo: solo ``core_logits`` tiene gradiente; el resto es constante."""
    import torch

    b, h, w = lote["core3"].shape
    g = h // 8
    return {"cls_logits": torch.zeros(b, 3), "det_scores": torch.zeros(b, 3, g, g),
            "det_ltrb": torch.full((b, 3, 4, g, g), 8.0), "seg_logits": torch.zeros(b, 4, h, w),
            "core_logits": core_logits}


CFG_CORE3 = {"seed": 42,
             "model": {"det_stride": 8, "seg_outputs": ["semantic4", "core3"]},
             "loss": {"seg_ce_weights": [1.0, 10.8, 8.7, 8.7], "core3_ce_weights": [1.0, 5.4, 54.8]}}


def test_core3_entra_dentro_del_termino_seg_y_no_como_cuarto_termino():
    """El enunciado §4.1 fija TRES términos (cls, det, seg): núcleo/borde va dentro de ``seg``."""
    torch = pytest.importorskip("torch")
    from pengwin.losses.losses import TERMS, MultiTaskLoss

    lote = _lote_de_un_corte(_corte_phantom())
    parts = MultiTaskLoss(CFG_CORE3)(_salida_falsa(lote, torch.zeros(1, 3, 32, 32)), lote)
    assert TERMS == ("cls", "det", "seg") and "core3" not in TERMS
    assert float(parts["seg"]) == pytest.approx(float(parts["seg_ce"] + parts["seg_dice"]
                                                      + parts["core3_ce"] + parts["core3_dice"]), rel=1e-5)
    assert float(parts["total"]) == pytest.approx(float(parts["cls"] + parts["det"] + parts["seg"]), rel=1e-5)
    assert "edge_bce" not in parts, "sin cabeza de borde binaria no hay término de borde binario"


def test_la_perdida_core3_distingue_el_objetivo_correcto():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    from pengwin.losses.losses import MultiTaskLoss

    lote = _lote_de_un_corte(_corte_phantom())
    perfecto = F.one_hot(lote["core3"], 3).permute(0, 3, 1, 2).float() * 10.0
    confundido = perfecto[:, [0, 2, 1]]                      # núcleo y borde intercambiados
    loss = MultiTaskLoss(CFG_CORE3)
    bien, mal = loss(_salida_falsa(lote, perfecto), lote), loss(_salida_falsa(lote, confundido), lote)
    assert float(bien["core3_ce"]) < float(mal["core3_ce"])
    assert float(bien["core3_dice"]) < float(mal["core3_dice"]) and float(bien["core3_dice"]) < 0.05


def test_la_perdida_core3_baja_optimizando_la_salida_de_tres_clases():
    """60 pasos de descenso sobre los logits bajan el término ``seg`` y el argmax acaba
    coincidiendo con el objetivo: la pérdida nueva es optimizable y apunta al sitio correcto."""
    torch = pytest.importorskip("torch")
    from pengwin.losses.losses import MultiTaskLoss

    lote = _lote_de_un_corte(_corte_phantom())
    logits = torch.zeros(1, 3, 32, 32, requires_grad=True)
    loss = MultiTaskLoss(CFG_CORE3)
    opt = torch.optim.Adam([logits], lr=0.3)
    seg, nucleo = [], []
    for _ in range(60):
        parts = loss(_salida_falsa(lote, logits), lote)
        opt.zero_grad(set_to_none=True)
        parts["seg"].backward()
        opt.step()
        seg.append(float(parts["seg"].detach()))
        nucleo.append(float((parts["core3_ce"] + parts["core3_dice"]).detach()))
    assert seg[-1] < seg[0], f"el término seg no baja: {seg[0]:.3f} -> {seg[-1]:.3f}"
    assert nucleo[-1] < 0.1 * nucleo[0], f"núcleo/borde: {nucleo[0]:.3f} -> {nucleo[-1]:.3f}"
    assert float((logits.detach().argmax(1) == lote["core3"]).float().mean()) > 0.99


def test_el_modelo_anade_core3_sin_tocar_las_otras_cabezas():
    torch = pytest.importorskip("torch")

    from conftest import REPO
    from pengwin.models.cbam import CBAM
    from pengwin.models.pengwin_net import PengwinNet, count_parameters
    from pengwin.utils.config import load_config

    torch.manual_seed(0)                                   # determinista: solo se miran formas
    base = PengwinNet(load_config(REPO / "configs" / "base.yaml")["model"]).eval()
    core3 = PengwinNet(load_config(REPO / "configs" / "tuning" / "f2b1_core3.yaml")["model"]).eval()
    out = core3(torch.rand(1, 3, 64, 64))
    assert out["core_logits"].shape == (1, 3, 64, 64), "3 clases a resolución completa"
    assert out["seg_logits"].shape == (1, 4, 64, 64), "la semántica de 4 clases no cambia"
    # P(borde) se sigue publicando, derivada del MISMO softmax: el posproceso edge/edt y
    # inference/volume.py funcionan igual sobre este checkpoint.
    assert torch.allclose(torch.sigmoid(out["edge_logits"]), out["core_logits"].softmax(1)[:, 2:3], atol=1e-6)
    for nombre in ("backbone", "neck", "cls_head", "det_head"):
        n_base = sum(p.numel() for p in getattr(base, nombre).parameters())
        assert sum(p.numel() for p in getattr(core3, nombre).parameters()) == n_base, nombre
    assert sum(isinstance(m, CBAM) for m in core3.modules()) == 2, "el CBAM sigue en los bloques 3 y 4"
    delta = count_parameters(core3)["seg_head"] - count_parameters(base)["seg_head"]
    assert delta == (3 * 32 + 3) - (32 + 1), "entra la conv 1×1 de 3 canales y sale la de borde binario"


def test_el_gradiente_llega_a_la_salida_nueva():
    """Cableado completo: objetivo -> lote -> cabeza de 3 clases -> pérdida -> gradiente."""
    torch = pytest.importorskip("torch")

    from conftest import REPO
    from pengwin.losses.losses import MultiTaskLoss
    from pengwin.models.pengwin_net import PengwinNet
    from pengwin.utils.config import load_config

    torch.manual_seed(0)
    cfg = load_config(REPO / "configs" / "tuning" / "f2b1_core3.yaml")
    model = PengwinNet(cfg["model"])
    lote = _lote_de_un_corte(_corte_phantom(64))
    lote["image"] = torch.rand(1, 3, 64, 64)
    MultiTaskLoss(cfg)(model(lote["image"]), lote)["total"].backward()
    for mod, nombre in ((model.seg_head.core, "seg_head.core"), (model.seg_head.semantic, "seg_head.semantic"),
                        (model.cls_head, "cls_head"), (model.det_head, "det_head")):
        assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in mod.parameters()), nombre
    assert model.seg_head.edge is None, "con core3 no hay cabeza de borde binaria"


def test_la_config_f2b1_solo_cambia_lo_necesario():
    pytest.importorskip("yaml")
    from conftest import REPO
    from pengwin.utils.config import load_config

    base = load_config(REPO / "configs" / "base.yaml")
    cfg = load_config(REPO / "configs" / "tuning" / "f2b1_core3.yaml")
    assert cfg["model"]["seg_outputs"] == ["semantic4", "core3"]
    assert cfg["postprocess"]["instance_method"] == "core3"
    assert cfg["loss"]["terms"] == ["cls", "det", "seg"], "la pérdida sigue siendo de 3 términos"
    distintas = {k for k in base["model"] if base["model"][k] != cfg["model"][k]}
    assert distintas == {"seg_outputs"}, f"la config toca más de lo necesario: {distintas}"
