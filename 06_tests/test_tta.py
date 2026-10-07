"""Tests de TTA en la inferencia (``pengwin.inference.volume``), con fantomas.

TTA promedia P(región) y P(borde) de varias versiones transformadas del mismo corte. Lo que
tiene que cumplirse:
    - transformar y deshacer la transformación devuelve la imagen original;
    - con un modelo que trabaja píxel a píxel (no depende de la posición), TTA da lo mismo que
      una sola pasada;
    - la detección y la clasificación salen de la pasada sin transformar.
"""

import numpy as np
import torch
import torch.nn as nn

from pengwin.inference.volume import TTA_DEFAULT, _tta_forward, _theta, _warp


def _blob(n: int = 64) -> torch.Tensor:
    """Gaussiana suave en el centro, (1, 1, n, n)."""
    yy, xx = np.mgrid[:n, :n]
    g = np.exp(-(((yy - n / 2) ** 2 + (xx - n / 2) ** 2) / (2 * (n / 8) ** 2)))
    return torch.from_numpy(g.astype(np.float32))[None, None]


class _PixelModel(nn.Module):
    """Modelo de juguete con convoluciones 1×1: su salida no depende de la posición."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.seg = nn.Conv2d(3, 4, 1)
        self.edge = nn.Conv2d(3, 1, 1)

    def forward(self, x):
        b = x.shape[0]
        return {"seg_logits": self.seg(x), "edge_logits": self.edge(x), "cls_logits": x.mean((2, 3)),
                "det_scores": torch.zeros(b, 3, 4, 4), "det_ltrb": torch.zeros(b, 3, 4, 4, 4)}


def test_transformar_y_deshacer_devuelve_el_original():
    img = _blob()
    for t in TTA_DEFAULT:
        th = _theta(*t)
        vuelta = _warp(_warp(img, th), torch.linalg.inv(th))
        c = slice(16, 48)                              # el centro nunca sale del lienzo
        assert torch.allclose(vuelta[..., c, c], img[..., c, c], atol=0.02), t


def test_theta_identidad():
    assert torch.allclose(_theta(1.0, 0.0, 0.0, 0.0), torch.eye(3))


def test_modelo_por_pixel_da_lo_mismo_con_y_sin_tta():
    model = _PixelModel().eval()
    x = _blob().repeat(2, 3, 1, 1)
    with torch.no_grad():
        uno = model(x)
        tta = _tta_forward(model, x, TTA_DEFAULT, amp=False, device=torch.device("cpu"))
    c = slice(16, 48)
    sem1 = uno["seg_logits"].softmax(1)[..., c, c]
    edge1 = torch.sigmoid(uno["edge_logits"])[..., c, c]
    assert torch.allclose(tta["tta_sem"][..., c, c], sem1, atol=0.02)
    assert torch.allclose(tta["tta_edge"][..., c, c], edge1, atol=0.02)
    # las probabilidades promediadas siguen sumando 1
    assert torch.allclose(tta["tta_sem"][..., c, c].sum(1), torch.ones(2, 32, 32), atol=1e-4)


def test_deteccion_y_clasificacion_salen_de_la_pasada_sin_transformar():
    model = _PixelModel().eval()
    x = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        uno = model(x)
        tta = _tta_forward(model, x, TTA_DEFAULT, amp=False, device=torch.device("cpu"))
    assert torch.equal(tta["cls_logits"], uno["cls_logits"])
    assert torch.equal(tta["det_scores"], uno["det_scores"])


def test_conjuntos_de_tta():
    from pengwin.inference.volume import TTA_AMPLIO, TTA_SETS

    assert TTA_SETS["5"] == TTA_DEFAULT and TTA_SETS["9"] == TTA_AMPLIO
    assert len(TTA_AMPLIO) == 9 and TTA_AMPLIO[0] == (1.0, 0.0, 0.0, 0.0)   # la primera es la identidad
