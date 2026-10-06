"""heads.py.

Las tres cabezas sobre el backbone compartido [enunciado §4.1, DD §2-3]. Todas se
entrenan desde cero (el transfer learning se permite solo en el backbone).

    ClassificationHead  C4 (stride 16) → pooling global → 3 logits multi-etiqueta:
                        qué regiones (SA, coxal izq., coxal der.) aparecen en el corte.
    GridDetectionHead   P3 (stride 8, 32×32) → por región: puntaje + (l, t, r, b).
    SegmentationHead    P3 + C2 + C1 → resolución completa: semántica de 4 clases
                        (fondo, SA, izq., der.) + las salidas que pida
                        ``model.seg_outputs`` a la MISMA resolución.

P3 sale de un cuello ligero (``TopDownNeck``) que mezcla C3 con C4 subido ×2, como el
camino descendente de una FPN: la detección a stride 8 recibe contexto del nivel más
profundo (el que pasó por el CBAM del bloque 4).

``model.seg_outputs`` (la semántica siempre está; es la etapa 1 de las dos etapas):
    semantic4       4 clases de región -> ``seg_logits``   (obligatoria)
    fracture_edge   borde binario      -> ``edge_logits``  (semanas 9-10)
    core3           3 clases fondo/núcleo/borde -> ``core_logits`` [F2B1]
    dist            distancia a la fractura (regresión) -> ``dist_logits`` [F2B2]
"""

from __future__ import annotations

import math
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

SEG_OUTPUTS = ("semantic4", "fracture_edge", "core3", "dist")
DEFAULT_SEG_OUTPUTS = ("semantic4", "fracture_edge")


def conv_bn_relu(c_in: int, c_out: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class TopDownNeck(nn.Module):
    """P3 = conv( lateral(C3) + γ · up×2( lateral(C4) ) ).

    Con ``gamma=True``, γ es un escalar aprendible que arranca en 1: dice cuánto contexto del
    nivel profundo (C4) usa la detección/segmentación frente al nivel de stride 8 (C3).
    """

    def __init__(self, c3: int, c4: int, out_channels: int = 128, gamma: bool = False):
        super().__init__()
        self.lat3 = nn.Conv2d(c3, out_channels, 1)
        self.lat4 = nn.Conv2d(c4, out_channels, 1)
        self.smooth = conv_bn_relu(out_channels, out_channels)
        self.gamma = nn.Parameter(torch.ones(1)) if gamma else None

    def forward(self, c3: torch.Tensor, c4: torch.Tensor) -> torch.Tensor:
        up = F.interpolate(self.lat4(c4), size=c3.shape[-2:], mode="nearest")
        if self.gamma is not None:
            up = self.gamma * up
        return self.smooth(self.lat3(c3) + up)


class ClassificationHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, c4: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(F.adaptive_avg_pool2d(c4, 1).flatten(1)))


class GridDetectionHead(nn.Module):
    """Cabeza anchor-free de stride 8 (ver ``detection/grid.py`` para asignación y decodificación)."""

    def __init__(self, in_channels: int, num_classes: int = 3, stride: int = 8, tower_depth: int = 2, prior: float = 0.01):
        super().__init__()
        self.num_classes = num_classes
        self.stride = stride
        self.cls_tower = nn.Sequential(*[conv_bn_relu(in_channels, in_channels) for _ in range(tower_depth)])
        self.box_tower = nn.Sequential(*[conv_bn_relu(in_channels, in_channels) for _ in range(tower_depth)])
        self.score = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.box = nn.Conv2d(in_channels, num_classes * 4, 3, padding=1)
        # Sesgo inicial: P(positivo) = prior. Con ~1 % de celdas positivas, arrancar en 0,5
        # haría que la pérdida focal de los negativos dominara las primeras iteraciones.
        nn.init.constant_(self.score.bias, -math.log((1 - prior) / prior))
        # Distancias iniciales ≈ 2 celdas (16 px) en vez de exp(0)·8
        nn.init.constant_(self.box.bias, math.log(2.0))

    def forward(self, p3: torch.Tensor):
        b, _, g, _ = p3.shape
        scores = self.score(self.cls_tower(p3))                                   # (B, K, G, G)
        raw = self.box(self.box_tower(p3)).view(b, self.num_classes, 4, g, g)
        ltrb = torch.exp(raw.float().clamp(max=6.0)) * self.stride               # px, siempre > 0
        return scores, ltrb


class SegmentationHead(nn.Module):
    """Decodificador tipo U-Net: P3 (s8) → s4 (+C2) → s2 (+C1) → s1.

    Las salidas son convoluciones 1×1 sobre el MISMO mapa de s1, así que añadir ``core3`` o
    ``dist`` cuesta C·(width/2)+C parámetros y no cambia el decodificador ni el CBAM.
    """

    def __init__(self, p3: int, c2: int, c1: int, num_classes: int = 4, width: int = 64,
                 spatial_dropout: float = 0.0, outputs: Sequence[str] = DEFAULT_SEG_OUTPUTS):
        super().__init__()
        self.outputs = tuple(outputs)
        desconocidas = [o for o in self.outputs if o not in SEG_OUTPUTS]
        if desconocidas:
            raise ValueError(f"model.seg_outputs desconocidas: {desconocidas}; válidas: {list(SEG_OUTPUTS)}")
        if "semantic4" not in self.outputs:
            raise ValueError("model.seg_outputs debe incluir 'semantic4': es la etapa 1 (región) [enunciado §3.2]")
        self.up4 = conv_bn_relu(p3 + c2, width)
        # Dropout espacial solo a stride 4 (64 canales): más arriba, a stride 2 y 1, quedan pocos
        # canales y apagarlos borraría el detalle fino del borde de fractura.
        self.drop = nn.Dropout2d(spatial_dropout) if spatial_dropout > 0 else nn.Identity()
        self.up2 = conv_bn_relu(width + c1, width // 2)
        self.up1 = conv_bn_relu(width // 2, width // 2)
        self.semantic = nn.Conv2d(width // 2, num_classes, 1)
        self.edge = nn.Conv2d(width // 2, 1, 1) if "fracture_edge" in self.outputs else None
        self.core = nn.Conv2d(width // 2, 3, 1) if "core3" in self.outputs else None
        # [F2B2] Regresión de la distancia a la superficie de fractura, a la MISMA resolución que
        # la salida de borde. Un solo canal: distancia = sigmoid(dist_logits) · loss.dist_max_mm.
        self.dist = nn.Conv2d(width // 2, 1, 1) if "dist" in self.outputs else None
        # Sin cabeza de borde binaria, P(borde) se deriva de core3 (o de la distancia) para que el
        # resto del pipeline (inference/volume.py, posproceso "edge"/"edt") siga funcionando igual.
        self.derive_edge = self.edge is None and (self.core is not None or self.dist is not None)

    @staticmethod
    def _up_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return torch.cat([F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip], dim=1)

    @staticmethod
    def edge_logit_from_core3(core: torch.Tensor) -> torch.Tensor:
        """logit de P(borde) a partir de los 3 logits: log(p2 / (1 − p2)) = l2 − logsumexp(l0, l1).

        Exacto: ``sigmoid(edge_logits)`` == ``softmax(core_logits)[:, 2]``.
        """
        c = core.float()
        return (c[:, 2:3] - torch.logsumexp(c[:, :2], dim=1, keepdim=True))

    @staticmethod
    def dist_mm_from_logits(dist_logits: torch.Tensor, dist_max_mm: float = 8.0) -> torch.Tensor:
        """Distancia a la fractura en mm = ``sigmoid(dist_logits) · dist_max_mm`` [F2B2].

        La sigmoide acota la salida a [0, ``dist_max_mm``] por construcción (no hay que recortar)
        y pone su zona de máxima pendiente en la mitad del rango, que es donde cae el umbral del
        posproceso (``seed_depth_mm`` ≈ 4 mm de 8). ``dist_max_mm`` = ``loss.dist_max_mm``
        (= ``pengwin.data.targets.DIST_MAX_MM`` por defecto).

        Con ``edge_logits = −dist_logits`` se cumple exactamente
        ``sigmoid(edge_logits) = 1 − dist/dist_max``: un "borde" monótono en la distancia que no
        se entrena como tal, pero mantiene vivo el pipeline de borde (volume.py, "edge"/"edt").
        """
        return torch.sigmoid(dist_logits.float()) * float(dist_max_mm)

    def forward(self, p3: torch.Tensor, c2: torch.Tensor, c1: torch.Tensor, out_size) -> Dict[str, torch.Tensor]:
        x = self.drop(self.up4(self._up_cat(p3, c2)))
        x = self.up2(self._up_cat(x, c1))
        x = self.up1(F.interpolate(x, size=out_size, mode="bilinear", align_corners=False))
        out = {"seg_logits": self.semantic(x)}
        if self.core is not None:
            out["core_logits"] = self.core(x)
        if self.dist is not None:
            out["dist_logits"] = self.dist(x)
        if self.edge is not None:
            out["edge_logits"] = self.edge(x)
        elif self.derive_edge:
            out["edge_logits"] = (self.edge_logit_from_core3(out["core_logits"]) if self.core is not None
                                  else -out["dist_logits"])
        return out
