"""heads.py.

Las tres cabezas sobre el backbone compartido [enunciado §4.1, DD §2-3]. Todas se
entrenan desde cero (el transfer learning se permite solo en el backbone).

    ClassificationHead  C4 (stride 16) → pooling global → 3 logits multi-etiqueta:
                        qué regiones (SA, coxal izq., coxal der.) aparecen en el corte.
    GridDetectionHead   P3 (stride 8, 32×32) → por región: puntaje + (l, t, r, b).
    SegmentationHead    P3 + C2 + C1 → resolución completa: semántica de 4 clases
                        (fondo, SA, izq., der.) + mapa de borde de fractura (1 canal).

P3 sale de un cuello ligero (``TopDownNeck``) que mezcla C3 con C4 subido ×2, como el
camino descendente de una FPN: la detección a stride 8 recibe contexto del nivel más
profundo (el que pasó por el CBAM del bloque 4).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(c_in: int, c_out: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class TopDownNeck(nn.Module):
    """P3 = conv( lateral(C3) + up×2( lateral(C4) ) )."""

    def __init__(self, c3: int, c4: int, out_channels: int = 128):
        super().__init__()
        self.lat3 = nn.Conv2d(c3, out_channels, 1)
        self.lat4 = nn.Conv2d(c4, out_channels, 1)
        self.smooth = conv_bn_relu(out_channels, out_channels)

    def forward(self, c3: torch.Tensor, c4: torch.Tensor) -> torch.Tensor:
        up = F.interpolate(self.lat4(c4), size=c3.shape[-2:], mode="nearest")
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
    """Decodificador tipo U-Net: P3 (s8) → s4 (+C2) → s2 (+C1) → s1."""

    def __init__(self, p3: int, c2: int, c1: int, num_classes: int = 4, width: int = 64):
        super().__init__()
        self.up4 = conv_bn_relu(p3 + c2, width)
        self.up2 = conv_bn_relu(width + c1, width // 2)
        self.up1 = conv_bn_relu(width // 2, width // 2)
        self.semantic = nn.Conv2d(width // 2, num_classes, 1)
        self.edge = nn.Conv2d(width // 2, 1, 1)

    @staticmethod
    def _up_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return torch.cat([F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip], dim=1)

    def forward(self, p3: torch.Tensor, c2: torch.Tensor, c1: torch.Tensor, out_size):
        x = self.up4(self._up_cat(p3, c2))
        x = self.up2(self._up_cat(x, c1))
        x = self.up1(F.interpolate(x, size=out_size, mode="bilinear", align_corners=False))
        return self.semantic(x), self.edge(x)
