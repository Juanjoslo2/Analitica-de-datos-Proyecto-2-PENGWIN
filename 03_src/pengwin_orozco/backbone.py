"""backbone.py.

FundidoraPC del curso (Semana 6, ``clases_notebook/S6_Bloque1_Clasificar_a_Detectar``)
extendida con un bloque CBAM propio, implementada desde cero sin tocar el módulo
compartido. El enunciado exige: "backbone compartido (FundidoraPC) + atención de
canal y espacial (CBAM) antes de la bifurcación hacia las cabezas".

Arquitectura (idéntica en espíritu a la de clase):
    conv1 (3 → 32) → BN → ReLU
    conv2 (32 → 64) → BN → ReLU → maxpool(2)
    conv3 (64 → 128) → BN → ReLU → maxpool(2) → CBAM(128)
    conv4 (128 → 256) → BN → ReLU → maxpool(2) → CBAM(256)
    salida: 256 canales a stride 8 (256×256 → 32×32), lista [C2, C3, C4]
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Atención de canal: media + máximo espacial → MLP 1×1 (C → C/r → C) → sigmoid."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        return torch.sigmoid(self.mlp(avg) + self.mlp(mx))


class SpatialAttention(nn.Module):
    """Atención espacial: concat(media, máx) por canal → conv 7×7 → sigmoid."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """CBAM (Woo et al., 2018): canal primero, espacial después. Guarda el mapa
    espacial de la última pasada (detach) para visualización."""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(kernel_size)
        self.last_spatial_map = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel(x)
        s = self.spatial(x)
        self.last_spatial_map = s.detach()
        return x * s


class FundidoraPC(nn.Module):
    """La FundidoraPC del curso con CBAM opcional en las dos últimas etapas.

    ``forward`` devuelve la lista ``[C2 (s2), C3 (s4), C4 (s8)]``; la detección usa
    C4 (stride 8). Se expone la lista por si más adelante se quiere un cuello o
    una cabeza de segmentación multiescala.
    """

    def __init__(self, in_channels: int = 3, widths=(32, 64, 128, 256),
                 cbam: bool = True, cbam_blocks=(3, 4), cbam_reduction: int = 16):
        super().__init__()
        w1, w2, w3, w4 = widths
        blocks = []
        self.in_channels = in_channels
        convs = [
            (in_channels, w1, False),   # bloque 1: sin pooling (la entrada ya es fina)
            (w1, w2, True),             # bloque 2: pool → stride 2
            (w2, w3, True),             # bloque 3: pool → stride 4
            (w3, w4, True),             # bloque 4: pool → stride 8
        ]
        self.stages = nn.ModuleList()
        for i, (ci, co, pool) in enumerate(convs, start=1):
            stage = nn.Sequential()
            stage.append(nn.Conv2d(ci, co, 3, padding=1, bias=False))
            stage.append(nn.BatchNorm2d(co))
            stage.append(nn.ReLU(inplace=True))
            if pool:
                stage.append(nn.MaxPool2d(2))
            if cbam and i in cbam_blocks:
                stage.append(CBAM(co, cbam_reduction))
            self.stages.append(stage)
        self.stride = 8
        self.out_channels = w4

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        c1 = self.stages[0](x)
        c2 = self.stages[1](c1)
        c3 = self.stages[2](c2)
        c4 = self.stages[3](c3)
        return [c2, c3, c4]

    def shared_parameters(self):
        """Parámetros del último bloque compartido (para calibrar λ por gradiente)."""
        return list(self.stages[-1].parameters())


def build_backbone(model_cfg: Dict) -> FundidoraPC:
    m = model_cfg
    return FundidoraPC(
        in_channels=3,
        widths=tuple(m.get("widths", (32, 64, 128, 256))),
        cbam=bool(m.get("cbam", True)),
        cbam_blocks=tuple(m.get("cbam_blocks", (3, 4))),
        cbam_reduction=int(m.get("cbam_reduction", 16)),
    )