"""cbam.py.

Convolutional Block Attention Module (Woo et al., ECCV 2018), implementación propia.
Atención de canal (QUÉ mapas importan) seguida de atención espacial (DÓNDE mirar):

    F'  = M_c(F) ⊙ F,      M_c(F)  = σ( MLP(AvgPool(F)) + MLP(MaxPool(F)) )
    F'' = M_s(F') ⊙ F',    M_s(F') = σ( conv7×7([mean_c(F'); max_c(F')]) )

En PENGWIN la atención espacial ayuda a concentrar la red en el hueso, que es ~3 % del
corte [EDA §8]; la de canal, a pesar distinto los mapas que responden a cortical vs. trabecular.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        # MLP compartido por las dos ramas de pooling, como en el artículo
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(x.mean(dim=(2, 3), keepdim=True))
        mx = self.mlp(x.amax(dim=(2, 3), keepdim=True))
        return torch.sigmoid(avg + mx)                        # (B, C, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        return torch.sigmoid(self.conv(pooled))               # (B, 1, H, W)


class CBAM(nn.Module):
    """Canal -> espacial, en ese orden (el artículo muestra que es mejor que en paralelo).

    Con ``gamma=True`` la salida es  y = x + γ · (CBAM(x) − x),  con γ escalar aprendible que
    arranca en 0: el bloque empieza como identidad y γ mide cuánto decide usar la red la
    atención (γ ≈ 0 la ignora, γ ≈ 1 la aplica completa).
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7, gamma: bool = False):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(kernel_size)
        self.gamma = nn.Parameter(torch.zeros(1)) if gamma else None
        self.last_spatial_map: torch.Tensor | None = None     # para visualizar dónde mira la red

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * self.channel(x)
        m = self.spatial(y)
        self.last_spatial_map = m.detach()
        y = y * m
        return y if self.gamma is None else x + self.gamma * (y - x)
