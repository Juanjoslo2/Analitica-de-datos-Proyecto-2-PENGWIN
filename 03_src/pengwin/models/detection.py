import torch
import torch.nn as nn


class DetectionHead(nn.Module):
    """
    Cabeza de detección anchor-free.

    Entrada:
        [B, C, H, W]

    Salidas:
        objectness:
            [B, 1, H, W]

        bbox:
            [B, 4, H, W]

    Las 4 coordenadas representan:

        l = distancia al borde izquierdo
        t = distancia al borde superior
        r = distancia al borde derecho
        b = distancia al borde inferior
    """

    def __init__(self, in_channels=256):

        super().__init__()

        self.shared = nn.Sequential(
            nn.Conv2d(
                in_channels,
                128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.objectness = nn.Conv2d(
            128,
            1,
            kernel_size=1,
        )

        self.bbox = nn.Conv2d(
            128,
            4,
            kernel_size=1,
        )

    def forward(self, x):

        features = self.shared(x)

        objectness = self.objectness(features)

        bbox = self.bbox(features)

        # Las distancias deben ser positivas
        bbox = torch.relu(bbox)

        return objectness, bbox