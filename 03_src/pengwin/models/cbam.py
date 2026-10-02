import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Atención sobre los canales de un mapa de características.

    Entrada:
        [B, C, H, W]

    Salida:
        [B, C, H, W]

    Cada canal recibe un peso entre 0 y 1.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()

        hidden_channels = channels // reduction

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden_channels,
                kernel_size=1,
                bias=False
            ),
            nn.ReLU(),
            nn.Conv2d(
                hidden_channels,
                channels,
                kernel_size=1,
                bias=False
            )
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # x: [B, C, H, W]
        avg = self.avg_pool(x)

        # avg: [B, C, 1, 1]
        attention = self.mlp(avg)

        # attention: [B, C, 1, 1]
        attention = self.sigmoid(attention)

        # Cada canal es reponderado.
        return x * attention


class SpatialAttention(nn.Module):
    """
    Atención sobre la posición espacial.

    Entrada:
        [B, C, H, W]

    Salida:
        [B, C, H, W]

    Aprende qué posiciones espaciales son más relevantes.
    """

    def __init__(self, kernel_size=7):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # Promedio entre todos los canales
        avg = torch.mean(x, dim=1, keepdim=True)

        # Máximo entre todos los canales
        max_values, _ = torch.max(x, dim=1, keepdim=True)

        # [B, 2, H, W]
        attention_input = torch.cat(
            [avg, max_values],
            dim=1,
        )

        # [B, 1, H, W]
        attention = self.conv(attention_input)

        attention = self.sigmoid(attention)

        # Broadcasting sobre los canales
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Aplica:
        1. Channel Attention
        2. Spatial Attention
    """

    def __init__(self, channels, reduction=4):
        super().__init__()

        self.channel_attention = ChannelAttention(
            channels=channels,
            reduction=reduction,
        )

        self.spatial_attention = SpatialAttention(
            kernel_size=7,
        )

    def forward(self, x):

        # Primero: atención sobre los canales
        x = self.channel_attention(x)

        # Segundo: atención sobre las posiciones espaciales
        x = self.spatial_attention(x)

        return x