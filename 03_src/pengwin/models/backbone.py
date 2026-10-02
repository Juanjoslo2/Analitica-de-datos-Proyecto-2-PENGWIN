import torch
import torch.nn as nn


class FundidoraPCBackbone(nn.Module):
    """
    FundidoraPC adaptado como backbone para detección.

    Conserva el mapa espacial de características en lugar
    de convertirlo en un vector mediante Global Average Pooling.
    """

    def __init__(self, in_channels=3):
        super().__init__()

        # Bloque 1: 3 -> 32
        self.conv1 = nn.Conv2d(
            in_channels, 32,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Bloque 2: 32 -> 64
        self.conv2 = nn.Conv2d(
            32, 64,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Bloque 3: 64 -> 128
        self.conv3 = nn.Conv2d(
            64, 128,
            kernel_size=3,
            padding=1
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # Bloque 4: 128 -> 256
        # Sin pooling para conservar stride 8.
        self.conv4 = nn.Conv2d(
            128, 256,
            kernel_size=3,
            padding=1
        )
        self.bn4 = nn.BatchNorm2d(256)

        # Información que utilizarán los componentes posteriores
        self.out_features = 256
        self.out_stride = 8

    def forward(self, x):

        x = self.pool1(
            torch.relu(
                self.bn1(
                    self.conv1(x)
                )
            )
        )

        x = self.pool2(
            torch.relu(
                self.bn2(
                    self.conv2(x)
                )
            )
        )

        x = self.pool3(
            torch.relu(
                self.bn3(
                    self.conv3(x)
                )
            )
        )

        x = torch.relu(
            self.bn4(
                self.conv4(x)
            )
        )

        return x