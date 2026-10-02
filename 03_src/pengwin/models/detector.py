import torch
import torch.nn as nn

from pengwin.models.backbone import FundidoraPCBackbone
from pengwin.models.cbam import CBAM
from pengwin.models.detection import DetectionHead


class PENGWINDetector(nn.Module):
    """
    Modelo inicial de detección.

    Flujo:

        Imagen
          ↓
        Backbone
          ↓
        CBAM
          ↓
        DetectionHead

    Salidas:

        objectness: [B, 1, 32, 32]
        bbox:       [B, 4, 32, 32]
    """

    def __init__(
        self,
        in_channels=3,
        backbone_channels=256,
    ):
        super().__init__()

        # ============================================
        # Backbone
        # ============================================

        self.backbone = FundidoraPCBackbone(
            in_channels=in_channels
        )

        # ============================================
        # CBAM
        # ============================================

        self.cbam = CBAM(
            channels=backbone_channels
        )

        # ============================================
        # Cabeza de detección
        # ============================================

        self.detection_head = DetectionHead(
            in_channels=backbone_channels
        )

    def forward(self, x):

        # Backbone
        features = self.backbone(x)

        # Atención
        features = self.cbam(features)

        # Detección
        objectness, bbox = self.detection_head(
            features
        )

        return {
            "features": features,
            "objectness": objectness,
            "bbox": bbox,
        }