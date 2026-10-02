"""backbone.py.

Backbone compartido: extensión propia de ``FundidoraPC`` (curso, semana 6) [DD §8, enunciado §4.1].

FundidoraPC del curso: 4 bloques Conv3×3 → BN → ReLU → MaxPool, canales 32 → 64 → 128 → 256,
y un pooling global al final. Sirve para clasificar, pero para PENGWIN le faltan tres cosas:

1. **Mapas multiescala, no un vector.** La detección usa un grid de stride 8 (39 % de las
   cajas del sacro mide < 16 px [EDA §8]) y la segmentación necesita resolución completa.
   Se exponen las salidas de los 4 bloques: C1 (stride 2), C2 (4), C3 (8), C4 (16).
2. **Más profundidad sin degradar el gradiente.** Variante ``fundidora_r``: tras la
   conv de cada bloque se añade un bloque residual (conv-BN-ReLU-conv-BN + identidad).
   La conv original del bloque se conserva con el mismo papel y forma, así que los
   pesos de una FundidoraPC ya entrenada se cargan tal cual (transfer learning).
3. **Atención CBAM** en los bloques 3 y 4, antes de la bifurcación hacia las tres
   cabezas (``cbam_blocks``). Con ``cbam: false`` (ablación) se reemplaza por identidad
   y el número de parámetros del resto de la red no cambia.

``fundidora`` (sin residuales) queda disponible para comparar con la versión del curso.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn as nn

from pengwin.models.cbam import CBAM


class ResidualBlock(nn.Module):
    """Bloque básico de ResNet con identidad (mismo número de canales a la entrada y salida)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv_a = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(channels)
        self.conv_b = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(channels)
        nn.init.zeros_(self.bn_b.weight)          # arranca como identidad: no altera la FundidoraPC cargada

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.bn_a(self.conv_a(x)))
        return torch.relu(x + self.bn_b(self.conv_b(y)))


class FundidoraStage(nn.Module):
    """Un bloque de FundidoraPC (+ residual y CBAM opcionales)."""

    def __init__(self, c_in: int, c_out: int, residual: bool, cbam: bool, cbam_reduction: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.res = ResidualBlock(c_out) if residual else nn.Identity()
        self.cbam = CBAM(c_out, cbam_reduction) if cbam else nn.Identity()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res(torch.relu(self.bn(self.conv(x))))
        return self.cbam(self.pool(x))


class FundidoraBackbone(nn.Module):
    """FundidoraPC extendida: devuelve [C1, C2, C3, C4] con strides 2, 4, 8, 16."""

    strides = (2, 4, 8, 16)

    def __init__(
        self,
        in_channels: int = 3,
        widths: Sequence[int] = (32, 64, 128, 256),
        residual: bool = True,
        cbam: bool = True,
        cbam_blocks: Sequence[int] = (3, 4),
        cbam_reduction: int = 16,
    ):
        super().__init__()
        stages, c_in = [], in_channels
        for k, c_out in enumerate(widths, start=1):
            stages.append(FundidoraStage(c_in, c_out, residual, cbam and k in cbam_blocks, cbam_reduction))
            c_in = c_out
        self.stages = nn.ModuleList(stages)
        self.widths = tuple(widths)
        self.out_features = widths[-1]                 # mismo atributo público que FundidoraPC

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats

    def shared_parameters(self) -> List[nn.Parameter]:
        """Parámetros de la última capa compartida (bloque 4 + su CBAM) [DD §4].

        Sobre ellos se mide ‖∇L_k‖ para calibrar los λ: es lo último que comparten las
        tres cabezas antes de bifurcarse.
        """
        return [p for p in self.stages[-1].parameters() if p.requires_grad]


def build_backbone(model_cfg: Dict, in_channels: int = 3) -> FundidoraBackbone:
    name = model_cfg.get("backbone", "fundidora_r")
    if name not in ("fundidora_r", "fundidora"):
        raise ValueError(f"Backbone desconocido: {name}")
    return FundidoraBackbone(
        in_channels=in_channels,
        widths=tuple(model_cfg.get("widths", (32, 64, 128, 256))),
        residual=name == "fundidora_r",
        cbam=bool(model_cfg.get("cbam", True)),
        cbam_blocks=tuple(model_cfg.get("cbam_blocks", (3, 4))),
        cbam_reduction=int(model_cfg.get("cbam_reduction", 16)),
    )


# --------------------------------------------------------------------------- transfer learning
_S6_KEY = re.compile(r"(?:^|\.)(conv|bn)(\d)\.(weight|bias|running_mean|running_var)$")
_T3_KEY = re.compile(r"(?:^|\.)blocks\.(\d+)\.([01])\.(weight|bias|running_mean|running_var)$")


def _canonical_fundidora_state(state: Dict[str, torch.Tensor]) -> Dict[int, Dict[str, torch.Tensor]]:
    """Normaliza los dos formatos de FundidoraPC del curso a {bloque: {conv_w, conv_b, bn_*}}.

    - Semana 6 (``S6_Bloque1``):  conv1.weight, conv1.bias, bn1.weight, ... conv4 / bn4
    - Taller 3 (RPN híbrido):    blocks.0.0.weight (conv), blocks.0.1.weight (BN), ...
    Cualquier prefijo (``backbone.``, ``module.``) se ignora.
    """
    out: Dict[int, Dict[str, torch.Tensor]] = {}
    for key, val in state.items():
        if (m := _S6_KEY.search(key)):
            kind, k, field = m.group(1), int(m.group(2)) - 1, m.group(3)
        elif (m := _T3_KEY.search(key)):
            kind, k, field = ("conv" if m.group(2) == "0" else "bn"), int(m.group(1)), m.group(3)
        else:
            continue
        name = {"weight": "conv_w", "bias": "conv_b"}[field] if kind == "conv" else f"bn_{field}"
        out.setdefault(k, {})[name] = val
    return out


def load_fundidora_weights(backbone: FundidoraBackbone, path: Path | str) -> Dict[str, List[str]]:
    """Carga en ``backbone`` las conv+BN de una FundidoraPC entrenada (transfer learning).

    - Si la conv del archivo tenía sesgo, se pliega en la media móvil de la BN
      (BN(x + b) con media μ  ≡  BN(x) con media μ − b), así la salida no cambia.
    - Si la primera conv tenía otro número de canales de entrada (p. ej. 1 en escala
      de grises), se replica el promedio a los 3 canales 2.5D y se reescala.
    - Las residuales y el CBAM no existen en FundidoraPC: arrancan desde cero
      (la residual como identidad), igual que las cabezas.
    Devuelve el informe {cargados, omitidos} para registrarlo en el entrenamiento.
    """
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    blocks = _canonical_fundidora_state(state)
    loaded, skipped = [], []
    for k, stage in enumerate(backbone.stages):
        src = blocks.get(k)
        if not src or "conv_w" not in src:
            skipped.append(f"stage{k + 1}: no está en el archivo")
            continue
        w = src["conv_w"]
        if w.shape[0] != stage.conv.out_channels:
            skipped.append(f"stage{k + 1}: {tuple(w.shape)} no coincide con {tuple(stage.conv.weight.shape)}")
            continue
        if w.shape[1] != stage.conv.in_channels:
            w = w.mean(dim=1, keepdim=True).repeat(1, stage.conv.in_channels, 1, 1) * (w.shape[1] / stage.conv.in_channels)
        with torch.no_grad():
            stage.conv.weight.copy_(w)
            for name in ("weight", "bias", "running_mean", "running_var"):
                if f"bn_{name}" in src:
                    getattr(stage.bn, name).copy_(src[f"bn_{name}"])
            if "conv_b" in src:
                stage.bn.running_mean.sub_(src["conv_b"])
        loaded.append(f"stage{k + 1}")
    return {"loaded": loaded, "skipped": skipped}
