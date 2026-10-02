import torch
import torch.nn.functional as F


def detection_loss(
    objectness_pred,
    bbox_pred,
    objectness_target,
    bbox_target,
    positive_weight=10.0,
):
    """
    Calcula la pérdida de detección anchor-free.

    Componentes:

        L_obj:
            Clasificación de cada celda como objeto/no objeto.

        L_bbox:
            Regresión l, t, r, b para las celdas positivas.

    Parámetros
    ----------
    objectness_pred : Tensor
        [B, 1, H, W]

    bbox_pred : Tensor
        [B, 4, H, W]

    objectness_target : Tensor
        [B, 1, H, W]

    bbox_target : Tensor
        [B, 4, H, W]

    positive_weight : float
        Peso aplicado a las celdas positivas de objectness.

    Retorna
    -------
    total_loss : Tensor

    loss_obj : Tensor

    loss_bbox : Tensor
    """

    # ================================================
    # 1. Pérdida de objectness
    # ================================================

    positive_mask = objectness_target == 1
    negative_mask = objectness_target == 0

    weights = torch.ones_like(
        objectness_target
    )

    weights[positive_mask] = positive_weight

    loss_obj = F.binary_cross_entropy_with_logits(
        objectness_pred,
        objectness_target,
        weight=weights,
    )

    # ================================================
    # 2. Pérdida de bounding boxes
    # ================================================

    # [B, 1, H, W] → [B, 4, H, W]
    positive_mask_bbox = positive_mask.expand_as(
        bbox_pred
    )

    pred_positive = bbox_pred[
        positive_mask_bbox
    ]

    target_positive = bbox_target[
        positive_mask_bbox
    ]

    if pred_positive.numel() == 0:

        loss_bbox = torch.zeros(
            device=bbox_pred.device,
            dtype=bbox_pred.dtype,
        )

    else:

        loss_bbox = F.smooth_l1_loss(
            pred_positive,
            target_positive,
            reduction="mean",
        )

    # ================================================
    # 3. Pérdida total
    # ================================================

    total_loss = (
        loss_obj +
        loss_bbox
    )

    return (
        total_loss,
        loss_obj,
        loss_bbox,
    )