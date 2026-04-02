"""
Focal loss for multi-label binary classification.

Focal loss down-weights easy negatives via a modulating factor (1-p)^gamma,
sharpening the training signal on hard examples (the drawn numbers).

Reference: Lin et al., "Focal Loss for Dense Object Detection" (RetinaNet), 2017.
"""

import torch
import torch.nn.functional as F


def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = None,
) -> torch.Tensor:
    """
    Compute focal loss for multi-label binary targets.

    Parameters
    ----------
    logits  : (batch, num_classes) raw model output (before sigmoid)
    targets : (batch, num_classes) binary float targets in {0, 1}
    gamma   : modulating exponent. 0.0 = standard BCE. Default 2.0.
    alpha   : positive-class weight in [0, 1], or None to skip alpha weighting.
              Default None.

    Returns
    -------
    Scalar mean loss over the batch and all classes.
    """
    # Numerically stable per-element BCE (same as BCEWithLogitsLoss reduction='none')
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    # p_t = predicted probability for the true class
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)

    focal_weight = (1.0 - p_t) ** gamma

    if alpha is not None:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        focal_weight = alpha_t * focal_weight

    return (focal_weight * bce).mean()
