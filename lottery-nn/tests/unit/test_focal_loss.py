"""Unit tests for focal_loss_with_logits."""
import pytest
import torch
import torch.nn.functional as F

from src.focal_loss import focal_loss_with_logits


def test_focal_equals_bce_when_gamma_zero():
    """With gamma=0 and alpha=None, focal loss must equal BCEWithLogitsLoss."""
    torch.manual_seed(0)
    logits  = torch.randn(8, 50)
    targets = (torch.rand(8, 50) > 0.86).float()  # ~14% positive, mirrors 7/50

    bce_loss   = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
    focal_loss = focal_loss_with_logits(logits, targets, gamma=0.0, alpha=None)

    assert abs(focal_loss.item() - bce_loss.item()) < 1e-5, (
        f"Expected focal (gamma=0, alpha=None) == BCE, "
        f"got focal={focal_loss.item():.6f} bce={bce_loss.item():.6f}"
    )


def test_focal_less_than_bce_on_easy_examples():
    """On easy (confident, correct) predictions focal loss must be < BCE."""
    logits  = torch.full((8, 50), 5.0)   # very confident positive
    targets = torch.ones(8, 50)           # all labels are 1

    bce_loss   = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
    focal_loss = focal_loss_with_logits(logits, targets, gamma=2.0, alpha=None)

    assert focal_loss.item() < bce_loss.item(), (
        f"Expected focal < BCE on easy examples, "
        f"got focal={focal_loss.item():.6f} bce={bce_loss.item():.6f}"
    )


def test_focal_returns_scalar():
    """focal_loss_with_logits must return a scalar tensor."""
    logits  = torch.randn(4, 50)
    targets = torch.zeros(4, 50)
    loss = focal_loss_with_logits(logits, targets)
    assert loss.shape == torch.Size([]), f"Expected scalar, got shape {loss.shape}"


def test_focal_alpha_scales_loss():
    """alpha=0.5 should produce a different (scaled) loss than alpha=None."""
    torch.manual_seed(1)
    logits  = torch.randn(8, 50)
    targets = (torch.rand(8, 50) > 0.86).float()

    loss_no_alpha   = focal_loss_with_logits(logits, targets, gamma=2.0, alpha=None)
    loss_with_alpha = focal_loss_with_logits(logits, targets, gamma=2.0, alpha=0.5)

    assert abs(loss_no_alpha.item() - loss_with_alpha.item()) > 1e-6, (
        "Expected alpha to change the loss value"
    )


def test_negative_gamma_raises():
    """gamma < 0 must raise ValueError."""
    logits  = torch.randn(4, 50)
    targets = torch.zeros(4, 50)
    with pytest.raises(ValueError, match="gamma"):
        focal_loss_with_logits(logits, targets, gamma=-1.0)


def test_alpha_out_of_range_raises():
    """alpha outside (0, 1) must raise ValueError."""
    logits  = torch.randn(4, 50)
    targets = torch.zeros(4, 50)
    with pytest.raises(ValueError, match="alpha"):
        focal_loss_with_logits(logits, targets, alpha=1.5)
    with pytest.raises(ValueError, match="alpha"):
        focal_loss_with_logits(logits, targets, alpha=0.0)
