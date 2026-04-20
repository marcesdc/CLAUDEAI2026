"""
Magnitude-based weight pruning for SharedLotteryTransformer.

Adapted from: The Lottery Ticket Hypothesis (Frankle & Carlin, 2018)
Strategy: global unstructured magnitude pruning across all nn.Linear weight
tensors. Biases, embeddings, and LayerNorm parameters are excluded because
they are too few and semantically different from connection weights.

Workflow
--------
1. Train model to convergence (save init_state before training).
2. Load best checkpoint.
3. prune_by_percent(model, percent) -- zero out smallest-magnitude weights.
4. reset_to_init(model, init_state, masks)  -- reload init values, re-apply masks.
5. Fine-tune the sparse subnetwork (re-apply masks each epoch via apply_masks).
"""

import copy

import torch
import torch.nn as nn


def _eligible_params(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    """Return (name, param) pairs eligible for pruning: nn.Linear weights only."""
    result = []
    for mod_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            param_name = f"{mod_name}.weight"
            result.append((param_name, module.weight))
    return result


def prune_by_percent(
    model: nn.Module,
    percent: float,
) -> dict[str, torch.Tensor]:
    """
    Globally prune the bottom *percent* of weights by absolute magnitude
    across all nn.Linear layers. Applies masks in-place.

    Args:
        model:   Model to prune (weights modified in-place).
        percent: Fraction to zero out, e.g. 0.65 removes 65%.

    Returns:
        masks: dict mapping param name -> binary float mask (1=keep, 0=pruned).

    Raises:
        ValueError: if percent is outside [0, 1).
    """
    if not (0.0 <= percent < 1.0):
        raise ValueError(f"percent must be in [0, 1), got {percent}")

    eligible = _eligible_params(model)
    if not eligible:
        return {}

    # percent=0.0 means keep everything -- skip threshold computation entirely
    if percent == 0.0:
        return {name: torch.ones_like(p.data) for name, p in eligible}

    # Pool all absolute values globally to find a single cutoff threshold
    all_abs = torch.cat([p.data.abs().flatten() for _, p in eligible])
    sorted_abs, _ = all_abs.sort()
    cutoff_idx = int(percent * sorted_abs.numel())
    cutoff = sorted_abs[cutoff_idx].item()

    masks: dict[str, torch.Tensor] = {}
    for name, param in eligible:
        # Strict > means weights exactly at cutoff are zeroed; actual pruned % may slightly exceed percent
        mask = (param.data.abs() > cutoff).float()
        masks[name] = mask.clone()
        param.data.mul_(mask)

    n_total  = sum(m.numel() for m in masks.values())
    n_pruned = sum(int((m == 0).sum().item()) for m in masks.values())
    print(
        f"[pruning] Pruned {n_pruned:,}/{n_total:,} weights "
        f"({100.0 * n_pruned / n_total:.1f}%)  cutoff={cutoff:.6f}"
    )
    return masks


def apply_masks(model: nn.Module, masks: dict[str, torch.Tensor]) -> None:
    """Re-zero all pruned weights after an optimizer step (keeps sparsity)."""
    params = dict(model.named_parameters())
    for name, mask in masks.items():
        if name in params:
            params[name].data.mul_(mask.to(params[name].device))


def reset_to_init(
    model: nn.Module,
    init_state: dict,
    masks: dict[str, torch.Tensor],
) -> None:
    """
    Reset model weights to their original initialization, then re-apply masks.

    This is the key insight from the Lottery Ticket Hypothesis: the winning
    ticket is the sparse subnetwork at its ORIGINAL initialization values,
    not the values it converged to after training.

    Args:
        model:      Model whose weights will be overwritten.
        init_state: state_dict snapshot taken before any training.
        masks:      Binary masks returned by prune_by_percent.
    """
    new_state = copy.deepcopy(init_state)
    for name, mask in masks.items():
        if name in new_state:
            new_state[name] = new_state[name] * mask.cpu()
    model.load_state_dict(new_state)
    print("[pruning] Weights reset to initialization values with masks applied.")


def sparsity(model: nn.Module) -> float:
    """Return fraction of zero weights across all nn.Linear layers (0.0-1.0)."""
    eligible = _eligible_params(model)
    if not eligible:
        return 0.0
    total  = sum(p.numel() for _, p in eligible)
    zeroed = sum(int((p.data == 0).sum().item()) for _, p in eligible)
    return zeroed / total
