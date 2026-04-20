"""Unit tests for src/pruning.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import copy
import torch
import torch.nn as nn
import pytest

from src.pruning import (
    apply_masks,
    prune_by_percent,
    reset_to_init,
    sparsity,
    _eligible_params,
)


# ---------------------------------------------------------------------------
# Fixture: a small deterministic model
# ---------------------------------------------------------------------------

class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8, bias=False)
        self.fc2 = nn.Linear(8, 4, bias=False)
        self.norm = nn.LayerNorm(8)   # should NOT be pruned

    def forward(self, x):
        return self.fc2(self.fc1(x))


@pytest.fixture()
def model():
    net = _TinyNet()
    # Fill with known values so tests are deterministic
    torch.manual_seed(0)
    nn.init.uniform_(net.fc1.weight, -1.0, 1.0)
    nn.init.uniform_(net.fc2.weight, -1.0, 1.0)
    return net


# ---------------------------------------------------------------------------
# _eligible_params
# ---------------------------------------------------------------------------

def test_eligible_params_includes_linear_weights(model):
    params = _eligible_params(model)
    names = [n for n, _ in params]
    assert "fc1.weight" in names
    assert "fc2.weight" in names


def test_eligible_params_excludes_layernorm(model):
    params = _eligible_params(model)
    names = [n for n, _ in params]
    assert not any("norm" in n for n in names)


# ---------------------------------------------------------------------------
# prune_by_percent
# ---------------------------------------------------------------------------

def test_prune_zeros_correct_fraction(model):
    masks = prune_by_percent(model, 0.5)
    s = sparsity(model)
    # Allow +-2% tolerance for ties at the cutoff
    assert 0.48 <= s <= 0.52


def test_prune_zero_percent_keeps_all(model):
    prune_by_percent(model, 0.0)
    assert sparsity(model) == 0.0


def test_prune_masks_are_binary(model):
    masks = prune_by_percent(model, 0.4)
    for mask in masks.values():
        unique = set(mask.flatten().tolist())
        assert unique <= {0.0, 1.0}


def test_prune_invalid_percent_raises(model):
    with pytest.raises(ValueError):
        prune_by_percent(model, 1.0)
    with pytest.raises(ValueError):
        prune_by_percent(model, -0.1)


def test_prune_modifies_model_in_place(model):
    w_before = model.fc1.weight.data.clone()
    prune_by_percent(model, 0.5)
    w_after = model.fc1.weight.data
    # At least some weights should have been zeroed
    assert not torch.equal(w_before, w_after)


def test_prune_small_weights_are_zeroed(model):
    # Manually set one weight to near-zero so it should be pruned
    model.fc1.weight.data[0, 0] = 1e-9
    prune_by_percent(model, 0.5)
    assert model.fc1.weight.data[0, 0].item() == 0.0


def test_prune_empty_model_returns_empty():
    class _NoLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(4)
    masks = prune_by_percent(_NoLinear(), 0.5)
    assert masks == {}


# ---------------------------------------------------------------------------
# apply_masks
# ---------------------------------------------------------------------------

def test_apply_masks_re_zeros_after_weight_update(model):
    masks = prune_by_percent(model, 0.5)
    # Simulate optimizer step that "fills in" zeros
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.ones_like(p))
    apply_masks(model, masks)
    # All positions that were masked must still be zero
    for name, mask in masks.items():
        param = dict(model.named_parameters())[name]
        zeroed = (mask == 0)
        assert (param.data[zeroed] == 0).all()


# ---------------------------------------------------------------------------
# reset_to_init
# ---------------------------------------------------------------------------

def test_reset_restores_init_values_for_unmasked_weights(model):
    init_state = copy.deepcopy(model.state_dict())
    masks = prune_by_percent(model, 0.5)
    reset_to_init(model, init_state, masks)

    for name, mask in masks.items():
        param = dict(model.named_parameters())[name]
        init_w = init_state[name]
        kept = (mask == 1)
        torch.testing.assert_close(param.data[kept], init_w[kept])


def test_reset_keeps_pruned_weights_at_zero(model):
    init_state = copy.deepcopy(model.state_dict())
    masks = prune_by_percent(model, 0.5)
    reset_to_init(model, init_state, masks)

    for name, mask in masks.items():
        param = dict(model.named_parameters())[name]
        zeroed = (mask == 0)
        assert (param.data[zeroed] == 0).all()


def test_reset_does_not_mutate_init_state(model):
    init_state = copy.deepcopy(model.state_dict())
    orig_fc1 = init_state["fc1.weight"].clone()
    masks = prune_by_percent(model, 0.5)
    reset_to_init(model, init_state, masks)
    torch.testing.assert_close(init_state["fc1.weight"], orig_fc1)


# ---------------------------------------------------------------------------
# sparsity
# ---------------------------------------------------------------------------

def test_sparsity_zero_before_pruning(model):
    assert sparsity(model) == 0.0


def test_sparsity_after_pruning(model):
    prune_by_percent(model, 0.6)
    s = sparsity(model)
    assert 0.58 <= s <= 0.62


def test_sparsity_no_linear_layers():
    class _NoLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(4)
    assert sparsity(_NoLinear()) == 0.0
