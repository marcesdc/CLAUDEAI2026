"""Unit tests for src/bandit.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from src.bandit import default_entry, update, sample_all, temperature_from_weight


def test_default_entry_keys():
    entry = default_entry()
    for key in ("alpha", "beta", "weight", "draws_scored"):
        assert key in entry


def test_update_increments_draws_scored():
    weights = {}
    weights = update(weights, "lottomax", hits=3, main_count=7)
    assert weights["lottomax"]["draws_scored"] == 1
    weights = update(weights, "lottomax", hits=2, main_count=7)
    assert weights["lottomax"]["draws_scored"] == 2


def test_update_alpha_increases_on_hits():
    weights = {"649": default_entry()}
    alpha_before = weights["649"]["alpha"]
    weights = update(weights, "649", hits=4, main_count=6)
    assert weights["649"]["alpha"] == alpha_before + 4


def test_update_beta_increases_on_misses():
    weights = {"649": default_entry()}
    beta_before = weights["649"]["beta"]
    weights = update(weights, "649", hits=2, main_count=6)
    assert weights["649"]["beta"] == beta_before + 4  # 6-2 misses


def test_sample_all_returns_all_keys():
    weights = {
        "lottomax":   default_entry(),
        "649":        default_entry(),
        "dailygrand": default_entry(),
    }
    result = sample_all(weights)
    assert set(result.keys()) == {"lottomax", "649", "dailygrand"}


def test_temperature_high_weight():
    t = temperature_from_weight(0.9)
    assert t < 1.2  # high confidence -> low temp (closer to t_min=0.8)


def test_temperature_low_weight():
    t = temperature_from_weight(0.1)
    assert t > 1.2  # low confidence -> high temp (closer to t_max=1.6)


def test_temperature_bounds():
    assert 0.8 <= temperature_from_weight(0.0) <= 1.6
    assert 0.8 <= temperature_from_weight(1.0) <= 1.6
