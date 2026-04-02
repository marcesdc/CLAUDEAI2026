"""Unit tests for src/bandit.py -- stubs filled by q_a1."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from src.bandit import default_entry, update, sample_all, temperature_from_weight


def test_default_entry_keys():
    pass


def test_update_increments_draws_scored():
    pass


def test_update_alpha_increases_on_hits():
    pass


def test_update_beta_increases_on_misses():
    pass


def test_sample_all_returns_all_keys():
    pass


def test_temperature_high_weight():
    pass


def test_temperature_low_weight():
    pass


def test_temperature_bounds():
    pass
