"""Unit tests for src/data_loader.py -- stubs filled by q_a1."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import pandas as pd
from src.data_loader import load_draws, generate_synthetic, _normalize_columns, _validate


def test_generate_synthetic_shape(tmp_path):
    pass


def test_generate_synthetic_range(tmp_path):
    pass


def test_load_draws_from_csv(tmp_path):
    pass


def test_normalize_columns_aliases():
    pass


def test_normalize_columns_grand_unchanged():
    pass


def test_validate_raises_missing_col():
    pass


def test_validate_raises_out_of_range():
    pass


def test_load_draws_fallback_synthetic():
    pass
