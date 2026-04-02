"""Unit tests for src/preprocessing.py -- stubs filled by q_a1."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from src.preprocessing import build_features


def test_build_features_shapes(df_lottomax):
    pass


def test_build_features_no_bonus(df_lottomax):
    pass


def test_multi_hot_values(df_lottomax):
    pass


def test_decode_multihot_count(df_lottomax):
    pass


def test_decode_onehot_range(df_lottomax):
    pass
