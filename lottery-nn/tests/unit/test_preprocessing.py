"""Unit tests for src/preprocessing.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
import config
from src.preprocessing import build_features, decode_multihot, decode_onehot


def test_build_features_shapes(df_lottomax):
    X, y_main, _ = build_features(df_lottomax)
    assert X.ndim == 3
    assert X.shape[0] == len(df_lottomax) - config.SEQUENCE_LEN
    assert X.shape[1] == config.SEQUENCE_LEN
    assert X.shape[2] == 2 * config.LOTTERY["main_max"]
    assert y_main.shape[1] == config.LOTTERY["main_max"]


def test_build_features_no_bonus(df_lottomax):
    df_no_bonus = df_lottomax.drop(columns=["bonus"])
    X, y_main, y_bonus = build_features(df_no_bonus)
    assert y_bonus is None


def test_multi_hot_values(df_lottomax):
    df_no_bonus = df_lottomax.drop(columns=["bonus"])
    X, y_main, _ = build_features(df_no_bonus)
    assert ((y_main == 0) | (y_main == 1)).all()
    assert (y_main.sum(axis=1) == config.LOTTERY["main_count"]).all()


def test_decode_multihot_count(df_lottomax):
    df_no_bonus = df_lottomax.drop(columns=["bonus"])
    X, y_main, _ = build_features(df_no_bonus)
    result = decode_multihot(y_main[0])
    assert len(result) == config.LOTTERY["main_count"]
    assert all(1 <= n <= config.LOTTERY["main_max"] for n in result)


def test_decode_onehot_range(df_lottomax):
    vec = np.zeros(config.LOTTERY["main_max"])
    vec[10] = 1.0  # ball 11
    assert decode_onehot(vec) == 11
