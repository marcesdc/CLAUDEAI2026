"""Unit tests for src/preprocessing_swarm.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from src.preprocessing_swarm import (
    LOTTERY_CONFIGS,
    POOL_MAX,
    SEQ_LEN,
    build_features,
    load_lottery_df,
)


def test_build_features_padded_shape(df_lottomax):
    cfg = LOTTERY_CONFIGS["lottomax"]
    X, y_main, y_bonus = build_features(df_lottomax, cfg)
    assert X.ndim == 3
    assert X.shape[1] == SEQ_LEN
    assert X.shape[2] == 2 * POOL_MAX  # 104


def test_build_features_lottomax_y_shape(df_lottomax):
    cfg = LOTTERY_CONFIGS["lottomax"]
    X, y_main, y_bonus = build_features(df_lottomax, cfg)
    assert y_main.shape[1] == cfg["main_max"]   # 52
    assert y_bonus.shape[1] == cfg["bonus_max"]  # 52 (dummy zeros, has_bonus=False)


def test_build_features_649_y_shape(df_649):
    cfg = LOTTERY_CONFIGS["649"]
    X, y_main, y_bonus = build_features(df_649, cfg)
    assert y_main.shape[1] == 49
    assert y_bonus.shape[1] == 49


def test_build_features_dailygrand_y_bonus(df_dailygrand_normalised):
    cfg = LOTTERY_CONFIGS["dailygrand"]
    X, y_main, y_bonus = build_features(df_dailygrand_normalised, cfg)
    assert y_bonus.shape[1] == 7
    assert y_bonus.max() <= 1.0
    assert y_bonus.min() >= 0.0


def test_lottery_configs_required_keys():
    required = {"id", "name", "main_count", "main_max", "bonus_max",
                "has_bonus", "bonus_col", "lines_per", "csv"}
    for name, cfg in LOTTERY_CONFIGS.items():
        for key in required:
            assert key in cfg, f"'{name}' config missing key '{key}'"


def test_load_lottery_df_renames_grand(dailygrand_csv, monkeypatch):
    import src.preprocessing_swarm as ps
    monkeypatch.setitem(ps.LOTTERY_CONFIGS["dailygrand"], "csv", dailygrand_csv)
    df = load_lottery_df("dailygrand")
    assert "bonus" in df.columns
    assert "grand" not in df.columns
