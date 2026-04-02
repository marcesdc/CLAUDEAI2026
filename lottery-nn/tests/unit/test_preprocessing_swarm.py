"""Unit tests for src/preprocessing_swarm.py -- stubs filled by q_a1."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from src.preprocessing_swarm import build_features, LOTTERY_CONFIGS


def test_build_features_padded_shape(df_lottomax):
    pass


def test_build_features_lottomax_y_shape(df_lottomax):
    pass


def test_build_features_649_y_shape(df_649):
    pass


def test_build_features_dailygrand_y_bonus(df_dailygrand_normalised):
    pass


def test_lottery_configs_required_keys():
    pass


def test_load_lottery_df_renames_grand(dailygrand_csv):
    pass
