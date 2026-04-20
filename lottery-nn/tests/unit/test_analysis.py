"""Unit tests for src/analysis.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import pytest
import config
from src.analysis import frequency_table, hot_cold, pair_frequency, gap_analysis


def test_frequency_table_shape(df_lottomax):
    ft = frequency_table(df_lottomax)
    assert len(ft) == config.LOTTERY["main_max"]


def test_frequency_table_count_sum(df_lottomax):
    ft = frequency_table(df_lottomax)
    expected = len(df_lottomax) * config.LOTTERY["main_count"]
    assert ft["count"].sum() == expected


def test_hot_cold_no_overlap(df_lottomax):
    result = hot_cold(df_lottomax)
    hot = set(result["hot"])
    cold = set(result["cold"])
    assert hot & cold == set()
    assert hot | cold == set(range(1, config.LOTTERY["main_max"] + 1))


def test_pair_frequency_returns_dataframe(df_lottomax):
    result = pair_frequency(df_lottomax)
    assert isinstance(result, pd.DataFrame)
    assert "pair" in result.columns
    assert "count" in result.columns


def test_gap_analysis_shape(df_lottomax):
    result = gap_analysis(df_lottomax)
    assert len(result) == config.LOTTERY["main_max"]


def test_gap_analysis_columns(df_lottomax):
    result = gap_analysis(df_lottomax)
    assert "number" in result.columns
    assert "avg_gap" in result.columns
