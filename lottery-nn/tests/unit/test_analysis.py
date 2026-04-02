"""Unit tests for src/analysis.py -- stubs filled by q_a1."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from src.analysis import frequency_table, hot_cold, pair_frequency, gap_analysis


def test_frequency_table_shape(df_lottomax):
    pass


def test_frequency_table_count_sum(df_lottomax):
    pass


def test_hot_cold_no_overlap(df_lottomax):
    pass


def test_pair_frequency_returns_dataframe(df_lottomax):
    pass


def test_gap_analysis_shape(df_lottomax):
    pass


def test_gap_analysis_columns(df_lottomax):
    pass
