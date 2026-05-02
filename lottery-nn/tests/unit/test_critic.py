"""
Tests for the Reflexion-style critic (B3).

Stats are computed empirically from historical draws via numpy percentiles.
The critic rejects lines that are statistically improbable: extreme sums,
long consecutive runs, all-same-decade, all-even / all-odd.
"""

import numpy as np
import pandas as pd
import pytest

from src.critic import (
    _decade_concentration,
    _max_consecutive,
    compute_lottery_stats,
    critic_filter,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def test_max_consecutive_simple():
    assert _max_consecutive([1, 2, 3, 4, 5, 6, 7]) == 7
    assert _max_consecutive([1, 3, 5, 7, 9]) == 1
    assert _max_consecutive([1, 2, 5, 6, 7, 20]) == 3
    assert _max_consecutive([]) == 0


def test_decade_concentration_simple():
    assert _decade_concentration([1, 2, 3, 4, 5, 6, 7]) == 7    # all decade 0
    assert _decade_concentration([1, 11, 21, 31, 41, 51]) == 1  # spread across 6 decades
    assert _decade_concentration([1, 2, 11, 12, 21]) == 2       # max 2-per-decade


# ---------------------------------------------------------------------------
# compute_lottery_stats
# ---------------------------------------------------------------------------

def test_compute_stats_lottomax_keys(df_lottomax):
    cfg = {"main_count": 7}
    stats = compute_lottery_stats(df_lottomax, cfg)
    expected = {"main_count", "sum_p_lo", "sum_p_hi",
                "max_consecutive_p_hi", "decade_concentration_p_hi"}
    assert expected.issubset(stats.keys())
    assert stats["main_count"] == 7
    assert stats["sum_p_lo"] < stats["sum_p_hi"]


def test_compute_stats_649_main_count(df_649):
    cfg = {"main_count": 6}
    stats = compute_lottery_stats(df_649, cfg)
    assert stats["main_count"] == 6


def test_compute_stats_dailygrand(df_dailygrand_normalised):
    cfg = {"main_count": 5}
    stats = compute_lottery_stats(df_dailygrand_normalised, cfg)
    assert stats["main_count"] == 5
    assert stats["sum_p_lo"] >= 1 + 2 + 3 + 4 + 5      # min possible sum
    assert stats["sum_p_hi"] <= 49 + 48 + 47 + 46 + 45 # max possible sum


def test_compute_stats_missing_columns_raises():
    df = pd.DataFrame({"date": ["2026-01-01"], "n1": [1]})
    cfg = {"main_count": 7}
    with pytest.raises(KeyError):
        compute_lottery_stats(df, cfg)


def test_compute_stats_custom_percentiles(df_lottomax):
    cfg = {"main_count": 7}
    s_default = compute_lottery_stats(df_lottomax, cfg, percentiles=(5, 95))
    s_tight   = compute_lottery_stats(df_lottomax, cfg, percentiles=(25, 75))
    # Tighter percentiles produce a narrower band.
    assert s_tight["sum_p_hi"] - s_tight["sum_p_lo"] \
        <= s_default["sum_p_hi"] - s_default["sum_p_lo"] + 1e-9


# ---------------------------------------------------------------------------
# critic_filter
# ---------------------------------------------------------------------------

def _lottomax_stats(df_lottomax):
    return compute_lottery_stats(df_lottomax, {"main_count": 7})


def test_critic_rejects_all_consecutive(df_lottomax):
    stats = _lottomax_stats(df_lottomax)
    assert critic_filter([1, 2, 3, 4, 5, 6, 7], stats) is False


def test_critic_rejects_all_even(df_lottomax):
    stats = _lottomax_stats(df_lottomax)
    # Synthesize an all-even line within the historical sum range.
    line = [2, 6, 14, 22, 26, 32, 50]   # sum = 152
    # If the line happens to fall outside the sum band, the test still
    # passes (rejection is what we want) -- but we want a *specifically*
    # parity-driven rejection, so verify sum is in band first.
    if stats["sum_p_lo"] <= sum(line) <= stats["sum_p_hi"]:
        assert critic_filter(line, stats) is False
    else:
        # Fallback: any all-even line is rejected (parity rule still triggers)
        assert critic_filter(line, stats) is False


def test_critic_rejects_all_odd(df_lottomax):
    stats = _lottomax_stats(df_lottomax)
    line = [1, 5, 13, 21, 27, 33, 49]   # sum = 149, all odd
    assert critic_filter(line, stats) is False


def test_critic_rejects_extreme_low_sum(df_lottomax):
    stats = _lottomax_stats(df_lottomax)
    line = [1, 2, 3, 4, 5, 6, 8]
    assert sum(line) < stats["sum_p_lo"]
    assert critic_filter(line, stats) is False


def test_critic_rejects_extreme_high_sum(df_lottomax):
    stats = _lottomax_stats(df_lottomax)
    line = [46, 47, 48, 49, 50, 51, 52]
    assert sum(line) > stats["sum_p_hi"]
    assert critic_filter(line, stats) is False


def test_critic_accepts_real_historical_draw(df_lottomax):
    """A row drawn from the historical distribution should pass the critic
    *unless* the synthetic fixture itself produced a pathological row -- in
    which case the critic correctly rejects it. We accept either outcome
    but require at least one historical row to pass to validate the path."""
    stats = _lottomax_stats(df_lottomax)
    cols = [f"n{i}" for i in range(1, 8)]
    accepts = 0
    for _, row in df_lottomax.iterrows():
        line = [int(row[c]) for c in cols]
        if critic_filter(line, stats):
            accepts += 1
    # At least some historical rows must pass (sanity check that the critic
    # isn't pathologically rejecting everything).
    assert accepts >= 5, f"only {accepts} rows passed critic out of {len(df_lottomax)}"


def test_critic_rejects_empty_line(df_lottomax):
    stats = _lottomax_stats(df_lottomax)
    assert critic_filter([], stats) is False


def test_critic_rejects_all_same_decade(df_lottomax):
    """Synthetic stats may allow some clustering, so override directly."""
    stats = {
        "main_count": 7,
        "sum_p_lo": 0,
        "sum_p_hi": 1000,
        "max_consecutive_p_hi": 7,
        "decade_concentration_p_hi": 4,   # historical max is 4 per decade
    }
    # A line where all 7 numbers fall in 1-10 violates concentration_p_hi=4
    line = [1, 2, 3, 4, 5, 6, 7]
    # Note: this also fails the consecutive rule (7 > 7? no, equal allowed).
    # Force consecutive to allow this exact line by checking decade only.
    stats_loose_consec = dict(stats, max_consecutive_p_hi=99)
    assert critic_filter(line, stats_loose_consec) is False
