"""Integration tests for src/backtest.py -- run a small rolling-origin pass."""

import math

import numpy as np
import pandas as pd
import pytest

import config
from src import backtest as bt


# ---------------------------------------------------------------------------
# realized_payout building blocks
# ---------------------------------------------------------------------------

def test_realized_payout_exact_match_fixed_tier(df_649, payouts_649_df):
    """If our combination equals the winning numbers, we 'match' 6/6 which is
    jackpot (pari-mutuel). Verify a pari-mutuel contribution is produced."""
    row = df_649.iloc[0]
    winning = sorted(int(row[f"n{i}"]) for i in range(1, 7))
    r = bt.realized_payout("649", winning, row, payouts_649_df)
    # Some positive dollar amount since pool > 0 for jackpot tier
    assert r > 0.0


def test_realized_payout_zero_matches_gives_zero(df_649, payouts_649_df):
    """A combination with no overlap with W should realize $0 (no tier has k=0 payout)."""
    row = df_649.iloc[0]
    W = set(int(row[f"n{i}"]) for i in range(1, 7))
    # Pick a combo guaranteed to miss
    combo = sorted(list(set(range(1, 50)) - W))[:6]
    r = bt.realized_payout("649", combo, row, payouts_649_df)
    # Per PRIZE_TIERS: (2, False) is a free-play tier with payout=0
    # and lower k tiers don't exist, so realized = 0
    assert math.isclose(r, 0.0, abs_tol=1e-12)


def test_realized_payout_fixed_tier_recovers_config_payout(df_lottomax, payouts_lottomax_df):
    """Crafted case: force a 4/7 main match (fixed $20 payout for lottomax)."""
    row = df_lottomax.iloc[0]
    W = sorted(int(row[f"n{i}"]) for i in range(1, 8))  # 7 winning numbers
    # Build a combo that matches EXACTLY 4 of W: take 4 winners + 3 non-winners
    non_W = [n for n in range(1, 51) if n not in set(W)]
    combo = sorted(W[:4] + non_W[:3])
    r = bt.realized_payout("lottomax", combo, row, payouts_lottomax_df)
    # (4, None, 20.0) is the only tier that fires -> expect $20
    assert math.isclose(r, 20.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Full rolling-origin backtest run
# ---------------------------------------------------------------------------

def test_backtest_run_on_649(patched_config, capsys):
    """End-to-end: backtest runs, emits per-draw CSV, and produces sensible numbers."""
    code = bt.run("649", min_train=20, refit_every=5, seed=42, save_csv=True)
    assert code == 0

    # Output CSV written
    assert config.BACKTEST_RESULTS.exists()
    df = pd.read_csv(config.BACKTEST_RESULTS)
    assert len(df) > 0
    assert set(["date", "combo_A", "combo_B", "k_A", "k_B",
                "realized_A", "realized_B", "diff"]).issubset(set(df.columns))
    # All k values should be within [0, main_count]
    assert df["k_A"].between(0, 6).all()
    assert df["k_B"].between(0, 6).all()

    out = capsys.readouterr().out
    assert "BACKTEST" in out
    assert "VERDICT" in out


def test_backtest_all_three_lotteries(patched_config):
    """Smoke test: backtest succeeds on each lottery with synthetic data."""
    for lot in ["lottomax", "649", "dailygrand"]:
        code = bt.run(lot, min_train=20, refit_every=10, seed=7, save_csv=False)
        assert code == 0, f"backtest failed for {lot}"


def test_backtest_errors_when_data_too_small(patched_config, monkeypatch):
    """If draws has fewer rows than min_train, report error and exit 1."""
    # Shrink the draws file by overwriting with just 5 rows
    import config
    df = pd.read_csv(config.DRAWS_CSV["649"]).head(5)
    df.to_csv(config.DRAWS_CSV["649"], index=False)
    code = bt.run("649", min_train=20, save_csv=False)
    assert code == 1


def test_backtest_errors_when_no_draws(monkeypatch, tmp_path):
    """If draws file is missing entirely, error out."""
    import config
    monkeypatch.setattr(config, "DRAWS_CSV",
                        {**config.DRAWS_CSV, "649": tmp_path / "missing.csv"})
    monkeypatch.setattr(config, "PAYOUTS_CSV",
                        {**config.PAYOUTS_CSV, "649": tmp_path / "missing_p.csv"})
    code = bt.run("649", min_train=20, save_csv=False)
    assert code == 1
