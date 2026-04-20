"""
conftest.py -- shared pytest fixtures for lottery-portfolio tests.

All synthetic data is written to pytest's tmp_path to avoid modifying real data files.
Python 3.14, ASCII-only output throughout. Forked from lottery-nn/tests/conftest.py
with new payout fixtures added.
"""

import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

# Ensure project root is on sys.path so `import config` works in src/ modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Lottery parameters (mirror LOTTERY_RULES in config.py)
# ---------------------------------------------------------------------------
_LOTTERY_PARAMS = {
    "lottomax":   {"main_count": 7, "main_max": 52, "bonus_max": 52, "bonus_col": "bonus"},
    "649":        {"main_count": 6, "main_max": 49, "bonus_max": 49, "bonus_col": "bonus"},
    "dailygrand": {"main_count": 5, "main_max": 49, "bonus_max": 7,  "bonus_col": "grand"},
}


def _make_draws(main_count, main_max, bonus_col, bonus_max, n=60, seed=42):
    """Generate a deterministic synthetic draw DataFrame."""
    rng = random.Random(seed)
    records = []
    date = datetime(2025, 1, 1)
    for _ in range(n):
        main = sorted(rng.sample(range(1, main_max + 1), main_count))
        bonus = rng.randint(1, bonus_max)
        row = {"date": date.strftime("%Y-%m-%d")}
        for i, v in enumerate(main, 1):
            row["n" + str(i)] = v
        row[bonus_col] = bonus
        records.append(row)
        date += timedelta(days=3)
    return pd.DataFrame(records)


def _make_payouts(draws_df, lottery, n_tickets_sold=1_000_000, seed=42):
    """Generate a synthetic payout table consistent with a draws DataFrame.

    For each draw, emit a few prize tiers with plausible winner counts and a jackpot.
    The numbers are synthetic but realistic -- tests only assert structural properties.
    """
    rng = random.Random(seed)
    records = []
    # Simple tier set per lottery for testing
    tier_schedule = {
        "lottomax":   [(7, "-", 50_000_000.0, None), (6, "Y", 0.0, None),
                       (6, "N", 0.0, None),          (5, "-", 0.0, None),
                       (4, "-", 0.0, 20.0),          (3, "Y", 0.0, 20.0)],
        "649":        [(6, "-", 7_000_000.0, None), (5, "Y", 0.0, None),
                       (5, "-", 0.0, None),          (4, "-", 0.0, 85.0),
                       (3, "-", 0.0, 10.0),          (2, "Y", 0.0, 5.0)],
        "dailygrand": [(5, "Y", 7_000_000.0, None), (5, "N", 0.0, 100_000.0),
                       (4, "Y", 0.0, 1_000.0),      (4, "N", 0.0, 500.0),
                       (3, "Y", 0.0, 100.0),        (3, "N", 0.0, 20.0),
                       (2, "Y", 0.0, 10.0),         (1, "Y", 0.0, 4.0)],
    }

    for _, row in draws_df.iterrows():
        # Jackpot: escalates if last draw had 0 jackpot winners
        jackpot = tier_schedule[lottery][0][2] + rng.uniform(0, 5_000_000.0)
        for (tmain, tbonus, _jack, fixed_pay) in tier_schedule[lottery]:
            # Winner count: fewer winners at higher tiers
            expected = max(1, int(n_tickets_sold * (0.0001 ** (tmain / 7))))
            n_winners = rng.randint(0, expected * 2)
            records.append({
                "date":              row["date"],
                "lottery":           lottery,
                "jackpot":           jackpot,
                "tier_main":         tmain,
                "tier_bonus":        tbonus,
                "n_winners":         n_winners,
                "payout_per_winner": fixed_pay if fixed_pay is not None else "",
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Per-lottery DataFrame fixtures (60 draws each, deterministic)
# ---------------------------------------------------------------------------

@pytest.fixture
def df_lottomax():
    p = _LOTTERY_PARAMS["lottomax"]
    return _make_draws(p["main_count"], p["main_max"], p["bonus_col"], p["bonus_max"], seed=42)


@pytest.fixture
def df_649():
    p = _LOTTERY_PARAMS["649"]
    return _make_draws(p["main_count"], p["main_max"], p["bonus_col"], p["bonus_max"], seed=43)


@pytest.fixture
def df_dailygrand():
    p = _LOTTERY_PARAMS["dailygrand"]
    return _make_draws(p["main_count"], p["main_max"], p["bonus_col"], p["bonus_max"], seed=44)


# ---------------------------------------------------------------------------
# CSV path fixtures (written to tmp_path for isolation)
# ---------------------------------------------------------------------------

@pytest.fixture
def lottomax_csv(tmp_path, df_lottomax):
    path = tmp_path / "draws_lottomax.csv"
    df_lottomax.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def draws_649_csv(tmp_path, df_649):
    path = tmp_path / "draws_649.csv"
    df_649.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def dailygrand_csv(tmp_path, df_dailygrand):
    path = tmp_path / "draws_dailygrand.csv"
    df_dailygrand.to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# Payout fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def payouts_lottomax_df(df_lottomax):
    return _make_payouts(df_lottomax, "lottomax", seed=42)


@pytest.fixture
def payouts_649_df(df_649):
    return _make_payouts(df_649, "649", seed=43)


@pytest.fixture
def payouts_dailygrand_df(df_dailygrand):
    return _make_payouts(df_dailygrand, "dailygrand", seed=44)


@pytest.fixture
def payouts_lottomax_csv(tmp_path, payouts_lottomax_df):
    path = tmp_path / "payouts_lottomax.csv"
    payouts_lottomax_df.to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# Full config-patched fixture (redirects DRAWS_CSV and PAYOUTS_CSV to tmp_path)
# ---------------------------------------------------------------------------

@pytest.fixture
def patched_config(tmp_path, monkeypatch,
                   df_lottomax, df_649, df_dailygrand,
                   payouts_lottomax_df, payouts_649_df, payouts_dailygrand_df):
    """Redirect all config CSV paths into tmp_path with synthetic data.

    Returns tmp_path for tests that want to probe files directly.
    """
    import config

    draws_map = {}
    payouts_map = {}
    for lottery, df, pdf in [
        ("lottomax",   df_lottomax,   payouts_lottomax_df),
        ("649",        df_649,        payouts_649_df),
        ("dailygrand", df_dailygrand, payouts_dailygrand_df),
    ]:
        draws_path = tmp_path / f"draws_{lottery}.csv"
        df.to_csv(draws_path, index=False)
        draws_map[lottery] = draws_path

        payouts_path = tmp_path / f"payouts_{lottery}.csv"
        pdf.to_csv(payouts_path, index=False)
        payouts_map[lottery] = payouts_path

    monkeypatch.setattr(config, "DRAWS_CSV",   draws_map)
    monkeypatch.setattr(config, "PAYOUTS_CSV", payouts_map)
    monkeypatch.setattr(config, "DATA_DIR",    tmp_path)
    return tmp_path
