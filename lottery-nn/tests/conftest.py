"""
conftest.py -- shared pytest fixtures for lottery-nn tests.

All synthetic data is written to pytest's tmp_path to avoid modifying real data files.
Python 3.14, ASCII-only output throughout.
"""

import sys
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

# Ensure project root is on sys.path so `import config` works in src/ modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Lottery parameters (mirrors LOTTERY_CONFIGS in preprocessing_swarm.py)
# ---------------------------------------------------------------------------

_LOTTERY_PARAMS = {
    "lottomax":   {"main_count": 7, "main_max": 50, "bonus_max": 50, "bonus_col": "bonus"},
    "649":        {"main_count": 6, "main_max": 49, "bonus_max": 49, "bonus_col": "bonus"},
    "dailygrand": {"main_count": 5, "main_max": 49, "bonus_max": 7,  "bonus_col": "grand"},
}


def _make_draws(main_count, main_max, bonus_col, bonus_max, n=50, seed=42):
    """Generate a deterministic synthetic draw DataFrame."""
    rng = random.Random(seed)
    records = []
    date = datetime(2020, 1, 1)
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


# ---------------------------------------------------------------------------
# Per-lottery DataFrame fixtures (50 draws each, deterministic)
# ---------------------------------------------------------------------------

@pytest.fixture
def df_lottomax():
    """Synthetic LottoMax draws: 7 from 50, bonus 1-50, 50 rows."""
    p = _LOTTERY_PARAMS["lottomax"]
    return _make_draws(p["main_count"], p["main_max"], p["bonus_col"], p["bonus_max"])


@pytest.fixture
def df_649():
    """Synthetic 6/49 draws: 6 from 49, bonus 1-49, 50 rows."""
    p = _LOTTERY_PARAMS["649"]
    return _make_draws(p["main_count"], p["main_max"], p["bonus_col"], p["bonus_max"], seed=43)


@pytest.fixture
def df_dailygrand():
    """Synthetic Daily Grand draws: 5 from 49, grand 1-7, 50 rows.
    Uses column name 'grand' matching real CSV format."""
    p = _LOTTERY_PARAMS["dailygrand"]
    return _make_draws(p["main_count"], p["main_max"], p["bonus_col"], p["bonus_max"], seed=44)


@pytest.fixture
def df_dailygrand_normalised(df_dailygrand):
    """Daily Grand df with 'grand' renamed to 'bonus', as produced by load_lottery_df()."""
    return df_dailygrand.rename(columns={"grand": "bonus"})


# ---------------------------------------------------------------------------
# CSV path fixtures (written to tmp_path for isolation)
# ---------------------------------------------------------------------------

@pytest.fixture
def lottomax_csv(tmp_path, df_lottomax):
    """Write synthetic LottoMax draws to a temp CSV, return path string."""
    path = tmp_path / "draws_lottomax_test.csv"
    df_lottomax.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def draws_649_csv(tmp_path, df_649):
    """Write synthetic 6/49 draws to a temp CSV, return path string."""
    path = tmp_path / "draws_649_test.csv"
    df_649.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def dailygrand_csv(tmp_path, df_dailygrand):
    """Write synthetic Daily Grand draws to a temp CSV, return path string.
    Column is 'grand' (not 'bonus') to match real file format."""
    path = tmp_path / "draws_dailygrand_test.csv"
    df_dailygrand.to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# Minimal draws.csv for feedback.py tests
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_draws_csv(tmp_path):
    """Write a minimal 3-row LottoMax draws.csv to tmp_path, return path string."""
    path = tmp_path / "draws.csv"
    rows = [
        {"date": "2026-01-01", "n1": 1,  "n2": 5,  "n3": 10, "n4": 20, "n5": 30, "n6": 40, "n7": 50},
        {"date": "2026-01-04", "n1": 2,  "n2": 6,  "n3": 11, "n4": 21, "n5": 31, "n6": 41, "n7": 49},
        {"date": "2026-01-07", "n1": 3,  "n2": 7,  "n3": 12, "n4": 22, "n5": 32, "n6": 42, "n7": 48},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# Feature array fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def swarm_features_all(df_lottomax, df_649, df_dailygrand):
    """Dict of {lottery_name: (X, y_main, y_bonus)} from synthetic data.
    Daily Grand 'grand' column is renamed to 'bonus' before build_features."""
    from src.preprocessing_swarm import build_features, LOTTERY_CONFIGS
    dg = df_dailygrand.rename(columns={"grand": "bonus"})
    dfs = {"lottomax": df_lottomax, "649": df_649, "dailygrand": dg}
    return {name: build_features(dfs[name], cfg) for name, cfg in LOTTERY_CONFIGS.items()}


# ---------------------------------------------------------------------------
# PyTorch tensor fixtures (CPU only)
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_swarm_batch():
    """Tensor (4, 10, 100) float32 on CPU for SharedLotteryTransformer tests."""
    import torch
    return torch.zeros(4, 10, 100, dtype=torch.float32)


@pytest.fixture
def dummy_main_batch():
    """Tensor (4, 10, 100) float32 on CPU for single-lottery model tests (2*MAIN_MAX=100)."""
    import torch
    return torch.zeros(4, 10, 100, dtype=torch.float32)
