"""
Data loading and generation utilities.

Supported sources
-----------------
1. A user-supplied CSV/Excel with columns:
       date, n1, n2, n3, n4, n5, n6, n7, bonus
2. Synthetic data generated for testing when no real data is available.
"""

import os
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import config

MAIN_COLS = [f"n{i}" for i in range(1, config.LOTTERY["main_count"] + 1)]  # n1..n7
REQUIRED_COLS = {"date"} | set(MAIN_COLS) | {"bonus"}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def load_draws(csv_path: str = config.RAW_CSV) -> pd.DataFrame:
    """
    Load draws from *csv_path* (CSV or Excel).  Returns a DataFrame with
    columns:  date  n1  n2  n3  n4  n5  n6  n7  bonus
    sorted oldest-first.

    If the file does not exist, falls back to :func:`generate_synthetic`.
    """
    path = Path(csv_path)
    if not path.exists():
        # Also look for an Excel file alongside the CSV path
        xlsx = path.with_suffix(".xlsx")
        xls = path.with_suffix(".xls")
        if xlsx.exists():
            path = xlsx
        elif xls.exists():
            path = xls
        else:
            print(f"[data_loader] '{csv_path}' not found - generating synthetic data.")
            return generate_synthetic()

    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path, parse_dates=["date"])
    else:
        df = pd.read_csv(path, parse_dates=["date"])

    df = _normalize_columns(df)
    _validate(df)
    df = df.sort_values("date").reset_index(drop=True)
    print(f"[data_loader] Loaded {len(df)} draws from '{path}'.")
    return df


def generate_synthetic(
    n_draws: int = 2000,
    save_path: str = config.RAW_CSV,
) -> pd.DataFrame:
    """
    Generate *n_draws* random lottery draws and save to *save_path*.
    Format: date, n1, n2, n3, n4, n5, n6, n7, bonus
    """
    rng = random.Random(config.SEED)
    main_max = config.LOTTERY["main_max"]
    main_count = config.LOTTERY["main_count"]
    bonus_max = config.LOTTERY["bonus_max"]

    start = datetime(2000, 1, 1)
    records = []
    date = start
    for _ in range(n_draws):
        main = sorted(rng.sample(range(1, main_max + 1), main_count))
        bonus = rng.randint(1, bonus_max)
        records.append([date, *main, bonus])
        date += timedelta(days=3 + rng.randint(0, 1))  # ~2x per week

    cols = ["date"] + MAIN_COLS + ["bonus"]
    df = pd.DataFrame(records, columns=cols)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[data_loader] Saved {n_draws} synthetic draws to '{save_path}'.")
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Maps common alternative column names -> canonical names used internally
_COL_ALIASES = {
    **{f"number{i}": f"n{i}" for i in range(1, 8)},   # number1..number7 -> n1..n7
    **{f"num{i}":    f"n{i}" for i in range(1, 8)},    # num1..num7
    **{f"ball{i}":   f"n{i}" for i in range(1, 8)},    # ball1..ball7
    "supplementary": "bonus",
    "supp": "bonus",
    "powerball": "bonus",
    "megaball": "bonus",
}

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common column name variants to canonical n1..n7 / bonus."""
    rename = {col: _COL_ALIASES[col.lower()] for col in df.columns if col.lower() in _COL_ALIASES}
    if rename:
        df = df.rename(columns=rename)
        print(f"[data_loader] Renamed columns: {rename}")
    return df


def _validate(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Data file is missing columns: {sorted(missing)}\n"
            f"Expected: date, n1, n2, n3, n4, n5, n6, n7, bonus"
        )

    # Range checks
    main_max = config.LOTTERY["main_max"]
    bonus_max = config.LOTTERY["bonus_max"]
    for col in MAIN_COLS:
        bad = df[(df[col] < 1) | (df[col] > main_max)]
        if not bad.empty:
            raise ValueError(f"Column '{col}' has values outside 1-{main_max}: {bad[col].tolist()[:5]}")
    bad_bonus = df[(df["bonus"] < 1) | (df["bonus"] > bonus_max)]
    if not bad_bonus.empty:
        raise ValueError(
            f"Column 'bonus' has values outside 1-{bonus_max}. "
            f"Adjust LOTTERY['bonus_max'] in config.py if needed."
        )
