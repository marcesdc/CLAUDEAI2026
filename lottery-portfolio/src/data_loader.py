"""
data_loader.py -- load and validate draws/payouts CSVs.

Schemas:

draws_<lottery>.csv
    date, n1..n{main_count}, <bonus_col>

payouts_<lottery>.csv  (LONG FORMAT: one row per draw per prize tier)
    date, lottery, jackpot, tier_main, tier_bonus, n_winners, payout_per_winner
        tier_main        int, main-number match count
        tier_bonus       {"Y", "N", "-"} where "-" means bonus irrelevant for this tier
        jackpot          total jackpot pool for that draw (CAD, same value repeats across tier rows)
        n_winners        integer count of winners at this tier
        payout_per_winner CAD per winner ("" for pari-mutuel tiers where it depends on splits)

All loaders normalize "grand" -> "bonus" internally so downstream code can use a
single column name regardless of lottery.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

import config


# ---------------------------------------------------------------------------
# Draws
# ---------------------------------------------------------------------------

def load_draws(lottery: str) -> pd.DataFrame:
    """Load draws_<lottery>.csv, validate, return sorted-by-date DataFrame.

    Returns empty DataFrame (with the expected columns) if the file is missing
    or empty -- callers should check ``df.empty`` before using it.
    """
    if lottery not in config.LOTTERY_RULES:
        raise ValueError(f"Unknown lottery {lottery!r}. Expected one of {list(config.LOTTERY_RULES)}")

    cfg = config.LOTTERY_RULES[lottery]
    path = config.DRAWS_CSV[lottery]
    expected_cols = _draws_columns(cfg)

    if not Path(path).exists():
        return pd.DataFrame(columns=expected_cols)

    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=expected_cols)

    # Normalize "grand" -> "bonus" for Daily Grand
    if cfg["bonus_col"] != "bonus" and cfg["bonus_col"] in df.columns:
        df = df.rename(columns={cfg["bonus_col"]: "bonus"})

    _validate_draws(df, cfg, lottery)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _draws_columns(cfg: dict) -> list[str]:
    """Expected columns for a draws CSV after normalization.

    Bonus column only required for lotteries where players pick it (Daily Grand).
    Machine-drawn bonuses (LottoMax, 6/49) are not included.
    """
    cols = ["date"] + [f"n{i}" for i in range(1, cfg["main_count"] + 1)]
    if cfg.get("has_bonus"):  # Only include bonus if players pick it
        cols.append("bonus")
    return cols


def _validate_draws(df: pd.DataFrame, cfg: dict, lottery: str) -> None:
    """Raise ValueError if draws DataFrame has structural problems."""
    expected = _draws_columns(cfg)
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(
            f"{lottery} draws CSV missing columns {missing}; have {list(df.columns)}"
        )

    main_cols = [f"n{i}" for i in range(1, cfg["main_count"] + 1)]

    # Range checks
    for c in main_cols:
        bad = df[(df[c] < 1) | (df[c] > cfg["main_max"])]
        if not bad.empty:
            raise ValueError(
                f"{lottery}: column {c} has {len(bad)} values outside "
                f"[1, {cfg['main_max']}]"
            )
    if "bonus" in df.columns:
        bad_b = df[(df["bonus"] < 1) | (df["bonus"] > cfg["bonus_max"])]
        if not bad_b.empty:
            raise ValueError(
                f"{lottery}: bonus has {len(bad_b)} values outside [1, {cfg['bonus_max']}]"
            )

    # No duplicates within a single draw's main numbers
    dup_rows = df[main_cols].apply(lambda r: len(set(r)) != len(r), axis=1)
    if dup_rows.any():
        n = int(dup_rows.sum())
        raise ValueError(f"{lottery}: {n} row(s) have duplicate main numbers")


# ---------------------------------------------------------------------------
# Payouts
# ---------------------------------------------------------------------------

PAYOUT_COLUMNS = [
    "date", "lottery", "jackpot",
    "tier_main", "tier_bonus",
    "n_winners", "payout_per_winner",
]


def load_payouts(lottery: str) -> pd.DataFrame:
    """Load payouts_<lottery>.csv, validate, sort by date+tier."""
    if lottery not in config.LOTTERY_RULES:
        raise ValueError(f"Unknown lottery {lottery!r}")

    path = config.PAYOUTS_CSV[lottery]
    if not Path(path).exists():
        return pd.DataFrame(columns=PAYOUT_COLUMNS)

    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=PAYOUT_COLUMNS)

    missing = [c for c in PAYOUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{lottery} payouts CSV missing columns {missing}; have {list(df.columns)}"
        )

    # Coerce types
    df["n_winners"] = df["n_winners"].fillna(0).astype(int)
    df["tier_main"] = df["tier_main"].astype(int)
    df["tier_bonus"] = df["tier_bonus"].astype(str)

    # Only keep rows for this lottery (the "lottery" column is mostly a sanity check)
    mask = df["lottery"].astype(str) == lottery
    if not mask.all():
        df = df[mask].reset_index(drop=True)

    df = df.sort_values(["date", "tier_main", "tier_bonus"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Appenders (used by main.py log and agent/olg_scraper.py)
# ---------------------------------------------------------------------------

def append_draw(
    lottery: str,
    date: str,
    numbers: list[int],
    bonus: int,
) -> None:
    """Append one draw row to draws_<lottery>.csv. Creates file with header if missing."""
    cfg = config.LOTTERY_RULES[lottery]
    if len(numbers) != cfg["main_count"]:
        raise ValueError(
            f"{lottery} expects {cfg['main_count']} main numbers, got {len(numbers)}"
        )

    row = {"date": date}
    for i, n in enumerate(sorted(numbers), 1):
        row[f"n{i}"] = int(n)
    row["bonus"] = int(bonus)
    # If Daily Grand, write column as "grand" to match the legacy CSV convention
    if cfg["bonus_col"] != "bonus":
        row[cfg["bonus_col"]] = row.pop("bonus")

    path = Path(config.DRAWS_CSV[lottery])
    path.parent.mkdir(parents=True, exist_ok=True)

    df_new = pd.DataFrame([row])
    if path.exists() and path.stat().st_size > 0:
        df_new.to_csv(path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(path, index=False)


def append_payouts(
    lottery: str,
    date: str,
    jackpot: float,
    tier_rows: list[dict],
) -> None:
    """Append all prize-tier rows for one draw to payouts_<lottery>.csv.

    tier_rows: list of dicts with keys
        tier_main, tier_bonus, n_winners, payout_per_winner
    """
    path = Path(config.PAYOUTS_CSV[lottery])
    path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for t in tier_rows:
        records.append({
            "date":              date,
            "lottery":           lottery,
            "jackpot":           float(jackpot),
            "tier_main":         int(t["tier_main"]),
            "tier_bonus":        str(t.get("tier_bonus", "-")),
            "n_winners":         int(t.get("n_winners", 0)),
            "payout_per_winner": t.get("payout_per_winner", ""),
        })

    df_new = pd.DataFrame(records, columns=PAYOUT_COLUMNS)
    if path.exists() and path.stat().st_size > 0:
        df_new.to_csv(path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Convenience -- combined view
# ---------------------------------------------------------------------------

def load_combined(lottery: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (draws_df, payouts_df) aligned by date where possible."""
    draws    = load_draws(lottery)
    payouts  = load_payouts(lottery)
    return draws, payouts


def summary_row_counts() -> dict:
    """Return {lottery: (n_draws, n_payout_rows)} for status output."""
    out = {}
    for lottery in config.LOTTERY_RULES:
        d = load_draws(lottery)
        p = load_payouts(lottery)
        out[lottery] = (len(d), len(p))
    return out
