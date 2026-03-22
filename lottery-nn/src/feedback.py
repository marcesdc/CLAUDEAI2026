"""
Feedback loop: log actual draw results, score predictions, and
compute recency-based sample weights for retraining.

Files managed here
------------------
data/draws.csv           – ground-truth draw history (appended on each log)
data/predictions_log.csv – predictions saved before each draw
data/score_log.csv       – hit scores for every logged draw
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.data_loader import MAIN_COLS


PRED_LOG   = "data/predictions_log.csv"
SCORE_LOG  = "data/score_log.csv"


# ---------------------------------------------------------------------------
# Logging a new draw
# ---------------------------------------------------------------------------

def log_draw(date: str, numbers: list[int], bonus: int) -> None:
    """
    Append a new draw to draws.csv.

    Parameters
    ----------
    date    : date string, e.g. '2026-03-25'
    numbers : list of 7 ints (main balls)
    bonus   : int
    """
    if len(numbers) != config.LOTTERY["main_count"]:
        raise ValueError(f"Expected {config.LOTTERY['main_count']} numbers, got {len(numbers)}.")

    # Basic date format check
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Date '{date}' is not valid. Use YYYY-MM-DD format, e.g. 2026-03-25.")

    main_max = config.LOTTERY["main_max"]
    for n in numbers:
        if not (1 <= n <= main_max):
            raise ValueError(f"Number {n} is outside 1-{main_max}.")
    if not (1 <= bonus <= config.LOTTERY["bonus_max"]):
        raise ValueError(f"Bonus {bonus} is outside 1-{config.LOTTERY['bonus_max']}.")

    row = {"date": date, **{f"n{i+1}": v for i, v in enumerate(sorted(numbers))}, "bonus": bonus}
    csv_path = config.RAW_CSV

    if Path(csv_path).exists():
        from src.data_loader import _normalize_columns
        df = _normalize_columns(pd.read_csv(csv_path))
        # Avoid duplicate dates
        if date in df["date"].astype(str).values:
            print(f"[feedback] Draw for {date} already exists - skipping append.")
            return
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(csv_path, index=False)
    print(f"[feedback] Draw logged: {date}  {sorted(numbers)}  bonus={bonus}")


# ---------------------------------------------------------------------------
# Saving / scoring predictions
# ---------------------------------------------------------------------------

def save_prediction(plays: list[dict], draw_date: str = "") -> None:
    """
    Persist the generated plays to predictions_log.csv so they can be
    scored once the actual result is known.
    """
    if not draw_date:
        draw_date = datetime.today().strftime("%Y-%m-%d")

    records = []
    for play_idx, play in enumerate(plays, 1):
        for line_idx, line in enumerate(play["lines"], 1):
            records.append({
                "pred_date": draw_date,
                "play": play_idx,
                "line": line_idx,
                "numbers": " ".join(str(n) for n in line),
                "bonus": play["bonus"],
            })

    df_new = pd.DataFrame(records)
    if Path(PRED_LOG).exists():
        df_new = pd.concat([pd.read_csv(PRED_LOG), df_new], ignore_index=True)
    df_new.to_csv(PRED_LOG, index=False)
    print(f"[feedback] Predictions saved to '{PRED_LOG}'.")


def score_last_prediction(actual_numbers: list[int], actual_bonus: int, draw_date: str = "") -> pd.DataFrame:
    """
    Score the most recently saved prediction against the actual draw.
    Prints a summary and appends results to score_log.csv.

    Returns a DataFrame with per-line hit counts.
    """
    if not Path(PRED_LOG).exists():
        print("[feedback] No prediction log found. Run 'predict' before logging a draw.")
        return pd.DataFrame()

    df_pred = pd.read_csv(PRED_LOG)
    if not draw_date:
        draw_date = df_pred["pred_date"].iloc[-1]

    latest = df_pred[df_pred["pred_date"] == draw_date].copy()
    if latest.empty:
        latest = df_pred[df_pred["pred_date"] == df_pred["pred_date"].max()].copy()

    actual_set = set(actual_numbers)
    rows = []
    for _, row in latest.iterrows():
        predicted = set(int(n) for n in str(row["numbers"]).split())
        hits = len(predicted & actual_set)
        bonus_hit = int(row["bonus"]) == actual_bonus
        rows.append({
            "draw_date":   draw_date,
            "play":        row["play"],
            "line":        row["line"],
            "predicted":   row["numbers"],
            "actual":      " ".join(str(n) for n in sorted(actual_numbers)),
            "hits":        hits,
            "bonus_hit":   bonus_hit,
        })

    df_scores = pd.DataFrame(rows)
    _print_score_summary(df_scores, actual_numbers, actual_bonus)

    # Append to score log
    if Path(SCORE_LOG).exists():
        df_scores = pd.concat([pd.read_csv(SCORE_LOG), df_scores], ignore_index=True)
    df_scores.to_csv(SCORE_LOG, index=False)
    return df_scores


# ---------------------------------------------------------------------------
# Recency weighting for training
# ---------------------------------------------------------------------------

def recency_weights(n_samples: int, decay: float = 0.92) -> np.ndarray:
    """
    Exponential recency weights: the most recent sample gets weight 1.0,
    older samples decay by *decay* per step.

    decay=0.92 means a draw 10 steps ago has weight ~0.43.
    Adjust upward (0.98) for slower decay with larger datasets.
    """
    weights = np.array([decay ** (n_samples - 1 - i) for i in range(n_samples)], dtype=np.float32)
    weights /= weights.sum()
    return weights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_score_summary(df: pd.DataFrame, actual: list[int], bonus: int) -> None:
    actual_str = "  ".join(f"{n:2d}" for n in sorted(actual))
    print(f"\n  Actual draw:  [ {actual_str} ]  bonus={bonus}")
    print(f"  {'Play':<6} {'Line':<6} {'Predicted':<30} {'Hits':<6} {'Bonus'}")
    print(f"  {'-'*60}")
    for _, r in df.iterrows():
        bonus_str = "YES" if r["bonus_hit"] else "-"
        print(f"  {int(r['play']):<6} {int(r['line']):<6} {r['predicted']:<30} {int(r['hits']):<6} {bonus_str}")
    best = df["hits"].max()
    print(f"\n  Best line: {best} match(es) out of {config.LOTTERY['main_count']}")
