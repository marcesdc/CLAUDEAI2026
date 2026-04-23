"""Unit tests for src/feedback.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import pytest
import config
import src.feedback as feedback_mod
from src.feedback import log_draw, recency_weights, score_last_prediction, save_prediction


def test_log_draw_appends_row(minimal_draws_csv, monkeypatch):
    monkeypatch.setattr(config, "RAW_CSV", minimal_draws_csv)
    log_draw("2026-02-01", [1, 5, 10, 15, 20, 25, 30])
    df = pd.read_csv(minimal_draws_csv)
    assert len(df) == 4
    assert "2026-02-01" in df["date"].values


def test_log_draw_duplicate_skipped(minimal_draws_csv, monkeypatch):
    monkeypatch.setattr(config, "RAW_CSV", minimal_draws_csv)
    log_draw("2026-01-01", [2, 6, 11, 21, 31, 41, 49])  # date exists
    df = pd.read_csv(minimal_draws_csv)
    assert len(df) == 3  # unchanged


def test_log_draw_bad_date_raises(minimal_draws_csv, monkeypatch):
    monkeypatch.setattr(config, "RAW_CSV", minimal_draws_csv)
    with pytest.raises(ValueError, match="not valid"):
        log_draw("2026-13-01", [1, 5, 10, 15, 20, 25, 30])


def test_log_draw_bad_count_raises(minimal_draws_csv, monkeypatch):
    monkeypatch.setattr(config, "RAW_CSV", minimal_draws_csv)
    with pytest.raises(ValueError, match="Expected"):
        log_draw("2026-02-01", [1, 5, 10])


def test_log_draw_raises_when_csv_missing(tmp_path, monkeypatch):
    """log_draw() must NOT bootstrap a 1-row file over a missing real-data path.

    Regression guard for incident 2026-04-23 (W1): the prior code created
    a fresh DataFrame with just the new row when the CSV was missing,
    silently replacing the entire history with one entry.
    """
    missing = str(tmp_path / "does_not_exist.csv")
    monkeypatch.setattr(config, "RAW_CSV", missing)
    with pytest.raises(FileNotFoundError, match="Refusing to bootstrap"):
        log_draw("2026-02-01", [1, 5, 10, 15, 20, 25, 30])
    # File must NOT have been created
    assert not Path(missing).exists()


def test_recency_weights_sum():
    w = recency_weights(20)
    assert abs(w.sum() - 1.0) < 1e-5


def test_recency_weights_length():
    w = recency_weights(30)
    assert len(w) == 30


def test_recency_weights_monotone():
    w = recency_weights(10)
    assert all(w[i] < w[i + 1] for i in range(len(w) - 1))


def test_score_last_prediction_no_match_returns_empty(tmp_path, monkeypatch):
    """Codex H1 guard: if draw_date has no saved prediction, return empty --
    do NOT fall back to an older prediction and stamp it with draw_date.
    """
    pred_log = tmp_path / "predictions_log.csv"
    score_log = tmp_path / "score_log.csv"
    monkeypatch.setattr(feedback_mod, "PRED_LOG", str(pred_log))
    monkeypatch.setattr(feedback_mod, "SCORE_LOG", str(score_log))

    # Saved prediction for 2026-04-01 only
    save_prediction(
        plays=[{"lines": [[1, 2, 3, 4, 5, 6, 7]]}],
        draw_date="2026-04-01",
    )

    # Caller asks to score for 2026-04-22 (no matching pred rows)
    result = score_last_prediction(
        actual_numbers=[1, 2, 3, 4, 5, 6, 7],
        draw_date="2026-04-22",
    )

    assert result.empty, "Must return empty DataFrame, not score old prediction"
    # Score log must NOT be written -- no scoring happened
    assert not score_log.exists(), "score_log.csv must not be created when no match"


def test_score_last_prediction_exact_match_scores(tmp_path, monkeypatch):
    """Positive case: when draw_date matches a saved pred_date, scoring proceeds."""
    pred_log = tmp_path / "predictions_log.csv"
    score_log = tmp_path / "score_log.csv"
    monkeypatch.setattr(feedback_mod, "PRED_LOG", str(pred_log))
    monkeypatch.setattr(feedback_mod, "SCORE_LOG", str(score_log))

    save_prediction(
        plays=[{"lines": [[1, 2, 3, 4, 5, 6, 7]]}],
        draw_date="2026-04-22",
    )
    result = score_last_prediction(
        actual_numbers=[1, 2, 3, 10, 20, 30, 40],
        draw_date="2026-04-22",
    )
    assert not result.empty
    assert int(result["hits"].iloc[0]) == 3
    assert result["draw_date"].iloc[0] == "2026-04-22"
