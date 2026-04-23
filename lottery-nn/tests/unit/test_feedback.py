"""Unit tests for src/feedback.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import pytest
import config
from src.feedback import log_draw, recency_weights


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
