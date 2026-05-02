"""
Tests for the --force / force=True overwrite path on `log` commands.

Covers both the single-lottery feedback.log_draw and the swarm cmd_log,
ensuring:
  - default (no force) still silently skips duplicate dates and preserves the existing row
  - force=True replaces the existing row in place with the corrected numbers
  - force on a missing date appends normally (no special behavior)
  - force does NOT bypass the 2026-04-23 missing-CSV guard
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Single-lottery: src.feedback.log_draw
# ---------------------------------------------------------------------------

def test_log_skip_without_force_preserves_old_row(monkeypatch, minimal_draws_csv):
    """Default behavior: duplicate date is skipped silently; old numbers stay."""
    import config
    from src.feedback import log_draw

    monkeypatch.setattr(config, "RAW_CSV", minimal_draws_csv)
    monkeypatch.setitem(config.LOTTERY, "main_count", 7)
    monkeypatch.setitem(config.LOTTERY, "main_max", 52)

    log_draw("2026-01-04", [10, 11, 12, 13, 14, 15, 16])  # date already exists

    df = pd.read_csv(minimal_draws_csv)
    row = df[df["date"] == "2026-01-04"].iloc[0]
    assert int(row["n1"]) == 2, "old numbers must be preserved when force=False"
    assert int(row["n7"]) == 49


def test_log_with_force_replaces_row_and_keeps_count(monkeypatch, minimal_draws_csv):
    import config
    from src.feedback import log_draw

    monkeypatch.setattr(config, "RAW_CSV", minimal_draws_csv)
    monkeypatch.setitem(config.LOTTERY, "main_count", 7)
    monkeypatch.setitem(config.LOTTERY, "main_max", 52)

    df_before = pd.read_csv(minimal_draws_csv)
    rows_before = len(df_before)

    log_draw("2026-01-04", [10, 11, 12, 13, 14, 15, 16], force=True)

    df_after = pd.read_csv(minimal_draws_csv)
    assert len(df_after) == rows_before, "force must not change row count"
    row = df_after[df_after["date"] == "2026-01-04"].iloc[0]
    assert sorted(int(row[f"n{i}"]) for i in range(1, 8)) == [10, 11, 12, 13, 14, 15, 16]


def test_log_with_force_on_missing_date_appends_normally(monkeypatch, minimal_draws_csv):
    import config
    from src.feedback import log_draw

    monkeypatch.setattr(config, "RAW_CSV", minimal_draws_csv)
    monkeypatch.setitem(config.LOTTERY, "main_count", 7)
    monkeypatch.setitem(config.LOTTERY, "main_max", 52)

    df_before = pd.read_csv(minimal_draws_csv)

    log_draw("2026-02-01", [4, 8, 15, 16, 23, 42, 45], force=True)

    df_after = pd.read_csv(minimal_draws_csv)
    assert len(df_after) == len(df_before) + 1
    assert "2026-02-01" in df_after["date"].astype(str).values


def test_log_force_does_not_bypass_missing_csv_guard(monkeypatch, tmp_path):
    """force must NOT create the file -- the 2026-04-23 guard wins."""
    import config
    from src.feedback import log_draw

    bogus = tmp_path / "definitely_does_not_exist.csv"
    monkeypatch.setattr(config, "RAW_CSV", str(bogus))
    monkeypatch.setitem(config.LOTTERY, "main_count", 7)
    monkeypatch.setitem(config.LOTTERY, "main_max", 52)

    with pytest.raises(FileNotFoundError) as excinfo:
        log_draw("2026-05-01", [1, 2, 3, 4, 5, 6, 7], force=True)

    assert "2026-04-23" in str(excinfo.value)
    assert not bogus.exists(), "force must NEVER create the CSV"


# ---------------------------------------------------------------------------
# Swarm: main_swarm.cmd_log
# ---------------------------------------------------------------------------

def _swarm_args(lottery, csv_path, date, numbers, force=False, bonus=None):
    """Mimic argparse.Namespace expected by cmd_log."""
    import argparse
    ns = argparse.Namespace()
    ns.lottery = lottery
    ns.date    = date
    ns.numbers = list(numbers)
    ns.bonus   = bonus
    ns.force   = force
    return ns


def test_swarm_log_skip_without_force(tmp_path, df_lottomax, monkeypatch):
    """Default swarm behavior: duplicate date is skipped, no bandit call."""
    csv = tmp_path / "draws.csv"
    df_lottomax.to_csv(csv, index=False)
    target_date = df_lottomax["date"].iloc[10]

    import main_swarm
    monkeypatch.setitem(main_swarm.LOTTERY_CONFIGS["lottomax"], "csv", str(csv))

    score_calls = []
    monkeypatch.setattr(main_swarm, "_score_swarm_prediction",
                        lambda *a, **kw: score_calls.append(a) or 0)
    monkeypatch.setattr(main_swarm, "_update_swarm_state", lambda **kw: None)

    args = _swarm_args("lottomax", str(csv), target_date, [1, 2, 3, 4, 5, 6, 7], force=False)
    main_swarm.cmd_log(args)

    df = pd.read_csv(csv)
    assert len(df) == len(df_lottomax), "row count must not change when skipping"
    assert score_calls == [], "no bandit update should fire on a skipped duplicate"


def test_swarm_log_with_force_replaces_row(tmp_path, df_lottomax, monkeypatch):
    csv = tmp_path / "draws.csv"
    df_lottomax.to_csv(csv, index=False)
    target_date = df_lottomax["date"].iloc[10]
    rows_before = len(df_lottomax)

    import main_swarm
    monkeypatch.setitem(main_swarm.LOTTERY_CONFIGS["lottomax"], "csv", str(csv))
    monkeypatch.setattr(main_swarm, "_score_swarm_prediction", lambda *a, **kw: 3)

    state_calls = []
    monkeypatch.setattr(main_swarm, "_update_swarm_state",
                        lambda **kw: state_calls.append(kw))

    args = _swarm_args("lottomax", str(csv), target_date, [10, 11, 12, 13, 14, 15, 16], force=True)
    main_swarm.cmd_log(args)

    df = pd.read_csv(csv)
    assert len(df) == rows_before, "force must keep row count constant"
    row = df[df["date"].astype(str) == str(target_date)].iloc[0]
    assert sorted(int(row[f"n{i}"]) for i in range(1, 8)) == [10, 11, 12, 13, 14, 15, 16]
    assert len(state_calls) == 1, "force-overwrite must trigger one bandit update"
    assert state_calls[0]["lottery"] == "lottomax"


def test_swarm_log_force_does_not_bypass_missing_csv_guard(tmp_path, monkeypatch):
    bogus = tmp_path / "definitely_missing.csv"

    import main_swarm
    monkeypatch.setitem(main_swarm.LOTTERY_CONFIGS["lottomax"], "csv", str(bogus))

    args = _swarm_args("lottomax", str(bogus), "2026-05-01",
                       [1, 2, 3, 4, 5, 6, 7], force=True)

    with pytest.raises(SystemExit):
        main_swarm.cmd_log(args)

    assert not bogus.exists(), "force must NEVER bootstrap the CSV"
