"""Unit tests for defensive guards in main_swarm.py."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import pytest
import main_swarm
from src.preprocessing_swarm import LOTTERY_CONFIGS


# ---------------------------------------------------------------------------
# Helpers and fixtures for cmd_log tests
# ---------------------------------------------------------------------------

def _args(lottery, numbers, bonus=None, date="2099-01-01"):
    """Build an argparse.Namespace matching what `log` subcommand parses."""
    return argparse.Namespace(lottery=lottery, numbers=list(numbers), bonus=bonus, date=date)


@pytest.fixture
def redirect_log_paths(tmp_path, monkeypatch):
    """Redirect all CSVs + swarm state/pred-log to tmp_path so cmd_log cannot touch real data.

    Note: patches the shared LOTTERY_CONFIGS module-level dict, so tests using this
    fixture are NOT safe under pytest-xdist (parallel workers share the same process
    memory via monkeypatch but run tests concurrently). Run with plain `pytest` only.
    """
    csv_map = {
        "lottomax":   str(tmp_path / "draws.csv"),
        "649":        str(tmp_path / "draws_649.csv"),
        "dailygrand": str(tmp_path / "draws_dailygrand.csv"),
    }
    for name, path in csv_map.items():
        monkeypatch.setitem(LOTTERY_CONFIGS[name], "csv", path)
    monkeypatch.setattr(main_swarm, "SWARM_STATE_FILE", str(tmp_path / "swarm_state.json"))
    monkeypatch.setattr(main_swarm, "SWARM_PRED_LOG",   str(tmp_path / "swarm_predictions_log.csv"))
    return csv_map


# ---------------------------------------------------------------------------
# cmd_log -- rejection paths (SystemExit with clear message)
# ---------------------------------------------------------------------------

def test_cmd_log_rejects_duplicate_numbers_lottomax(redirect_log_paths, capsys):
    with pytest.raises(SystemExit) as exc:
        main_swarm.cmd_log(_args("lottomax", [7, 7, 12, 29, 38, 39, 44]))
    assert exc.value.code == 1
    assert "unique" in capsys.readouterr().out.lower()


def test_cmd_log_rejects_duplicate_numbers_649(redirect_log_paths, capsys):
    with pytest.raises(SystemExit) as exc:
        main_swarm.cmd_log(_args("649", [1, 1, 2, 3, 4, 5]))
    assert exc.value.code == 1
    assert "unique" in capsys.readouterr().out.lower()


def test_cmd_log_rejects_bonus_on_lottomax(redirect_log_paths, capsys):
    with pytest.raises(SystemExit) as exc:
        main_swarm.cmd_log(_args("lottomax", [1, 2, 3, 4, 5, 6, 7], bonus=10))
    assert exc.value.code == 1
    msg = capsys.readouterr().out.lower()
    assert "no player-picked bonus" in msg or "drop --bonus" in msg


def test_cmd_log_rejects_bonus_on_649(redirect_log_paths, capsys):
    with pytest.raises(SystemExit) as exc:
        main_swarm.cmd_log(_args("649", [1, 2, 3, 4, 5, 6], bonus=10))
    assert exc.value.code == 1
    msg = capsys.readouterr().out.lower()
    assert "no player-picked bonus" in msg or "drop --bonus" in msg


def test_cmd_log_rejects_bonus_zero_on_no_bonus_lottery(redirect_log_paths):
    """--bonus 0 is not None, so the no-bonus guard must still reject it."""
    with pytest.raises(SystemExit) as exc:
        main_swarm.cmd_log(_args("lottomax", [1, 2, 3, 4, 5, 6, 7], bonus=0))
    assert exc.value.code == 1


def test_cmd_log_rejects_out_of_range_number(redirect_log_paths, capsys):
    """LottoMax main_max is 52; 53 must be rejected."""
    with pytest.raises(SystemExit) as exc:
        main_swarm.cmd_log(_args("lottomax", [1, 2, 3, 4, 5, 6, 53]))
    assert exc.value.code == 1
    assert "outside" in capsys.readouterr().out.lower()


def test_cmd_log_rejects_wrong_count(redirect_log_paths, capsys):
    """LottoMax requires exactly 7 numbers."""
    with pytest.raises(SystemExit) as exc:
        main_swarm.cmd_log(_args("lottomax", [1, 2, 3, 4, 5, 6]))
    assert exc.value.code == 1
    assert "expected 7" in capsys.readouterr().out.lower()


def test_cmd_log_rejects_bonus_missing_for_dailygrand(redirect_log_paths, capsys):
    """Daily Grand requires --bonus."""
    with pytest.raises(SystemExit) as exc:
        main_swarm.cmd_log(_args("dailygrand", [1, 2, 3, 4, 5], bonus=None))
    assert exc.value.code == 1
    assert "--bonus is required" in capsys.readouterr().out.lower()


def test_cmd_log_rejects_bonus_zero_for_dailygrand(redirect_log_paths, capsys):
    """Daily Grand bonus must be 1..7; 0 is out of range."""
    with pytest.raises(SystemExit) as exc:
        main_swarm.cmd_log(_args("dailygrand", [1, 2, 3, 4, 5], bonus=0))
    assert exc.value.code == 1
    assert "1-7" in capsys.readouterr().out


def test_cmd_log_rejects_bonus_above_max_for_dailygrand(redirect_log_paths, capsys):
    """Daily Grand bonus 8 is out of range."""
    with pytest.raises(SystemExit) as exc:
        main_swarm.cmd_log(_args("dailygrand", [1, 2, 3, 4, 5], bonus=8))
    assert exc.value.code == 1
    assert "1-7" in capsys.readouterr().out


def test_cmd_log_rejects_invalid_date(redirect_log_paths, capsys):
    """Non-YYYY-MM-DD date is rejected."""
    with pytest.raises(SystemExit) as exc:
        main_swarm.cmd_log(_args("lottomax", [1, 2, 3, 4, 5, 6, 7], date="04/20/2026"))
    assert exc.value.code == 1
    assert "invalid date" in capsys.readouterr().out.lower()


def test_cmd_log_rejects_iso_extended_date(redirect_log_paths, capsys):
    """ISO-extended format (with time component) is rejected by strict strptime."""
    with pytest.raises(SystemExit) as exc:
        main_swarm.cmd_log(_args("lottomax", [1, 2, 3, 4, 5, 6, 7], date="2026-04-20T00:00:00"))
    assert exc.value.code == 1
    assert "invalid date" in capsys.readouterr().out.lower()


def test_cmd_log_rejects_number_zero(redirect_log_paths, capsys):
    """0 is below the 1..main_max range and must be rejected."""
    with pytest.raises(SystemExit) as exc:
        main_swarm.cmd_log(_args("lottomax", [0, 2, 3, 4, 5, 6, 7]))
    assert exc.value.code == 1
    assert "outside" in capsys.readouterr().out.lower()


# ---------------------------------------------------------------------------
# cmd_log -- happy paths (CSV row written correctly)
# ---------------------------------------------------------------------------

# Header schemas matching LOTTERY_CONFIGS (cmd_log requires CSV to exist with header)
_HEADERS = {
    "lottomax":   "date,n1,n2,n3,n4,n5,n6,n7\n",
    "649":        "date,n1,n2,n3,n4,n5,n6\n",
    "dailygrand": "date,n1,n2,n3,n4,n5,grand\n",
}


def _seed_header(csv_path: str, lottery: str) -> None:
    """Write a header-only CSV at csv_path. cmd_log appends to it."""
    Path(csv_path).write_text(_HEADERS[lottery])


def test_cmd_log_accepts_valid_lottomax_draw(redirect_log_paths):
    _seed_header(redirect_log_paths["lottomax"], "lottomax")
    main_swarm.cmd_log(_args("lottomax", [12, 7, 29, 52, 44, 39, 38], date="2099-01-01"))
    df = pd.read_csv(redirect_log_paths["lottomax"])
    assert len(df) == 1
    row = df.iloc[0]
    assert row["date"] == "2099-01-01"
    # Numbers are stored sorted regardless of input order
    assert [row[f"n{i}"] for i in range(1, 8)] == [7, 12, 29, 38, 39, 44, 52]
    assert "bonus" not in df.columns and "grand" not in df.columns


def test_cmd_log_accepts_valid_649_draw(redirect_log_paths):
    _seed_header(redirect_log_paths["649"], "649")
    main_swarm.cmd_log(_args("649", [45, 37, 35, 32, 30, 27], date="2099-01-02"))
    df = pd.read_csv(redirect_log_paths["649"])
    assert len(df) == 1
    row = df.iloc[0]
    assert [row[f"n{i}"] for i in range(1, 7)] == [27, 30, 32, 35, 37, 45]
    assert "bonus" not in df.columns


def test_cmd_log_accepts_valid_dailygrand_draw(redirect_log_paths):
    _seed_header(redirect_log_paths["dailygrand"], "dailygrand")
    main_swarm.cmd_log(_args("dailygrand", [47, 37, 34, 21, 18], bonus=6, date="2099-01-03"))
    df = pd.read_csv(redirect_log_paths["dailygrand"])
    assert len(df) == 1
    row = df.iloc[0]
    assert [row[f"n{i}"] for i in range(1, 6)] == [18, 21, 34, 37, 47]
    # Daily Grand bonus column is 'grand' in the CSV (not 'bonus')
    assert "grand" in df.columns
    assert row["grand"] == 6


def test_cmd_log_skips_duplicate_date(redirect_log_paths, capsys):
    """Logging the same date twice prints a skip message and does not grow the CSV."""
    _seed_header(redirect_log_paths["lottomax"], "lottomax")
    main_swarm.cmd_log(_args("lottomax", [1, 2, 3, 4, 5, 6, 7], date="2099-01-01"))
    main_swarm.cmd_log(_args("lottomax", [8, 9, 10, 11, 12, 13, 14], date="2099-01-01"))
    df = pd.read_csv(redirect_log_paths["lottomax"])
    assert len(df) == 1
    # First row (1..7) must be preserved; second call was a no-op
    assert [df.iloc[0][f"n{i}"] for i in range(1, 8)] == [1, 2, 3, 4, 5, 6, 7]
    assert "already exists" in capsys.readouterr().out.lower()


def test_cmd_log_rejects_missing_csv(redirect_log_paths, capsys):
    """cmd_log() must NOT bootstrap a 1-row file over a missing real-data path.

    Regression guard for incident 2026-04-23: parallel to the W1 fix in
    feedback.py, the swarm cmd_log path must also refuse to silently create
    a fresh file masquerading as full history.
    """
    csv_path = redirect_log_paths["649"]
    assert not Path(csv_path).exists(), "fixture should leave path empty"
    with pytest.raises(SystemExit) as exc:
        main_swarm.cmd_log(_args("649", [1, 2, 3, 4, 5, 6], date="2099-01-04"))
    assert exc.value.code == 1
    msg = capsys.readouterr().out.lower()
    assert "not found" in msg and "refusing to bootstrap" in msg
    # File must NOT have been created
    assert not Path(csv_path).exists()


# ---------------------------------------------------------------------------
# _prune_and_finetune guards (pre-existing)
# ---------------------------------------------------------------------------

def test_prune_and_finetune_missing_checkpoint_returns_pretrain_best_val(tmp_path, monkeypatch):
    """_prune_and_finetune returns pretrain_best_val immediately if checkpoint is absent."""
    monkeypatch.setattr(main_swarm, "SWARM_CHECKPOINT", str(tmp_path / "nonexistent.pt"))
    result = main_swarm._prune_and_finetune(
        model=None,
        init_state=None,
        train_loaders=None,
        val_loaders=None,
        args=None,
        history={},
        pretrain_best_val=0.42,
    )
    assert result == pytest.approx(0.42)


def test_prune_and_finetune_missing_checkpoint_default_pretrain_val(tmp_path, monkeypatch):
    """When pretrain_best_val is omitted, default float('inf') is returned."""
    monkeypatch.setattr(main_swarm, "SWARM_CHECKPOINT", str(tmp_path / "nonexistent.pt"))
    result = main_swarm._prune_and_finetune(
        model=None,
        init_state=None,
        train_loaders=None,
        val_loaders=None,
        args=None,
        history={},
    )
    assert result == float("inf")
