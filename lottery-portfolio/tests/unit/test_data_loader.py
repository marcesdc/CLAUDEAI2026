"""Unit tests for src/data_loader.py."""

from pathlib import Path

import pandas as pd
import pytest

from src import data_loader


# ---------------------------------------------------------------------------
# load_draws
# ---------------------------------------------------------------------------

def test_load_draws_missing_file_returns_empty(monkeypatch, tmp_path):
    import config
    fake_paths = {k: tmp_path / f"missing_{k}.csv" for k in config.LOTTERY_RULES}
    monkeypatch.setattr(config, "DRAWS_CSV", fake_paths)
    for lottery in config.LOTTERY_RULES:
        df = data_loader.load_draws(lottery)
        assert df.empty


def test_load_draws_validates_and_sorts(patched_config):
    df = data_loader.load_draws("lottomax")
    assert not df.empty
    assert list(df["date"]) == sorted(df["date"])


def test_load_draws_renames_grand_to_bonus(patched_config):
    df = data_loader.load_draws("dailygrand")
    assert "bonus" in df.columns
    assert "grand" not in df.columns


def test_load_draws_rejects_unknown_lottery():
    with pytest.raises(ValueError, match="Unknown lottery"):
        data_loader.load_draws("not_a_lottery")


def test_load_draws_rejects_out_of_range(tmp_path, monkeypatch):
    import config
    path = tmp_path / "draws_649.csv"
    bad = pd.DataFrame([{
        "date": "2025-01-01",
        "n1": 1, "n2": 2, "n3": 3, "n4": 4, "n5": 5, "n6": 99,  # 99 > 49
        "bonus": 10,
    }])
    bad.to_csv(path, index=False)
    monkeypatch.setattr(config, "DRAWS_CSV", {"649": path, **{k: v for k, v in config.DRAWS_CSV.items() if k != "649"}})
    with pytest.raises(ValueError, match="outside"):
        data_loader.load_draws("649")


def test_load_draws_rejects_duplicate_main_numbers(tmp_path, monkeypatch):
    import config
    path = tmp_path / "draws_649.csv"
    bad = pd.DataFrame([{
        "date": "2025-01-01",
        "n1": 5, "n2": 5, "n3": 3, "n4": 4, "n5": 6, "n6": 7,  # n1==n2
        "bonus": 10,
    }])
    bad.to_csv(path, index=False)
    monkeypatch.setattr(config, "DRAWS_CSV", {"649": path, **{k: v for k, v in config.DRAWS_CSV.items() if k != "649"}})
    with pytest.raises(ValueError, match="duplicate main numbers"):
        data_loader.load_draws("649")


# ---------------------------------------------------------------------------
# load_payouts
# ---------------------------------------------------------------------------

def test_load_payouts_missing_file_returns_empty(monkeypatch, tmp_path):
    import config
    fake_paths = {k: tmp_path / f"missing_{k}.csv" for k in config.LOTTERY_RULES}
    monkeypatch.setattr(config, "PAYOUTS_CSV", fake_paths)
    for lottery in config.LOTTERY_RULES:
        df = data_loader.load_payouts(lottery)
        assert df.empty
        assert list(df.columns) == data_loader.PAYOUT_COLUMNS


def test_load_payouts_returns_sorted(patched_config):
    df = data_loader.load_payouts("lottomax")
    assert not df.empty
    # Sorted by date first, then tier
    assert list(df["date"]) == sorted(df["date"]) or df["date"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# append_draw
# ---------------------------------------------------------------------------

def test_append_draw_creates_file(tmp_path, monkeypatch):
    import config
    path = tmp_path / "draws_649.csv"
    monkeypatch.setattr(config, "DRAWS_CSV", {
        **config.DRAWS_CSV, "649": path,
    })
    data_loader.append_draw("649", "2026-04-17", [3, 7, 18, 24, 31, 42], 15)
    df = pd.read_csv(path)
    assert len(df) == 1
    assert df.iloc[0]["n1"] == 3  # Sorted ascending
    assert df.iloc[0]["bonus"] == 15


def test_append_draw_sorts_numbers(tmp_path, monkeypatch):
    import config
    path = tmp_path / "draws_649.csv"
    monkeypatch.setattr(config, "DRAWS_CSV", {**config.DRAWS_CSV, "649": path})
    data_loader.append_draw("649", "2026-04-17", [42, 3, 31, 7, 24, 18], 15)
    df = pd.read_csv(path)
    nums = [df.iloc[0][f"n{i}"] for i in range(1, 7)]
    assert nums == sorted(nums)


def test_append_draw_dailygrand_writes_grand_column(tmp_path, monkeypatch):
    import config
    path = tmp_path / "draws_dailygrand.csv"
    monkeypatch.setattr(config, "DRAWS_CSV", {**config.DRAWS_CSV, "dailygrand": path})
    data_loader.append_draw("dailygrand", "2026-04-17", [8, 17, 28, 37, 46], 4)
    df = pd.read_csv(path)
    assert "grand" in df.columns
    assert "bonus" not in df.columns
    assert df.iloc[0]["grand"] == 4


def test_append_draw_rejects_wrong_count(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "DRAWS_CSV", {**config.DRAWS_CSV, "649": tmp_path / "d.csv"})
    with pytest.raises(ValueError, match="expects 6"):
        data_loader.append_draw("649", "2026-04-17", [1, 2, 3], 4)


def test_append_draw_appends_to_existing(tmp_path, monkeypatch):
    import config
    path = tmp_path / "draws_649.csv"
    monkeypatch.setattr(config, "DRAWS_CSV", {**config.DRAWS_CSV, "649": path})
    data_loader.append_draw("649", "2026-04-17", [1, 2, 3, 4, 5, 6], 10)
    data_loader.append_draw("649", "2026-04-20", [7, 8, 9, 10, 11, 12], 20)
    df = pd.read_csv(path)
    assert len(df) == 2


# ---------------------------------------------------------------------------
# append_payouts
# ---------------------------------------------------------------------------

def test_append_payouts_creates_rows(tmp_path, monkeypatch):
    import config
    path = tmp_path / "payouts_649.csv"
    monkeypatch.setattr(config, "PAYOUTS_CSV", {**config.PAYOUTS_CSV, "649": path})
    tiers = [
        {"tier_main": 6, "tier_bonus": "-", "n_winners": 0, "payout_per_winner": ""},
        {"tier_main": 5, "tier_bonus": "Y", "n_winners": 2, "payout_per_winner": ""},
        {"tier_main": 4, "tier_bonus": "-", "n_winners": 150, "payout_per_winner": 85.0},
    ]
    data_loader.append_payouts("649", "2026-04-17", 5_000_000.0, tiers)
    df = pd.read_csv(path)
    assert len(df) == 3
    assert df.iloc[0]["jackpot"] == 5_000_000.0
    assert df.iloc[0]["tier_main"] == 6
    assert df.iloc[2]["n_winners"] == 150


# ---------------------------------------------------------------------------
# summary_row_counts
# ---------------------------------------------------------------------------

def test_summary_row_counts_shape(patched_config):
    counts = data_loader.summary_row_counts()
    assert set(counts.keys()) == {"lottomax", "649", "dailygrand"}
    for lottery, (n_draws, n_pay) in counts.items():
        assert n_draws > 0
        assert n_pay > 0
