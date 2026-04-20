"""Unit tests for agent/olg_scraper.py -- parser only (no live network)."""

import pytest

from agent import olg_scraper


SAMPLE_LOTTOMAX = """
DATE: 2026-04-15
NUMBERS: 3 12 18 24 31 40 47
BONUS: 9
JACKPOT: 55000000
PAYOUTS:
TIER_MAIN | TIER_BONUS | N_WINNERS | PAYOUT_PER_WINNER
7 | - | 0 |
6 | Y | 1 |
6 | N | 22 |
5 | - | 350 |
4 | - | 14500 | 20.00
3 | Y | 29000 | 20.00
""".strip()


def test_parse_full_lottomax_block():
    parsed = olg_scraper._parse_result(SAMPLE_LOTTOMAX, "lottomax")
    assert parsed is not None
    assert parsed["date"] == "2026-04-15"
    assert parsed["numbers"] == [3, 12, 18, 24, 31, 40, 47]
    assert parsed["bonus"] == 9
    assert parsed["jackpot"] == 55_000_000.0
    assert len(parsed["tiers"]) == 6
    assert parsed["tiers"][0] == {
        "tier_main": 7, "tier_bonus": "-", "n_winners": 0, "payout_per_winner": "",
    }
    assert parsed["tiers"][4]["payout_per_winner"] == 20.0


def test_parse_handles_failed_result():
    assert olg_scraper._parse_result("FAILED: page timed out", "lottomax") is None


def test_parse_rejects_wrong_main_count():
    bad = """
DATE: 2026-04-15
NUMBERS: 1 2 3 4
BONUS: 5
JACKPOT: 1000
PAYOUTS:
""".strip()
    assert olg_scraper._parse_result(bad, "lottomax") is None


def test_parse_rejects_missing_fields():
    bad = "DATE: 2026-04-15\nNUMBERS: 1 2 3 4 5 6 7\nBONUS: 8\n"  # no JACKPOT
    assert olg_scraper._parse_result(bad, "lottomax") is None


def test_parse_dailygrand_uses_correct_main_count():
    dg = """
DATE: 2026-04-14
NUMBERS: 8 17 28 37 46
BONUS: 4
JACKPOT: 7000000
PAYOUTS:
TIER_MAIN | TIER_BONUS | N_WINNERS | PAYOUT_PER_WINNER
5 | Y | 0 |
5 | N | 3 | 100000.00
""".strip()
    parsed = olg_scraper._parse_result(dg, "dailygrand")
    assert parsed is not None
    assert parsed["numbers"] == [8, 17, 28, 37, 46]
    assert parsed["bonus"] == 4
    assert parsed["tiers"][1]["payout_per_winner"] == 100_000.0


def test_parse_tolerates_commas_in_jackpot():
    text = SAMPLE_LOTTOMAX.replace("JACKPOT: 55000000", "JACKPOT: 55,000,000")
    parsed = olg_scraper._parse_result(text, "lottomax")
    assert parsed is not None
    assert parsed["jackpot"] == 55_000_000.0


def test_save_skips_duplicate_date(tmp_path, monkeypatch):
    import config
    from src import data_loader

    draws_path   = tmp_path / "draws_649.csv"
    payouts_path = tmp_path / "payouts_649.csv"
    monkeypatch.setattr(config, "DRAWS_CSV",   {**config.DRAWS_CSV,   "649": draws_path})
    monkeypatch.setattr(config, "PAYOUTS_CSV", {**config.PAYOUTS_CSV, "649": payouts_path})

    # Pre-seed one draw
    data_loader.append_draw("649", "2026-04-17", [1, 2, 3, 4, 5, 6], 10)

    # Try to save another with same date
    duplicate = {
        "date": "2026-04-17",
        "numbers": [7, 8, 9, 10, 11, 12],
        "bonus": 20,
        "jackpot": 1_000_000.0,
        "tiers": [],
    }
    saved = olg_scraper.save("649", duplicate)
    assert not saved
