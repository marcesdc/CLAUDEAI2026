"""Unit tests for src/feedback.py -- stubs filled by q_a1."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from src.feedback import log_draw, recency_weights


def test_log_draw_appends_row(minimal_draws_csv, monkeypatch):
    pass


def test_log_draw_duplicate_skipped(minimal_draws_csv, monkeypatch):
    pass


def test_log_draw_bad_date_raises(minimal_draws_csv, monkeypatch):
    pass


def test_log_draw_bad_count_raises(minimal_draws_csv, monkeypatch):
    pass


def test_recency_weights_sum():
    pass


def test_recency_weights_length():
    pass


def test_recency_weights_monotone():
    pass
