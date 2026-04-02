"""Unit tests for src/model.py -- stubs filled by q_a1."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from src.model import build_model, count_params, LotteryTransformer, LotteryLSTM


def test_transformer_forward_main_shape(dummy_main_batch):
    pass


def test_transformer_forward_no_bonus(dummy_main_batch):
    pass


def test_lstm_forward_main_shape(dummy_main_batch):
    pass


def test_build_model_transformer():
    pass


def test_build_model_lstm():
    pass


def test_build_model_invalid():
    pass


def test_count_params_positive():
    pass
