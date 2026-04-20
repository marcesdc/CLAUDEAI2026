"""Unit tests for src/model.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import pytest
import config
from src.model import build_model, count_params, LotteryTransformer, LotteryLSTM


def test_transformer_forward_main_shape(dummy_main_batch):
    model = LotteryTransformer(has_bonus=False)
    model.eval()
    with torch.no_grad():
        main_logits, bonus_logits = model(dummy_main_batch)
    assert main_logits.shape == (4, config.LOTTERY["main_max"])


def test_transformer_forward_no_bonus(dummy_main_batch):
    model = LotteryTransformer(has_bonus=False)
    model.eval()
    with torch.no_grad():
        _, bonus_logits = model(dummy_main_batch)
    assert bonus_logits is None


def test_lstm_forward_main_shape(dummy_main_batch):
    model = LotteryLSTM(has_bonus=False)
    model.eval()
    with torch.no_grad():
        main_logits, _ = model(dummy_main_batch)
    assert main_logits.shape == (4, config.LOTTERY["main_max"])


def test_build_model_transformer():
    model = build_model(arch="transformer")
    assert isinstance(model, LotteryTransformer)


def test_build_model_lstm():
    model = build_model(arch="lstm")
    assert isinstance(model, LotteryLSTM)


def test_build_model_invalid():
    with pytest.raises(ValueError, match="Unknown arch"):
        build_model(arch="bad_arch")


def test_count_params_positive():
    model = build_model(arch="transformer")
    assert count_params(model) > 0
