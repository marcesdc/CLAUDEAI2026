"""Unit tests for src/model_swarm.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import pytest
from src.model_swarm import (
    SharedLotteryTransformer,
    count_params,
    MAIN_HEAD_SIZES,
    BONUS_HEAD_SIZES,
    INPUT_DIM,
)


def test_shared_transformer_lottomax_shapes(dummy_swarm_batch):
    model = SharedLotteryTransformer()
    model.eval()
    with torch.no_grad():
        main_logits, bonus_logits = model(dummy_swarm_batch, lottery_id=0)
    assert main_logits.shape == (4, MAIN_HEAD_SIZES[0])
    assert bonus_logits.shape == (4, BONUS_HEAD_SIZES[0])


def test_shared_transformer_649_shapes(dummy_swarm_batch):
    model = SharedLotteryTransformer()
    model.eval()
    with torch.no_grad():
        main_logits, bonus_logits = model(dummy_swarm_batch, lottery_id=1)
    assert main_logits.shape == (4, MAIN_HEAD_SIZES[1])
    assert bonus_logits.shape == (4, BONUS_HEAD_SIZES[1])


def test_shared_transformer_dailygrand_shapes(dummy_swarm_batch):
    model = SharedLotteryTransformer()
    model.eval()
    with torch.no_grad():
        main_logits, bonus_logits = model(dummy_swarm_batch, lottery_id=2)
    assert main_logits.shape == (4, MAIN_HEAD_SIZES[2])
    assert bonus_logits.shape == (4, BONUS_HEAD_SIZES[2])


def test_shared_transformer_count_params():
    model = SharedLotteryTransformer()
    assert count_params(model) > 100_000


def test_input_dim_matches_batch():
    # Passing a tensor with the wrong last dim must raise, not silently produce garbage
    model = SharedLotteryTransformer()
    bad_batch = torch.zeros(2, 10, INPUT_DIM + 1)
    with pytest.raises(Exception):
        model(bad_batch, lottery_id=0)
