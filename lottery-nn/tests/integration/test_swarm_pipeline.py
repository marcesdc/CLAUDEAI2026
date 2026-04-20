"""Integration tests for the multi-lottery swarm pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import pytest
from src.model_swarm import SharedLotteryTransformer, MAIN_HEAD_SIZES, BONUS_HEAD_SIZES
from src.preprocessing_swarm import LOTTERY_CONFIGS, POOL_MAX


def test_swarm_build_all_features(swarm_features_all):
    for name, (X, y_main, y_bonus) in swarm_features_all.items():
        cfg = LOTTERY_CONFIGS[name]
        assert X.shape[2] == 2 * POOL_MAX, f"{name}: expected INPUT_DIM={2*POOL_MAX}, got {X.shape[2]}"
        assert y_main.shape[1] == cfg["main_max"], f"{name}: y_main width mismatch"
        assert y_bonus.shape[1] == cfg["bonus_max"], f"{name}: y_bonus width mismatch"


def test_swarm_forward_all_lottery_ids(dummy_swarm_batch):
    model = SharedLotteryTransformer()
    model.eval()
    for lid in range(3):
        with torch.no_grad():
            main_logits, bonus_logits = model(dummy_swarm_batch, lottery_id=lid)
        assert main_logits.shape == (4, MAIN_HEAD_SIZES[lid]), f"lid={lid} main shape wrong"
        assert bonus_logits.shape == (4, BONUS_HEAD_SIZES[lid]), f"lid={lid} bonus shape wrong"
