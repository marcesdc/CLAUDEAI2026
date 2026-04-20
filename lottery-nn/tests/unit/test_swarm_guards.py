"""Unit tests for defensive guards in main_swarm.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import main_swarm


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
