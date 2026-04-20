"""Unit tests for import-time guards in src/predict.py."""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import config
import src.predict as predict_module


def test_assert_fires_when_has_bonus_true_and_bonus_max_missing():
    """Import-time guard raises AssertionError when has_bonus=True but bonus_max absent."""
    original = config.LOTTERY.copy()
    try:
        config.LOTTERY["has_bonus"] = True
        config.LOTTERY.pop("bonus_max", None)
        with pytest.raises(AssertionError, match="bonus_max"):
            importlib.reload(predict_module)
    finally:
        config.LOTTERY.clear()
        config.LOTTERY.update(original)
        importlib.reload(predict_module)  # restore module to good state


def test_no_assert_when_has_bonus_false_and_bonus_max_missing():
    """No AssertionError when has_bonus=False, even without bonus_max."""
    original = config.LOTTERY.copy()
    try:
        config.LOTTERY["has_bonus"] = False
        config.LOTTERY.pop("bonus_max", None)
        importlib.reload(predict_module)  # must not raise
    finally:
        config.LOTTERY.clear()
        config.LOTTERY.update(original)
        importlib.reload(predict_module)


def test_no_assert_when_has_bonus_true_and_bonus_max_present():
    """No AssertionError when has_bonus=True and bonus_max is set."""
    original = config.LOTTERY.copy()
    try:
        config.LOTTERY["has_bonus"] = True
        config.LOTTERY["bonus_max"] = 50
        importlib.reload(predict_module)  # must not raise
    finally:
        config.LOTTERY.clear()
        config.LOTTERY.update(original)
        importlib.reload(predict_module)
