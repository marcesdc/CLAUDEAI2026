"""Unit tests for src/utils.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from src.utils import temperature_softmax


def test_sums_to_one():
    logits = np.array([1.0, 2.0, 3.0, 4.0])
    result = temperature_softmax(logits, temperature=1.0)
    assert abs(result.sum() - 1.0) < 1e-6


def test_all_finite():
    logits = np.array([1.0, 2.0, 3.0])
    result = temperature_softmax(logits, temperature=1.0)
    assert np.all(np.isfinite(result))


def test_uniform_input_gives_uniform_output():
    logits = np.zeros(5)
    result = temperature_softmax(logits, temperature=1.0)
    np.testing.assert_allclose(result, np.full(5, 0.2), atol=1e-6)


def test_high_temperature_flattens_distribution():
    logits = np.array([0.0, 10.0])
    low_temp = temperature_softmax(logits, temperature=0.1)
    high_temp = temperature_softmax(logits, temperature=10.0)
    # High temperature -> more uniform -> lower max probability
    assert high_temp.max() < low_temp.max()


def test_low_temperature_concentrates_on_argmax():
    logits = np.array([1.0, 5.0, 2.0])
    result = temperature_softmax(logits, temperature=0.01)
    assert result.argmax() == 1


def test_zero_temperature_clamped_no_crash():
    logits = np.array([1.0, 2.0, 3.0])
    result = temperature_softmax(logits, temperature=0.0)
    assert np.all(np.isfinite(result))
    assert abs(result.sum() - 1.0) < 1e-6


def test_large_logits_no_overflow():
    logits = np.array([1000.0, 1001.0, 999.0])
    result = temperature_softmax(logits, temperature=1.0)
    assert np.all(np.isfinite(result))
    assert abs(result.sum() - 1.0) < 1e-6


def test_single_element():
    logits = np.array([42.0])
    result = temperature_softmax(logits, temperature=1.0)
    np.testing.assert_allclose(result, [1.0], atol=1e-6)
