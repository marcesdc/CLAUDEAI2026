"""
Shared utility functions used across predict.py, main_swarm.py, and other modules.
"""

import numpy as np


def temperature_softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to logits and return a probability distribution.

    Args:
        logits: Raw model output scores (1-D array).
        temperature: Scaling factor. >1 increases diversity; <1 concentrates mass.

    Returns:
        Probability array of the same shape as logits, summing to 1.
    """
    logits = logits / max(temperature, 1e-6)
    logits -= logits.max()   # numerical stability
    exp = np.exp(logits)
    return exp / exp.sum()
