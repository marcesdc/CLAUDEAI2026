"""
Thompson sampling bandit for per-lottery agent weights.

Each lottery maintains a Beta(alpha, beta) posterior where:
  alpha = 1 + cumulative main-ball hits across all scored draws
  beta  = 1 + cumulative main-ball misses across all scored draws

After each scored draw, alpha/beta are updated and a new weight is sampled
from the posterior. A higher weight reflects better recent prediction accuracy.

The sampled weight can be used to:
  - Adjust prediction temperature (lower weight -> higher temp -> more exploration)
  - Display relative agent confidence in `status`
  - Prioritise retraining focus in future Actor-Critic phase
"""

import numpy as np

_DEFAULT_ALPHA = 1.0
_DEFAULT_BETA  = 1.0


def default_entry() -> dict:
    """Return a fresh Thompson entry with uninformative Beta(1,1) prior."""
    return {
        "alpha":  _DEFAULT_ALPHA,
        "beta":   _DEFAULT_BETA,
        "weight": 0.5,
        "draws_scored": 0,
    }


def update(agent_weights: dict, lottery: str, hits: int, main_count: int) -> dict:
    """
    Update Thompson parameters for *lottery* after a scored draw.

    Parameters
    ----------
    agent_weights : the full agent_weights dict from swarm_state.json
    lottery       : 'lottomax' | '649' | 'dailygrand'
    hits          : main-ball hits on the best-scoring line (0..main_count)
    main_count    : total main balls drawn for this lottery

    Returns the mutated agent_weights dict.
    """
    entry  = agent_weights.get(lottery, default_entry())
    misses = max(main_count - hits, 0)

    entry["alpha"]        = entry.get("alpha", _DEFAULT_ALPHA) + hits
    entry["beta"]         = entry.get("beta",  _DEFAULT_BETA)  + misses
    entry["weight"]       = float(np.random.beta(entry["alpha"], entry["beta"]))
    entry["draws_scored"] = entry.get("draws_scored", 0) + 1

    agent_weights[lottery] = entry
    return agent_weights


def sample_all(agent_weights: dict) -> dict:
    """
    Re-sample weights for every lottery from their current Beta posteriors.
    Useful when you want a fresh Thompson draw without a new scored result.
    """
    result = {}
    for lottery, entry in agent_weights.items():
        alpha = entry.get("alpha", _DEFAULT_ALPHA)
        beta  = entry.get("beta",  _DEFAULT_BETA)
        result[lottery] = {
            **entry,
            "weight": float(np.random.beta(alpha, beta)),
        }
    return result


def temperature_from_weight(weight: float,
                             t_min: float = 0.8,
                             t_max: float = 1.6) -> float:
    """
    Map a Thompson weight in [0, 1] to a sampling temperature.

    High confidence (weight near 1) -> low temperature (more exploitation).
    Low confidence  (weight near 0) -> high temperature (more exploration).
    """
    # Clamp to avoid degenerate Beta samples at exact 0 or 1
    w = max(min(weight, 0.99), 0.01)
    return round(t_max - w * (t_max - t_min), 4)


def summary(agent_weights: dict, lottery_configs: dict) -> str:
    """Return a human-readable summary string for cmd_status."""
    lines = []
    for lottery, entry in agent_weights.items():
        cfg    = lottery_configs.get(lottery, {})
        label  = cfg.get("name", lottery)
        alpha  = entry.get("alpha", _DEFAULT_ALPHA)
        beta   = entry.get("beta",  _DEFAULT_BETA)
        weight = entry.get("weight", 0.5)
        scored = entry.get("draws_scored", 0)
        mean   = alpha / (alpha + beta)          # Beta posterior mean
        lines.append(
            f"    {label:14s}  weight={weight:.3f}  mean={mean:.3f}"
            f"  alpha={alpha:.1f}  beta={beta:.1f}  draws={scored}"
        )
    return "\n".join(lines)
