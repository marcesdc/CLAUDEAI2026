"""
Thompson sampling bandit over portfolio STRATEGIES.

Forked and reframed from lottery-nn/src/bandit.py. The original tracked per-lottery
"hit rate" posteriors -- meaningless because lottery draws are i.i.d. uniform. Here
we track per-strategy "did this strategy beat the uniform-random baseline on this draw"
posteriors, which IS a real Bernoulli signal because strategies differ in their
expected EV/$ on the popularity-of-tickets channel.

Each strategy maintains a Beta(alpha, beta) posterior where:
  alpha = 1 + cumulative draws where strategy realized EV/$ >= baseline EV/$
  beta  = 1 + cumulative draws where strategy realized EV/$ <  baseline EV/$

After each scored draw, alpha/beta are updated and a new weight is sampled
from the posterior. Higher weight reflects stronger empirical edge over baseline.

The sampled weight can be used to:
  - Choose which strategy to play next draw (Thompson sampling)
  - Display relative strategy confidence in `status`
  - Mix strategies via softmax (weighted ensemble)

Reproducibility fix: uses np.random.default_rng(seed) instead of legacy np.random.
"""

import numpy as np

_DEFAULT_ALPHA = 1.0
_DEFAULT_BETA  = 1.0

# Module-level RNG. Seeded by set_seed() from main.py at startup.
_rng = np.random.default_rng()


def set_seed(seed: int) -> None:
    """Reseed the module RNG. Call once at program startup for reproducibility."""
    global _rng
    _rng = np.random.default_rng(seed)


def default_entry() -> dict:
    """Return a fresh Thompson entry with uninformative Beta(1,1) prior."""
    return {
        "alpha":  _DEFAULT_ALPHA,
        "beta":   _DEFAULT_BETA,
        "weight": 0.5,
        "draws_scored": 0,
    }


def update(strategy_weights: dict, strategy: str, beat_baseline: bool) -> dict:
    """
    Update Thompson parameters for *strategy* after a scored draw.

    Parameters
    ----------
    strategy_weights : full dict from portfolio_state.json
    strategy         : strategy name, e.g. 'anti_popular', 'uniform_random'
    beat_baseline    : True if this strategy's realized EV/$ exceeded the baseline

    Returns the mutated strategy_weights dict.
    """
    entry = strategy_weights.get(strategy, default_entry())

    if beat_baseline:
        entry["alpha"] = entry.get("alpha", _DEFAULT_ALPHA) + 1.0
    else:
        entry["beta"] = entry.get("beta", _DEFAULT_BETA) + 1.0

    entry["weight"]       = float(_rng.beta(entry["alpha"], entry["beta"]))
    entry["draws_scored"] = entry.get("draws_scored", 0) + 1

    strategy_weights[strategy] = entry
    return strategy_weights


def sample_all(strategy_weights: dict) -> dict:
    """
    Re-sample weights for every strategy from current Beta posteriors.
    Useful when you want a fresh Thompson draw without a new scored result.
    """
    result = {}
    for strategy, entry in strategy_weights.items():
        alpha = entry.get("alpha", _DEFAULT_ALPHA)
        beta  = entry.get("beta",  _DEFAULT_BETA)
        result[strategy] = {
            **entry,
            "weight": float(_rng.beta(alpha, beta)),
        }
    return result


def pick_strategy(strategy_weights: dict) -> str:
    """Thompson sampling: sample one weight per strategy, pick the argmax."""
    if not strategy_weights:
        raise ValueError("strategy_weights is empty -- nothing to pick from")
    samples = {
        s: float(_rng.beta(e.get("alpha", _DEFAULT_ALPHA), e.get("beta", _DEFAULT_BETA)))
        for s, e in strategy_weights.items()
    }
    return max(samples, key=samples.get)


def summary(strategy_weights: dict) -> str:
    """Return a human-readable summary string for cmd_status."""
    lines = []
    for strategy, entry in strategy_weights.items():
        alpha  = entry.get("alpha", _DEFAULT_ALPHA)
        beta   = entry.get("beta",  _DEFAULT_BETA)
        weight = entry.get("weight", 0.5)
        scored = entry.get("draws_scored", 0)
        mean   = alpha / (alpha + beta)
        lines.append(
            f"    {strategy:18s}  weight={weight:.3f}  mean={mean:.3f}"
            f"  alpha={alpha:.1f}  beta={beta:.1f}  draws={scored}"
        )
    return "\n".join(lines)
