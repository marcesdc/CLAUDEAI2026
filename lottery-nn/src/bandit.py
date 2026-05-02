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

# Tier table for tier_reward(): rewards 3+ matches exponentially more than
# 1-2 matches, mirroring real OLG prize structures where small matches pay
# a token amount and large matches pay orders of magnitude more.
# Capped at 50 alpha-units per draw so a single fluke can't dominate the
# Beta posterior.
_TIER_ALPHA_TABLE = {0: 0.0, 1: 0.5, 2: 1.0, 3: 4.0, 4: 12.0, 5: 30.0, 6: 80.0, 7: 200.0}
_TIER_ALPHA_CAP   = 50.0


def default_entry() -> dict:
    """Return a fresh Thompson entry with uninformative Beta(1,1) prior."""
    return {
        "alpha":  _DEFAULT_ALPHA,
        "beta":   _DEFAULT_BETA,
        "weight": 0.5,
        "draws_scored": 0,
    }


def tier_reward(hits: int, main_count: int, lottery: str = "") -> tuple[float, float]:
    """
    Map a hit count to (alpha_delta, beta_delta) for tier-weighted Thompson.

    The alpha increment scales super-linearly with hits so a single 3-match
    contributes more than three 1-matches combined. Beta_delta is just the
    miss count, keeping the Beta distribution well-defined.

    Parameters
    ----------
    hits       : main-ball hits on the best-scoring line (clamped to main_count)
    main_count : total main balls drawn for this lottery (used for misses)
    lottery    : reserved for future per-lottery scaling; currently unused

    Returns (alpha_delta, beta_delta), both non-negative floats.
    """
    h = max(0, min(int(hits), int(main_count)))
    alpha_delta = min(_TIER_ALPHA_TABLE.get(h, 0.0), _TIER_ALPHA_CAP)
    beta_delta  = float(max(int(main_count) - h, 0))
    return alpha_delta, beta_delta


def update(agent_weights: dict,
           lottery: str,
           hits: int,
           main_count: int,
           *,
           tier_weighted: bool = False) -> dict:
    """
    Update Thompson parameters for *lottery* after a scored draw.

    Parameters
    ----------
    agent_weights : the full agent_weights dict from swarm_state.json
    lottery       : 'lottomax' | '649' | 'dailygrand'
    hits          : main-ball hits on the best-scoring line (0..main_count)
    main_count    : total main balls drawn for this lottery
    tier_weighted : when True, alpha grows according to tier_reward() (3+ hits
                    weighted exponentially). When False (default), behavior is
                    the original linear `alpha += hits`.

    Returns the mutated agent_weights dict.
    """
    entry  = agent_weights.get(lottery, default_entry())

    if tier_weighted:
        alpha_d, beta_d = tier_reward(hits, main_count, lottery)
    else:
        alpha_d = float(hits)
        beta_d  = float(max(main_count - hits, 0))

    entry["alpha"]        = entry.get("alpha", _DEFAULT_ALPHA) + alpha_d
    entry["beta"]         = entry.get("beta",  _DEFAULT_BETA)  + beta_d
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
