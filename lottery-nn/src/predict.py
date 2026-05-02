"""
Inference: generate lottery play suggestions from a trained model.

A "play" (ticket) contains LINES_PER_PLAY independent selections of
MAIN_COUNT numbers each, plus one bonus number suggestion.

The model outputs a probability distribution over all MAIN_MAX numbers.
Temperature scaling controls diversity across lines within a play.
"""

import numpy as np
import torch

import config
from src.checkpoint_meta import assert_compatible as assert_ckpt_compatible
from src.model import build_model
from src.utils import temperature_softmax


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAIN_MAX = config.LOTTERY["main_max"]
MAIN_COUNT = config.LOTTERY["main_count"]
BONUS_MAX = config.LOTTERY.get("bonus_max")
if config.LOTTERY.get("has_bonus", True):
    assert BONUS_MAX is not None, "config.LOTTERY must set bonus_max when has_bonus=True"
LINES_PER_PLAY = config.LINES_PER_PLAY


def load_model(checkpoint: str = config.CHECKPOINT, arch: str = "transformer", has_bonus: bool = True):
    main_max  = int(config.LOTTERY["main_max"])
    bonus_max = int(config.LOTTERY.get("bonus_max", 0)) if has_bonus else 0
    assert_ckpt_compatible(
        checkpoint,
        expected_pool_max=main_max,
        expected_main_head_sizes=[main_max],
        expected_bonus_head_sizes=[bonus_max] if has_bonus else [],
    )
    model = build_model(arch=arch, has_bonus=has_bonus)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
    model.eval()
    model.to(DEVICE)
    return model


def predict(
    model,
    history_window: np.ndarray,
    n: int = config.NUM_PLAYS,
    temperature: float = config.TEMPERATURE,
) -> list[dict]:
    """
    Parameters
    ----------
    history_window : np.ndarray of shape (SEQ_LEN, 2*MAIN_MAX)
        The most recent SEQ_LEN draws, preprocessed (from preprocessing.py).
    n : number of plays (tickets) to generate.
    temperature : sampling temperature (>1 more diverse, <1 more concentrated).

    Returns
    -------
    List of play dicts:
        {"lines": [[7 ints], [7 ints], [7 ints]], "bonus": int | None}
    """
    x = torch.tensor(history_window[None], dtype=torch.float32).to(DEVICE)  # (1, T, F)

    with torch.no_grad():
        main_logits, bonus_logits = model(x)

    main_probs = temperature_softmax(main_logits[0].cpu().numpy(), temperature)
    bonus_probs = (
        temperature_softmax(bonus_logits[0].cpu().numpy(), temperature)
        if bonus_logits is not None
        else None
    )

    diversity_on = bool(getattr(config, "DIVERSITY_GUARD_ENABLED", True))
    critic_on    = bool(getattr(config, "CRITIC_ENABLED", True))
    max_attempts = int(getattr(config, "DIVERSITY_MAX_ATTEMPTS", 20))

    lottery_stats = None
    if critic_on:
        try:
            from src.critic import compute_lottery_stats
            from src.data_loader import load_draws
            df_hist = load_draws()
            lottery_stats = compute_lottery_stats(
                df_hist, config.LOTTERY,
                percentiles=getattr(config, "CRITIC_PERCENTILES", (5, 95)),
            )
        except Exception as e:
            print(f"[predict] critic disabled: could not compute stats ({e})")
            lottery_stats = None

    from src.diversity import is_too_similar, resolve_max_overlap
    max_overlap = resolve_max_overlap(getattr(config, "DIVERSITY_MAX_OVERLAP", None), MAIN_COUNT)

    rng = np.random.default_rng()
    plays = []

    for _ in range(n):
        lines = []
        for _ in range(LINES_PER_PLAY):
            nums = _sample_one_line(
                rng, main_probs, MAIN_MAX, MAIN_COUNT,
                existing_lines=lines,
                diversity_on=diversity_on,
                max_overlap=max_overlap,
                lottery_stats=lottery_stats if critic_on else None,
                max_attempts=max_attempts,
            )
            lines.append(nums)

        bonus_num = None
        if bonus_probs is not None:
            bonus_num = int(rng.choice(BONUS_MAX, p=bonus_probs) + 1)

        plays.append({"lines": lines, "bonus": bonus_num})

    # Auto-save so they can be scored after the draw
    from src.feedback import save_prediction
    save_prediction(plays)

    return plays


def _sample_one_line(rng, main_probs, main_max, main_count, *,
                     existing_lines, diversity_on, max_overlap,
                     lottery_stats, max_attempts):
    """Single-line sampler with shared budget across diversity + critic."""
    import sys as _sys
    from src.critic import critic_filter
    from src.diversity import is_too_similar

    last_nums = []
    for _ in range(max_attempts):
        nums = sorted(
            int(v) + 1
            for v in rng.choice(main_max, size=main_count, replace=False, p=main_probs)
        )
        last_nums = nums
        if diversity_on and existing_lines and is_too_similar(nums, existing_lines, max_overlap):
            continue
        if lottery_stats is not None and not critic_filter(nums, lottery_stats):
            continue
        return nums

    print(
        f"[critic+diversity] gave up after {max_attempts} attempts -- returning last sample.",
        file=_sys.stderr,
    )
    return last_nums


def print_plays(plays: list[dict]) -> None:
    game = config.LOTTERY["name"]
    width = 50
    print(f"\n{'=' * width}")
    print(f"  {game} - Suggested Plays")
    print(f"{'=' * width}")

    for play_idx, play in enumerate(plays, 1):
        print(f"\n  Play #{play_idx}")
        print(f"  {'-' * (width - 2)}")
        for line_idx, line in enumerate(play["lines"], 1):
            nums_str = "  ".join(f"{n:2d}" for n in line)
            print(f"  Line {line_idx}:  {nums_str}")
        if play["bonus"] is not None:
            print(f"  Bonus:   {play['bonus']:2d}")

    print(f"\n{'=' * width}")
    print("  Reminder: lottery outcomes are random. Play responsibly.")
    print(f"{'=' * width}\n")


