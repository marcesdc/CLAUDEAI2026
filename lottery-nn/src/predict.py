"""
Inference: generate lottery play suggestions from a trained model.

A "play" (ticket) contains LINES_PER_PLAY independent selections of
MAIN_COUNT numbers each, plus one bonus number suggestion.

The model outputs a probability distribution over all 50 numbers.
Temperature scaling controls diversity across lines within a play.
"""

import numpy as np
import torch

import config
from src.model import build_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAIN_MAX = config.LOTTERY["main_max"]
MAIN_COUNT = config.LOTTERY["main_count"]
BONUS_MAX = config.LOTTERY["bonus_max"]
LINES_PER_PLAY = config.LINES_PER_PLAY


def load_model(checkpoint: str = config.CHECKPOINT, arch: str = "transformer", has_bonus: bool = True):
    model = build_model(arch=arch, has_bonus=has_bonus)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
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

    main_probs = _temperature_softmax(main_logits[0].cpu().numpy(), temperature)
    bonus_probs = (
        _temperature_softmax(bonus_logits[0].cpu().numpy(), temperature)
        if bonus_logits is not None
        else None
    )

    rng = np.random.default_rng()
    plays = []

    for _ in range(n):
        lines = []
        for _ in range(LINES_PER_PLAY):
            nums = sorted(
                int(v) + 1
                for v in rng.choice(MAIN_MAX, size=MAIN_COUNT, replace=False, p=main_probs)
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


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _temperature_softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    logits = logits / max(temperature, 1e-6)
    logits -= logits.max()   # numerical stability
    exp = np.exp(logits)
    return exp / exp.sum()
