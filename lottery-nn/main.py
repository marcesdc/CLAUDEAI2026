"""
lottery-nn - command-line entry point.

Commands
--------
  python main.py train      - load data, train model, save checkpoint
  python main.py predict    - load checkpoint, print play suggestions
  python main.py log        - record actual draw result, score last prediction, retrain
  python main.py evaluate   - evaluate checkpoint on held-out test set
  python main.py data       - (re)generate synthetic draw data

Options (all commands)
-----------------------
  --arch transformer|lstm   model architecture (default: transformer)
  --checkpoint PATH         override checkpoint path
  --epochs N                override training epochs
  --plays N                 number of plays to generate (predict only)

log-specific options
---------------------
  --date DATE               draw date, e.g. 2026-03-25  (default: today)
  --numbers N N N N N N N   the 7 drawn numbers
  --bonus N                 the bonus number
  --no-retrain              score only, skip retraining

Examples
---------
  python main.py train
  python main.py predict --plays 5
  python main.py log --date 2026-03-25 --numbers 7 9 22 24 34 36 37 --bonus 19
  python main.py evaluate
"""

import argparse
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

import config
from src.data_loader import load_draws, generate_synthetic
from src.preprocessing import build_features, split


def cmd_train(args):
    from src.train import train
    from src.evaluate import plot_training, plot_number_frequency

    df = load_draws()
    X, y_main, y_bonus = build_features(df)
    train_data, val_data, test_data = split(X, y_main, y_bonus)

    model, history = train(
        train_data,
        val_data,
        arch=args.arch,
        epochs=args.epochs,
        checkpoint=args.checkpoint,
    )
    plot_training(history)
    plot_number_frequency(df)
    print(f"[main] Training complete. Checkpoint: {args.checkpoint}")


def cmd_predict(args):
    from src.predict import load_model, predict, print_plays

    df = load_draws()
    X, y_main, y_bonus = build_features(df)
    has_bonus = y_bonus is not None

    if not os.path.exists(args.checkpoint):
        print(f"[main] No checkpoint at '{args.checkpoint}'. Run 'train' first.")
        sys.exit(1)

    model = load_model(args.checkpoint, arch=args.arch, has_bonus=has_bonus)
    window = X[-1] if len(X) > 0 else np.zeros((config.SEQUENCE_LEN, 2 * config.LOTTERY["main_max"]))
    plays = predict(model, window, n=args.plays)
    print_plays(plays)


def cmd_evaluate(args):
    from src.predict import load_model
    from src.evaluate import evaluate

    df = load_draws()
    X, y_main, y_bonus = build_features(df)
    has_bonus = y_bonus is not None
    _, _, test_data = split(X, y_main, y_bonus)

    if not os.path.exists(args.checkpoint):
        print(f"[main] No checkpoint at '{args.checkpoint}'. Run 'train' first.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, arch=args.arch, has_bonus=has_bonus)
    evaluate(model, test_data, device=device)


def cmd_log(args):
    from src.feedback import log_draw, score_last_prediction, recency_weights
    from src.train import train
    from src.evaluate import plot_training

    date = args.date or datetime.today().strftime("%Y-%m-%d")

    # 1. Score last prediction against the actual draw
    score_last_prediction(args.numbers, args.bonus, draw_date="")

    # 2. Append the new draw to draws.csv
    log_draw(date, args.numbers, args.bonus)

    if args.no_retrain:
        return

    # 3. Retrain with recency weighting
    print("\n[main] Retraining with updated data + recency weighting...")
    df = load_draws()
    X, y_main, y_bonus = build_features(df)
    train_data, val_data, _ = split(X, y_main, y_bonus)

    # Weights only apply to the training split
    n_train = len(train_data[0])
    weights = recency_weights(n_train, decay=0.92)

    model, history = train(
        train_data,
        val_data,
        arch=args.arch,
        epochs=args.epochs,
        checkpoint=args.checkpoint,
        sample_weights=weights,
    )
    plot_training(history)
    print(f"[main] Retrain complete. Checkpoint updated: {args.checkpoint}")


def cmd_data(_args):
    generate_synthetic()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="lottery-nn - neural-network lottery prediction toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("command", choices=["train", "predict", "log", "evaluate", "data"])
    parser.add_argument("--arch", default="transformer", choices=["transformer", "lstm"])
    parser.add_argument("--checkpoint", default=config.CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--plays", type=int, default=config.NUM_PLAYS)
    # log command
    parser.add_argument("--date", default="", help="Draw date e.g. 2026-03-25")
    parser.add_argument("--numbers", type=int, nargs=7, metavar="N", help="The 7 drawn numbers")
    parser.add_argument("--bonus", type=int, help="The bonus number")
    parser.add_argument("--no-retrain", action="store_true", help="Score only, skip retraining")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    dispatch = {
        "train": cmd_train,
        "predict": cmd_predict,
        "log": cmd_log,
        "evaluate": cmd_evaluate,
        "data": cmd_data,
    }
    dispatch[args.command](args)
