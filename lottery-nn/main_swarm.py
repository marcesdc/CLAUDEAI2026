"""
Swarm entry point — joint training and per-lottery prediction.

Commands
--------
  python main_swarm.py joint-train
      Train the shared encoder on all 3 lotteries simultaneously.
      Saves a single checkpoint: models/best_swarm.pt

  python main_swarm.py predict --lottery lottomax|649|dailygrand
      Generate play suggestions for the specified lottery using the
      shared encoder checkpoint.

  python main_swarm.py log --lottery 649|dailygrand
      --date YYYY-MM-DD --numbers N [N ...] --bonus N
      Append a new draw result for 6/49 or Daily Grand.
      (LottoMax logging uses the original main.py)

  python main_swarm.py status
      Show swarm_state.json — last scores, training count, agent weights.

Options
-------
  --epochs N        override training epochs  (default: 100)
  --patience N      early-stopping patience   (default: 15)
  --lr FLOAT        learning rate             (default: 1e-3)
  --plays N         number of plays to generate (predict only, default: 5)
  --temperature T   sampling temperature      (predict only, default: 1.2)

Examples
--------
  python main_swarm.py joint-train
  python main_swarm.py predict --lottery 649 --plays 5
  python main_swarm.py log --lottery dailygrand --date 2026-03-23 --numbers 8 17 28 37 46 --bonus 7
  python main_swarm.py status
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))

from src.model_swarm import SharedLotteryTransformer, count_params, SEQ_LEN
from src.preprocessing_swarm import (
    LOTTERY_CONFIGS,
    build_all_lottery_data,
    build_features,
    get_last_window,
    load_lottery_df,
    split,
)

SWARM_CHECKPOINT = "models/best_swarm.pt"
SWARM_STATE_FILE = "data/swarm_state.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_joint_train(args):
    os.makedirs("models", exist_ok=True)

    all_data = build_all_lottery_data(seq_len=SEQ_LEN)

    # Split each lottery chronologically
    splits = {}
    for name, (X, y_main, y_bonus) in all_data.items():
        splits[name] = split(X, y_main, y_bonus)

    # Build loaders per lottery per split
    train_loaders = _make_loaders(splits, "train", args.batch_size, shuffle=True)
    val_loaders   = _make_loaders(splits, "val",   args.batch_size, shuffle=False)

    model = SharedLotteryTransformer().to(DEVICE)
    print(f"[swarm] SharedLotteryTransformer  params={count_params(model):,}  device={DEVICE}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val   = float("inf")
    patience_c = 0
    history    = {"train_loss": [], "val_loss": []}

    for epoch in range(1, args.epochs + 1):
        t0       = time.time()
        tr_loss  = _run_joint_epoch(model, train_loaders, optimizer, train=True)
        val_loss = _run_joint_epoch(model, val_loaders,   optimizer=None, train=False)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"train={tr_loss:.4f}  val={val_loss:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  {time.time()-t0:.1f}s"
        )

        if val_loss < best_val:
            best_val   = val_loss
            patience_c = 0
            torch.save(model.state_dict(), SWARM_CHECKPOINT)
            print(f"  [saved] {SWARM_CHECKPOINT}  (val={best_val:.4f})")
        else:
            patience_c += 1
            if patience_c >= args.patience:
                print(f"[swarm] Early stopping at epoch {epoch}.")
                break

    print(f"[swarm] Joint training complete. Best val_loss={best_val:.4f}")
    _update_swarm_state(joint_val_loss=best_val)
    _plot_training(history)


def cmd_predict(args):
    lottery = args.lottery
    if lottery not in LOTTERY_CONFIGS:
        print(f"[swarm] Unknown lottery '{lottery}'. Choose: {list(LOTTERY_CONFIGS)}")
        sys.exit(1)

    if not Path(SWARM_CHECKPOINT).exists():
        print(f"[swarm] No checkpoint at '{SWARM_CHECKPOINT}'. Run joint-train first.")
        sys.exit(1)

    cfg = LOTTERY_CONFIGS[lottery]
    model = SharedLotteryTransformer().to(DEVICE)
    model.load_state_dict(torch.load(SWARM_CHECKPOINT, map_location=DEVICE))
    model.eval()

    window = get_last_window(lottery, seq_len=SEQ_LEN)
    plays  = _sample_plays(
        model, window, cfg,
        n=args.plays, temperature=args.temperature,
    )
    _print_plays(plays, cfg)


def cmd_log(args):
    lottery = args.lottery
    if lottery not in ("649", "dailygrand"):
        print("[swarm] Use main.py log for LottoMax. This command handles 649 and dailygrand.")
        sys.exit(1)

    cfg      = LOTTERY_CONFIGS[lottery]
    csv_path = cfg["csv"]
    date     = args.date or datetime.today().strftime("%Y-%m-%d")

    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        print(f"[swarm] Invalid date '{date}'. Use YYYY-MM-DD.")
        sys.exit(1)

    main_count = cfg["main_count"]
    if len(args.numbers) != main_count:
        print(f"[swarm] Expected {main_count} numbers for {cfg['name']}, got {len(args.numbers)}.")
        sys.exit(1)

    main_max  = cfg["main_max"]
    bonus_max = cfg["bonus_max"]
    for n in args.numbers:
        if not (1 <= n <= main_max):
            print(f"[swarm] Number {n} is outside 1-{main_max}.")
            sys.exit(1)
    if not (1 <= args.bonus <= bonus_max):
        print(f"[swarm] Bonus {args.bonus} is outside 1-{bonus_max}.")
        sys.exit(1)

    bonus_col = cfg["bonus_col"]
    row_cols  = ["date"] + [f"n{i}" for i in range(1, main_count + 1)] + [bonus_col]
    row_vals  = [date] + sorted(args.numbers) + [args.bonus]
    row       = dict(zip(row_cols, row_vals))

    import pandas as pd
    if Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        if date in df["date"].astype(str).values:
            print(f"[swarm] Draw for {date} already exists — skipping.")
            return
        import pandas as pd
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(csv_path, index=False)
    print(f"[swarm] {cfg['name']} draw logged: {date}  {sorted(args.numbers)}  bonus={args.bonus}")

    _update_swarm_state(lottery=lottery, hits=None, draw_date=date)


def cmd_status(_args):
    if not Path(SWARM_STATE_FILE).exists():
        print("[swarm] No swarm_state.json found.")
        return
    with open(SWARM_STATE_FILE) as f:
        state = json.load(f)

    print("\n=== Swarm Status ===")
    print(f"  Joint training runs : {state.get('joint_train_count', 0)}")
    print(f"  Last val loss       : {state.get('joint_val_loss', 'n/a')}")
    print(f"  Last updated        : {state.get('last_updated', 'n/a')}")
    print()
    print("  Last scores:")
    for name, info in state.get("last_scores", {}).items():
        hits = info.get("hits", "n/a")
        date = info.get("date", "n/a")
        cfg  = LOTTERY_CONFIGS.get(name, {})
        label = cfg.get("name", name)
        print(f"    {label:14s}  hits={hits}  date={date}")
    print()
    print("  Agent weights:")
    for name, weights in state.get("agent_weights", {}).items():
        cfg   = LOTTERY_CONFIGS.get(name, {})
        label = cfg.get("name", name)
        print(f"    {label:14s}  {weights}")
    print()


# ---------------------------------------------------------------------------
# Training internals
# ---------------------------------------------------------------------------

def _make_loaders(splits: dict, split_name: str, batch_size: int, shuffle: bool) -> dict:
    """Build a DataLoader per lottery for the given split."""
    idx = {"train": 0, "val": 1, "test": 2}[split_name]
    loaders = {}
    for name, lottery_splits in splits.items():
        X, y_main, y_bonus = lottery_splits[idx]
        ds     = TensorDataset(
            torch.tensor(X),
            torch.tensor(y_main),
            torch.tensor(y_bonus),
        )
        loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return loaders


def _run_joint_epoch(model, loaders: dict, optimizer, train: bool) -> float:
    """One epoch iterating through all lottery loaders in round-robin."""
    model.train(train)
    main_crit  = nn.BCEWithLogitsLoss()
    bonus_crit = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_n    = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        # Interleave batches across lotteries
        iters = {name: iter(loader) for name, loader in loaders.items()}
        exhausted = set()

        while len(exhausted) < len(loaders):
            for name, it in iters.items():
                if name in exhausted:
                    continue
                try:
                    batch = next(it)
                except StopIteration:
                    exhausted.add(name)
                    continue

                lid = LOTTERY_CONFIGS[name]["id"]
                x, y_main, y_bonus = [b.to(DEVICE) for b in batch]

                main_logits, bonus_logits = model(x, lottery_id=lid)

                loss = main_crit(main_logits, y_main)
                bonus_target = y_bonus.argmax(dim=1)
                loss = loss + 0.3 * bonus_crit(bonus_logits, bonus_target)

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                total_loss += loss.item() * len(x)
                total_n    += len(x)

    return total_loss / total_n if total_n > 0 else 0.0


# ---------------------------------------------------------------------------
# Prediction internals
# ---------------------------------------------------------------------------

def _sample_plays(model, window: np.ndarray, cfg: dict, n: int, temperature: float) -> list:
    x = torch.tensor(window[None], dtype=torch.float32).to(DEVICE)  # (1, T, F)
    lid = cfg["id"]

    with torch.no_grad():
        main_logits, bonus_logits = model(x, lottery_id=lid)

    main_probs  = _temperature_softmax(main_logits[0].cpu().numpy(),  temperature)
    bonus_probs = _temperature_softmax(bonus_logits[0].cpu().numpy(), temperature)

    main_count = cfg["main_count"]
    main_max   = cfg["main_max"]
    bonus_max  = cfg["bonus_max"]
    lines_per  = 3

    rng   = np.random.default_rng()
    plays = []
    for _ in range(n):
        lines = []
        for _ in range(lines_per):
            nums = sorted(
                int(v) + 1
                for v in rng.choice(main_max, size=main_count, replace=False, p=main_probs[:main_max])
            )
            lines.append(nums)
        bonus_num = int(rng.choice(bonus_max, p=bonus_probs[:bonus_max]) + 1)
        plays.append({"lines": lines, "bonus": bonus_num})
    return plays


def _temperature_softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    logits = logits / max(temperature, 1e-6)
    logits -= logits.max()
    exp = np.exp(logits)
    return exp / exp.sum()


def _print_plays(plays: list, cfg: dict) -> None:
    name      = cfg["name"]
    bonus_col = cfg.get("bonus_col", "bonus").capitalize()
    width     = 52
    print(f"\n{'=' * width}")
    print(f"  {name} - Suggested Plays")
    print(f"{'=' * width}")
    for play_idx, play in enumerate(plays, 1):
        print(f"\n  Play #{play_idx}")
        print(f"  {'-' * (width - 2)}")
        for line_idx, line in enumerate(play["lines"], 1):
            nums_str = "  ".join(f"{n:2d}" for n in line)
            print(f"  Line {line_idx}:  {nums_str}")
        print(f"  {bonus_col}:  {play['bonus']:2d}")
    print(f"\n{'=' * width}")
    print("  Reminder: lottery outcomes are random. Play responsibly.")
    print(f"{'=' * width}\n")


# ---------------------------------------------------------------------------
# Swarm state helpers
# ---------------------------------------------------------------------------

def _update_swarm_state(lottery: str = None, hits=None, draw_date: str = None, joint_val_loss=None):
    state = {}
    if Path(SWARM_STATE_FILE).exists():
        with open(SWARM_STATE_FILE) as f:
            state = json.load(f)

    state["last_updated"] = datetime.today().strftime("%Y-%m-%d")

    if joint_val_loss is not None:
        state["joint_val_loss"]    = round(float(joint_val_loss), 6)
        state["joint_train_count"] = state.get("joint_train_count", 0) + 1

    if lottery and hits is not None:
        state.setdefault("last_scores", {})[lottery] = {
            "hits": hits,
            "date": draw_date or datetime.today().strftime("%Y-%m-%d"),
        }

    with open(SWARM_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _plot_training(history: dict) -> None:
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(history["train_loss"], label="train")
        ax.plot(history["val_loss"],   label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Swarm Joint Training")
        ax.legend()
        path = "models/swarm_training_curve.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"[swarm] Training curve saved to '{path}'.")
    except Exception:
        pass   # matplotlib optional


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="lottery-nn swarm — joint training and multi-lottery prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")
    sub.required = True

    # joint-train
    p_train = sub.add_parser("joint-train", help="Train shared encoder on all 3 lotteries")
    p_train.add_argument("--epochs",     type=int,   default=100)
    p_train.add_argument("--patience",   type=int,   default=15)
    p_train.add_argument("--lr",         type=float, default=1e-3)
    p_train.add_argument("--batch-size", type=int,   default=32, dest="batch_size")

    # predict
    p_pred = sub.add_parser("predict", help="Generate plays for a specific lottery")
    p_pred.add_argument("--lottery",     required=True, choices=list(LOTTERY_CONFIGS))
    p_pred.add_argument("--plays",       type=int,   default=5)
    p_pred.add_argument("--temperature", type=float, default=1.2)

    # log
    p_log = sub.add_parser("log", help="Log a new draw result (649 or dailygrand)")
    p_log.add_argument("--lottery", required=True, choices=["649", "dailygrand"])
    p_log.add_argument("--date",    default="")
    p_log.add_argument("--numbers", type=int, nargs="+", required=True)
    p_log.add_argument("--bonus",   type=int, required=True)

    # status
    sub.add_parser("status", help="Show swarm state")

    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args   = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    dispatch = {
        "joint-train": cmd_joint_train,
        "predict":     cmd_predict,
        "log":         cmd_log,
        "status":      cmd_status,
    }
    dispatch[args.command](args)
