"""
autorun.py -- Autonomous overnight hyperparameter search for lottery-nn.

Each experiment:
  1. Samples a random config from the search space
  2. Runs training in an isolated subprocess (fresh imports, no module caching)
  3. Reads val_loss from subprocess stdout
  4. Keeps the best checkpoint found so far
  5. Logs every result to data/experiment_log.csv

Usage:
    # Default: 8-hour search, 3-min budget per experiment
    C:\\Python314\\python.exe autorun.py

    # Shorter run
    C:\\Python314\\python.exe autorun.py --hours 2 --budget 120

    # Review best config found
    type models\\auto_best_config.json

Output files:
    models/auto_best.pt          best checkpoint found
    models/auto_best_config.json config that produced the best checkpoint
    data/experiment_log.csv      full log of every experiment
"""

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PYTHON = r"C:\Python314\python.exe"
PROJECT_ROOT = Path(__file__).resolve().parent
LOG_FILE = PROJECT_ROOT / "data" / "experiment_log.csv"
TMP_CHECKPOINT = PROJECT_ROOT / "models" / "_auto_tmp.pt"
AUTO_BEST_PT = PROJECT_ROOT / "models" / "auto_best.pt"
AUTO_BEST_CFG = PROJECT_ROOT / "models" / "auto_best_config.json"

LOG_FIELDS = [
    "id", "timestamp", "arch",
    "EMBED_DIM", "NUM_HEADS", "NUM_LAYERS", "FF_DIM", "DROPOUT",
    "LR", "WEIGHT_DECAY", "BATCH_SIZE", "SEQUENCE_LEN", "PATIENCE", "EPOCHS",
    "val_loss", "duration_s", "status",
]

# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------
SEARCH_SPACE = {
    "arch":         ["transformer", "lstm"],
    "EMBED_DIM":    [32, 64, 128, 256],
    "NUM_LAYERS":   [1, 2, 3, 4, 5],
    "FF_DIM":       [128, 256, 512],
    "DROPOUT":      [0.0, 0.1, 0.2, 0.3],
    "LR":           [1e-4, 3e-4, 1e-3, 3e-3],
    "WEIGHT_DECAY": [0.0, 1e-5, 1e-4, 1e-3],
    "BATCH_SIZE":   [16, 32, 64, 128],
    "SEQUENCE_LEN": [5, 10, 15, 20, 30],
    "PATIENCE":     [8, 12, 15],
    "EPOCHS":       [50, 75, 100],
}


def sample_config() -> dict:
    """Sample one random valid config from the search space."""
    cfg = {k: random.choice(v) for k, v in SEARCH_SPACE.items()}
    # NUM_HEADS must divide EMBED_DIM for the Transformer
    if cfg["arch"] == "transformer":
        valid_heads = [h for h in [2, 4, 8] if cfg["EMBED_DIM"] % h == 0]
        cfg["NUM_HEADS"] = random.choice(valid_heads)
    else:
        cfg["NUM_HEADS"] = 4  # not used by LSTM but kept in log for completeness
    return cfg


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_worker(cfg: dict, budget_s: int) -> tuple:
    """
    Spawn a subprocess that trains with the given config overrides.
    Returns (val_loss: float, status: str).
    """
    env = os.environ.copy()
    env["LOTTERY_OVERRIDES"] = json.dumps(cfg)

    try:
        result = subprocess.run(
            [PYTHON, str(Path(__file__).resolve()), "--worker"],
            capture_output=True,
            text=True,
            timeout=budget_s + 60,   # hard kill after budget + 1-min grace
            cwd=str(PROJECT_ROOT),
            env=env,
        )
    except subprocess.TimeoutExpired:
        return float("inf"), "timeout"

    if result.returncode != 0:
        last_err = result.stderr.strip().splitlines()
        if last_err:
            print(f"\n         ! {last_err[-1]}")
        return float("inf"), "error"

    # Parse the val_loss printed by train.py: "[train] Best val_loss=X.XXXX"
    for line in result.stdout.splitlines():
        if "[train] Best val_loss=" in line:
            try:
                return float(line.split("=")[-1].strip()), "ok"
            except ValueError:
                pass

    return float("inf"), "parse_error"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_loop(hours: float, budget_s: int):
    best_val = float("inf")
    exp_id = _next_id()
    deadline = time.time() + hours * 3600

    print(f"[autorun] Search started  {hours}h window | {budget_s}s budget/experiment")
    print(f"[autorun] Log      -> {LOG_FILE.relative_to(PROJECT_ROOT)}")
    print(f"[autorun] Best pt  -> {AUTO_BEST_PT.relative_to(PROJECT_ROOT)}")
    print(f"[autorun] Ctrl+C to stop early\n")

    try:
        while time.time() < deadline:
            cfg = sample_config()
            t0 = time.time()

            # Print a short summary of the key config choices while training runs
            short = {k: cfg[k] for k in ["arch", "EMBED_DIM", "NUM_LAYERS", "LR", "SEQUENCE_LEN"]}
            print(f"[{exp_id:04d}] {short} ...", end="", flush=True)

            val_loss, status = run_worker(cfg, budget_s)
            duration = time.time() - t0

            flag = ""
            if val_loss < best_val:
                best_val = val_loss
                _save_best(cfg)
                flag = "  *** BEST ***"

            print(f"  val={val_loss:.4f}  {duration:.0f}s  [{status}]{flag}")
            _log(exp_id, cfg, val_loss, duration, status)
            exp_id += 1

    except KeyboardInterrupt:
        print("\n[autorun] Stopped by user.")

    print(f"\n[autorun] Done.  Best val_loss={best_val:.4f}")
    if AUTO_BEST_CFG.exists():
        print(f"[autorun] Winning config: {AUTO_BEST_CFG.name}")
        with open(AUTO_BEST_CFG) as f:
            print(json.dumps(json.load(f), indent=2))


# ---------------------------------------------------------------------------
# Worker (subprocess entry point -- do not call directly)
# ---------------------------------------------------------------------------

def worker_main():
    """
    Apply config overrides passed via LOTTERY_OVERRIDES env var, then train.
    Runs in an isolated subprocess so module-level constants re-evaluate cleanly.
    """
    sys.path.insert(0, str(PROJECT_ROOT))
    overrides = json.loads(os.environ.get("LOTTERY_OVERRIDES", "{}"))

    # Apply overrides to config BEFORE any other project imports so that
    # module-level constants (e.g. SEQ_LEN = config.SEQUENCE_LEN) pick them up.
    import config
    arch = overrides.pop("arch", "transformer")
    for key, val in overrides.items():
        if hasattr(config, key):
            setattr(config, key, val)

    # Now import project modules (first-time import reads updated config values)
    from src.data_loader import load_draws
    from src.preprocessing import build_features, split
    from src.train import train

    df = load_draws()
    X, y_main, y_bonus = build_features(df)
    train_data, val_data, _ = split(X, y_main, y_bonus)

    os.makedirs(str(PROJECT_ROOT / "models"), exist_ok=True)
    train(
        train_data,
        val_data,
        arch=arch,
        epochs=config.EPOCHS,
        checkpoint=str(TMP_CHECKPOINT),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_id() -> int:
    if not LOG_FILE.exists():
        return 1
    with open(LOG_FILE, newline="") as f:
        rows = list(csv.DictReader(f))
    return (max(int(r["id"]) for r in rows) + 1) if rows else 1


def _log(exp_id: int, cfg: dict, val_loss: float, duration: float, status: str):
    write_header = not LOG_FILE.exists()
    val_str = f"{val_loss:.6f}" if val_loss != float("inf") else "inf"
    row = {
        "id": exp_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "val_loss": val_str,
        "duration_s": f"{duration:.0f}",
        "status": status,
    }
    row.update({k: cfg.get(k, "") for k in LOG_FIELDS if k in cfg})

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _save_best(cfg: dict):
    if TMP_CHECKPOINT.exists():
        shutil.copy(TMP_CHECKPOINT, AUTO_BEST_PT)
    with open(AUTO_BEST_CFG, "w") as f:
        json.dump(cfg, f, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Overnight hyperparameter search for lottery-nn",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--hours", type=float, default=8.0,
        help="Total search duration in hours (default: 8)"
    )
    parser.add_argument(
        "--budget", type=int, default=180,
        help="Per-experiment wall-clock budget in seconds (default: 180)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible search order"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        worker_main()
    else:
        if args.seed is not None:
            random.seed(args.seed)
        run_loop(args.hours, args.budget)
