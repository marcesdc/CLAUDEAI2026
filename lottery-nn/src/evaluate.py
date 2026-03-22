"""
Evaluation metrics and visualisation for lottery-nn.

Metrics
-------
hit_rate_k  : fraction of draws where ≥ k predicted numbers matched actual.
coverage    : mean number of actual numbers captured in top-20 predictions.
"""

import numpy as np
import matplotlib.pyplot as plt

import config
from src.preprocessing import decode_multihot


MAIN_COUNT = config.LOTTERY["main_count"]
MAIN_MAX = config.LOTTERY["main_max"]


def evaluate(model, test_data: tuple, device="cpu") -> dict:
    """
    Run model on test set and return evaluation metrics.

    test_data : (X, y_main, y_bonus)  numpy arrays
    """
    import torch
    X, y_main, y_bonus = test_data
    model.eval().to(device)

    all_main_probs = []
    batch = 256
    with torch.no_grad():
        for start in range(0, len(X), batch):
            x_t = torch.tensor(X[start : start + batch], dtype=torch.float32).to(device)
            logits, _ = model(x_t)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_main_probs.append(probs)

    pred_probs = np.concatenate(all_main_probs, axis=0)   # (N, MAIN_MAX)

    metrics = {}
    for k in range(1, MAIN_COUNT + 1):
        metrics[f"hit_rate_{k}"] = _hit_rate_k(pred_probs, y_main, k)

    metrics["mean_coverage"] = _mean_coverage(pred_probs, y_main)
    metrics["top20_coverage"] = _mean_coverage(pred_probs, y_main, top=20)

    _print_metrics(metrics)
    return metrics


def plot_training(history: dict, save_path: str = "models/training_curve.png") -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train_loss"], label="Train loss")
    ax.plot(history["val_loss"], label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[evaluate] Training curve saved to '{save_path}'.")


def plot_number_frequency(df, save_path: str = "models/frequency.png") -> None:
    """Bar chart of historical draw frequency for each main ball."""
    import matplotlib.pyplot as plt

    main_cols = [c for c in df.columns if c.startswith("n")]
    counts = np.zeros(MAIN_MAX, dtype=int)
    for row in df[main_cols].values:
        for v in row:
            if 1 <= v <= MAIN_MAX:
                counts[v - 1] += 1

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(1, MAIN_MAX + 1), counts, color="steelblue")
    ax.set_xlabel("Ball number")
    ax.set_ylabel("Draw count")
    ax.set_title(f"{config.LOTTERY['name']} – historical frequency")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[evaluate] Frequency chart saved to '{save_path}'.")


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _hit_rate_k(pred_probs: np.ndarray, y_true: np.ndarray, k: int) -> float:
    hits = 0
    for p, y in zip(pred_probs, y_true):
        top_k = set(np.argsort(p)[::-1][:MAIN_COUNT])
        actual = set(np.where(y > 0.5)[0])
        if len(top_k & actual) >= k:
            hits += 1
    return hits / len(pred_probs)


def _mean_coverage(pred_probs: np.ndarray, y_true: np.ndarray, top: int = MAIN_COUNT) -> float:
    coverages = []
    for p, y in zip(pred_probs, y_true):
        top_k = set(np.argsort(p)[::-1][:top])
        actual = set(np.where(y > 0.5)[0])
        coverages.append(len(top_k & actual))
    return float(np.mean(coverages))


def _print_metrics(metrics: dict) -> None:
    print("\n[evaluate] Test-set metrics:")
    for k, v in metrics.items():
        print(f"  {k:<20s} {v:.4f}")
    print()
