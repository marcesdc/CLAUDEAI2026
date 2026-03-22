"""
Training loop with early stopping, learning-rate scheduling, and
checkpoint saving.

Usage (from project root):
    python main.py train [--arch transformer|lstm] [--epochs N]
"""

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config
from src.model import build_model, count_params


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def train(
    train_data: tuple,
    val_data: tuple,
    arch: str = "transformer",
    epochs: int = config.EPOCHS,
    checkpoint: str = config.CHECKPOINT,
    sample_weights: np.ndarray = None,
) -> nn.Module:
    """
    Parameters
    ----------
    train_data / val_data : (X, y_main, y_bonus)
        Arrays from preprocessing.split().  y_bonus may be None.
    arch : 'transformer' | 'lstm'
    checkpoint : path where best model weights are saved.

    Returns the model loaded with best validation weights.
    """
    X_tr, ym_tr, yb_tr = train_data
    X_v, ym_v, yb_v = val_data
    has_bonus = yb_tr is not None

    model = build_model(arch=arch, has_bonus=has_bonus).to(DEVICE)
    print(f"[train] Model: {arch}  |  params: {count_params(model):,}  |  device: {DEVICE}")

    train_loader = _make_loader(X_tr, ym_tr, yb_tr, shuffle=True, weights=sample_weights)
    val_loader = _make_loader(X_v, ym_v, yb_v, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    main_criterion = nn.BCEWithLogitsLoss()
    bonus_criterion = nn.CrossEntropyLoss() if has_bonus else None

    best_val_loss = float("inf")
    patience_counter = 0
    os.makedirs(os.path.dirname(checkpoint), exist_ok=True)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss = _run_epoch(model, train_loader, optimizer, main_criterion, bonus_criterion, train=True)
        val_loss = _run_epoch(model, val_loader, None, main_criterion, bonus_criterion, train=False)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{epochs}  "
            f"train={tr_loss:.4f}  val={val_loss:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint)
            print(f"  [saved] checkpoint  (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"[train] Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    print(f"[train] Best val_loss={best_val_loss:.4f}")
    return model, history


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _make_loader(X, y_main, y_bonus, shuffle: bool, weights: np.ndarray = None) -> DataLoader:
    from torch.utils.data import WeightedRandomSampler
    tensors = [torch.tensor(X), torch.tensor(y_main)]
    if y_bonus is not None:
        tensors.append(torch.tensor(y_bonus))
    ds = TensorDataset(*tensors)
    if weights is not None and shuffle:
        sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.float32), num_samples=len(ds), replacement=True)
        return DataLoader(ds, batch_size=config.BATCH_SIZE, sampler=sampler, pin_memory=True)
    return DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=shuffle, pin_memory=True)


def _run_epoch(model, loader, optimizer, main_crit, bonus_crit, train: bool) -> float:
    model.train(train)
    total_loss = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for batch in loader:
            x = batch[0].to(DEVICE)
            y_main = batch[1].to(DEVICE)
            y_bonus = batch[2].to(DEVICE) if len(batch) == 3 else None

            main_logits, bonus_logits = model(x)
            loss = main_crit(main_logits, y_main)

            if bonus_crit is not None and bonus_logits is not None and y_bonus is not None:
                bonus_target = y_bonus.argmax(dim=1)
                loss = loss + 0.3 * bonus_crit(bonus_logits, bonus_target)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * len(x)

    return total_loss / len(loader.dataset)
