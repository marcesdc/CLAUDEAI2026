# Focal Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `BCEWithLogitsLoss` on the main-number head with focal loss to improve training signal on the heavily imbalanced multi-label target (7 drawn from 50).

**Architecture:** A new `src/focal_loss.py` module exposes a single function `focal_loss_with_logits`. Both `src/train.py` and `main_swarm.py` import and call it instead of `BCEWithLogitsLoss`. Gamma and alpha are configurable via `config.py`.

**Tech Stack:** PyTorch (`torch.nn.functional`), Python 3.14, pytest

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/focal_loss.py` | Create | Focal loss implementation |
| `config.py` | Modify | Add `FOCAL_GAMMA`, `FOCAL_ALPHA` |
| `src/train.py` | Modify | Swap loss in `_run_epoch` |
| `main_swarm.py` | Modify | Swap loss in `_run_joint_epoch` |
| `tests/unit/test_focal_loss.py` | Create | Unit tests for focal loss |

---

### Task 1: Add config constants

**Files:**
- Modify: `config.py`

- [ ] **Step 1: Open `config.py` and append focal loss constants after the Training section**

The file currently ends with:
```python
# Prediction
NUM_PLAYS = 5
TEMPERATURE = 1.2
```

Add this block after the Training section (after `PATIENCE = 15`), before the Prediction section:

```python
# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
FOCAL_GAMMA = 2.0   # focal loss modulating exponent; 0 = standard BCE
FOCAL_ALPHA = 0.25  # positive-class weight; None = no alpha weighting
```

- [ ] **Step 2: Verify the file parses cleanly**

```bash
C:\Python314\python.exe -c "import config; print(config.FOCAL_GAMMA, config.FOCAL_ALPHA)"
```

Expected output:
```
2.0 0.25
```

- [ ] **Step 3: Commit**

```bash
cd "d:/AI - 2026/CLAUDEAI2026/lottery-nn"
git add config.py
git commit -m "config: add FOCAL_GAMMA and FOCAL_ALPHA constants"
```

---

### Task 2: Implement focal loss with tests (TDD)

**Files:**
- Create: `tests/unit/test_focal_loss.py`
- Create: `src/focal_loss.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_focal_loss.py`:

```python
"""Unit tests for focal_loss_with_logits."""
import torch
import torch.nn.functional as F
import pytest

from src.focal_loss import focal_loss_with_logits


def test_focal_equals_bce_when_gamma_zero():
    """With gamma=0 and alpha=None, focal loss must equal BCEWithLogitsLoss."""
    torch.manual_seed(0)
    logits  = torch.randn(8, 50)
    targets = (torch.rand(8, 50) > 0.86).float()  # ~14% positive, mirrors 7/50

    bce_loss   = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
    focal_loss = focal_loss_with_logits(logits, targets, gamma=0.0, alpha=None)

    assert abs(focal_loss.item() - bce_loss.item()) < 1e-5, (
        f"Expected focal (gamma=0, alpha=None) == BCE, "
        f"got focal={focal_loss.item():.6f} bce={bce_loss.item():.6f}"
    )


def test_focal_less_than_bce_on_easy_examples():
    """On easy (confident, correct) predictions focal loss must be < BCE."""
    # Model is very confident and correct: large positive logits where target=1
    logits  = torch.full((8, 50), 5.0)   # very confident positive
    targets = torch.ones(8, 50)           # all labels are 1

    bce_loss   = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
    focal_loss = focal_loss_with_logits(logits, targets, gamma=2.0, alpha=None)

    assert focal_loss.item() < bce_loss.item(), (
        f"Expected focal < BCE on easy examples, "
        f"got focal={focal_loss.item():.6f} bce={bce_loss.item():.6f}"
    )


def test_focal_returns_scalar():
    """focal_loss_with_logits must return a scalar tensor."""
    logits  = torch.randn(4, 50)
    targets = torch.zeros(4, 50)
    loss = focal_loss_with_logits(logits, targets)
    assert loss.shape == torch.Size([]), f"Expected scalar, got shape {loss.shape}"


def test_focal_alpha_scales_loss():
    """alpha=0.5 should produce a different (scaled) loss than alpha=None."""
    torch.manual_seed(1)
    logits  = torch.randn(8, 50)
    targets = (torch.rand(8, 50) > 0.86).float()

    loss_no_alpha   = focal_loss_with_logits(logits, targets, gamma=2.0, alpha=None)
    loss_with_alpha = focal_loss_with_logits(logits, targets, gamma=2.0, alpha=0.5)

    assert abs(loss_no_alpha.item() - loss_with_alpha.item()) > 1e-6, (
        "Expected alpha to change the loss value"
    )
```

- [ ] **Step 2: Run tests — verify they all FAIL (module not found)**

```bash
cd "d:/AI - 2026/CLAUDEAI2026/lottery-nn"
C:\Python314\python.exe -m pytest tests/unit/test_focal_loss.py -v --tb=short --no-header -p no:warnings
```

Expected: 4 errors — `ModuleNotFoundError: No module named 'src.focal_loss'`

- [ ] **Step 3: Implement `src/focal_loss.py`**

Create `src/focal_loss.py`:

```python
"""
Focal loss for multi-label binary classification.

Focal loss down-weights easy negatives via a modulating factor (1-p)^gamma,
sharpening the training signal on hard examples (the drawn numbers).

Reference: Lin et al., "Focal Loss for Dense Object Detection" (RetinaNet), 2017.
"""

import torch
import torch.nn.functional as F


def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = None,
) -> torch.Tensor:
    """
    Compute focal loss for multi-label binary targets.

    Parameters
    ----------
    logits  : (batch, num_classes) raw model output (before sigmoid)
    targets : (batch, num_classes) binary float targets in {0, 1}
    gamma   : modulating exponent. 0.0 = standard BCE. Default 2.0.
    alpha   : positive-class weight in [0, 1], or None to skip alpha weighting.
              Default None.

    Returns
    -------
    Scalar mean loss over the batch and all classes.
    """
    # Numerically stable per-element BCE (same as BCEWithLogitsLoss reduction='none')
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    # p_t = predicted probability for the true class
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)

    focal_weight = (1.0 - p_t) ** gamma

    if alpha is not None:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        focal_weight = alpha_t * focal_weight

    return (focal_weight * bce).mean()
```

- [ ] **Step 4: Run tests — verify all 4 PASS**

```bash
C:\Python314\python.exe -m pytest tests/unit/test_focal_loss.py -v --tb=short --no-header -p no:warnings
```

Expected:
```
tests/unit/test_focal_loss.py::test_focal_equals_bce_when_gamma_zero PASSED
tests/unit/test_focal_loss.py::test_focal_less_than_bce_on_easy_examples PASSED
tests/unit/test_focal_loss.py::test_focal_returns_scalar PASSED
tests/unit/test_focal_loss.py::test_focal_alpha_scales_loss PASSED
4 passed
```

- [ ] **Step 5: Commit**

```bash
git add src/focal_loss.py tests/unit/test_focal_loss.py
git commit -m "feat: implement focal_loss_with_logits with unit tests"
```

---

### Task 3: Swap loss in `src/train.py`

**Files:**
- Modify: `src/train.py`

- [ ] **Step 1: Add import at the top of `src/train.py`**

The current imports end at line 19:
```python
import config
from src.model import build_model, count_params
```

Add one line after them:
```python
import config
from src.model import build_model, count_params
from src.focal_loss import focal_loss_with_logits
```

- [ ] **Step 2: Remove the `main_criterion` variable from `train()`**

Currently at line 61:
```python
    main_criterion = nn.BCEWithLogitsLoss()
    bonus_criterion = nn.CrossEntropyLoss() if has_bonus else None
```

Replace with:
```python
    bonus_criterion = nn.CrossEntropyLoss() if has_bonus else None
```

(`main_criterion` is removed entirely — focal loss will be called directly.)

- [ ] **Step 3: Update the `_run_epoch` signature and call site**

Currently `_run_epoch` is called at lines 72-73 as:
```python
        tr_loss = _run_epoch(model, train_loader, optimizer, main_criterion, bonus_criterion, train=True)
        val_loss = _run_epoch(model, val_loader, None, main_criterion, bonus_criterion, train=False)
```

Replace both with:
```python
        tr_loss = _run_epoch(model, train_loader, optimizer, bonus_criterion, train=True)
        val_loss = _run_epoch(model, val_loader, None, bonus_criterion, train=False)
```

- [ ] **Step 4: Update `_run_epoch` definition and body**

Currently at line 118:
```python
def _run_epoch(model, loader, optimizer, main_crit, bonus_crit, train: bool) -> float:
```

Replace signature and the loss line inside it:

```python
def _run_epoch(model, loader, optimizer, bonus_crit, train: bool) -> float:
```

Inside the loop, currently at line 130:
```python
            loss = main_crit(main_logits, y_main)
```

Replace with:
```python
            loss = focal_loss_with_logits(
                main_logits, y_main,
                gamma=config.FOCAL_GAMMA,
                alpha=config.FOCAL_ALPHA,
            )
```

- [ ] **Step 5: Run the full unit test suite to confirm nothing regressed**

```bash
C:\Python314\python.exe -m pytest tests/unit/ -v --tb=short --no-header -p no:warnings
```

Expected: all tests PASS (no failures).

- [ ] **Step 6: Commit**

```bash
git add src/train.py
git commit -m "feat: use focal loss in single-lottery training loop"
```

---

### Task 4: Swap loss in `main_swarm.py`

**Files:**
- Modify: `main_swarm.py`

- [ ] **Step 1: Add import near the top of `main_swarm.py`**

The current `src` imports (around lines 54-66) end with:
```python
from src.preprocessing_swarm import (
    LOTTERY_CONFIGS,
    build_all_lottery_data,
    get_last_window,
    split,
)
```

Add one line after:
```python
from src.focal_loss import focal_loss_with_logits
```

- [ ] **Step 2: Update `_run_joint_epoch` — remove local `main_crit` and swap the loss call**

Currently at line 271-275 inside `_run_joint_epoch`:
```python
def _run_joint_epoch(model, loaders: dict, optimizer, train: bool) -> float:
    ...
    main_crit  = nn.BCEWithLogitsLoss()
    bonus_crit = nn.CrossEntropyLoss()
```

Remove the `main_crit` line:
```python
def _run_joint_epoch(model, loaders: dict, optimizer, train: bool) -> float:
    ...
    bonus_crit = nn.CrossEntropyLoss()
```

Inside the batch loop, currently at line 301:
```python
                loss = main_crit(main_logits, y_main)
```

Replace with:
```python
                loss = focal_loss_with_logits(
                    main_logits, y_main,
                    gamma=config.FOCAL_GAMMA,
                    alpha=config.FOCAL_ALPHA,
                )
```

Also add `import config` at the top of `main_swarm.py` if it is not already imported (check the existing imports — it may already be there via `sys.path` but should be explicit):

At the top-level imports section add:
```python
import config
```

- [ ] **Step 3: Run the integration tests to confirm swarm pipeline still works**

```bash
C:\Python314\python.exe -m pytest tests/integration/ -v --tb=short --no-header -p no:warnings
```

Expected: all integration tests PASS.

- [ ] **Step 4: Smoke-test joint training with 5 epochs**

```bash
C:\Python314\python.exe main_swarm.py joint-train --epochs 5
```

Expected: prints epoch loss lines, saves checkpoint, no exceptions.

- [ ] **Step 5: Commit**

```bash
git add main_swarm.py
git commit -m "feat: use focal loss in swarm joint-training loop"
```

---

### Task 5: Full test run and final commit

**Files:** none (validation only)

- [ ] **Step 1: Run the complete test suite**

```bash
C:\Python314\python.exe -m pytest tests/ -v --tb=short --no-header -p no:warnings
```

Expected: all tests PASS.

- [ ] **Step 2: Verify config imports cleanly in both entry points**

```bash
C:\Python314\python.exe -c "import config; print('FOCAL_GAMMA:', config.FOCAL_GAMMA, 'FOCAL_ALPHA:', config.FOCAL_ALPHA)"
```

Expected:
```
FOCAL_GAMMA: 2.0 FOCAL_ALPHA: 0.25
```

- [ ] **Step 3: Final commit tagging the feature complete**

```bash
git add -A
git commit -m "feat: focal loss complete -- replaces BCEWithLogitsLoss on main-number head"
```
