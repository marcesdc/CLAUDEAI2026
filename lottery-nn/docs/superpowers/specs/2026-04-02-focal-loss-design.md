# Focal Loss — Design Spec
Date: 2026-04-02

## Problem

`BCEWithLogitsLoss` treats all 50 ball positions equally. Only 7 of 50 are drawn per
game, so 86% of labels are 0. The loss is dominated by easy negatives, giving the model
a weak training signal on the numbers that actually matter.

## Solution

Replace `BCEWithLogitsLoss` on the main-number head with **focal loss**, which
down-weights easy negatives via a modulating factor `(1 - p)^gamma`. This shifts
training focus toward hard positives — the drawn numbers.

## Architecture

### New file: `src/focal_loss.py`

Single public function:

```python
focal_loss_with_logits(logits, targets, gamma=2.0, alpha=0.25) -> Tensor
```

- Numerically stable: uses `sigmoid` + manual BCE formulation (not `log(sigmoid)` directly)
- `gamma`: modulating exponent. 0 = standard BCE. 2.0 is the standard default.
- `alpha`: class-balance weight for positives (drawn numbers). 0.25 is standard default.
- Returns scalar loss (mean over batch and positions), matching BCEWithLogitsLoss API.

### Config additions (`config.py`)

```python
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25
```

### Changes to `src/train.py`

- Remove `main_criterion = nn.BCEWithLogitsLoss()`
- In `_run_epoch`: replace `main_crit(main_logits, y_main)` with
  `focal_loss_with_logits(main_logits, y_main, config.FOCAL_GAMMA, config.FOCAL_ALPHA)`

### Changes to `main_swarm.py`

- In `_run_joint_epoch` (line 274): same swap — remove local `BCEWithLogitsLoss`,
  call `focal_loss_with_logits` instead.

### New test file: `tests/unit/test_focal_loss.py`

Two tests:
1. `test_focal_equals_bce_when_gamma_zero`: with `gamma=0, alpha=0.5`, focal loss
   must equal BCEWithLogitsLoss within 1e-5.
2. `test_focal_less_than_bce_on_easy_examples`: construct logits where the model
   is very confident and correct; focal loss must be less than BCE loss (easy
   examples are down-weighted).

## Parameters

| Param | Default | Notes |
|---|---|---|
| `FOCAL_GAMMA` | 2.0 | Standard from Lin et al. (RetinaNet). Higher = stronger down-weighting of easy negatives. |
| `FOCAL_ALPHA` | 0.25 | Positive-class weight. 0.25 is standard for imbalanced binary tasks. |

## Files Touched

| File | Change |
|---|---|
| `src/focal_loss.py` | New — implements `focal_loss_with_logits` |
| `config.py` | Add `FOCAL_GAMMA`, `FOCAL_ALPHA` |
| `src/train.py` | Swap loss function in `_run_epoch` |
| `main_swarm.py` | Swap loss function in `_run_joint_epoch` |
| `tests/unit/test_focal_loss.py` | New — 2 unit tests |

## Out of Scope

- Bonus head: remains `CrossEntropyLoss` (single label, no imbalance problem)
- `LINES_PER_PLAY`, prediction logic, bandit, feedback — untouched
