# program.md — Research Instructions for lottery-nn

This file guides the autonomous search and helps Claude understand the research
goals when reviewing results or launching a new run.

---

## Project context

**Game**: Lotto Max — pick 7 numbers from 1–50, plus a bonus ball (1–50).
**Goal**: Train a neural network that learns statistical patterns in draw history
and produces better-than-random ticket suggestions over time.
**Data**: `data/draws.csv` (real draws) or synthetic fallback (2000 draws).
**Normal routine**: after each real draw → `log` → `predict`.

---

## How to launch a search

```bash
# Default: 8 hours, 3-min budget per experiment (~160 experiments overnight)
C:\Python314\python.exe autorun.py

# Shorter daytime run
C:\Python314\python.exe autorun.py --hours 2 --budget 120

# Reproducible run
C:\Python314\python.exe autorun.py --hours 8 --seed 42
```

Results land in `data/experiment_log.csv`.
Best checkpoint: `models/auto_best.pt`.
Best config: `models/auto_best_config.json`.

---

## Search space

| Parameter      | Values searched                         | Notes                               |
|----------------|-----------------------------------------|-------------------------------------|
| `arch`         | transformer, lstm                       | Two backbone families               |
| `EMBED_DIM`    | 32, 64, 128, 256                        | Transformer only                    |
| `NUM_HEADS`    | 2, 4, 8 (auto-valid)                    | Must divide EMBED_DIM               |
| `NUM_LAYERS`   | 1–5                                     | Depth of encoder/LSTM               |
| `FF_DIM`       | 128, 256, 512                           | Transformer FFN width               |
| `DROPOUT`      | 0.0, 0.1, 0.2, 0.3                      | Regularization                      |
| `LR`           | 1e-4, 3e-4, 1e-3, 3e-3                  | AdamW learning rate                 |
| `WEIGHT_DECAY` | 0, 1e-5, 1e-4, 1e-3                     | L2 regularization                   |
| `BATCH_SIZE`   | 16, 32, 64, 128                         |                                     |
| `SEQUENCE_LEN` | 5, 10, 15, 20, 30                       | History window fed to the model     |
| `PATIENCE`     | 8, 12, 15                               | Early-stopping patience (epochs)    |
| `EPOCHS`       | 50, 75, 100                             | Max training epochs per experiment  |

Strategy: random search (uniform sampling). ~160 experiments per 8-hour run.

---

## Metric

**`val_loss`** — validation BCE loss on held-out draws (chronological split).
Lower is better. Improvement over the default config baseline (~0.35–0.45 range
on synthetic data) is the target.

Note: val_loss measures fit to historical patterns, not lottery prediction
accuracy (which is fundamentally random). It is a proxy for "the model learned
something") — improvement should be understood in that context.

---

## Hypotheses to investigate

1. **Transformer vs LSTM**: Does the attention mechanism capture draw-history
   patterns better than the LSTM baseline on this small dataset?

2. **Sequence length**: Is a longer history window (20–30 draws) helpful, or
   does it hurt by adding noise on limited real data?

3. **Model size**: Is underfitting or overfitting the bigger problem? (Small
   models: EMBED_DIM=32-64 vs larger: 128-256)

4. **Dropout**: Small dataset (~few hundred real draws) may benefit from
   heavier dropout (0.2–0.3) to prevent overfitting.

5. **Learning rate**: Does a lower LR (1e-4) with more patience outperform the
   default 1e-3 on small data?

---

## Reading the experiment log

```bash
# View top 10 experiments by val_loss (PowerShell)
Import-Csv data\experiment_log.csv | Sort-Object val_loss | Select-Object -First 10 | Format-Table

# Or ask Claude to analyze data/experiment_log.csv directly
```

When reviewing results, look for:
- Which `arch` family wins most often in the top 20?
- Is there a `SEQUENCE_LEN` sweet spot?
- Does `DROPOUT` > 0.1 consistently help (suggests overfitting)?
- Any `EMBED_DIM` / `NUM_LAYERS` combinations dominating the top results?

---

## Applying the best config

After a search run, the best config is in `models/auto_best_config.json`.
To apply it permanently, update `config.py` with those values and retrain:

```bash
C:\Python314\python.exe main.py train --arch transformer
```

Or use `auto_best.pt` directly as the checkpoint for predictions:

```bash
C:\Python314\python.exe main.py predict --checkpoint models/auto_best.pt
```

---

## Hardware notes

- CPU: AMD Ryzen 7 3800X (16 logical cores), 64 GB RAM, no GPU confirmed yet.
- Device auto-detected in `train.py` (CUDA if available, else CPU).
- On CPU, each experiment typically completes in 1–3 minutes for the current
  model sizes — well within the 3-min budget.
- If a GPU is added later: reduce `--budget` to 60s and increase `--hours`
  to search more configurations overnight.
