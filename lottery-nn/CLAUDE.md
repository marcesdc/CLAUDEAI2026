# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Use `C:\Python314\python.exe` (Python 3.14 — the environment where `torch` is installed). `python` on PATH resolves to 3.12 which lacks the dependencies.

```bash
# Install dependencies
pip install -r requirements.txt          # pip is already Python 3.14's pip

# Generate synthetic data (when no real CSV is available)
C:\Python314\python.exe main.py data

# Train the model
C:\Python314\python.exe main.py train
C:\Python314\python.exe main.py train --arch lstm --epochs 50

# Generate play suggestions (requires a trained checkpoint)
C:\Python314\python.exe main.py predict --plays 10

# Log actual draw result, score last prediction, and retrain
C:\Python314\python.exe main.py log --date 2026-03-25 --numbers 3 6 12 21 28 35 41 --bonus 47

# Score only, skip retraining
C:\Python314\python.exe main.py log --date 2026-03-25 --numbers 3 6 12 21 28 35 41 --bonus 47 --no-retrain

# Evaluate on the test split
C:\Python314\python.exe main.py evaluate
```

All commands accept `--arch transformer|lstm` and `--checkpoint PATH`.

**Normal routine after each draw:**
```
log → predict
```
Only run `train` separately when adding a large batch of historical data at once.

## Architecture

**Entry point**: `main.py` — dispatches `train / predict / evaluate / data` subcommands.

**Data flow**:
```
data/draws.csv
  → src/data_loader.py   (load or generate synthetic draws)
  → src/preprocessing.py (sliding-window multi-hot features + rolling frequency)
  → src/train.py         (training loop, early stopping, checkpoint)
  → src/predict.py       (temperature-sampled ticket generation)
  → src/evaluate.py      (hit-rate metrics, training curve & frequency plots)
```

**Models** (`src/model.py`):
- `LotteryTransformer` — transformer encoder with attention pooling, two heads (main + bonus). Default architecture.
- `LotteryLSTM` — bidirectional LSTM baseline for comparison.
- Both share the same `(main_logits, bonus_logits)` forward signature.

**Feature representation** (`src/preprocessing.py`):
- Each draw is encoded as a multi-hot vector of length `MAIN_MAX`.
- A rolling-frequency vector (mean over the window) is concatenated, so each time-step has dimension `2 * MAIN_MAX`.
- Sequences of length `SEQUENCE_LEN` (default 20) form each sample.

**Loss**: `BCEWithLogitsLoss` for main numbers (multi-label), `CrossEntropyLoss` for bonus (weighted 0.3×).

## Configuration

All hyperparameters live in `config.py`. Key settings:

| Variable | Default | Purpose |
|---|---|---|
| `LOTTERY` | 7 from 50 + bonus | Ball counts and pool sizes |
| `LINES_PER_PLAY` | 3 | Lines per play (ticket) |
| `SEQUENCE_LEN` | 20 | History window fed to the model |
| `EPOCHS` / `PATIENCE` | 100 / 15 | Training schedule |
| `TEMPERATURE` | 1.2 | Prediction sampling diversity |
| `NUM_PLAYS` | 5 | Plays generated per predict call |

## Real Data

Place the Excel file at `data/draws.xlsx` (or `data/draws.csv`).  Required columns:

```
date  n1  n2  n3  n4  n5  n6  n7  bonus
```

- `n1`–`n7`: main numbers, each 1–50
- `bonus`: bonus ball (adjust `LOTTERY["bonus_max"]` in `config.py` to match your game's range)
- `date`: any format pandas can parse

Without this file, synthetic random draws are generated automatically.

A **play** = 3 independent lines of 7 numbers + 1 bonus suggestion.

## Statistical Analysis

`src/analysis.py` provides standalone helpers (`frequency_table`, `hot_cold`, `pair_frequency`, `gap_analysis`) that can be imported in notebooks without running the full pipeline.

## Feedback Loop

`src/feedback.py` manages the draw-by-draw learning cycle:
- `log_draw()` — appends a new result to `draws.csv` with date validation
- `save_prediction()` — auto-called by `predict`, saves plays to `data/predictions_log.csv`
- `score_last_prediction()` — scores the saved prediction against the actual draw
- `recency_weights()` — exponential weights (decay=0.92) so recent draws count more during retraining

`data/score_log.csv` accumulates hit scores over time for tracking model improvement.

**Known data quirk**: `draws.csv` originally uses `number1`–`number7` column names. `data_loader._normalize_columns()` renames these to `n1`–`n7` on load. New rows logged via `feedback.log_draw()` are written as `n1`–`n7` directly.

## Agent Interface

Three agent modes live in `agent/`, all launched via `agent/run.py`:

```bash
# Conversational + analysis (natural language)
C:\Python314\python.exe agent/run.py chat

# Check OLG website for new draw once (uses Playwright MCP)
C:\Python314\python.exe agent/run.py monitor

# Poll OLG every 6 hours automatically
C:\Python314\python.exe agent/run.py watch
```

**chat** — maintains session context across turns; can log draws, generate predictions, and answer analysis questions in plain English. Uses `Bash`, `Read`, `Glob`, `Grep` tools.

**monitor / watch** — uses `@playwright/mcp@latest` (Node.js) to render the OLG JavaScript page and extract the latest Lotto Max draw. Compares draw date with last row in `draws.csv`; auto-logs and retrains if new. OLG page: https://www.olg.ca/en/lottery/winning-numbers-results.html#game-item-lottomax

**Dependencies**: `pip install claude-agent-sdk anyio` + Node.js/npx for Playwright.

## Applied Learning

Lessons from real usage — updated whenever something breaks, causes confusion, or a fix proves itself.

- **Python executable**: `python` on PATH is 3.12 and lacks `torch`. Always use `C:\Python314\python.exe`. `pip` already points to 3.14 so installs go to the right place.
- **matplotlib not installed by default**: First `train` run failed with `ModuleNotFoundError: No module named 'matplotlib'`. Fixed by running `pip install matplotlib seaborn tqdm`.
- **Unicode in print() crashes on Windows**: Characters like `→` and `✓` raise `UnicodeEncodeError` on Windows terminals using cp1252 encoding. Use ASCII alternatives (`->`, `[saved]`) in all print statements.
- **`log` corrupts draws.csv if columns aren't normalized first**: `log_draw()` originally read the CSV without renaming columns, then concatenated a row with `n1`–`n7` keys against a file with `number1`–`number7` headers — producing duplicate columns and a pandas reindex crash. Fix: call `_normalize_columns()` inside `log_draw()` before concat.
- **Date typos cause silent bad data**: User typed `2026-06-2026` instead of `2026-03-06`. The bad row was written to `draws.csv` before the crash, requiring manual cleanup. Fix: added `datetime.strptime` validation in `log_draw()` to reject malformed dates immediately.
- **`NUM_SUGGESTIONS` renamed to `NUM_PLAYS`**: When the user edited `config.py` directly, they reverted to `NUM_SUGGESTIONS`. The rest of the code references `config.NUM_PLAYS` — always check config keys match after manual edits.
- **OLG lottery page requires JavaScript** — plain `WebFetch` returns no numbers. The backend API (`gateway.www.olg.ca`, `gateway.can2.mkodo.io`) requires auth. Solution: Playwright MCP renders the page in a real browser.
- **`LINES_PER_PLAY` and `"name"` key can go missing from config**: User's manual edit dropped both. Code in `predict.py` and `evaluate.py` depends on them — restore if missing.
