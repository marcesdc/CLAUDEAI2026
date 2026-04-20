# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project IS and IS NOT

**This project is NOT a lottery-number predictor.** Lottery draws are i.i.d. uniform random variables. `I(past_draws; next_draw) = 0`. No model can predict which numbers come out of the machine.

**This project IS a ticket-portfolio optimizer.** It models *human ticket-pick behavior* (which combinations players over-choose: birthdays 1-31, lucky 7s, patterns) to find combinations that, if they win, would have to be split with fewer other winners. Win probability stays at chance; **expected payout per dollar** rises 5-30% (literature: Cook & Clotfelter 1993; Simon 1999; DeBoer 1990; Skiena 2004).

If the headline backtest metric (realized EV/$ vs uniform-random baseline) does NOT show a statistically significant lift, that is a valid null result -- record it honestly in Applied Learning. The project's framing is defensible whether or not the empirical lift is large on Canadian lotteries.

## Commands

Use `C:\Python314\python.exe` (Python 3.14). `python` on PATH resolves to 3.12 and lacks deps.

```bash
# Install dependencies
pip install -r requirements.txt

# Show data + state summary
C:\Python314\python.exe main.py status

# Fit the popularity model (needs both draws_*.csv AND payouts_*.csv with rows)
C:\Python314\python.exe main.py fit --lottery lottomax
C:\Python314\python.exe main.py fit --lottery 649
C:\Python314\python.exe main.py fit --lottery dailygrand

# Get the recommended combination for the next draw (runs qa_gate first)
C:\Python314\python.exe main.py optimize --lottery lottomax
C:\Python314\python.exe main.py optimize --lottery 649
C:\Python314\python.exe main.py optimize --lottery dailygrand

# Log a real draw (winning numbers only)
C:\Python314\python.exe main.py log --lottery 649 --date 2026-04-17 \
    --numbers 3 7 18 24 31 42 --bonus 15

# Pass --jackpot and --tickets-sold to optimize for a specific next-draw scenario
C:\Python314\python.exe main.py optimize --lottery 649 --jackpot 5000000 --tickets-sold 4000000

# Backtest realized EV/$ vs uniform-random baseline (rolling-origin)
C:\Python314\python.exe main.py backtest --lottery lottomax

# Scrape the latest draw + payouts from OLG (Playwright via MCP)
C:\Python314\python.exe agent/run.py scrape --lottery lottomax
```

All commands accept `--lottery {lottomax,649,dailygrand}`.

## Architecture

**Entry point**: `main.py` -- dispatches `status / scrape / fit / optimize / log / backtest` subcommands.

**Data flow**:
```
OLG website
  -> agent/olg_scraper.py     (Playwright MCP, extracts numbers + payout tiers)
  -> data/draws_*.csv         (winning numbers per draw)
  -> data/payouts_*.csv       (jackpot + winner counts per tier per draw)
  -> src/data_loader.py       (load + validate)
  -> src/popularity_model.py  (fit P(player picks combination C))
  -> src/ev_optimizer.py      (argmax over C of E[payout(C) / cost])
  -> data/portfolio_log.csv   (chosen combinations, predicted EV)
  -> src/backtest.py          (rolling-origin realized EV/$ vs baseline)
  -> data/realized_ev_log.csv (after-the-fact dollars won vs spent)
```

**Core modules** (`src/`):
- `popularity_model.py` -- small (~10-param) parametric model of P(combination chosen by random player). Fit by Poisson-NLL matching of observed winner counts per prize tier per draw, with Gaussian regularization to literature priors (`POPULARITY_PRIORS` in `config.py`).
- `ev_optimizer.py` -- For each candidate combination C, compute `E[reward(C)] = sum_k P_win_k * effective_payout_k - ticket_cost`. Because `popularity_model` is fully factored (`weight(C) = prod_i w_i`), the min-popularity combination is the closed-form `main_count` smallest-weight numbers -- no search is needed. `best_combination` uses `np.argsort(w, kind="stable")[:main_count]`. The non-jackpot pari-mutuel tiers still use a uniform-player split approximation.
- `bandit.py` -- Thompson sampling over portfolio STRATEGIES (`anti_popular`, `uniform_random`, `max_ev_search`). Beta posteriors track "did this strategy beat baseline EV/$ on this draw". Reproducible via `bandit.set_seed()`.
- `backtest.py` -- Rolling-origin: fit on draws `[0, t]`, optimize for `t+1`, score against actual; slide forward. Output realized EV/$ per strategy.
- `data_loader.py` -- Validates row counts, sorts main columns ascending, normalizes `grand` -> `bonus` for Daily Grand internally.
- `qa_gate.py`, `utils.py` -- Forked from lottery-nn unchanged.

**Scraper** (`agent/olg_scraper.py`):
- Forked + extended from `lottery-nn/agent/monitor.py`. Same Playwright MCP pattern; the prompt is rewritten per lottery to extract:
  - Winning numbers (DATE, NUMBERS, BONUS)
  - Jackpot for that draw (JACKPOT)
  - Winner counts per prize tier (PAYOUTS block)
- Falls back to manual CLI entry if scraping fails.

## Configuration

All hyperparameters live in `config.py`. Key settings:

| Variable | Default | Purpose |
|---|---|---|
| `LOTTERY_RULES` | (3 lotteries) | main_count, main_max, bonus, ticket_cost per lottery |
| `PRIZE_TIERS` | OLG schedule | `(main_match, bonus_match, payout_cad)` per tier |
| `POPULARITY_PRIORS` | 8 weights | Literature-derived starting point for the popularity model |
| `POPULARITY_PRIOR_SIGMA` | 0.20 | Gaussian regularization std around priors |
| `STRATEGIES` | 3 | Strategy bandit registry |
| `LINES_PER_PLAY` | 1 | Tickets recommended per draw, per lottery |
| `SEARCH_RESTARTS` | 1000 | Local-search restarts in `ev_optimizer` |
| `HELDOUT_FRACTION` | 0.25 | Last 25% of draws reserved for held-out NLL |
| `DEFAULT_TICKETS_SOLD` | per-lottery | Fallback when OLG doesn't publish total sales |
| `SEED` | 42 | RNG seed for `bandit` and any sampling |

Verify `PRIZE_TIERS` against `https://www.olg.ca` periodically -- prize structures get restructured.

## Real Data

OLG only retains ~1 year of historical results on the public results page (because tickets expire after 1 year). Scrape on every draw to build history forward. The user can also enter draws manually via `main.py log`.

CSV schemas:

```
data/draws_lottomax.csv
  date, n1, n2, n3, n4, n5, n6, n7, bonus

data/draws_649.csv
  date, n1, n2, n3, n4, n5, n6, bonus

data/draws_dailygrand.csv
  date, n1, n2, n3, n4, n5, grand

data/payouts_*.csv  (one row per draw per prize tier)
  date, lottery, jackpot, tier_main, tier_bonus, n_winners, payout_per_winner
```

## Statistical Reality Check

For each lottery, the win probability of any single ticket is fixed by combinatorics:

| Lottery | Jackpot odds | C(N, k) |
|---|---|---|
| LottoMax (7/52) | 1 in 133,784,560 | 133.78M |
| 6/49 (6/49)     | 1 in 13,983,816 | 13.98M |
| Daily Grand (5/49 + 1/7) | 1 in 13,348,188 | 13.35M |

This system does not change those probabilities. It only changes the **expected dollar payoff conditional on winning**, by recommending combinations that fewer other players are likely to have picked.

## Feedback Loop

`main.py log` accepts both the winning numbers AND the payout tier breakdown for a draw, appending to both `draws_*.csv` and `payouts_*.csv`. `main.py score` reads the most recent `portfolio_log.csv` row, computes realized payout from the actual draw, appends to `realized_ev_log.csv`, and updates the strategy bandit posteriors via `bandit.update`.

## QA Agents

Three Claude Code sub-agents live at `.claude/agents/`. Forked from lottery-nn with paths and scope updated for this project.

```
@q_a_lead I just finished implementing popularity_model.py
```

The lead agent launches `q_a1` (code review + tests) and `q_a2` (perf + edge cases) in parallel, returns GREEN/RED.

**MANDATORY**: invoke `@q_a_lead` after every code change to `src/` or root scripts. Don't tell the user a task is done until verdict is GREEN.

## Applied Learning

Lessons from real usage -- updated whenever something breaks, causes confusion, or a fix proves itself.

- **Python executable**: `python` on PATH is 3.12 and lacks deps. Always use `C:\Python314\python.exe`. `pip` already points to 3.14 so installs go to the right place.
- **Unicode in print() crashes on Windows**: cp1252 encoding. Use ASCII alternatives (`->`, `[saved]`) in all print statements.
- **OLG retention**: ~1 year of history. Scrape continuously to build forward; do not assume historical depth.
- **Daily Grand uses column `grand` in CSV**, not `bonus`. `data_loader.py` renames it internally.
- **Factored popularity model has a closed-form optimizer**: `weight(C) = prod_i w_i` means the min-popularity combination is the `main_count` numbers with the smallest per-number weights. No Monte-Carlo search required -- `np.argsort` is optimal. The original plan specified 1000-restart local search; that's unnecessary under this factorization.
- **PRIZE_TIERS must be disjoint events**: early config had both `(5, None, None)` and `(5, True, None)` for 649, which double-counts the 5+Bonus case. Fixed to `(5, True, None)` and `(5, False, None)`. Any tier with `tier_bonus=None` means "bonus irrelevant for this tier" -- check that it does not overlap with a `tier_bonus=True` or `tier_bonus=False` entry at the same `tier_main`.
- **Honest null result**: if backtest shows no statistical EV lift, document it here -- do not bury it.
- **LottoMax rule change (2026)**: the main pool expanded from 1-50 to 1-52. `C(52,7) = 133,784,560` jackpot odds. `config.LOTTERY_RULES["lottomax"]` now has `main_max=52, bonus_max=52`. When scraping OLG, numbers up to 52 are valid. Any hardcoded `50` in tests or docs that refers to LottoMax must be updated.
- **Scraper requires `permission_mode="bypassPermissions"`**: the spawned SDK process hits "tool not authorized" errors otherwise. Set explicit `allowed_tools=["mcp__playwright__browser_navigate", ...]` alongside the bypass so only the scraper-needed tools are enabled.
- **Scraper parsing is fragile for odd tiers**: on Daily Grand, the jackpot value parsed was $1,000 (the secondary per-day prize display) rather than the $7M grand-prize annuity; a `(0, Y)` tier row also appeared. Use `DAILYGRAND_GRAND_CASH_VALUE` for EV math and treat the scraper jackpot value as advisory for Daily Grand.
- **Real-data backtest is blocked on history**: OLG only retains ~1 year on the public page. Scraping seeds one draw per run; `backtest.run` needs > `min_train=20` draws, so a meaningful realized-EV backtest requires 2-3 months of continuous scraping before it can run.
- **LottoMax and 6/49 bonus is machine-drawn, not player-picked**: Players select only the 7 (LottoMax) or 6 (6/49) main numbers. The bonus ball is drawn by the lottery at draw time from the remaining pool -- it never appears on the player's ticket. Consequences for this codebase: (a) `popularity_model` scores only the main-number combination -- it does not weight the bonus; (b) `ev_optimizer.best_combination` returns only main numbers, no bonus recommendation; (c) `bonus_match_prob` for LottoMax/6/49 treats the bonus as drawn from `main_max - main_count` remaining numbers (see `src/ev_optimizer.py:58`); (d) the `bonus` column in `draws_*.csv` stores the DRAWN bonus (needed to compute realized tier matches in backtest), not a player pick. Daily Grand is different: the Grand Number (1-7) IS player-picked and is drawn from an independent pool.

## Reusable forks (provenance)

These files were forked from `lottery-nn/` at the start of this project:

| File | Source | Modification |
|---|---|---|
| `src/utils.py` | `lottery-nn/src/utils.py` | unchanged |
| `src/qa_gate.py` | `lottery-nn/src/qa_gate.py` | unchanged |
| `src/bandit.py` | `lottery-nn/src/bandit.py` | reframed for strategies; `np.random.default_rng` for reproducibility |
| `agent/olg_scraper.py` | `lottery-nn/agent/monitor.py` | extended prompt for payouts |
| `agent/run.py` | `lottery-nn/agent/run.py` | adapted CLI subcommands |
| `tests/conftest.py` | `lottery-nn/tests/conftest.py` | fork pattern + payout fixtures |
| `.claude/agents/*` | `lottery-nn/.claude/agents/*` | path/scope updates |

Do NOT import from `lottery-nn/` -- the projects are independent. The fork is the contract.
