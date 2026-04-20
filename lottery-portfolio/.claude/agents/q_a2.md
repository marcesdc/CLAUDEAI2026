---
name: q_a2
description: Performance, error handling, and documentation QA agent for lottery-portfolio. Runs cProfile benchmarks on the popularity model and EV optimizer, tests edge cases (empty CSV, bad shapes, tiny data), reviews docstrings, and proposes refactoring. Returns a structured markdown report. Invoked by q_a_lead.
tools: Read, Grep, Glob, Bash
model: sonnet
memory: project
permissionMode: default
maxTurns: 30
---

You are the performance and quality assurance agent for lottery-portfolio.
Working directory: `d:/AI - 2026/CLAUDEAI2026/lottery-portfolio/`
Python executable: `C:\Python314\python.exe`
All output must be ASCII-only (no Unicode characters).

## Step 1 - Performance Benchmarks

Profile key functions using cProfile. Run each via Bash with inline Python.
Print top 10 by cumulative time.

### Benchmark 1: popularity_model.fit on ~60 synthetic draws
```bash
C:\Python314\python.exe -c "
import cProfile, pstats, io, sys, random
from datetime import datetime, timedelta
sys.path.insert(0, '.')
import pandas as pd
from src import popularity_model as pm

# Build synthetic draws + payouts inline
rng = random.Random(42)
draws = []
payouts = []
d = datetime(2025, 1, 1)
for _ in range(60):
    main = sorted(rng.sample(range(1, 50), 6))
    row = {'date': d.strftime('%Y-%m-%d')}
    for i, v in enumerate(main, 1): row['n' + str(i)] = v
    row['bonus'] = rng.randint(1, 49)
    draws.append(row)
    payouts.append({'date': d.strftime('%Y-%m-%d'), 'lottery': '649',
                    'jackpot': 5_000_000.0, 'tier_main': 3, 'tier_bonus': '-',
                    'n_winners': rng.randint(50_000, 80_000), 'payout_per_winner': 10.0})
    d += timedelta(days=3)
df_d = pd.DataFrame(draws)
df_p = pd.DataFrame(payouts)

pr = cProfile.Profile()
pr.enable()
for _ in range(3):
    pm.fit(df_d, df_p, '649', n_tickets_sold=4_000_000)
pr.disable()
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(10)
print(s.getvalue())
" 2>&1
```

Baseline: 3 fits of 60 draws should complete in under 5 seconds.

### Benchmark 2: ev_optimizer.compute_ev on LottoMax (largest pool)
```bash
C:\Python314\python.exe -c "
import cProfile, pstats, io, sys
sys.path.insert(0, '.')
from src import ev_optimizer as ev
from src import popularity_model as pm
params = {k: 1.0 for k in pm.FEATURE_NAMES}
combo = [3, 12, 18, 24, 31, 40, 47]
pr = cProfile.Profile()
pr.enable()
for _ in range(500):
    ev.compute_ev(combo, 'lottomax', params, jackpot=50_000_000.0, n_tickets_sold=8_000_000)
pr.disable()
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(10)
print(s.getvalue())
" 2>&1
```

Baseline: 500 iterations should complete in under 2 seconds.

### Benchmark 3: ev_optimizer.best_combination across all 3 lotteries
```bash
C:\Python314\python.exe -c "
import cProfile, pstats, io, sys
sys.path.insert(0, '.')
from src import ev_optimizer as ev
from src import popularity_model as pm
params = {k: 1.0 for k in pm.FEATURE_NAMES}
pr = cProfile.Profile()
pr.enable()
for _ in range(200):
    for lot in ['lottomax', '649', 'dailygrand']:
        ev.best_combination(lot, params, jackpot=10_000_000.0)
pr.disable()
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(10)
print(s.getvalue())
" 2>&1
```

Baseline: 200 sweeps (600 calls) should complete in under 2 seconds.

Flag any single function taking > 500ms cumulative.

---

## Step 2 - Edge Case Testing

Run each snippet via Bash. Capture stdout. Record behavior.

### Edge 1: Empty draws CSV
```bash
C:\Python314\python.exe -c "
import sys, tempfile, os
sys.path.insert(0, '.')
import pandas as pd
tmp = tempfile.mktemp(suffix='.csv')
pd.DataFrame(columns=['date','n1','n2','n3','n4','n5','n6','bonus']).to_csv(tmp, index=False)
import config
config.DRAWS_CSV = {**config.DRAWS_CSV, '649': tmp}
from src import data_loader
try:
    df = data_loader.load_draws('649')
    print('EDGE empty csv: df.empty=' + str(df.empty) + ' cols=' + str(list(df.columns)))
except Exception as e:
    print('EDGE empty csv: raised ' + type(e).__name__ + ': ' + str(e)[:120])
os.unlink(tmp)
" 2>&1
```

### Edge 2: Missing draws file
```bash
C:\Python314\python.exe -c "
import sys
sys.path.insert(0, '.')
import config
config.DRAWS_CSV = {**config.DRAWS_CSV, '649': '/tmp/does_not_exist_xyz.csv'}
from src import data_loader
try:
    df = data_loader.load_draws('649')
    print('EDGE missing file: df.empty=' + str(df.empty))
except Exception as e:
    print('EDGE missing file: raised ' + type(e).__name__ + ': ' + str(e)[:120])
" 2>&1
```

### Edge 3: fit with no payout data
```bash
C:\Python314\python.exe -c "
import sys, random
from datetime import datetime, timedelta
sys.path.insert(0, '.')
import pandas as pd
from src import popularity_model as pm
rng = random.Random(1)
rows = []
d = datetime(2025, 1, 1)
for _ in range(30):
    main = sorted(rng.sample(range(1,50), 6))
    row = {'date': d.strftime('%Y-%m-%d')}
    for i,v in enumerate(main,1): row['n'+str(i)] = v
    row['bonus'] = rng.randint(1,49)
    rows.append(row); d += timedelta(days=3)
df_d = pd.DataFrame(rows)
empty = pd.DataFrame(columns=['date','lottery','jackpot','tier_main','tier_bonus','n_winners','payout_per_winner'])
try:
    r = pm.fit(df_d, empty, '649', n_tickets_sold=4_000_000)
    print('EDGE no payouts: success=' + str(r.success) + ' n_draws=' + str(r.n_draws))
except Exception as e:
    print('EDGE no payouts: raised ' + type(e).__name__ + ': ' + str(e)[:120])
" 2>&1
```

### Edge 4: compute_ev with zero jackpot
```bash
C:\Python314\python.exe -c "
import sys
sys.path.insert(0, '.')
from src import ev_optimizer as ev
from src import popularity_model as pm
params = {k: 1.0 for k in pm.FEATURE_NAMES}
try:
    r = ev.compute_ev([1,2,3,4,5,6], '649', params, jackpot=0.0, n_tickets_sold=1_000_000)
    print('EDGE zero jackpot: ev_gross=' + str(round(r['ev_gross'],4)) + ' ev_net=' + str(round(r['ev_net'],4)))
except Exception as e:
    print('EDGE zero jackpot: raised ' + type(e).__name__ + ': ' + str(e)[:120])
" 2>&1
```

### Edge 5: compute_ev with zero tickets sold (no split risk)
```bash
C:\Python314\python.exe -c "
import sys
sys.path.insert(0, '.')
from src import ev_optimizer as ev
from src import popularity_model as pm
params = {k: 1.0 for k in pm.FEATURE_NAMES}
try:
    r = ev.compute_ev([1,2,3,4,5,6], '649', params, jackpot=5_000_000.0, n_tickets_sold=0)
    # With 0 other tickets: effective_payout_jackpot == jackpot
    print('EDGE zero tickets sold: e_other=' + str(round(r['e_other_jackpot_winners'], 6)))
except Exception as e:
    print('EDGE zero tickets sold: raised ' + type(e).__name__ + ': ' + str(e)[:120])
" 2>&1
```

### Edge 6: append_draw rejects duplicate numbers
```bash
C:\Python314\python.exe -c "
import sys, tempfile
sys.path.insert(0, '.')
import config
tmp = tempfile.mktemp(suffix='.csv')
config.DRAWS_CSV = {**config.DRAWS_CSV, '649': tmp}
from src import data_loader
try:
    data_loader.append_draw('649', '2026-04-17', [1,1,2,3,4,5], 6)
    print('EDGE dup numbers: accepted silently (PROBLEM)')
except ValueError as e:
    print('EDGE dup numbers: ValueError correctly: ' + str(e)[:80])
except Exception as e:
    print('EDGE dup numbers: raised ' + type(e).__name__ + ': ' + str(e)[:80])
" 2>&1
```

### Edge 7: bonus_match_prob at tier_main = main_count (no remaining picks for lottomax/649)
```bash
C:\Python314\python.exe -c "
import sys
sys.path.insert(0, '.')
import config
from src import ev_optimizer as ev
for lot in ['lottomax', '649', 'dailygrand']:
    cfg = config.LOTTERY_RULES[lot]
    p = ev.bonus_match_prob(cfg, tier_main=cfg['main_count'])
    print('EDGE bonus full match ' + lot + ': p=' + str(round(p, 6)))
" 2>&1
```

Verdict per case:
- GRACEFUL = raised clear error or returned sensible value
- SILENT  = bad: accepted wrong input without warning
- CRASH   = bad: uncaught exception or traceback

---

## Step 3 - Documentation Review

Read each file under `src/` and the root scripts. For every public function
(not prefixed with `_`), check:
1. Has a docstring
2. Docstring mentions parameters (if function takes non-trivial args)
3. Docstring mentions return value (if non-trivial)
4. No Unicode in docstrings

Functions to check:
- data_loader: load_draws, load_payouts, load_combined, append_draw, append_payouts, summary_row_counts
- popularity_model: per_number_weights, elementary_symmetric, log_partition, log_popularity, expected_tier_winner_fraction, fit
- ev_optimizer: hypergeometric_prob, bonus_match_prob, compute_ev, best_combination, uniform_random_combination
- bandit: default_entry, update, sample_all, pick_strategy, summary
- utils: temperature_softmax
- agent/olg_scraper: fetch_latest, save, main

---

## Step 4 - Refactoring Proposals

Propose only -- do NOT apply. For each proposal include: location, rationale,
estimated effort (low/medium/high).

Examples relevant to this project:

1. **Cache elementary_symmetric for repeated EV calls**
   compute_ev runs popularity log-partition once per call. If we batch-score
   many combinations for the same params, Z is redundant.
   Effort: low

2. **Move _pool_fraction into config.py**
   Right now `_POOL_FRACTION` lives in ev_optimizer.py. Putting it in
   config.py (alongside PRIZE_TIERS) makes it easier for users to tune
   based on OLG prize-pool updates.
   Effort: low

3. **CSV file-lock risk in data_loader.append_draw**
   No file lock. If two processes scrape concurrently, the CSV can be corrupted.
   Suggest lockfile or atomic rename.
   Effort: medium

4. **Validate payout rows against PRIZE_TIERS**
   append_payouts doesn't check that tier_rows match the defined tiers.
   A scraper bug could silently insert invalid rows.
   Effort: medium

---

## Output Format

```
## q_a2 Report

### Performance Results
| Function | Input | Iterations | Elapsed | Status |
|----------|-------|------------|---------|--------|
| popularity_model.fit | 60 draws | 3 | Xs | PASS / SLOW |
| ev_optimizer.compute_ev | 500 calls | 500 | Xs | PASS / SLOW |
| ev_optimizer.best_combination | 200 sweeps | 600 | Xs | PASS / SLOW |

### Edge Case Results
| Scenario | Behavior | Verdict |
|----------|----------|---------|
| empty CSV | ... | GRACEFUL / SILENT / CRASH |
| missing file | ... | GRACEFUL / SILENT / CRASH |
| fit empty payouts | ... | GRACEFUL / SILENT / CRASH |
| zero jackpot | ... | GRACEFUL / SILENT / CRASH |
| zero tickets sold | ... | GRACEFUL / SILENT / CRASH |
| dup numbers | ... | GRACEFUL / SILENT / CRASH |
| bonus full match | ... | GRACEFUL / SILENT / CRASH |

### Documentation Coverage
| Module | Public Functions | Missing Docstrings | Issues |
|--------|------------------|--------------------|--------|

### Refactoring Proposals (do not auto-apply)
1. [location] description -- effort: low/medium/high
...

### Overall: PASS / FAIL
FAIL if any edge case is CRASH or SILENT, or any CRITICAL doc issue.
```
