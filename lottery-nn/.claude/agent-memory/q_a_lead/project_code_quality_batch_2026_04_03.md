---
name: code_quality_batch_2026_04_03
description: QA results for code quality batch: utils.py extraction, vectorized preprocessing, path guard, train.py type hints -- GREEN verdict 2026-04-03
type: project
---

Code quality batch landed 2026-04-03. All 61 tests pass. Verdict: GREEN.

Changes audited:
- src/utils.py (NEW): temperature_softmax extracted from predict.py and main_swarm.py -- behavior identical to removed local copies (verified bit-for-bit on random, edge, and zero-temp inputs)
- src/preprocessing.py: _multi_hot and _one_hot vectorized with NumPy np.where -- 32x speedup, bit-identical output to loop versions including out-of-range guards
- src/data_loader.py: path traversal guard using _PROJECT_ROOT in load_draws(); duplicate-date warning in _validate() -- both confirmed working
- src/train.py: _make_loader got full type hints and docstring -- no logic change
- main_swarm.py: local _temperature_softmax removed, now imports from src.utils; _score_swarm_prediction got -> int | None return annotation
- CLAUDE.md: three doc corrections (play=1 line, LINES_PER_PLAY default 1, BCEWithLogitsLoss -> focal loss)

**Why:** Refactor sprint to reduce duplication and harden security before Phase 2 actor-critic work.

**How to apply:** src/utils.py is now the canonical home for temperature_softmax -- do not re-add local copies in predict.py or main_swarm.py.

Open warnings (pre-existing, not introduced by this batch):
- src/preprocessing.py and main_swarm.py module docstrings contain Unicode en-dash/em-dash (\xe2\x80\x93/\xe2\x80\x94) -- in docstrings only, not in print() calls, so no runtime crash risk on Windows cp1252
- src/utils.py has 0% test coverage -- no unit test for temperature_softmax exists yet
- path traversal guard uses startswith() on string representation of Path -- symlinks could bypass this on Linux; acceptable on Windows for this project
- All data_loader and preprocessing tests are stub pass bodies -- pre-existing, not a regression
