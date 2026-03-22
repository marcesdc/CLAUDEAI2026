---
name: summarize
description: Summarize what was accomplished in this session — draws logged, model retrained, predictions generated, bugs fixed, or code changes made.
disable-model-invocation: true
---

Review what was done in this conversation and produce a concise end-of-session summary.

Structure it as:

**What was done**
- Bullet list of actions taken (draws logged, model retrained, predictions generated, code changes, bugs fixed, etc.)

**Current state**
- Total draws in dataset
- Last draw logged (date + numbers)
- Model checkpoint status

**Next steps**
- What the user should do next (e.g. wait for next draw, run predict, etc.)

To get the current state, run:
```bash
C:\Python314\python.exe -c "
import sys; sys.path.insert(0, '.')
import pandas as pd
from src.data_loader import load_draws
from pathlib import Path
df = load_draws()
print('Total draws:', len(df))
print('Last draw:', df.iloc[-1].to_dict())
checkpoint = Path('models/best.pt')
print('Checkpoint exists:', checkpoint.exists())
"
```

Keep the summary short — one screen.
