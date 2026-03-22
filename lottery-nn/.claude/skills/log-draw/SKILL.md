---
name: log-draw
description: Log an actual lottery draw result, score the last prediction against it, and retrain the model. Use when the user provides real draw results.
disable-model-invocation: true
allowed-tools: Bash
---

The user wants to log a lottery draw result. $ARGUMENTS

Run the following command using the date and numbers provided by the user:

```bash
C:\Python314\python.exe main.py log --date YYYY-MM-DD --numbers N1 N2 N3 N4 N5 N6 N7 --bonus N
```

Rules:
- Date must be in YYYY-MM-DD format. If the user gives another format, convert it.
- There must be exactly 7 main numbers, each between 1 and 50.
- The bonus number must be between 1 and 50.
- If any value is missing or unclear, ask before running.

After the command succeeds, remind the user to run /predict-plays to generate new suggestions.
