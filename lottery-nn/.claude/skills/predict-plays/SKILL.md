---
name: predict-plays
description: Generate lottery play suggestions from the trained model. Use after logging a draw or when the user asks for ticket suggestions.
disable-model-invocation: true
allowed-tools: Bash
---

Generate lottery play suggestions using the trained model.

```bash
C:\Python314\python.exe main.py predict $ARGUMENTS
```

If `$ARGUMENTS` contains a number, pass it as `--plays N` (e.g. "predict-plays 10" → `--plays 10`).
If no argument is given, run without flags (uses default of 5 plays).

If the command fails with "No checkpoint", tell the user to run:
```bash
C:\Python314\python.exe main.py train
```
