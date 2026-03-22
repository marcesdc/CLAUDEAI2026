"""
Conversational + Analysis interface for lottery-nn.

The agent understands natural language, runs project commands,
and can answer analysis questions about the draw history.

Usage:
    C:\\Python314\\python.exe agent/run.py chat
"""

import anyio
from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    ResultMessage,
    SystemMessage,
    CLINotFoundError,
    CLIConnectionError,
)

PROJECT_DIR = "d:/AI - 2026/lottery-nn"
PYTHON     = r"C:\Python314\python.exe"

SYSTEM_PROMPT = f"""You are an assistant for a Lotto Max neural network prediction project.

Project directory : {PROJECT_DIR}
Python executable : {PYTHON}
Data file         : data/draws.csv  (columns: date, number1..number7, bonus  OR  n1..n7, bonus)
Game format       : 7 numbers from 1-50, plus 1 bonus number (1-50)
A "play"          : 3 independent lines of 7 numbers + 1 bonus suggestion

── Available CLI commands ───────────────────────────────────────────────────
Log a draw result and retrain:
  {PYTHON} main.py log --date YYYY-MM-DD --numbers N1 N2 N3 N4 N5 N6 N7 --bonus N

Score only (no retrain):
  {PYTHON} main.py log --date YYYY-MM-DD --numbers N1 N2 N3 N4 N5 N6 N7 --bonus N --no-retrain

Generate play predictions:
  {PYTHON} main.py predict --plays N

Train from scratch (large data batch):
  {PYTHON} main.py train

Evaluate model on test set:
  {PYTHON} main.py evaluate

── Analysis ─────────────────────────────────────────────────────────────────
For analysis questions (hot/cold numbers, pairs, gaps, frequency), run Python
inline with Bash using the helpers in src/analysis.py.  For example:

  {PYTHON} -c "
  import sys; sys.path.insert(0, '.')
  import pandas as pd
  from src.data_loader import load_draws
  from src.analysis import frequency_table, hot_cold, pair_frequency, gap_analysis
  df = load_draws()
  print(frequency_table(df).head(10))
  "

Always run Bash commands from the project directory ({PROJECT_DIR}).

── Behaviour rules ──────────────────────────────────────────────────────────
- When the user gives draw results in ANY format, extract date + numbers and
  run the log command.  Convert dates to YYYY-MM-DD.
- When the user asks for predictions, run the predict command and show output.
- When the user asks analysis questions, write and run a quick Python snippet.
- Keep responses concise.  Show command output directly.
"""


async def run_chat():
    print()
    print("=" * 52)
    print("  Lottery NN  –  Chat & Analysis Interface")
    print("  Type 'quit' to exit.")
    print("=" * 52)

    session_id = None

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        print("\nAssistant:", flush=True)

        options = ClaudeAgentOptions(
            cwd=PROJECT_DIR,
            allowed_tools=["Bash", "Read", "Glob", "Grep"],
            system_prompt=SYSTEM_PROMPT if session_id is None else None,
            permission_mode="acceptEdits",
            max_turns=15,
            **({"resume": session_id} if session_id else {}),
        )

        try:
            async for message in query(prompt=user_input, options=options):
                if isinstance(message, SystemMessage) and message.subtype == "init":
                    session_id = message.data.get("session_id")
                elif isinstance(message, ResultMessage):
                    print(message.result)

        except CLINotFoundError:
            print("Claude Code CLI not found. Run: pip install claude-agent-sdk")
            break
        except CLIConnectionError as e:
            print(f"Connection error: {e}")


def main():
    anyio.run(run_chat)


if __name__ == "__main__":
    main()
