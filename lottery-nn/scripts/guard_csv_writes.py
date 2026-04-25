"""
PreToolUse hook -- block Edit/Write to real-data lottery CSVs.

Triggered by .claude/settings.json on Edit|Write tool calls.

Rationale: incident 2026-04-23 silently overwrote draws.csv with a 1-row
synthetic file. Direct Edit/Write to data/draws*.csv is almost never the
correct path -- new draws should go through main_swarm.py log, and
historical edits should be deliberate. This hook turns that guideline
into a hard gate.

Override for a deliberate manual edit:
    set LOTTERY_CSV_WRITE_OK=1   (Windows cmd)
    $env:LOTTERY_CSV_WRITE_OK="1" (PowerShell)
"""

import json
import os
import re
import sys

# Match canonical real-data CSVs. Backups (e.g. draws.csv.bak.2026-04-23)
# and synthetic CSVs in tmp paths are intentionally NOT matched.
PROTECTED_PATTERN = re.compile(
    r"data[\\/]draws(_649|_dailygrand)?\.csv$",
    re.IGNORECASE,
)


def main() -> None:
    raw = sys.stdin.read()
    if not raw.strip():
        # Nothing to inspect -- fail open
        sys.exit(0)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        # Bad payload -- log and fail open so we don't break unrelated tools.
        print(
            f"[guard_csv_writes] could not parse hook payload: {exc}",
            file=sys.stderr,
        )
        sys.exit(0)

    tool_name = payload.get("tool_name", "")
    if tool_name not in ("Edit", "Write"):
        sys.exit(0)

    file_path = payload.get("tool_input", {}).get("file_path", "")
    if not file_path:
        sys.exit(0)

    norm = file_path.replace("\\", "/")
    if not PROTECTED_PATTERN.search(norm):
        sys.exit(0)

    if os.environ.get("LOTTERY_CSV_WRITE_OK") == "1":
        print(
            f"[guard_csv_writes] override active (LOTTERY_CSV_WRITE_OK=1) -- "
            f"allowing edit to '{file_path}'",
            file=sys.stderr,
        )
        sys.exit(0)

    print(
        f"[guard_csv_writes] BLOCKED: '{file_path}' is real-data lottery history "
        f"(incident 2026-04-23). New draws must go through "
        f"'main_swarm.py log --lottery <name> --date <date> --numbers ...'. "
        f"For a deliberate manual edit, set env LOTTERY_CSV_WRITE_OK=1 first.",
        file=sys.stderr,
    )
    # Exit 2 = block the tool call; stderr is shown to Claude.
    sys.exit(2)


if __name__ == "__main__":
    main()
