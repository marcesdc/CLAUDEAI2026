"""
PostToolUse hook -- run fast unit tests after edits to source / tests / config.

Triggered by .claude/settings.json on Edit|Write tool calls. Only runs pytest
when the edited file is in src/, tests/, or one of the root *.py entry points
(main.py, main_swarm.py, autorun.py, config.py).

Rationale: CLAUDE.md mandates `@q_a_lead` after every code change. This hook
catches obvious regressions immediately so Claude sees the failure on the
next turn, while @q_a_lead is still used for full GREEN/RED verdicts on
substantive changes.

Never blocks (PostToolUse runs after the tool already executed). On failure,
writes a short summary to stderr which Claude sees as a system message.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
PYTHON = r"C:\Python314\python.exe"

# Trigger only for these path patterns (project-relative).
TRIGGERS = [
    re.compile(r"^src/.+\.py$"),
    re.compile(r"^tests/.+\.py$"),
    re.compile(r"^(main|main_swarm|autorun|config)\.py$"),
]

# Skip running when only a test file we just wrote is being edited mid-flight
# (avoid recursive flakiness during long edit sessions).
TIMEOUT_SECONDS = 45


def _project_relative(file_path: str) -> str:
    """Reduce a path to project-relative form using forward slashes."""
    p = file_path.replace("\\", "/")
    project_str = str(PROJECT).replace("\\", "/")
    if p.lower().startswith(project_str.lower()):
        p = p[len(project_str):].lstrip("/")
    return p


def _matches_trigger(file_path: str) -> bool:
    rel = _project_relative(file_path)
    return any(pat.search(rel) for pat in TRIGGERS)


def main() -> None:
    raw = sys.stdin.read()
    if not raw.strip():
        sys.exit(0)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        sys.exit(0)

    if payload.get("tool_name", "") not in ("Edit", "Write"):
        sys.exit(0)

    file_path = payload.get("tool_input", {}).get("file_path", "")
    if not file_path or not _matches_trigger(file_path):
        sys.exit(0)

    # Allow opt-out for noisy edit sessions.
    if os.environ.get("LOTTERY_SKIP_PYTEST_HOOK") == "1":
        sys.exit(0)

    try:
        result = subprocess.run(
            [
                PYTHON, "-m", "pytest", "tests/unit",
                "-q", "--no-header", "--tb=line", "-p", "no:warnings",
            ],
            cwd=str(PROJECT),
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        print(
            f"[post_edit_pytest] timeout after {TIMEOUT_SECONDS}s "
            f"(edit was: {_project_relative(file_path)})",
            file=sys.stderr,
        )
        sys.exit(0)
    except FileNotFoundError:
        # Python interpreter missing -- silent skip rather than nag.
        sys.exit(0)

    if result.returncode == 0:
        # Quiet success: no output, don't pollute Claude's context.
        sys.exit(0)

    print(
        f"[post_edit_pytest] FAIL after edit to '{_project_relative(file_path)}':",
        file=sys.stderr,
    )
    combined = (result.stdout + "\n" + result.stderr).strip().splitlines()
    # Show only the last 25 lines -- usually enough for the failing test
    # summary plus a stack trace tail. Never echo gigabytes.
    for line in combined[-25:]:
        print(line, file=sys.stderr)
    sys.exit(0)


if __name__ == "__main__":
    main()
