"""
qa_gate.py -- Pre-action QA gate.

Runs the pytest test suite before allowing optimize/backtest/log to proceed.
If any test fails the process exits with code 1 and prints a FAIL message.

Forked from lottery-nn/src/qa_gate.py unchanged. Path constants resolve relative
to this module's parent so the gate works inside any project root.
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TESTS_DIR = PROJECT_ROOT / "tests"


def run_qa_gate() -> None:
    """Run pytest against tests/. Exit(1) on any failure or collection error.

    Called at the start of every CLI command that can affect data on disk
    or produce a recommendation the user might act on.
    """
    if not TESTS_DIR.exists():
        print("[qa_gate] SKIP - tests/ directory not found, proceeding without QA check.")
        return

    print("[qa_gate] Running QA gate (pytest tests/) ...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(TESTS_DIR),
         "-q", "--tb=line", "--no-header", "-p", "no:warnings"],
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        print("[qa_gate] FAIL - QA gate failed. Fix the failing tests before proceeding.")
        print("[qa_gate] Run manually: C:\\Python314\\python.exe -m pytest tests/ -v --tb=short")
        sys.exit(1)

    print("[qa_gate] PASS - all tests passed, proceeding.")
