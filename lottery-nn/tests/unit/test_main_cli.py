"""Unit tests for main.py CLI argument handling."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import pytest

import main as main_module


def test_cmd_log_rejects_missing_numbers(capsys):
    """Codex M1 guard: 'log' without --numbers must exit cleanly with a clear
    error, not crash with TypeError/ValueError deep inside feedback.py."""
    parser = main_module.build_parser()
    args = parser.parse_args(["log", "--date", "2026-04-22"])
    assert args.numbers is None  # confirm parser allows omission
    with pytest.raises(SystemExit) as exc_info:
        main_module.cmd_log(args)
    # sys.exit(str) exits with code 1 and the string as message
    err = capsys.readouterr().err + str(exc_info.value)
    assert "--numbers" in err
