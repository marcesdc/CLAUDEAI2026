"""
Automated draw monitor – uses Playwright (via MCP) to fetch the latest
Lotto Max result from the OLG website, then logs it and retrains if new.

Usage:
    C:\\Python314\\python.exe agent/run.py monitor        # check once
    C:\\Python314\\python.exe agent/run.py watch          # check every 6 hours
"""

import anyio
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    ResultMessage,
    CLINotFoundError,
    CLIConnectionError,
)

PROJECT_DIR = "d:/AI - 2026/lottery-nn"
PYTHON      = r"C:\Python314\python.exe"
OLG_URL     = "https://www.olg.ca/en/lottery/winning-numbers-results.html#game-item-lottomax"

# Days to check: Wednesday=2, Saturday=5 (Monday=0 ... Sunday=6)
CHECK_DAYS = {2, 5}
CHECK_HOUR  = 9   # run at 9 AM on check days

MONITOR_PROMPT = f"""
You are a lottery result extractor. Your only job is to retrieve the most recent
Lotto Max draw result from the OLG website.

Steps:
1. Navigate to: {OLG_URL}
2. Wait for the page to fully load (it uses JavaScript — wait several seconds).
3. Locate the "Lotto Max" section. It shows the most recent draw date and 7 winning
   numbers plus a bonus number.
4. Extract:
   - Draw date  →  convert to YYYY-MM-DD format
   - The 7 main numbers (each between 1 and 50)
   - The 1 bonus number (between 1 and 50)

Return the result in EXACTLY this format and nothing else:
DATE: YYYY-MM-DD
NUMBERS: N1 N2 N3 N4 N5 N6 N7
BONUS: N

If you cannot find the results or the page fails to load, return:
FAILED: <brief reason>
"""


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

async def fetch_latest_draw() -> dict | None:
    """Use Playwright MCP to get the latest draw from OLG."""
    result_text = ""

    try:
        async for message in query(
            prompt="Extract the latest Lotto Max draw result from the OLG website.",
            options=ClaudeAgentOptions(
                cwd=PROJECT_DIR,
                system_prompt=MONITOR_PROMPT,
                mcp_servers={
                    "playwright": {
                        "command": "npx",
                        "args": ["@playwright/mcp@latest"],
                    }
                },
                max_turns=20,
            ),
        ):
            if isinstance(message, ResultMessage):
                result_text = message.result

    except (CLINotFoundError, CLIConnectionError) as e:
        print(f"[monitor] SDK error: {e}")
        return None

    if not result_text or "FAILED" in result_text:
        print(f"[monitor] Could not fetch draw: {result_text or 'no response'}")
        return None

    return _parse_result(result_text)


def _parse_result(text: str) -> dict | None:
    data = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith("DATE:"):
            data["date"] = line.replace("DATE:", "").strip()
        elif line.startswith("NUMBERS:"):
            try:
                data["numbers"] = [int(n) for n in line.replace("NUMBERS:", "").strip().split()]
            except ValueError:
                pass
        elif line.startswith("BONUS:"):
            try:
                data["bonus"] = int(line.replace("BONUS:", "").strip())
            except ValueError:
                pass

    if not all(k in data for k in ("date", "numbers", "bonus")):
        print(f"[monitor] Incomplete result: {text}")
        return None
    if len(data["numbers"]) != config.LOTTERY["main_count"]:
        print(f"[monitor] Wrong number count: {data['numbers']}")
        return None

    return data


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_last_logged_date() -> str | None:
    csv_path = Path(config.RAW_CSV)
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"])
    return df["date"].max().strftime("%Y-%m-%d")


def log_draw(draw: dict) -> bool:
    cmd = [
        PYTHON, "main.py", "log",
        "--date", draw["date"],
        "--numbers", *[str(n) for n in draw["numbers"]],
        "--bonus", str(draw["bonus"]),
    ]
    result = subprocess.run(cmd, cwd=PROJECT_DIR, capture_output=False, text=True)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Main check
# ---------------------------------------------------------------------------

async def check_once():
    print(f"[monitor] {datetime.now().strftime('%Y-%m-%d %H:%M')}  Checking OLG for new draw...")

    draw = await fetch_latest_draw()
    if draw is None:
        return

    print(f"[monitor] Latest draw  :  {draw['date']}  {draw['numbers']}  bonus={draw['bonus']}")

    last_date = get_last_logged_date()
    if last_date:
        # Normalise to YYYY-MM-DD for comparison
        try:
            last_dt = datetime.strptime(last_date, "%Y-%m-%d")
            draw_dt = datetime.strptime(draw["date"], "%Y-%m-%d")
            if draw_dt <= last_dt:
                print(f"[monitor] Already up-to-date  (last logged: {last_date})")
                return
        except ValueError:
            pass

    print("[monitor] New draw found — logging and retraining...")
    success = log_draw(draw)
    if success:
        print("[monitor] Done.  Run predict to generate new plays.")
    else:
        print("[monitor] Log command failed — check output above.")


def _seconds_until_next_check() -> int:
    """Return seconds until CHECK_HOUR on the next Wednesday or Saturday."""
    now = datetime.now()
    for days_ahead in range(1, 8):
        candidate = (now + timedelta(days=days_ahead)).replace(
            hour=CHECK_HOUR, minute=0, second=0, microsecond=0
        )
        if candidate.weekday() in CHECK_DAYS:
            return max(1, int((candidate - now).total_seconds()))
    return 7 * 24 * 3600  # fallback


async def watch():
    print("[monitor] Watch mode — checks at 9 AM every Wednesday and Saturday.  Ctrl+C to stop.")
    while True:
        if datetime.now().weekday() in CHECK_DAYS:
            await check_once()

        secs = _seconds_until_next_check()
        wake = datetime.now() + timedelta(seconds=secs)
        print(f"[monitor] Next check: {wake.strftime('%A %Y-%m-%d at %H:%M')}")
        await anyio.sleep(secs)


def main(mode: str = "once"):
    if mode == "watch":
        anyio.run(watch)
    else:
        anyio.run(check_once)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "once"
    main(mode)
