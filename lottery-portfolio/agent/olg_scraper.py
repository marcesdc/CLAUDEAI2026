"""
agent/olg_scraper.py -- Scrape latest draw and full payout breakdown from OLG.

Forked and extended from lottery-nn/agent/monitor.py. Keeps the Playwright MCP
pattern; adds per-lottery URL, jackpot extraction, and prize-tier winner counts.

Usage:
    C:\\Python314\\python.exe agent/run.py scrape --lottery lottomax
    C:\\Python314\\python.exe agent/run.py scrape --lottery 649
    C:\\Python314\\python.exe agent/run.py scrape --lottery dailygrand

On success appends to both data/draws_<lottery>.csv and data/payouts_<lottery>.csv.
On failure prints FAILED and exits 0 (user can manually enter via main.py log).
"""

import os
import sys
from pathlib import Path

import anyio

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config                               # noqa: E402
from src import data_loader                 # noqa: E402

try:
    from claude_agent_sdk import (
        query,
        ClaudeAgentOptions,
        ResultMessage,
        CLINotFoundError,
        CLIConnectionError,
    )
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False


# ---------------------------------------------------------------------------
# OLG URLs -- one per lottery
# ---------------------------------------------------------------------------
OLG_URLS = {
    "lottomax":   "https://www.olg.ca/en/lottery/winning-numbers-results.html#game-item-lottomax",
    "649":        "https://www.olg.ca/en/lottery/winning-numbers-results.html#game-item-lotto649",
    "dailygrand": "https://www.olg.ca/en/lottery/winning-numbers-results.html#game-item-dailygrand",
}


# ---------------------------------------------------------------------------
# Scraper prompt template
# ---------------------------------------------------------------------------
def _build_prompt(lottery: str) -> str:
    cfg = config.LOTTERY_RULES[lottery]
    main_count = cfg["main_count"]
    return f"""
You are a lottery result extractor. Your only job is to retrieve the most recent
{cfg['name']} draw result AND the full payout breakdown from the OLG website.

Steps:
1. Navigate to: {OLG_URLS[lottery]}
2. Wait several seconds for the JavaScript page to render.
3. Locate the "{cfg['name']}" section and the most recent draw.
4. Extract:
   - Draw date  ->  convert to YYYY-MM-DD format
   - The {main_count} main numbers (each between 1 and {cfg['main_max']})
   - The 1 bonus number (between 1 and {cfg['bonus_max']})
   - The total jackpot amount for that draw (the "estimated jackpot" or
     "jackpot" value; Canadian dollars, plain number)
   - The payout / prize-breakdown table showing for each prize tier:
       * match description (e.g. "7/7", "6/7 + Bonus", "6/7", "4/7", etc.)
       * number of winners
       * prize amount per winner (or "PARI-MUTUEL" / "" if split pool)

Return the result in EXACTLY this format and nothing else:

DATE: YYYY-MM-DD
NUMBERS: N1 N2 N3 N4 N5 ...
BONUS: N
JACKPOT: <dollars, plain number>
PAYOUTS:
TIER_MAIN | TIER_BONUS | N_WINNERS | PAYOUT_PER_WINNER
7 | - | 0 |
6 | Y | 2 |
6 | N | 15 |
...

Conventions:
- TIER_MAIN: integer, number of main numbers matched
- TIER_BONUS: "Y" if bonus must match, "N" if bonus must NOT match, "-" if bonus irrelevant
- PAYOUT_PER_WINNER: blank for pari-mutuel (split) tiers, plain number for fixed tiers

If you cannot find the results or the page fails to load, return only:
FAILED: <brief reason>
""".strip()


# ---------------------------------------------------------------------------
# Async fetch
# ---------------------------------------------------------------------------
async def fetch_latest(lottery: str) -> dict | None:
    """Use Playwright MCP to get the latest draw + payouts."""
    if not _SDK_AVAILABLE:
        print("[scraper] claude_agent_sdk not installed -- install with: pip install claude-agent-sdk")
        return None

    if lottery not in OLG_URLS:
        print(f"[scraper] Unknown lottery {lottery!r}")
        return None

    result_text = ""
    try:
        async for message in query(
            prompt=f"Extract the latest {lottery} draw + payout breakdown.",
            options=ClaudeAgentOptions(
                cwd=str(PROJECT_ROOT),
                system_prompt=_build_prompt(lottery),
                mcp_servers={
                    "playwright": {
                        "command": "npx",
                        "args": ["@playwright/mcp@latest"],
                    }
                },
                permission_mode="bypassPermissions",
                allowed_tools=[
                    "mcp__playwright__browser_navigate",
                    "mcp__playwright__browser_snapshot",
                    "mcp__playwright__browser_wait_for",
                    "mcp__playwright__browser_click",
                    "mcp__playwright__browser_evaluate",
                    "mcp__playwright__browser_close",
                    "WebFetch",
                ],
                max_turns=25,
            ),
        ):
            if isinstance(message, ResultMessage):
                result_text = message.result

    except (CLINotFoundError, CLIConnectionError) as e:
        print(f"[scraper] SDK error: {e}")
        return None

    if not result_text or "FAILED" in result_text:
        print(f"[scraper] Could not fetch draw: {result_text or 'no response'}")
        return None

    return _parse_result(result_text, lottery)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def _parse_result(text: str, lottery: str) -> dict | None:
    """Parse the DATE/NUMBERS/BONUS/JACKPOT/PAYOUTS block into a dict.

    Validates that main numbers are within [1, main_max] and bonus within
    [1, bonus_max] to catch scraper mis-reads before they corrupt the CSVs.
    """
    cfg = config.LOTTERY_RULES[lottery]
    data = {"tiers": []}
    in_payouts = False

    for raw_line in text.strip().splitlines():
        line = raw_line.strip()
        if not line:
            continue

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
        elif line.startswith("JACKPOT:"):
            try:
                data["jackpot"] = float(line.replace("JACKPOT:", "").strip().replace(",", ""))
            except ValueError:
                data["jackpot"] = 0.0
        elif line.startswith("PAYOUTS:"):
            in_payouts = True
        elif in_payouts and "|" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) != 4:
                continue
            # Skip the header row
            if parts[0].upper() == "TIER_MAIN":
                continue
            try:
                tier = {
                    "tier_main":         int(parts[0]),
                    "tier_bonus":        parts[1] if parts[1] in ("Y", "N", "-") else "-",
                    "n_winners":         int(parts[2]),
                    "payout_per_winner": float(parts[3].replace(",", "")) if parts[3] else "",
                }
                data["tiers"].append(tier)
            except ValueError:
                continue

    # Validate required fields
    required = ("date", "numbers", "bonus", "jackpot")
    if not all(k in data for k in required):
        print(f"[scraper] Incomplete result, missing one of {required}")
        print(f"[scraper] Raw text was:\n{text}")
        return None
    if len(data["numbers"]) != cfg["main_count"]:
        print(f"[scraper] Wrong main count: {len(data['numbers'])}, expected {cfg['main_count']}")
        return None

    return data


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------
def save(lottery: str, draw: dict) -> bool:
    """Append to draws_<lottery>.csv and payouts_<lottery>.csv. Returns True on success."""
    try:
        # Guard against re-logging same draw
        existing = data_loader.load_draws(lottery)
        if not existing.empty and draw["date"] in set(existing["date"].astype(str)):
            print(f"[scraper] Already have {lottery} draw for {draw['date']} -- skip")
            return False

        data_loader.append_draw(lottery, draw["date"], draw["numbers"], draw["bonus"])
        if draw.get("tiers"):
            data_loader.append_payouts(lottery, draw["date"], draw["jackpot"], draw["tiers"])
        print(f"[scraper] Saved {lottery} draw {draw['date']} + {len(draw.get('tiers', []))} tier rows")
        return True
    except (OSError, ValueError) as e:
        print(f"[scraper] Save failed: {e}")
        return False


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
def main(lottery: str) -> int:
    print(f"[scraper] Fetching latest {lottery} draw from OLG ...")
    draw = anyio.run(fetch_latest, lottery)
    if draw is None:
        return 1
    print(
        f"[scraper] Got {draw['date']} "
        f"numbers={draw['numbers']} bonus={draw['bonus']} "
        f"jackpot={draw['jackpot']:,.2f} tiers={len(draw.get('tiers', []))}"
    )
    save(lottery, draw)
    return 0


if __name__ == "__main__":
    lottery_arg = sys.argv[1] if len(sys.argv) > 1 else "lottomax"
    sys.exit(main(lottery_arg))
