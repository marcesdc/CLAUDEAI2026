"""agent/run.py -- CLI wrapper for the scraper."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import olg_scraper


def main() -> int:
    parser = argparse.ArgumentParser(prog="agent/run.py")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_scrape = sub.add_parser("scrape", help="Fetch latest draw + payouts from OLG")
    p_scrape.add_argument("--lottery", required=True,
                          choices=["lottomax", "649", "dailygrand"])

    args = parser.parse_args()
    if args.cmd == "scrape":
        return olg_scraper.main(args.lottery)
    return 1


if __name__ == "__main__":
    sys.exit(main())
