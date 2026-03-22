"""
Entry point for all agent modes.

Usage
-----
    C:\\Python314\\python.exe agent/run.py chat       # conversational + analysis
    C:\\Python314\\python.exe agent/run.py monitor    # check OLG for new draw once
    C:\\Python314\\python.exe agent/run.py watch      # check OLG every 6 hours
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

USAGE = """
Usage:
  C:\\Python314\\python.exe agent/run.py chat       Start conversational interface
  C:\\Python314\\python.exe agent/run.py monitor    Check OLG for new draw (once)
  C:\\Python314\\python.exe agent/run.py watch      Check OLG every 6 hours
"""

def main():
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "chat":
        from agent.chat import main as chat_main
        chat_main()

    elif mode == "monitor":
        from agent.monitor import main as monitor_main
        monitor_main("once")

    elif mode == "watch":
        from agent.monitor import main as monitor_main
        monitor_main("watch")

    else:
        print(f"Unknown mode: {mode}")
        print(USAGE)
        sys.exit(1)


if __name__ == "__main__":
    main()
