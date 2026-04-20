"""
main.py -- CLI entry point for lottery-portfolio.

Subcommands
-----------
  status
      Show per-lottery draw/payout row counts and last fit/prediction info.

  scrape --lottery lottomax|649|dailygrand
      Delegate to agent/olg_scraper.py to fetch the latest draw + payouts.

  fit --lottery L [--heldout FRAC]
      Fit the popularity model to draws+payouts for lottery L. Persists params
      to data/portfolio_state.json.

  optimize --lottery L --jackpot DOLLARS [--tickets-sold N] [--skip-qa]
      Compute the min-popularity combination and its EV estimate. Logs the
      prediction to data/portfolio_log.csv. Runs the QA gate first unless --skip-qa.

  log --lottery L --date YYYY-MM-DD --numbers N... --bonus N
      Append one observed draw row to draws_<L>.csv. Validates date and range.

  backtest --lottery L [--min-train N]
      Rolling-origin backtest of realized EV/dollar vs uniform-random baseline.
      (Implementation in src/backtest.py.)

Python: C:\\Python314\\python.exe  (ASCII print only)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import config
from src import data_loader, popularity_model, ev_optimizer, qa_gate


LOTTERY_CHOICES = list(config.LOTTERY_RULES.keys())


# ---------------------------------------------------------------------------
# Portfolio state I/O
# ---------------------------------------------------------------------------

def _load_state() -> dict:
    path = Path(config.PORTFOLIO_STATE)
    if not path.exists() or path.stat().st_size == 0:
        return {"lotteries": {k: {} for k in LOTTERY_CHOICES}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return {"lotteries": {k: {} for k in LOTTERY_CHOICES}}
    data.setdefault("lotteries", {})
    for k in LOTTERY_CHOICES:
        data["lotteries"].setdefault(k, {})
    return data


def _save_state(state: dict) -> None:
    path = Path(config.PORTFOLIO_STATE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def _current_params(state: dict, lottery: str) -> dict:
    """Return fitted params if available, else literature priors."""
    fit = state["lotteries"].get(lottery, {}).get("fit")
    if fit and "params" in fit:
        return dict(fit["params"])
    return {k: float(v) for k, v in config.POPULARITY_PRIORS.items()}


# ---------------------------------------------------------------------------
# Portfolio log (predictions)
# ---------------------------------------------------------------------------

PORTFOLIO_LOG_COLUMNS = [
    "timestamp", "lottery", "target_date",
    "combination", "jackpot", "n_tickets_sold",
    "ev_gross", "ev_net", "p_jackpot", "e_other_jackpot_winners",
]


def _append_portfolio_log(row: dict) -> None:
    import pandas as pd
    path = Path(config.PORTFOLIO_LOG)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{c: row.get(c, "") for c in PORTFOLIO_LOG_COLUMNS}])
    if path.exists() and path.stat().st_size > 0:
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_status(_args) -> int:
    counts = data_loader.summary_row_counts()
    state = _load_state()

    print("=" * 64)
    print("lottery-portfolio status")
    print("=" * 64)
    print(f"Data dir: {config.DATA_DIR}")
    for lot in LOTTERY_CHOICES:
        cfg = config.LOTTERY_RULES[lot]
        n_draws, n_payouts = counts.get(lot, (0, 0))
        print(f"\n[{cfg['name']}]  ({lot})")
        print(f"  draws:     {n_draws:5d} rows   ({config.DRAWS_CSV[lot]})")
        print(f"  payouts:   {n_payouts:5d} rows   ({config.PAYOUTS_CSV[lot]})")

        lstate = state["lotteries"].get(lot, {})
        fit = lstate.get("fit")
        if fit:
            print(f"  last fit:  {fit.get('date', '?')}  n_draws={fit.get('n_draws', '?')}"
                  f"  heldout_nll={fit.get('heldout_nll', 'NA')}")
            params = fit.get("params", {})
            pstr = ", ".join(f"{k}={v:.3f}" for k, v in params.items())
            print(f"  params:    {pstr}")
        else:
            print("  last fit:  (none -- using literature priors)")

        last = lstate.get("last_prediction")
        if last:
            print(f"  last pred: {last.get('target_date', '?')}  "
                  f"C={last.get('combination')}  "
                  f"ev_net={last.get('ev_net', 0.0):.4f}")
    print()
    return 0


def cmd_scrape(args) -> int:
    from agent import olg_scraper
    return olg_scraper.main(args.lottery)


def cmd_fit(args) -> int:
    draws = data_loader.load_draws(args.lottery)
    payouts = data_loader.load_payouts(args.lottery)
    if draws.empty:
        print(f"[fit] ERROR -- no draws found for {args.lottery}. Add data first.")
        return 1
    n_tix = config.DEFAULT_TICKETS_SOLD[args.lottery]
    print(f"[fit] lottery={args.lottery}  draws={len(draws)}  payouts={len(payouts)}"
          f"  n_tickets_sold_est={n_tix}")
    result = popularity_model.fit(
        draws, payouts, args.lottery,
        n_tickets_sold=n_tix,
        heldout_fraction=args.heldout,
    )

    print(f"[fit] success={result.success}  message={result.message}")
    print(f"[fit] n_draws={result.n_draws}  nll={result.nll:.4f}  reg={result.reg:.4f}"
          f"  heldout_nll={result.heldout_nll}")
    for k, v in result.params.items():
        prior = config.POPULARITY_PRIORS.get(k, 1.0)
        print(f"  {k:20s}  fitted={v:.4f}  prior={prior:.4f}")

    state = _load_state()
    state["lotteries"][args.lottery]["fit"] = {
        "date":         datetime.now().strftime("%Y-%m-%d"),
        "params":       result.params,
        "n_draws":      result.n_draws,
        "nll":          float(result.nll),
        "reg":          float(result.reg),
        "heldout_nll":  (None if result.heldout_nll is None else float(result.heldout_nll)),
        "success":      bool(result.success),
    }
    _save_state(state)
    print(f"[fit] saved params to {config.PORTFOLIO_STATE}")
    return 0


def cmd_optimize(args) -> int:
    if not args.skip_qa:
        qa_gate.run_qa_gate()

    state = _load_state()
    params = _current_params(state, args.lottery)

    draws = data_loader.load_draws(args.lottery)
    recent = popularity_model._recent_winning_set(draws, len(draws), k=5) if not draws.empty else set()

    # For Daily Grand only, collect recent Grand Numbers so the picker can
    # pick the least-recently-drawn one. LottoMax/6/49 pass None.
    recent_grand = None
    if config.LOTTERY_RULES[args.lottery]["bonus_col"] != "bonus" and not draws.empty:
        recent_grand = [int(x) for x in draws["bonus"].tail(10).tolist()]

    n_tix = args.tickets_sold if args.tickets_sold else config.DEFAULT_TICKETS_SOLD[args.lottery]

    result = ev_optimizer.best_combination(
        args.lottery, params, jackpot=args.jackpot,
        n_tickets_sold=n_tix, recent_winning=recent,
        recent_grand=recent_grand,
    )
    combo = result["combination"]
    grand = result.get("grand_number")
    evres = result["ev"]

    print("=" * 64)
    print(f"OPTIMIZE -- {config.LOTTERY_RULES[args.lottery]['name']}")
    print("=" * 64)
    print(f"  combination:              {combo}")
    if grand is not None:
        print(f"  grand number:             {grand}")
    print(f"  jackpot:                  ${args.jackpot:,.2f}")
    print(f"  n_tickets_sold (assumed): {n_tix:,}")
    print(f"  recent_winning_bias:      {sorted(recent)}")
    print(f"  P(combination | params):  {evres['p_combination_given_params']:.3e}")
    print(f"  P(jackpot match):         {evres['p_jackpot']:.3e}")
    print(f"  E[other jackpot winners]: {evres['e_other_jackpot_winners']:.3f}")
    print(f"  EV gross:                 ${evres['ev_gross']:.4f}")
    print(f"  EV net (minus ticket):    ${evres['ev_net']:.4f}")
    print(f"  ticket cost:              ${evres['ticket_cost']:.2f}")
    print()
    print("  tier breakdown:")
    for t in evres["tier_breakdown"]:
        bonus = "-" if t["tier_bonus"] is None else ("Y" if t["tier_bonus"] else "N")
        contrib = t.get("contribution", 0.0)
        print(f"    {t['tier_main']}+{bonus}  "
              f"p={t['p_tier']:.3e}  "
              f"model={t['payout_model']:32s}  "
              f"contrib=${contrib:.4f}")

    target_date = args.target_date or datetime.now().strftime("%Y-%m-%d")
    combo_str = " ".join(str(n) for n in combo)
    if grand is not None:
        combo_str = f"{combo_str} | grand={grand}"
    _append_portfolio_log({
        "timestamp":      datetime.now().isoformat(timespec="seconds"),
        "lottery":        args.lottery,
        "target_date":    target_date,
        "combination":    combo_str,
        "jackpot":        args.jackpot,
        "n_tickets_sold": n_tix,
        "ev_gross":       evres["ev_gross"],
        "ev_net":         evres["ev_net"],
        "p_jackpot":      evres["p_jackpot"],
        "e_other_jackpot_winners": evres["e_other_jackpot_winners"],
    })
    state["lotteries"][args.lottery]["last_prediction"] = {
        "target_date": target_date,
        "combination": combo,
        "grand_number": grand,
        "jackpot":     args.jackpot,
        "ev_gross":    evres["ev_gross"],
        "ev_net":      evres["ev_net"],
    }
    _save_state(state)
    print(f"\n[optimize] logged to {config.PORTFOLIO_LOG}")
    return 0


def cmd_log(args) -> int:
    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print(f"[log] ERROR -- date {args.date!r} must be YYYY-MM-DD")
        return 1

    cfg = config.LOTTERY_RULES[args.lottery]
    if len(args.numbers) != cfg["main_count"]:
        print(f"[log] ERROR -- {args.lottery} expects {cfg['main_count']} numbers, "
              f"got {len(args.numbers)}")
        return 1
    for n in args.numbers:
        if not (1 <= n <= cfg["main_max"]):
            print(f"[log] ERROR -- number {n} outside [1, {cfg['main_max']}]")
            return 1
    if not (1 <= args.bonus <= cfg["bonus_max"]):
        print(f"[log] ERROR -- bonus {args.bonus} outside [1, {cfg['bonus_max']}]")
        return 1
    if len(set(args.numbers)) != len(args.numbers):
        print(f"[log] ERROR -- duplicate main numbers: {args.numbers}")
        return 1

    data_loader.append_draw(args.lottery, args.date, args.numbers, args.bonus)
    print(f"[log] appended  {args.lottery}  {args.date}  "
          f"main={sorted(args.numbers)}  bonus={args.bonus}")
    return 0


def cmd_backtest(args) -> int:
    try:
        from src import backtest
    except ImportError:
        print("[backtest] ERROR -- src/backtest.py not yet implemented.")
        return 1
    return backtest.run(args.lottery, min_train=args.min_train)


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="main.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="Show data + fit status")

    p_scrape = sub.add_parser("scrape", help="Fetch latest draw + payouts from OLG")
    p_scrape.add_argument("--lottery", required=True, choices=LOTTERY_CHOICES)

    p_fit = sub.add_parser("fit", help="Fit popularity model")
    p_fit.add_argument("--lottery", required=True, choices=LOTTERY_CHOICES)
    p_fit.add_argument("--heldout", type=float, default=config.HELDOUT_FRACTION)

    p_opt = sub.add_parser("optimize", help="Recommend best combination")
    p_opt.add_argument("--lottery", required=True, choices=LOTTERY_CHOICES)
    p_opt.add_argument("--jackpot", type=float, required=True)
    p_opt.add_argument("--tickets-sold", type=int, default=None)
    p_opt.add_argument("--target-date", type=str, default=None,
                       help="YYYY-MM-DD of upcoming draw (default: today)")
    p_opt.add_argument("--skip-qa", action="store_true",
                       help="Skip pytest QA gate (default: off -- gate runs)")

    p_log = sub.add_parser("log", help="Append an observed draw")
    p_log.add_argument("--lottery", required=True, choices=LOTTERY_CHOICES)
    p_log.add_argument("--date", required=True, help="YYYY-MM-DD")
    p_log.add_argument("--numbers", type=int, nargs="+", required=True)
    p_log.add_argument("--bonus", type=int, required=True)

    p_bt = sub.add_parser("backtest", help="Rolling-origin backtest vs uniform baseline")
    p_bt.add_argument("--lottery", required=True, choices=LOTTERY_CHOICES)
    p_bt.add_argument("--min-train", type=int, default=20,
                      help="Minimum draws needed before first prediction")

    return p


def main() -> int:
    args = build_parser().parse_args()
    dispatch = {
        "status":   cmd_status,
        "scrape":   cmd_scrape,
        "fit":      cmd_fit,
        "optimize": cmd_optimize,
        "log":      cmd_log,
        "backtest": cmd_backtest,
    }
    handler = dispatch.get(args.cmd)
    if handler is None:
        print(f"Unknown command: {args.cmd}")
        return 1
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
