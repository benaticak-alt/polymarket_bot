"""CLI for the paper core portfolio.

  python -m portfolio status   --prices prices.csv
  python -m portfolio plan     --prices prices.csv [--apply]
  python -m portfolio deposit  250
  python -m portfolio sweep    [--apply]
  python -m portfolio backtest (--prices prices.csv | --synthetic)

Prices for status/plan come from the last row of a CSV (see load_prices_csv)
or from repeated --price ASSET=VALUE flags.
"""

import argparse
import sys

import pandas as pd

import bankroll_state
from config import PORTFOLIO_EXAMPLE_PATH, PORTFOLIO_STATE_PATH, PORTFOLIO_TARGETS_PATH
from portfolio import backtest, ledger as ledger_mod, settings as settings_mod, sweep
from portfolio.broker import PaperBroker
from portfolio.rebalancer import plan


def _settings():
    path = PORTFOLIO_TARGETS_PATH
    if not path.exists():
        print(f"[!] {path.name} not found; using ILLUSTRATIVE placeholders from "
              f"{PORTFOLIO_EXAMPLE_PATH.name}. Copy it and set your own targets.\n")
        path = PORTFOLIO_EXAMPLE_PATH
    return settings_mod.load(path)


def _prices(args) -> dict[str, float]:
    px = {}
    if args.prices:
        last = backtest.load_prices_csv(args.prices).ffill().iloc[-1]
        px.update({k: float(v) for k, v in last.items() if pd.notna(v)})
    for kv in args.price or []:
        k, _, v = kv.partition("=")
        px[k] = float(v)
    if not px:
        sys.exit("need --prices CSV or --price ASSET=VALUE")
    return px


def _print_drift(p):
    print(f"Portfolio value: ${p.value:,.2f}")
    if p.value <= 0:
        print("Empty. Add cash with `deposit` or `sweep --apply`.")
        return
    print(f"{'asset':<16}{'weight':>9}{'target':>9}{'dev':>9}{'band':>8}")
    for r in p.drift:
        flag = "  <-- outside band" if r.breached else ""
        print(f"{r.asset:<16}{r.weight:>9.2%}{r.target:>9.2%}{r.deviation:>+9.2%}"
              f"{r.band:>8.2%}{flag}")


def cmd_status(args):
    s, led = _settings(), ledger_mod.load(PORTFOLIO_STATE_PATH)
    p = plan(led, _prices(args), s)
    _print_drift(p)
    print(f"Cash: ${led.cash:,.2f} (target {s.cash_target:.1%})   "
          f"contributed: ${led.contributed:,.2f}   costs paid: ${led.costs_paid:,.2f}")
    print(f"Next action: {p.mode}")


def cmd_plan(args):
    s, led = _settings(), ledger_mod.load(PORTFOLIO_STATE_PATH)
    px = _prices(args)
    p = plan(led, px, s)
    _print_drift(p)
    if not p.orders:
        print("\nNo trades needed. Everything is inside its band.")
        return
    print(f"\nMode: {p.mode}")
    for o in p.orders:
        print(f"  {o.side.upper():<5}{o.asset:<16}${o.notional:>11,.2f}  "
              f"({o.units:,.6f} units, est cost ${o.est_cost:,.2f})")
    if args.apply:
        PaperBroker(s).execute(p, led, px)
        ledger_mod.save(led, PORTFOLIO_STATE_PATH)
        print(f"\nApplied to paper ledger ({PORTFOLIO_STATE_PATH.name}).")
    else:
        print("\n(dry run: add --apply to fill these on the paper ledger)")


def cmd_deposit(args):
    led = ledger_mod.load(PORTFOLIO_STATE_PATH)
    led.deposit(args.amount)
    ledger_mod.save(led, PORTFOLIO_STATE_PATH)
    print(f"Deposited ${args.amount:,.2f}. Cash now ${led.cash:,.2f}.")


def cmd_sweep(args):
    sp = sweep.plan_sweep(bankroll_state.load()["bankroll"])
    print(sp.reason)
    if sp.amount <= 0:
        return
    if not args.apply:
        print(f"Would sweep ${sp.amount:,.2f} to the core. Add --apply to do it on paper.")
        return
    led = ledger_mod.load(PORTFOLIO_STATE_PATH)
    sweep.apply_sweep(sp, led)
    ledger_mod.save(led, PORTFOLIO_STATE_PATH)
    print(f"Swept ${sp.amount:,.2f}. Run `plan` to deploy it.")


def cmd_backtest(args):
    s = _settings()
    if args.synthetic:
        print("[!] SYNTHETIC prices: this tests the code, not any strategy.\n")
        prices = backtest.synthetic_prices(list(s.targets), seed=args.seed)
    elif args.prices:
        prices = backtest.load_prices_csv(args.prices)
    else:
        sys.exit("need --prices CSV or --synthetic")
    table = backtest.compare(prices, s, backtest.default_strategies(),
                             initial=args.initial, contribution=args.contribution)
    fmt = {"final": "${:,.0f}", "contributed": "${:,.0f}", "CAGR": "{:.2%}",
           "vol": "{:.2%}", "sharpe": "{:.2f}", "max_dd": "{:.1%}",
           "turnover/yr": "{:.1%}", "costs": "${:,.2f}", "trades": "{:d}"}
    out = table.copy().astype(object)
    for col, f in fmt.items():
        out[col] = [f.format(v) for v in table[col]]
    years = (prices.index[-1] - prices.index[0]).days / 365.25
    print(f"{len(prices)} rows, {prices.index[0].date()} -> {prices.index[-1].date()} "
          f"({years:.1f}y)\n")
    print(out.to_string())
    print("\nA policy beats buy_hold only if it wins on CAGR or drawdown AFTER costs "
          "and the win holds on data it wasn't tuned on.")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m portfolio")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def price_args(p):
        p.add_argument("--prices", help="wide CSV: date + one column per asset")
        p.add_argument("--price", action="append", metavar="ASSET=VALUE")

    price_args(sp := sub.add_parser("status")); sp.set_defaults(fn=cmd_status)
    price_args(sp := sub.add_parser("plan")); sp.set_defaults(fn=cmd_plan)
    sp.add_argument("--apply", action="store_true")
    sp = sub.add_parser("deposit"); sp.set_defaults(fn=cmd_deposit)
    sp.add_argument("amount", type=float)
    sp = sub.add_parser("sweep"); sp.set_defaults(fn=cmd_sweep)
    sp.add_argument("--apply", action="store_true")
    sp = sub.add_parser("backtest"); sp.set_defaults(fn=cmd_backtest)
    sp.add_argument("--prices")
    sp.add_argument("--synthetic", action="store_true")
    sp.add_argument("--seed", type=int, default=7)
    sp.add_argument("--initial", type=float, default=10_000)
    sp.add_argument("--contribution", type=float, default=500)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
