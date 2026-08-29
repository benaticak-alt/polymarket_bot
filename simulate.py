"""
Bankroll simulation from paper trading logs.
Replays logged trades against a starting bankroll and projects forward.

Usage:
    python simulate.py                    # default $100 start, all retrains, 30-day projection
    python simulate.py 250                # custom starting bankroll
    python simulate.py 250 15             # only trades from retrain >= 15
    python simulate.py 250 15 14          # project 14 days instead of 30
"""

import csv
import sys
import time
from pathlib import Path
from collections import defaultdict

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR   = Path(__file__).parent
LOG_PATHS  = [BASE_DIR / "papertrade_log_5m.csv", BASE_DIR / "papertrade_log_15m.csv"]

MAX_POSITION_USDC = 500.0  # must match config.py / .env
MIN_POSITION_USDC = 1.0    # must match config.py
KELLY_MAX_PCT     = 0.15   # must match config.py — max bet as % of bankroll
STARTING_BANKROLL = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
MIN_RETRAIN       = int(sys.argv[2])   if len(sys.argv) > 2 else 0
PROJ_DAYS         = int(sys.argv[3])   if len(sys.argv) > 3 else 30


# ── Load trades ────────────────────────────────────────────────────────────────

def load_active_trades():
    rows = []
    for path in LOG_PATHS:
        if path.exists():
            with open(path, "r", newline="") as f:
                rows.extend(csv.DictReader(f))
    rows = [
        r for r in rows
        if r["outcome"] in ("WIN", "LOSS")
        and float(r["ev_fraction"]) > 0
        and int(r.get("train_count") or 0) >= MIN_RETRAIN
        # order_placed column added later; missing means old row, treat as placed
        and int(r.get("order_placed") or 1) == 1
    ]
    rows.sort(key=lambda r: float(r["timestamp"]))
    return rows


# ── P&L per trade ─────────────────────────────────────────────────────────────

POLYMARKET_FEE = 0.02  # must match config.py

def pnl(trade: dict, stake: float) -> float:
    """Binary market P&L after Polymarket's 2% fee on winnings."""
    price = float(trade["market_price"])
    if trade["outcome"] == "WIN":
        return stake * (1.0 - price) / price * (1.0 - POLYMARKET_FEE)
    return -stake


# ── Simulation ─────────────────────────────────────────────────────────────────

def _dd_multiplier(peak: float, bankroll: float) -> float:
    """Same brake curve as bankroll_state.drawdown_multiplier."""
    if peak <= 0:
        return 1.0
    dd = (peak - bankroll) / peak
    if dd <= 0.05:
        return 1.0
    return 1.0 - min((dd - 0.05) / 0.25, 1.0) * 0.75


MIN_POSITION_USDC = 1.0   # must match config.py


def simulate(trades: list, starting: float, proportional: bool,
             braked: bool = False, capped: bool = False, auto_scale: bool = False) -> dict:
    """
    proportional=False: use logged stake_usdc directly (fixed dollar amounts).
    proportional=True:  rescale stakes to current bankroll (Kelly grows with wins).
    braked=True:        proportional Kelly + drawdown brake.
    capped=True:        proportional Kelly but hard-capped to $1–$10 per trade.
    auto_scale=True:    max bet = KELLY_MAX_PCT % of current bankroll — grows and
                        shrinks with the account automatically.
    """
    bankroll   = starting
    peak       = starting
    max_dd     = 0.0
    equity     = [starting]
    daily_pnl  = defaultdict(float)
    wins = losses = 0

    for t in trades:
        logged_stake = float(t["stake_usdc"])
        kelly_frac   = min(1.0, logged_stake / MAX_POSITION_USDC)  # fraction of max used (0–1)

        if auto_scale:
            dyn_max = max(MIN_POSITION_USDC, bankroll * KELLY_MAX_PCT)
            stake   = kelly_frac * dyn_max
            if braked:
                stake *= _dd_multiplier(peak, bankroll)
            stake = max(MIN_POSITION_USDC, min(dyn_max, stake))
        elif proportional:
            stake = bankroll * (logged_stake / MAX_POSITION_USDC) * (MAX_POSITION_USDC / starting)
            stake = max(0.01, min(stake, bankroll * 0.30))
            if braked:
                stake *= _dd_multiplier(peak, bankroll)
            if capped:
                stake = max(MIN_POSITION_USDC, min(MAX_POSITION_USDC, stake))
        else:
            stake = logged_stake

        gain     = pnl(t, stake)
        bankroll = max(0.01, bankroll + gain)
        equity.append(bankroll)

        peak   = max(peak, bankroll)
        dd     = (peak - bankroll) / peak
        max_dd = max(max_dd, dd)

        day = time.strftime("%Y-%m-%d", time.localtime(float(t["timestamp"])))
        daily_pnl[day] += gain

        if t["outcome"] == "WIN":
            wins += 1
        else:
            losses += 1

    return {
        "final":    bankroll,
        "equity":   equity,
        "max_dd":   max_dd,
        "daily":    daily_pnl,
        "wins":     wins,
        "losses":   losses,
    }


# ── Display ────────────────────────────────────────────────────────────────────

def display(trades: list) -> None:
    if not trades:
        print("No settled active trades found in logs.")
        return

    first_ts = float(trades[0]["timestamp"])
    last_ts  = float(trades[-1]["timestamp"])
    span_days = max((last_ts - first_ts) / 86400, 1 / 24)

    total    = len(trades)
    wins     = sum(1 for t in trades if t["outcome"] == "WIN")
    win_rate = wins / total

    # Average trade edge: E[pnl / stake]
    edges = []
    for t in trades:
        p  = float(t["market_price"])
        wr = 1.0 if t["outcome"] == "WIN" else 0.0
        edges.append(wr * (1 - p) / p - (1 - wr))
    avg_edge = sum(edges) / len(edges)

    retrain_label = f"retrain >= {MIN_RETRAIN}" if MIN_RETRAIN > 0 else "all retrains"
    print("=" * 58)
    print(f"  BANKROLL SIMULATION — starting ${STARTING_BANKROLL:.2f}")
    print(f"  Filter: {retrain_label}")
    print("=" * 58)
    print(f"\n  Sample:  {total} active trades over {span_days:.1f} day(s)")
    print(f"  Win rate: {win_rate:.1%}  |  Avg edge per trade: {avg_edge:+.3f}")

    # ── Fixed-amount simulation ────────────────────────────────────────────────
    r_fixed = simulate(trades, STARTING_BANKROLL, proportional=False)
    fixed_return = (r_fixed["final"] - STARTING_BANKROLL) / STARTING_BANKROLL

    # ── Proportional Kelly simulation ──────────────────────────────────────────
    r_prop = simulate(trades, STARTING_BANKROLL, proportional=True)
    prop_return = (r_prop["final"] - STARTING_BANKROLL) / STARTING_BANKROLL

    # ── Kelly + drawdown brake simulation ──────────────────────────────────────
    r_brake = simulate(trades, STARTING_BANKROLL, proportional=True, braked=True)
    brake_return = (r_brake["final"] - STARTING_BANKROLL) / STARTING_BANKROLL

    # ── Capped Kelly ($1–$10) — mirrors live bot behaviour ─────────────────────
    r_capped = simulate(trades, STARTING_BANKROLL, proportional=True, capped=True)
    capped_return = (r_capped["final"] - STARTING_BANKROLL) / STARTING_BANKROLL

    # ── Auto-scale Kelly — max bet = 10% of current bankroll ───────────────────
    r_auto = simulate(trades, STARTING_BANKROLL, proportional=True, auto_scale=True)
    auto_return = (r_auto["final"] - STARTING_BANKROLL) / STARTING_BANKROLL

    r_auto_braked = simulate(trades, STARTING_BANKROLL, proportional=True, auto_scale=True, braked=True)
    auto_braked_return = (r_auto_braked["final"] - STARTING_BANKROLL) / STARTING_BANKROLL

    col = 14
    print(f"\n  {'':─<94}")
    print(f"  {'MODE':<22} {'FIXED($1-10)':>{col}}  {'CAPPED($1-10)':>{col}}  {'AUTO-SCALE':>{col}}  {'AUTO+BRAKE':>{col}}")
    print(f"  {'':─<94}")
    for label, rf, rc, ra, rab in [
        ("Starting",   STARTING_BANKROLL,   STARTING_BANKROLL,   STARTING_BANKROLL,   STARTING_BANKROLL),
        ("Final",      r_fixed["final"],    r_capped["final"],   r_auto["final"],     r_auto_braked["final"]),
    ]:
        print(f"  {label+' bankroll':<22} {'${:.2f}'.format(rf):>{col}}  {'${:.2f}'.format(rc):>{col}}  {'${:.2f}'.format(ra):>{col}}  {'${:.2f}'.format(rab):>{col}}")
    print(f"  {'Return':<22} {fixed_return:>+{col-1}.1%}  {capped_return:>+{col-1}.1%}  {auto_return:>+{col-1}.1%}  {auto_braked_return:>+{col-1}.1%}")
    print(f"  {'Max drawdown':<22} {r_fixed['max_dd']:>+{col-1}.1%}  {r_capped['max_dd']:>+{col-1}.1%}  {r_auto['max_dd']:>+{col-1}.1%}  {r_auto_braked['max_dd']:>+{col-1}.1%}")
    print(f"  {'':─<94}")
    print(f"  AUTO-SCALE: max bet = {KELLY_MAX_PCT:.0%} of current bankroll — grows with wins, shrinks with losses.")

    # ── Per-day breakdown ──────────────────────────────────────────────────────
    print(f"\n  DAILY P&L  (fixed stakes)")
    print(f"  {'Date':<12} {'Trades':>7} {'P&L':>8} {'Balance':>9}")
    print(f"  {'':─<40}")

    # Rebuild equity day by day for fixed mode
    running = STARTING_BANKROLL
    daily_counts = defaultdict(int)
    for t in trades:
        day = time.strftime("%Y-%m-%d", time.localtime(float(t["timestamp"])))
        daily_counts[day] += 1

    running = STARTING_BANKROLL
    for day in sorted(r_fixed["daily"]):
        dpnl  = r_fixed["daily"][day]
        running += dpnl
        print(f"  {day:<12} {daily_counts[day]:>7} {dpnl:>+7.2f}  ${running:>8.2f}")

    # ── 30-day projection ──────────────────────────────────────────────────────
    trades_per_day = total / span_days
    proj_days      = PROJ_DAYS

    # Average market price and stake across all trades
    avg_price = sum(float(t["market_price"]) for t in trades) / total
    avg_stake = sum(float(t["stake_usdc"]) for t in trades) / total
    avg_odds  = (1.0 - avg_price) / avg_price   # payout ratio per $ staked

    # Bayesian shrinkage: blend observed win rate toward 50% using N pseudo-observations.
    # With 87 trades the observed rate is noisy; with 500+ it converges to the true rate.
    PRIOR = 100   # equivalent to 100 coin-flip observations at 50%
    shrunk_wr = (wins + PRIOR * 0.5) / (total + PRIOR)

    def proj_fixed(wr: float, days: int) -> float:
        """Linear extrapolation — fixed stakes don't compound."""
        daily = (wr * avg_stake * avg_odds - (1 - wr) * avg_stake) * trades_per_day
        return STARTING_BANKROLL + daily * days

    def proj_capped(wr: float, days: int) -> float:
        """Kelly scaling capped at $1–$10: compounds while small, then linear at $10/trade."""
        bankroll = STARTING_BANKROLL
        per_trade_edge = wr * avg_odds - (1 - wr)
        n_trades = int(trades_per_day * days)
        for _ in range(n_trades):
            stake_frac = avg_stake / STARTING_BANKROLL
            stake = bankroll * stake_frac
            stake = max(MIN_POSITION_USDC, min(MAX_POSITION_USDC, stake))
            bankroll += stake * per_trade_edge
            bankroll = max(0.01, bankroll)
        return bankroll

    def proj_kelly(wr: float, days: int) -> float:
        """Geometric compounding — stake scales with bankroll each trade."""
        avg_stake_frac = avg_stake / STARTING_BANKROLL
        per_trade_edge = wr * avg_odds - (1 - wr)
        per_trade_mult = max(0.5, 1.0 + avg_stake_frac * per_trade_edge)
        return STARTING_BANKROLL * (per_trade_mult ** (trades_per_day * days))

    if total < 200:
        reliability = f"LOW ({total} trades) — treat as rough estimate only"
    elif total < 500:
        reliability = f"MEDIUM ({total} trades)"
    else:
        reliability = f"HIGH ({total} trades)"

    print(f"\n  PROJECTION  ({trades_per_day:.0f} trades/day)")
    print(f"  Reliability: {reliability}")
    print(f"  Conservative win rate: {shrunk_wr:.1%} (Bayesian-shrunk from observed {win_rate:.1%})")
    print()
    kelly_days  = min(proj_days, 14)
    kelly_label = f"{kelly_days}-day" + (" (capped; grows too fast beyond 14d)" if proj_days > 14 else "")

    short     = min(7, proj_days)
    two_cols  = short < proj_days

    print(f"  FIXED STAKES ($1-$10) — linear growth")
    print(f"  {'':─<52}")
    if two_cols:
        print(f"  {'Scenario':<18} {'Win rate':>9}  {f'{short}-day':>8}  {f'{proj_days}-day':>8}")
    else:
        print(f"  {'Scenario':<18} {'Win rate':>9}  {f'{proj_days}-day':>8}")
    print(f"  {'':─<52}")
    for label, wr in [("Conservative", shrunk_wr), ("Observed", win_rate)]:
        ps = proj_fixed(wr, short)
        pl = proj_fixed(wr, proj_days)
        if two_cols:
            print(f"  {label:<18} {wr:>8.1%}  {'${:.0f}'.format(ps):>8}  {'${:.0f}'.format(pl):>8}")
        else:
            print(f"  {label:<18} {wr:>8.1%}  {'${:.0f}'.format(pl):>8}")
    print()
    print(f"  CAPPED KELLY ($1-$10) — how the bot actually works")
    print(f"  {'':─<52}")
    if two_cols:
        print(f"  {'Scenario':<18} {'Win rate':>9}  {f'{short}-day':>8}  {f'{proj_days}-day':>8}")
    else:
        print(f"  {'Scenario':<18} {'Win rate':>9}  {f'{proj_days}-day':>8}")
    print(f"  {'':─<52}")
    for label, wr in [("Conservative", shrunk_wr), ("Observed", win_rate)]:
        ps = proj_capped(wr, short)
        pl = proj_capped(wr, proj_days)
        if two_cols:
            print(f"  {label:<18} {wr:>8.1%}  {'${:.0f}'.format(ps):>8}  {'${:.0f}'.format(pl):>8}")
        else:
            print(f"  {label:<18} {wr:>8.1%}  {'${:.0f}'.format(pl):>8}")
    print()
    def proj_kelly_braked(wr: float, days: int) -> float:
        """Kelly + brake: assume average 15% drawdown drag (conservative estimate)."""
        avg_stake_frac = avg_stake / STARTING_BANKROLL
        per_trade_edge = wr * avg_odds - (1 - wr)
        # Brake reduces stakes by ~15% on average during normal variance
        braked_frac    = avg_stake_frac * 0.85
        per_trade_mult = max(0.5, 1.0 + braked_frac * per_trade_edge)
        return STARTING_BANKROLL * (per_trade_mult ** (trades_per_day * days))

    print(f"  AUTO-SCALE KELLY ({KELLY_MAX_PCT:.0%}/trade, {kelly_label})")
    print(f"  {'':─<54}")
    print(f"  {'Scenario':<18} {'Win rate':>9}  {f'{kelly_days}-day (no brake)':>18}  {f'{kelly_days}-day (+brake)':>15}")
    print(f"  {'':─<54}")
    for label, wr in [("Conservative", shrunk_wr), ("Observed", win_rate)]:
        per_trade_edge = wr * avg_odds - (1 - wr)
        avg_kelly_frac = (avg_stake / MAX_POSITION_USDC) * KELLY_MAX_PCT
        try:
            pa  = STARTING_BANKROLL * (max(0.5, 1 + avg_kelly_frac * per_trade_edge) ** (trades_per_day * kelly_days))
        except OverflowError:
            pa = float("inf")
        try:
            pab = STARTING_BANKROLL * (max(0.5, 1 + avg_kelly_frac * 0.85 * per_trade_edge) ** (trades_per_day * kelly_days))
        except OverflowError:
            pab = float("inf")
        pa_str  = ">$1T" if pa  == float("inf") or pa  > 1e12 else "${:.0f}".format(pa)
        pab_str = ">$1T" if pab == float("inf") or pab > 1e12 else "${:.0f}".format(pab)
        print(f"  {label:<18} {wr:>8.1%}  {pa_str:>18}  {pab_str:>15}")
    print("=" * 58)


if __name__ == "__main__":
    trades = load_active_trades()
    display(trades)
