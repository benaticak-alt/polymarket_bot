"""
Live paper trading performance dashboard.
Run in a separate terminal: python paper_stats.py
Refreshes every 10 seconds.
"""

import csv, re, time, os, sys, json
from pathlib import Path
from collections import defaultdict
from typing import Optional
import numpy as np

# Only show data for assets the bot is configured to trade
try:
    from config import TRADING_ASSETS as _TRADING_ASSETS
    _KNOWN_ASSETS = {a["name"] for a in _TRADING_ASSETS}
except Exception:
    _KNOWN_ASSETS = {"btc"}

try:
    from config import MIN_TRAIN_SAMPLES as _MIN_TRAIN, RETRAIN_INTERVAL as _RETRAIN_INT
except Exception:
    _MIN_TRAIN, _RETRAIN_INT = 50, 25

try:
    from config import POLYMARKET_FEE as FEE
except Exception:
    FEE = 0.02

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).parent


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def _discover_log_paths(prefix: str) -> list[Path]:
    """Return all papertrade log CSV files sorted by name."""
    return sorted(BASE_DIR.glob(f"{prefix}_*.csv"))


def _asset_from_stem(stem: str) -> str | None:
    """
    Parse the asset name from a log filename stem.
    papertrade_log_5m        → "btc"
    papertrade_log_5m_eth    → "eth"
    papertrade_log_5m_pre51signals → None  (unknown / archive — skip)
    """
    m = re.match(r"papertrade_log_(?:5m|15m)(?:_(.+))?$", stem)
    if not m:
        return None
    asset = m.group(1) or "btc"
    return asset if asset in _KNOWN_ASSETS else None


def _valid_ts(row) -> bool:
    try:
        float(row.get("timestamp", ""))
        return True
    except (TypeError, ValueError):
        return False


def load_trades():
    """Load all papertrade logs across all assets and intervals.

    Hardened against corrupted rows (e.g. null-byte damage from two bot instances
    writing the same file): skips any row without a parseable timestamp and never
    lets one bad file crash the whole dashboard.
    """
    try:
        csv.field_size_limit(2**31 - 1)   # tolerate an oversized corrupt field
    except (OverflowError, ValueError):
        pass
    trades = []
    for path in _discover_log_paths("papertrade_log"):
        asset = _asset_from_stem(path.stem)
        if asset is None:
            continue  # skip archive / unknown files
        try:
            # errors="replace" strips null-byte corruption; DictReader on a header
            # already lost to corruption just yields junk rows, which _valid_ts drops.
            with open(path, "r", newline="", encoding="utf-8", errors="replace") as f:
                for row in csv.DictReader(f):
                    if not _valid_ts(row) or row.get("interval") not in ("5m", "15m"):
                        continue
                    row["_asset"] = asset
                    trades.append(row)
        except Exception:
            continue  # a mangled file never takes the dashboard down
    trades.sort(key=lambda t: float(t["timestamp"]))
    return trades


def load_model_status() -> list:
    """Read per-model replay buffers directly — the real source of truth for
    retraining. The CSV train_count column lags and resets when logs are cleared.
    Buffer file naming mirrors main.py: btc → '5m'/'15m', others → '5m_<asset>'."""
    weights_dir = BASE_DIR / "model" / "weights"
    out = []
    for asset in sorted(_KNOWN_ASSETS):
        suffix = "" if asset == "btc" else f"_{asset}"
        for interval in ("5m", "15m"):
            name = f"{interval}{suffix}"
            path = weights_dir / f"replay_buffer_{name}.npz"
            n, tc = 0, 0
            if path.exists():
                try:
                    # Close the handle immediately: the bot rewrites this file via
                    # os.replace, which fails on Windows if we hold it open.
                    with np.load(path, allow_pickle=True) as d:
                        n = len(d["X"])
                        if "meta" in d.files:
                            tc = int(d["meta"][0])
                except Exception:
                    pass
            out.append({"label": f"{interval} {asset.upper()}", "samples": n, "retrains": tc})
    return out


def _print_model_section() -> None:
    print(f"\n  MODEL  (per-model; buffer counts every settled outcome, bet or not)")
    print(f"  {'Model':<10} {'Buffer':<10} {'Retrains':<9} {'Next retrain'}")
    print(f"  {'-'*50}")
    for m in load_model_status():
        n, tc = m["samples"], m["retrains"]
        if tc == 0:
            need = max(0, _MIN_TRAIN - n)
            buf  = f"{n}/{_MIN_TRAIN}"
            nxt  = f"in {need} outcomes" if need > 0 else "imminent"
        else:
            # _new_since_train resets on restart, so this is the worst-case wait
            buf  = str(n)
            nxt  = f"≤{_RETRAIN_INT} outcomes"
        print(f"  {m['label']:<10} {buf:<10} {tc:<9} {nxt}")


def _print_shadow_section() -> None:
    """Chainlink-vs-Binance settlement divergence. Silent unless shadow logging
    has written rows (i.e. the Data Streams client is enabled and trades settled)."""
    path = BASE_DIR / "chainlink_shadow.csv"
    if not path.exists():
        return
    try:
        with open(path, "r", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return
    if not rows:
        return

    def stats(rs):
        n = len(rs)
        fl = sum(1 for r in rs if r.get("label_flip") == "1")
        return n, fl, (fl / n * 100 if n else 0.0)

    n, fl, rate = stats(rows)
    print(f"\n  CHAINLINK SHADOW  (Binance vs Chainlink settlement; flip = corrupted label)")
    print(f"  {'Label flips:':<22} {fl}/{n}  ({rate:.1f}%)")

    by_int = defaultdict(list)
    for r in rows:
        by_int[r.get("interval", "?")].append(r)
    parts = []
    for interval in sorted(by_int):
        gn, gf, gr = stats(by_int[interval])
        parts.append(f"{interval} {gf}/{gn} ({gr:.0f}%)")
    print(f"  {'By interval:':<22} " + "   ".join(parts))

    rn, rf, rr = stats(rows[-50:])
    print(f"  {'Last 50:':<22} {rf} flips ({rr:.1f}%)")


def load_bankroll():
    p = Path(__file__).parent / "bankroll.json"
    try:
        if p.exists():
            return json.loads(p.read_text())
    except Exception:
        pass
    return None


def display():
    trades  = load_trades()
    settled = [t for t in trades if t["outcome"] in ("WIN", "LOSS")]
    pending = [t for t in trades if t["outcome"] == "PENDING"]

    clear()
    print("=" * 55)
    print("  POLYMARKET BOT — PAPER TRADING DASHBOARD")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    # Bankroll status
    br = load_bankroll()
    if br:
        peak = br["peak"]
        bankroll = br["bankroll"]
        dd = (peak - bankroll) / peak if peak > 0 else 0
        dd_pct = dd * 100
        mult = max(0.25, 1.0 - min((dd - 0.05) / 0.25, 1.0) * 0.75) if dd > 0.05 else 1.0
        print(f"\n  BANKROLL")
        print(f"  {'Current:':<22} ${bankroll:.2f}  (peak ${peak:.2f})")
        print(f"  {'Drawdown:':<22} {dd_pct:.1f}%  →  stake multiplier {mult:.2f}x")

        # Live-test probe status — only shown while a real-money trial is armed.
        try:
            from config import LIVE_TEST_MODE, LIVE_TEST_STATE_PATH
            if LIVE_TEST_MODE:
                st = json.load(open(LIVE_TEST_STATE_PATH)) if LIVE_TEST_STATE_PATH.exists() else {}
                o = st.get("orders", 0); dep = st.get("deployed", 0.0)
                mo = st.get("max_orders", "?"); bud = st.get("budget", "?")
                print(f"\n  LIVE TEST  (REAL MONEY)")
                print(f"  {'Orders placed:':<22} {o}/{mo}")
                print(f"  {'Deployed:':<22} ${dep:.2f} / ${bud}")
        except Exception:
            pass

    # Active trades: only decisions where a bet was actually placed.
    # Positive-EV decisions that were blocked (confluence gate, circuit breaker)
    # are logged for training but never staked, so they don't count as trades
    # and don't move the bankroll.
    traded = [t for t in settled if str(t.get("order_placed", "0")) == "1"]

    if not traded:
        print("\n  OVERALL")
        print(f"  No active trades yet — bot is collecting training data.")
        print(f"  Decisions logged: {len(settled)}  |  Pending: {len(pending)}")
        _print_model_section()
        _print_shadow_section()
        _print_early_shadow_section()
        _print_last_decisions(settled)
        log_names = ", ".join(
        p.name for p in _discover_log_paths("papertrade_log")
        if _asset_from_stem(p.stem) is not None
    )
        print(f"\n  Logs: {log_names}")
        print(f"  Refreshing every 10s — Ctrl+C to exit")
        print("=" * 55)
        return

    total  = len(traded)
    wins   = sum(1 for t in traded if t["outcome"] == "WIN")
    losses = total - wins

    print(f"\n  OVERALL")
    print(f"  {'Active trades:':<22} {total}  (decisions logged: {len(settled)})")
    print(f"  {'Wins / Losses:':<22} {wins} / {losses}")
    print(f"  {'Win rate:':<22} {wins/total:.1%}")
    print(f"  {'Pending:':<22} {len(pending)}")

    # By asset — always show when multiple assets are configured
    if len(_KNOWN_ASSETS) > 1:
        print(f"\n  BY ASSET")
        for asset in sorted(_KNOWN_ASSETS):
            at = [t for t in traded if t["_asset"] == asset]
            if not at:
                print(f"  {asset.upper():<6} no trades yet")
                continue
            aw = sum(1 for t in at if t["outcome"] == "WIN")
            print(f"  {asset.upper():<6} {aw}/{len(at)} = {aw/len(at):.1%}")

    print(f"\n  BY INTERVAL")
    by_interval = defaultdict(list)
    for t in traded:
        by_interval[t["interval"]].append(t)
    for interval, ts in sorted(by_interval.items()):
        w = sum(1 for t in ts if t["outcome"] == "WIN")
        print(f"  {interval:<6} {w}/{len(ts)} = {w/len(ts):.1%}")

    # ── Counterfactual: what 5m WOULD have done if TRADE_5M were on ───────────
    # 5m decisions are still logged while order placement is disabled, so we can
    # price out the road not taken. Only rows blocked *solely* by "5m_disabled"
    # count — those passed every other gate and would really have been bet.
    # Two corrections keep this honest:
    #  1. require book_src=="clob" — the 5m_disabled gate fires BEFORE the
    #     no_clob_book gate, so gamma-book rows are tagged 5m_disabled even
    #     though the clob gate would really have blocked them.
    #  2. drop 2026-07-07..07-10, when frozen Chainlink prices corrupted labels
    #     (those days inflate every win rate, 5m included).
    _CORRUPT_DAYS = {"07-07", "07-08", "07-09", "07-10"}
    would5 = [t for t in settled
              if t["interval"] == "5m"
              and t.get("block_reason") == "5m_disabled"
              and t["outcome"] in ("WIN", "LOSS")
              and t.get("book_src") == "clob"
              and time.strftime("%m-%d", time.localtime(float(t["timestamp"])))
                  not in _CORRUPT_DAYS]
    if would5:
        w5    = sum(1 for t in would5 if t["outcome"] == "WIN")
        stake = sum(float(t["stake_usdc"]) for t in would5)
        pnl   = sum(
            float(t["stake_usdc"]) * (1 - float(t["market_price"]))
            / float(t["market_price"]) * (1 - FEE)
            if t["outcome"] == "WIN" else -float(t["stake_usdc"])
            for t in would5
        )
        avg_p = sum(float(t["market_price"]) for t in would5) / len(would5)
        print(f"\n  IF WE USED 5m  (counterfactual — clob-book only, corrupt days excl.)")
        print(f"  {'Would-be trades:':<22} {len(would5)}")
        print(f"  {'Win rate:':<22} {w5/len(would5):.1%}   (avg price paid {avg_p:.2f})")
        print(f"  {'Hypothetical P&L:':<22} ${pnl:+,.2f}  on ${stake:,.0f} staked")
        print(f"  {'EV per $1:':<22} {pnl/stake:+.3f}"
              f"   {'<- would LOSE money' if pnl < 0 else '<- would profit'}")


# Last 20 active trades + circuit breaker status
    recent   = traded[-20:]
    r_wins   = sum(1 for t in recent if t["outcome"] == "WIN")
    recent10 = traded[-10:]
    r10_wr   = sum(1 for t in recent10 if t["outcome"] == "WIN") / max(len(recent10), 1)
    paused   = len(recent10) >= 10 and r10_wr < 0.40
    print(f"\n  RECENT (last {len(recent)} active)")
    print(f"  {'Win rate:':<22} {r_wins/len(recent):.1%}")
    if paused:
        print(f"  *** CIRCUIT BREAKER: PAUSED ({r10_wr:.0%} last 10 < 40%) ***")

    _print_model_section()
    _print_shadow_section()
    _print_early_shadow_section()
    _print_last_decisions(settled)

    log_names = ", ".join(
        p.name for p in _discover_log_paths("papertrade_log")
        if _asset_from_stem(p.stem) is not None
    )
    print(f"\n  Logs: {log_names}")
    print(f"  Refreshing every 10s — Ctrl+C to exit")
    print("=" * 55)


def _short_reason(raw: str) -> str:
    """Compact a block_reason into a dashboard-friendly token."""
    if not raw:
        return "no_bet"
    if raw.startswith("ev_below_threshold"):
        return "low_ev"
    if raw.startswith("low_conviction"):
        return "low_conv"
    if raw == "model_untrained":
        return "untrained"
    if raw.startswith("contrarian_5m"):
        return "contra_5m"
    return raw   # confluence / circuit_breaker / spread_too_wide / invalid_book


def _print_early_shadow_section() -> None:
    """Early-entry probe: what the 15m model would decide at T-330s (~63% through)
    vs the real T-180s (~80%). Silent until shadow_early_entry.csv has rows."""
    path = BASE_DIR / "shadow_early_entry.csv"
    if not path.exists():
        return
    try:
        with open(path, "r", newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("outcome") in ("WIN", "LOSS")]
    except Exception:
        return
    if not rows:
        return
    try:
        from config import MAX_ENTRY_PRICE_15M as CEIL, BLOCK_15M_TOSSUP_LO as TLO, BLOCK_15M_TOSSUP_HI as THI
    except Exception:
        CEIL, TLO, THI = 0.60, 0.40, 0.55

    def price(r):
        try: return float(r["price"])
        except (TypeError, ValueError): return -1.0

    def bettable(r):
        p = price(r)
        return 0 <= p and (p < TLO or THI <= p < CEIL)

    print(f"\n  EARLY-ENTRY SHADOW  (would-be 15m decisions vs real T-180s; probe only)")
    print(f"  {'kind':<7} {'n':>4} {'bettable':>9} {'would-place':>12} {'win rate':>9}")
    print(f"  {'-'*46}")
    # Group by kind; legacy rows (pre-'kind' column) are the old T330 probe.
    kinds = {}
    for r in rows:
        kinds.setdefault(r.get("kind") or "T330(old)", []).append(r)
    order = ["T600", "T330", "dyn", "noconv", "T330(old)"]
    for k in sorted(kinds, key=lambda x: order.index(x) if x in order else 99):
        rs = kinds[k]
        n  = len(rs)
        bet = sum(1 for r in rs if bettable(r))
        wp = [r for r in rs if r.get("would_place") == "1"]
        wr = ""
        if wp:
            w = sum(1 for r in wp if r.get("outcome") == "WIN")
            wr = f"{w}/{len(wp)}={w/len(wp):.0%}"
        label = {"T600": "T600 33%", "T330": "T330 63%", "dyn": "dynamic",
                 "noconv": "no-conv"}.get(k, k)
        print(f"  {label:<9} {n:>4} {bet/n:>8.0%} {len(wp):>6} ({len(wp)/n:>3.0%}) {wr:>10}")


def _print_last_decisions(settled: list) -> None:
    print(f"\n  LAST 10 DECISIONS  (Result column shows why no bet was placed)")
    print(f"  {'Time':<9} {'Asset':<6} {'Int':<5} {'Dir':<5} {'Prob':<7} {'EV':<7} {'Result'}")
    print(f"  {'-'*60}")
    for t in settled[-10:]:
        ts     = time.strftime("%H:%M:%S", time.localtime(float(t["timestamp"])))
        asset  = t.get("_asset", "btc").upper()
        ev     = float(t["ev_fraction"])
        placed = str(t.get("order_placed", "0")) == "1"
        if placed:
            res = "WIN " if t["outcome"] == "WIN" else "LOSS"
        else:
            res = _short_reason(t.get("block_reason", ""))
        print(
            f"  {ts:<9} {asset:<6} {t['interval']:<5} {t['direction']:<5} "
            f"{float(t['model_prob']):.3f}  {ev:.3f}  {res}"
        )


if __name__ == "__main__":
    print("Starting paper trading dashboard...")
    try:
        while True:
            display()
            time.sleep(10)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
