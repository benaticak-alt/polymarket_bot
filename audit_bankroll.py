"""Bankroll audit: independently reconstruct the bankroll from the trade record
and reconcile it against bankroll.json.

Each placed+settled trade moves the bankroll exactly once (a normal settlement or
an early exit). Going forward every such row carries an exact `realized_pnl`, so
the audit is simply START + sum(realized_pnl). For historical rows written before
that column existed, realized_pnl is blank and we reconstruct it: settlement-basis
P&L from the CSV, corrected for early exits using the [EARLY EXIT] log lines
(matched by market_id when present, else interval/dir/stake/entry-price + time).

Usage:  python audit_bankroll.py [START]      (START defaults to 100.0)
"""

import csv, glob, json, re, sys, datetime as dt
from config import POLYMARKET_FEE, BANKROLL_STATE_PATH

FEE = POLYMARKET_FEE
START = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
# Boundary of the last manual bankroll reset (data_archive_20260610_012753).
RESET = dt.datetime(2026, 6, 10, 1, 27, 53)


def _ts(line):
    m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
    return dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") if m else None


def settle_pnl(stake, price, won):
    return stake * (1 - price) / price * (1 - FEE) if won else -stake


def exit_pnl(stake, entry, ex):
    p = stake * (ex / entry - 1.0)
    return p * (1 - FEE) if p > 0 else p


def load_rows():
    rows = []
    for f in glob.glob("papertrade_log_*.csv"):
        if "pre51" in f:
            continue
        interval = "15m" if "15m" in f else "5m"
        for r in csv.DictReader(open(f)):
            if r["order_placed"] == "1" and r["outcome"] in ("WIN", "LOSS"):
                rows.append({
                    "interval": interval, "dir": r["direction"],
                    "mid": r["market_id"], "stake": float(r["stake_usdc"]),
                    "price": float(r["market_price"]), "won": r["outcome"] == "WIN",
                    "rp": r.get("realized_pnl", ""),
                    "ts": dt.datetime.fromtimestamp(float(r["timestamp"])),
                    "used": False,
                })
    return rows


def load_early_exits():
    ee = []
    for lf in ["bot.log.2", "bot.log.1", "bot.log"]:
        try:
            txt = open(lf, encoding="utf-8", errors="ignore").read()
        except FileNotFoundError:
            continue
        for ln in txt.splitlines():
            # new format carries market_id; old format does not
            m = re.search(r"\[EARLY EXIT\] (?:(\S+) \| )?(\S+) \| (\w+) \| sold @ "
                          r"([0-9.]+) \(entry ([0-9.]+)\) \| \$([0-9.]+)", ln)
            t = _ts(ln)
            if m and t and t >= RESET:
                ee.append({"mid": m.group(1), "interval": m.group(2), "dir": m.group(3),
                           "exit": float(m.group(4)), "entry": float(m.group(5)),
                           "stake": float(m.group(6)), "ts": t})
    return ee


def main():
    rows = load_rows()
    total = START
    have_rp = used_logs = 0

    # 1) rows with an exact recorded realized_pnl
    blank = []
    for t in rows:
        if t["rp"] not in ("", None):
            total += float(t["rp"]); have_rp += 1
        else:
            blank.append(t)

    # 2) reconstruct the rest: held-to-settlement basis...
    total += sum(settle_pnl(t["stake"], t["price"], t["won"]) for t in blank)

    # ...corrected for early exits among the blank rows
    ee = load_early_exits()
    matched = 0
    for e in ee:
        cands = [t for t in blank if not t["used"] and t["interval"] == e["interval"]
                 and t["dir"] == e["dir"] and abs(t["stake"] - e["stake"]) < 0.5
                 and round(t["price"], 3) == round(e["entry"], 3) and t["ts"] <= e["ts"]
                 and (e["mid"] is None or t["mid"][:20] == e["mid"][:20])]
        if not cands:
            continue
        t = min(cands, key=lambda t: abs((e["ts"] - t["ts"]).total_seconds()))
        t["used"] = True; matched += 1; used_logs += 1
        total += exit_pnl(e["stake"], e["entry"], e["exit"]) \
            - settle_pnl(t["stake"], t["price"], t["won"])

    actual = json.load(open(BANKROLL_STATE_PATH))["bankroll"]
    print(f"Placed+settled trades        : {len(rows)}")
    print(f"  via recorded realized_pnl  : {have_rp}")
    print(f"  via log reconstruction     : {len(blank)} ({matched} early exits corrected)")
    print(f"Start bankroll               : ${START:,.2f}")
    print(f"Reconstructed bankroll       : ${total:,.2f}")
    print(f"bankroll.json                : ${actual:,.2f}")
    print(f"Residual                     : ${total - actual:,.2f}")
    print("RECONCILED" if abs(total - actual) < 1.0 else "*** MISMATCH ***")


if __name__ == "__main__":
    main()
