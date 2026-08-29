"""
Summarize chainlink_shadow.csv — the Binance-vs-Chainlink settlement divergence.

    python chainlink_shadow_stats.py

The headline number is the LABEL-FLIP RATE: how often the two price sources
disagree on the actual UP/DOWN outcome. Only flips corrupt the training label;
price differences that don't change the outcome are harmless. 15m flips matter
most — that's the only +EV interval.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

PATH = Path(__file__).parent / "chainlink_shadow.csv"


def _wr(rows, key):
    n = len(rows)
    flips = sum(1 for r in rows if r["label_flip"] == "1")
    return n, flips, (flips / n * 100 if n else 0.0)


def main() -> int:
    if not PATH.exists():
        print("No chainlink_shadow.csv yet.")
        print("It's written on the first settlement after you run the bot with")
        print("CHAINLINK_STREAMS_ENABLED=true. Let it run a few settlement cycles, then rerun.")
        return 1

    rows = list(csv.DictReader(open(PATH)))
    if not rows:
        print("chainlink_shadow.csv is empty — no settlements logged yet.")
        return 1

    n, flips, rate = _wr(rows, None)
    print(f"Chainlink vs Binance settlement divergence  ({n} comparisons)")
    print("=" * 56)
    print(f"  LABEL FLIPS: {flips}/{n}  ({rate:.1f}%)")
    print("  (a flip = the two sources disagreed on UP/DOWN → a corrupted label)\n")

    # Breakdown by asset + interval
    groups = defaultdict(list)
    for r in rows:
        groups[(r["asset"], r["interval"])].append(r)
    print("  By asset / interval:")
    for (asset, interval), grp in sorted(groups.items()):
        gn, gf, gr = _wr(grp, None)
        star = "  *15m = the +EV interval*" if interval == "15m" and gf else ""
        print(f"    {asset:<4} {interval:<4}  {gf:>3}/{gn:<4} flips  ({gr:4.1f}%){star}")

    # Recent trend
    recent = rows[-50:]
    rn, rf, rr = _wr(recent, None)
    print(f"\n  Last {rn}: {rf} flips ({rr:.1f}%)")

    # Show the actual flip rows — these are the ones worth eyeballing
    flip_rows = [r for r in rows if r["label_flip"] == "1"]
    if flip_rows:
        print(f"\n  Flip detail (last {min(10, len(flip_rows))}):")
        for r in flip_rows[-10:]:
            print(
                f"    {r['asset']} {r['interval']} {r['direction']}: "
                f"binance {r['binance_entry']}->{r['binance_exit']} "
                f"(won={r['binance_won']}) | "
                f"chainlink {r['chainlink_entry']}->{r['chainlink_exit']} "
                f"(won={r['chainlink_won']})"
            )

    print("\n  Read: a high 15m flip rate is the signal to cut CHAINLINK_SETTLEMENT_SOURCE")
    print("  over to 'chainlink'. Near-zero flips → Binance was already fine, leave it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
