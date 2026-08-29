"""
Shadow logger for Chainlink-vs-Binance settlement divergence.

Phase-1 instrumentation: at every settlement, record BOTH price sources and
whether the WIN/LOSS label would FLIP if the trade had been settled on the other
source. This quantifies — before any cutover — how often Binance futures mid
(what the bot settles on today) disagrees with Chainlink Data Streams (what
Polymarket actually resolves on). Pure measurement; never raises into the
settlement loop.

A flip is the thing that matters: prices can differ by a few dollars every time
without ever changing an outcome. Only flips corrupt the training label.
"""

import csv
import time
from pathlib import Path

from utils.logger import get_logger

log = get_logger("cl_shadow")

_PATH = Path(__file__).parent / "chainlink_shadow.csv"
_FIELDS = [
    "timestamp", "asset", "interval", "direction",
    "binance_entry", "binance_exit", "binance_up", "binance_won",
    "chainlink_entry", "chainlink_exit", "chainlink_up", "chainlink_won",
    "label_flip",
]


def _ensure_csv() -> None:
    if not _PATH.exists():
        with open(_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=_FIELDS).writeheader()


def record(
    *,
    asset: str,
    interval: str,
    direction: str,
    binance_entry: float,
    binance_exit: float,
    chainlink_entry: float,
    chainlink_exit: float,
) -> None:
    """Append one settlement comparison row. No-op if Chainlink prices missing."""
    try:
        if not (chainlink_entry and chainlink_exit and binance_entry and binance_exit):
            return
        b_up = binance_exit > binance_entry
        c_up = chainlink_exit > chainlink_entry
        b_won = (direction == "YES") == b_up
        c_won = (direction == "YES") == c_up
        flip = b_won != c_won

        _ensure_csv()
        with open(_PATH, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=_FIELDS).writerow({
                "timestamp": round(time.time(), 3),
                "asset": asset,
                "interval": interval,
                "direction": direction,
                "binance_entry": round(binance_entry, 2),
                "binance_exit": round(binance_exit, 2),
                "binance_up": int(b_up),
                "binance_won": int(b_won),
                "chainlink_entry": round(chainlink_entry, 2),
                "chainlink_exit": round(chainlink_exit, 2),
                "chainlink_up": int(c_up),
                "chainlink_won": int(c_won),
                "label_flip": int(flip),
            })

        if flip:
            log.warning(
                f"[CL SHADOW] LABEL FLIP {asset} {interval} {direction}: "
                f"binance {'UP' if b_up else 'DOWN'} "
                f"({binance_entry:.2f}->{binance_exit:.2f}) vs "
                f"chainlink {'UP' if c_up else 'DOWN'} "
                f"({chainlink_entry:.2f}->{chainlink_exit:.2f})"
            )
    except Exception as e:
        log.debug(f"shadow record failed: {e}")
