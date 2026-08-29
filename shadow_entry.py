"""Shadow early-entry logger.

Records what the 15m model WOULD decide at SHADOW_EARLY_ENTRY_S before close —
earlier than the real T-180s entry — to answer two questions with real data:
  1. Does an earlier entry surface more bettable prices? (the market is less
     "decided" earlier in the window, so the model's chosen side may sit in a
     tradeable range instead of an extreme.)
  2. Are the model's earlier calls actually profitable?

Pure measurement, like chainlink_shadow.py: it never places an order, commits
capital, or feeds the model (no training). The decision is held in memory until
its market settles — at which point the real settlement loop calls settle() with
the true up/down, and one complete row is written to shadow_early_entry.csv.
"""

import csv
import time
from pathlib import Path

from utils.logger import get_logger

log = get_logger("shadow_entry")

_PATH = Path(__file__).parent / "shadow_early_entry.csv"
_FIELDS = [
    "timestamp", "asset", "market_id", "kind", "seconds_left", "price", "direction",
    "model_prob", "ev_fraction", "would_place", "book_src", "outcome", "correct",
]
_pending: dict[str, dict] = {}   # market_id -> decision row awaiting settlement


def _ensure_csv() -> None:
    if _PATH.exists():
        try:
            with open(_PATH, "r", newline="") as f:
                header = next(csv.reader(f), None)
            if header == _FIELDS:
                return
            # Old-format file (e.g. before the "kind" column) — archive and start
            # fresh so appends stay aligned. Keeps the old rows for reference.
            _PATH.replace(_PATH.with_name("shadow_early_entry_old.csv"))
        except Exception:
            pass
    with open(_PATH, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=_FIELDS).writeheader()


def record(*, market_id: str, asset: str, kind: str, seconds_left: float,
           price: float, direction: str, model_prob: float, ev_fraction: float,
           would_place: bool, book_src: str) -> None:
    """Hold a shadow decision until its market settles. `kind` distinguishes the
    fixed points (T600/T330) from the dynamic catch (dyn). A market can have
    several shadow rows (one per kind), keyed market_id:kind."""
    _pending[f"{market_id}:{kind}"] = {
        "timestamp": round(time.time(), 3), "asset": asset,
        "market_id": market_id[:20], "kind": kind,
        "seconds_left": round(seconds_left, 1),
        "price": round(price, 4), "direction": direction or "?",
        "model_prob": round(model_prob, 4), "ev_fraction": round(ev_fraction, 4),
        "would_place": int(bool(would_place)), "book_src": book_src,
        "outcome": "", "correct": "",
    }
    # Drop orphans (markets that never reached settlement) so the dict can't grow.
    if len(_pending) > 1200:
        for k in list(_pending)[:300]:
            _pending.pop(k, None)


def settle(market_id: str, went_up: bool) -> None:
    """Called from the real settlement loop with the true outcome. Settles every
    shadow kind (T600/T330/dyn) recorded for this market. No-op if none."""
    keys = [k for k in _pending if k.startswith(f"{market_id}:")]
    if not keys:
        return
    try:
        _ensure_csv()
        with open(_PATH, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_FIELDS)
            for k in keys:
                d = _pending.pop(k)
                won = (d["direction"] == "YES") == bool(went_up)
                d["outcome"] = "WIN" if won else "LOSS"
                d["correct"] = 1 if won else 0
                w.writerow(d)
    except Exception as e:
        log.debug(f"shadow_entry settle write failed: {e}")
