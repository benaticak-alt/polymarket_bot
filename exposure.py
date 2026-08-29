"""Process-wide open-position exposure tracker.

Every placed order commits capital until its market settles. Sizing reads the
committed total so the bot never has more at risk across all simultaneously-open
bets than the bankroll allows — the over-leverage the flat per-bet cap misses
(each bet otherwise sizes off the full bankroll, ignoring other open positions).

In-memory only: pending positions don't survive a restart, so neither should this.
"""

import threading

_lock = threading.Lock()
_committed = 0.0


def add(stake: float) -> None:
    """Commit `stake` to the open-exposure pool when an order is placed."""
    global _committed
    if stake > 0:
        with _lock:
            _committed += stake


def release(stake: float) -> None:
    """Free `stake` back when its position settles."""
    global _committed
    if stake > 0:
        with _lock:
            _committed = max(0.0, _committed - stake)


def current() -> float:
    """Total capital currently committed across all open positions."""
    return _committed
