"""Process-wide tracker of open directional exposure for correlation-aware sizing.

Our four crypto assets (BTC/ETH/SOL/XRP) are highly positively correlated, and
their up/down markets settle on the same wall-clock boundaries. When several open
bets all lean the same way, they're effectively one larger bet on "crypto up"
(or down): the bankroll swing on a single adverse macro tick is the SUM of them,
not independent draws. Full Kelly assumes independence, so stacking correlated
bets at full size over-bets — adding variance without the matching growth.

This module records each placed bet's direction (UP / DOWN) while it's open, so
sizing can shrink a new bet in proportion to how many same-direction bets are
already live. Mirrors exposure.py: in-memory only, keyed by market id, released
at settlement.

"UP" means a bet that profits if the underlying rises — i.e. a "YES" on these
up/down markets. "DOWN" is the "NO" side.
"""

import threading

_lock = threading.Lock()
# market_id -> "UP" | "DOWN"
_open: dict[str, str] = {}


def _side(direction: str) -> str:
    return "UP" if direction == "YES" else "DOWN"


def add(market_id: str, direction: str) -> None:
    """Register an open bet's direction when its order is placed."""
    with _lock:
        _open[market_id] = _side(direction)


def release(market_id: str) -> None:
    """Drop a bet from the open set when its position settles."""
    with _lock:
        _open.pop(market_id, None)


def count_same_direction(direction: str) -> int:
    """How many currently-open bets share this direction (UP/DOWN)."""
    side = _side(direction)
    with _lock:
        return sum(1 for s in _open.values() if s == side)


def haircut(direction: str, rho: float) -> float:
    """Stake multiplier for a new bet given the open same-direction bets.

    Classic correlated-Kelly down-scaling: for n equally-correlated bets with
    pairwise correlation rho, each should be sized ~1 / (1 + rho*(n-1)) of its
    standalone Kelly. n here is the *new* bet plus the already-open
    same-direction bets, so n-1 = count of open same-direction bets.
    rho=0 → 1.0 (no haircut); rho=1 → 1/n (split a single bet n ways).
    """
    n_open = count_same_direction(direction)
    if n_open <= 0 or rho <= 0:
        return 1.0
    return 1.0 / (1.0 + rho * n_open)
