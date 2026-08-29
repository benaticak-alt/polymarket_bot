"""Process-wide guardrail counters for the capped live-money test.

Hard-stops the live probe after a fixed number of orders or a fixed total of
USDC deployed, independent of the (paper) bankroll sizing. In-memory only:
counters reset on restart, which is the desired behavior for a bounded trial.
Mirrors the exposure.py pattern.
"""

import json
import threading
from config import LIVE_TEST_BUDGET_USDC, LIVE_TEST_MAX_ORDERS, LIVE_TEST_STATE_PATH
from utils.logger import get_logger

log = get_logger("live_test")

_lock = threading.Lock()
_orders = 0
_deployed = 0.0


def _persist() -> None:
    """Write counters to a small JSON the (separate) dashboard process can read."""
    try:
        with open(LIVE_TEST_STATE_PATH, "w") as f:
            json.dump({"orders": _orders, "deployed": round(_deployed, 2),
                       "max_orders": LIVE_TEST_MAX_ORDERS,
                       "budget": LIVE_TEST_BUDGET_USDC}, f)
    except Exception:
        pass


def reset() -> None:
    """Zero the counters and the status file — call at startup so a new trial
    doesn't inherit the previous run's count."""
    global _orders, _deployed
    with _lock:
        _orders = 0
        _deployed = 0.0
    _persist()


def can_place(stake: float) -> tuple[bool, str]:
    """Whether another live-test order of `stake` is allowed. Returns (ok, reason)."""
    with _lock:
        if _orders >= LIVE_TEST_MAX_ORDERS:
            return False, "max_orders"
        if _deployed + stake > LIVE_TEST_BUDGET_USDC:
            return False, "budget"
        return True, ""


def record(stake: float) -> None:
    """Count a placed live-test order against the budget/order caps."""
    global _orders, _deployed
    with _lock:
        _orders += 1
        _deployed += stake
        log.warning(
            f"[LIVE TEST] order #{_orders}/{LIVE_TEST_MAX_ORDERS} placed | "
            f"deployed ${_deployed:.2f}/${LIVE_TEST_BUDGET_USDC:.2f}"
        )
    _persist()


def status() -> tuple[int, float]:
    return _orders, _deployed
