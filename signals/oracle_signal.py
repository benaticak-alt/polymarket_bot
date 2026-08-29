"""
Oracle-specific signals.
These are the highest-value signals at T-10s.
Signals 23–26 in the feature vector (4 signals).
"""

import time
import numpy as np
from data.binance_ws import MarketState
from data.oracle_feed import OracleState
from utils.ring_buffer import TimeSeriesBuffer


def _clip(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ── Per-signal functions ────────────────────────────────────────────────────────

def oracle_vs_spot(oracle: OracleState, spot_price: float) -> float:
    """
    Signed fractional divergence of oracle price from spot.
    Positive → oracle lags below spot (spot moved up, oracle hasn't caught up → YES edge).
    Negative → oracle lags above spot.
    Normalised to [-1, 1] — typical divergence is 0.001–0.005.
    """
    div = oracle.divergence_from(spot_price)
    # A 0.5% divergence is large — amplify by 200x
    return _clip(div * 200.0)


def oracle_lag_indicator(oracle: OracleState) -> float:
    """
    Normalised oracle lag.
    0 seconds lag → -1 (just updated, no opportunity).
    30+ seconds lag → +1 (stale oracle, high opportunity window).
    """
    lag = oracle.lag_seconds
    # 0–30s range → map to [−1, +1]
    normalised = (lag / 30.0) * 2.0 - 1.0
    return _clip(normalised)


def oracle_momentum(oracle: OracleState, oracle_history: TimeSeriesBuffer) -> float:
    """
    Direction and magnitude of recent oracle price changes.
    Uses a short history of oracle prices.
    Positive → oracle trending up (feeds into YES position).
    """
    hist = oracle_history.last(10)
    if len(hist) < 3:
        return 0.0
    changes = np.diff(hist)
    if changes.std() < 1e-10:
        return 0.0
    trend = changes.mean() / (hist.mean() + 1e-10)
    return _clip(trend * 10000.0)


def multi_exchange_divergence(state: MarketState, oracle: OracleState) -> float:
    """
    Divergence between Binance futures mid and oracle price.
    Large positive → futures running hot above oracle (momentum).
    """
    spot = state.mid
    if not spot or not oracle.price:
        return 0.0
    div = (spot - oracle.price) / oracle.price
    return _clip(div * 200.0)


class OracleSignalComputer:
    """Stateful wrapper that maintains oracle price history for momentum."""

    def __init__(self):
        self.oracle_history = TimeSeriesBuffer(maxlen=500)

    def compute(self, market_state: MarketState, oracle: OracleState) -> list[float]:
        """Returns 4 oracle signals."""
        # Update oracle history on each new oracle round
        if oracle.price > 0:
            last = self.oracle_history.last_value()
            if last is None or abs(oracle.price - last) > 1e-6:
                self.oracle_history.append(oracle.price, oracle.updated_at)

        spot = market_state.mid

        return [
            oracle_vs_spot(oracle, spot),
            oracle_lag_indicator(oracle),
            oracle_momentum(oracle, self.oracle_history),
            multi_exchange_divergence(market_state, oracle),
        ]
