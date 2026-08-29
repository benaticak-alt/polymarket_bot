"""
Market regime detector.
Classifies current market conditions as HIGH_VOL, LOW_VOL, or NORMAL
using realized volatility from recent trades and Deribit IV.
Used to dynamically adjust EV thresholds — not added to the signal vector.
"""

import numpy as np
from data.binance_ws import MarketState
from data.deribit_iv import DeribitIVState

# Realized vol thresholds (std dev of per-trade returns)
HIGH_VOL_THRESHOLD = 0.0006   # ~0.06% per trade = very active
LOW_VOL_THRESHOLD  = 0.00015  # ~0.015% per trade = very quiet

# Deribit IV score thresholds (already normalised to [-1, 1])
HIGH_IV_THRESHOLD  =  0.4
LOW_IV_THRESHOLD   = -0.3


def detect_regime(
    market_state: MarketState,
    deribit_state: DeribitIVState | None = None,
) -> str:
    """
    Returns 'HIGH_VOL', 'LOW_VOL', or 'NORMAL'.

    HIGH_VOL → tighten EV threshold (only take high-conviction trades).
    LOW_VOL  → loosen EV threshold slightly (more opportunities in calm markets).
    NORMAL   → use base threshold.
    """
    prices = market_state.price.last(200)
    if len(prices) >= 20:
        returns = np.diff(prices) / (prices[:-1] + 1e-9)
        rv      = float(np.std(returns))
    else:
        rv = 0.0003  # assume normal if not enough data

    iv_score = float(deribit_state.iv_score) if deribit_state else 0.0

    if rv > HIGH_VOL_THRESHOLD or iv_score > HIGH_IV_THRESHOLD:
        return "HIGH_VOL"
    if rv < LOW_VOL_THRESHOLD and iv_score < LOW_IV_THRESHOLD:
        return "LOW_VOL"
    return "NORMAL"


def threshold_multiplier(regime: str, recent_win_rate: float) -> float:
    """
    Combines regime and recent performance into a single EV threshold multiplier.
    Return value multiplies the base EV threshold. Clamped to [0.7, 2.0].
    """
    regime_mult = {"HIGH_VOL": 1.3, "LOW_VOL": 0.9, "NORMAL": 1.0}[regime]

    if recent_win_rate < 0.40:
        perf_mult = 1.5
    elif recent_win_rate < 0.48:
        perf_mult = 1.2
    elif recent_win_rate > 0.58:
        perf_mult = 0.85
    else:
        perf_mult = 1.0

    return float(max(0.7, min(2.0, regime_mult * perf_mult)))
