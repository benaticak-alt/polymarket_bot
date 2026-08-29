"""
Barrier / first-passage signal for mid-window entry.

At T-Nmin the settlement question is purely: will price still be on the same
side of the window-start reference price when the window closes?  This models
it as a driftless random walk and returns:

  barrier_z     normalized displacement z-score  (how far past the line, in
                units of expected remaining-window volatility)
  barrier_prob  centered analytic P(settle up) = 2*Phi(z) - 1

These give the model the analytic fair-value baseline for the random-walk
component, so the rest of the signal set only has to learn the residual
(drift) edge — when flow / OBI / momentum push the outcome away from the
driftless prior.
"""

import math
import numpy as np
from data.binance_ws import MarketState

VOL_LOOKBACK_S = 120   # seconds of price history for the realized-vol estimate


def _vol_per_sec(state: MarketState, lookback_s: float = VOL_LOOKBACK_S) -> float:
    """Realized volatility in fractional (log-return) units, per second.

    Uses realized variance / actual elapsed span so it is robust to the
    irregular tick spacing of the aggTrade feed.
    """
    times = list(state.price._times)
    vals  = list(state.price._values)
    if len(vals) < 5:
        return 0.0

    cutoff = times[-1] - lookback_s
    pts = [(t, v) for t, v in zip(times, vals) if t >= cutoff and v > 0]
    if len(pts) < 5:
        return 0.0

    ts = np.array([p[0] for p in pts], dtype=np.float64)
    px = np.array([p[1] for p in pts], dtype=np.float64)
    span = ts[-1] - ts[0]
    if span < 1.0:
        return 0.0

    log_ret = np.diff(np.log(px))
    realized_var = float(np.sum(log_ret ** 2))   # realized variance over the span
    var_per_sec = realized_var / span
    return math.sqrt(max(var_per_sec, 0.0))


def compute(
    state: MarketState,
    reference_price: float,
    seconds_remaining: float,
    current_override: float = 0.0,
) -> list[float]:
    """Return [barrier_z, barrier_prob] both in [-1, 1].

    Returns [0.0, 0.0] when inputs are unavailable so the model treats the
    signal as no-information rather than a spurious directional cue.

    current_override: when > 0, use this as the current price instead of the
    Binance mid — set to the Chainlink Data Streams price so the displacement is
    measured against the same source Polymarket settles on. Realized vol is still
    estimated from the dense Binance tick series (the two are near-identical and
    Binance has far higher tick density).
    """
    current = current_override if current_override > 0 else state.mid
    if reference_price <= 0 or current <= 0 or seconds_remaining <= 0:
        return [0.0, 0.0]

    sigma_sec = _vol_per_sec(state)
    if sigma_sec <= 0:
        return [0.0, 0.0]

    displacement    = (current - reference_price) / reference_price
    sigma_remaining = sigma_sec * math.sqrt(seconds_remaining)
    if sigma_remaining < 1e-9:
        return [0.0, 0.0]

    z = displacement / sigma_remaining
    # P(settle above reference) under driftless GBM = Phi(z)
    prob_up = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    barrier_z    = max(-1.0, min(1.0, z / 3.0))
    barrier_prob = max(-1.0, min(1.0, 2.0 * prob_up - 1.0))
    return [barrier_z, barrier_prob]
