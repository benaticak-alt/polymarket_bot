"""BTC→altcoin lead-lag signal.

BTC leads the major alts (ETH/SOL/XRP) at minute horizons: a BTC move propagates
into alt prices with a short delay, and intraday BTC-alt correlation regularly
runs 0.65–0.99. Each alt's model otherwise sees only its own feed, so BTC's most
recent move is real information it doesn't have.

Feature: BTC's net move over BTC_LEAD_WINDOW_S, z-scaled by the window's own
price dispersion (same normalization as signals/lag.py) → [-1, 1].

For BTC's own models the feature is 0.0 (its mom_* signals already cover it);
the ensemble prunes it there automatically.
"""

from config import BTC_LEAD_WINDOW_S, LAG_Z_SCALE


def compute(btc_market_state, *, window_s: float = BTC_LEAD_WINDOW_S,
            z_scale: float = LAG_Z_SCALE) -> list[float]:
    """Return [btc_lead] in [-1, 1]. Pass None (e.g. for BTC itself) → [0.0]."""
    if btc_market_state is None:
        return [0.0]

    prices = btc_market_state.price.window(window_s)
    if prices.size < 3:
        return [0.0]

    p0, p1 = float(prices[0]), float(prices[-1])
    std = float(prices.std())
    if p0 <= 0 or std <= 0:
        return [0.0]

    z = (p1 - p0) / (std * z_scale)
    return [max(-1.0, min(1.0, z))]
