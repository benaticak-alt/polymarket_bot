"""Spot-vs-Polymarket lag signal.

Latency-arb premise: the spot/futures price moves first; the Polymarket implied
probability reprices with a lag. So when spot has just moved in a direction the
Polymarket mid hasn't yet absorbed, the market is more likely to resolve that way
than its current price implies.

The feature is the gap between a recent, volatility-normalized spot move and
Polymarket's current lean:

    lag = clip( spot_dir - pm_dir , -1, 1 )

  spot_dir : recent spot return over LAG_WINDOW_S, z-scaled by the window's
             own price dispersion (asset-agnostic across BTC/ETH/SOL/XRP).
  pm_dir   : Polymarket implied prob centered & scaled to [-1, 1].

A large positive lag = spot surged up but PM still near a coin flip → expect PM
to catch up upward. Mirror for negative. Auto-pruned by the ensemble if it adds
no signal, exactly like the AVWAP features.
"""

from data.binance_ws import MarketState
from data.polymarket_api import OrderBookState
from config import LAG_WINDOW_S, LAG_Z_SCALE


def compute(
    market_state: MarketState,
    pm_book: OrderBookState | None,
    *,
    window_s: float = LAG_WINDOW_S,
    z_scale: float = LAG_Z_SCALE,
) -> list[float]:
    """Return [lag_dev] in [-1, 1]. 0.0 when data is insufficient."""
    if pm_book is None:
        return [0.0]

    prices = market_state.price.window(window_s)
    if prices.size < 3:
        return [0.0]

    p0, p1 = float(prices[0]), float(prices[-1])
    std = float(prices.std())
    if p0 <= 0 or std <= 0:
        return [0.0]

    # Volatility-normalized recent spot move (z of the net move vs window noise).
    spot_dir = max(-1.0, min(1.0, (p1 - p0) / (std * z_scale)))

    pm_prob = pm_book.yes_implied_prob()
    pm_dir = max(-1.0, min(1.0, (pm_prob - 0.5) * 2.0)) if pm_prob > 0 else 0.0

    lag = max(-1.0, min(1.0, spot_dir - pm_dir))
    return [lag]
