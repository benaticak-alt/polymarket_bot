"""
CoinGecko-derived signals.
Lower frequency (60s updates) — useful as early-window regime context,
not T-10s execution signals.

Signals added to vector: 4 signals (indices 27–30).
"""

from data.coingecko_api import CoinGeckoState


def _clip(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def btc_dominance_trend(cg: CoinGeckoState) -> float:
    """
    Direction and strength of BTC dominance change.
    Rising dominance → risk-on BTC momentum → slight bullish lean.
    Range [-1, 1].
    """
    hist = cg._dominance_history
    if len(hist) < 3:
        return 0.0
    # Slope over recent history
    recent_avg = sum(hist[-3:]) / 3
    older_avg  = sum(hist[:3])  / 3
    delta = recent_avg - older_avg
    return _clip(delta * 5.0)  # 0.2% dominance shift → ±1


def volume_anomaly(cg: CoinGeckoState) -> float:
    """
    Volume anomaly vs rolling average.
    High positive = volume surge (regime event, increased uncertainty).
    Range [-1, 1].
    """
    return _clip(cg.btc_vol_change_pct)


def price_momentum_1h(cg: CoinGeckoState) -> float:
    """
    1-hour BTC price change from CoinGecko (macro momentum context).
    Range [-1, 1]. ±5% mapped to ±1.
    """
    return _clip(cg.btc_price_change_1h / 5.0)


def sentiment_proxy(cg: CoinGeckoState) -> float:
    """
    Vol-to-market-cap ratio as a fear/greed proxy.
    High ratio = high activity relative to market cap = elevated sentiment.
    Typical range: 0.03–0.12. Map 0.03→-1, 0.08→0, 0.15→+1.
    """
    r = cg.vol_to_cap_ratio
    if r == 0:
        return 0.0
    normalised = (r - 0.06) / 0.05
    return _clip(normalised)


def compute(cg: CoinGeckoState) -> list[float]:
    """Returns 4 CoinGecko signals. Returns zeros if data is stale."""
    if cg.is_stale:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        btc_dominance_trend(cg),
        volume_anomaly(cg),
        price_momentum_1h(cg),
        sentiment_proxy(cg),
    ]
