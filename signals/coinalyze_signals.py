"""
CoinAlyze-derived signals (derivatives market context).
Signals 27–30 in the feature vector (4 signals).

These are early-to-mid window regime signals, not T-10s execution signals.
The model learns their weight automatically.
"""

from data.coinalyze_api import CoinAlyzeState


def _clip(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def funding_rate_signal(ca: CoinAlyzeState) -> float:
    """
    Funding rate sentiment signal.
    High positive funding = crowded longs = slight bearish lean (mean reversion).
    High negative funding = crowded shorts = slight bullish lean.
    Typical range: -0.001 to +0.001 per 8h.
    Inverted: positive funding → negative signal (longs overextended).
    """
    # 0.001 (0.1% per 8h) is extreme — map ±0.001 → ∓1
    return _clip(-ca.funding_rate / 0.001)


def oi_change_signal(ca: CoinAlyzeState) -> float:
    """
    Open interest change direction.
    Rising OI = new money entering = conviction behind current move.
    Falling OI = liquidations/exits = weakening trend.
    Range [-1, 1]. ±5% OI change maps to ±1.
    """
    return _clip(ca.oi_change_pct / 0.05)


def liquidation_imbalance_signal(ca: CoinAlyzeState) -> float:
    """
    Liquidation imbalance: which side is getting squeezed.
    More long liquidations (positive imbalance) = downward pressure = bearish.
    More short liquidations (negative imbalance) = upward squeeze = bullish.
    Inverted so that bullish conditions → positive signal.
    """
    return _clip(-ca.liq_imbalance)


def compute(ca: CoinAlyzeState) -> list[float]:
    """Returns 4 signals. The 4th slot (stoch_k) is overwritten by the aggregator."""
    if ca.is_stale:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        funding_rate_signal(ca),
        oi_change_signal(ca),
        liquidation_imbalance_signal(ca),
        0.0,   # placeholder — replaced with stochastic_k in aggregator
    ]
