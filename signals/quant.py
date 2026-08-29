"""
Quantitative signals: volatility, z-score, autocorrelation, regime metrics.
Signals 14–22 in the feature vector (9 signals).
"""

import numpy as np
from scipy.stats import pearsonr
from data.binance_ws import MarketState
from config import VOL_WINDOW_SHORT, VOL_WINDOW_LONG, ZSCORE_WINDOW


def _clip(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def realized_vol_short(state: MarketState) -> float:
    """
    Realized volatility over VOL_WINDOW_SHORT seconds.
    Annualised and normalised: 0 → -1 (calm), high → +1 (volatile).
    """
    prices = state.price.window(VOL_WINDOW_SHORT)
    if len(prices) < 3:
        return 0.0
    returns = np.diff(np.log(prices + 1e-10))
    rv = returns.std() * np.sqrt(86400)  # daily proxy
    return _clip(rv * 5.0 - 1.0)         # ~0.1–0.6 daily vol typical


def realized_vol_long(state: MarketState) -> float:
    """Realized volatility over VOL_WINDOW_LONG seconds."""
    prices = state.price.window(VOL_WINDOW_LONG)
    if len(prices) < 5:
        return 0.0
    returns = np.diff(np.log(prices + 1e-10))
    rv = returns.std() * np.sqrt(86400)
    return _clip(rv * 5.0 - 1.0)


def vol_ratio(state: MarketState) -> float:
    """
    Short-window vol / long-window vol.
    > 1 → vol expanding (turbulent); < 1 → vol contracting (calm).
    Normalised to [-1, 1].
    """
    p_short = state.price.window(VOL_WINDOW_SHORT)
    p_long  = state.price.window(VOL_WINDOW_LONG)
    if len(p_short) < 3 or len(p_long) < 5:
        return 0.0
    rv_s = np.diff(np.log(p_short + 1e-10)).std()
    rv_l = np.diff(np.log(p_long  + 1e-10)).std()
    if rv_l < 1e-12:
        return 0.0
    ratio = rv_s / rv_l
    return _clip((ratio - 1.0) * 2.0)


def price_zscore(state: MarketState) -> float:
    """
    Z-score of current price relative to ZSCORE_WINDOW-second window.
    Clipped to [-3, 3] then scaled to [-1, 1].
    """
    prices = state.price.window(ZSCORE_WINDOW)
    if len(prices) < 10:
        return 0.0
    mu  = prices.mean()
    std = prices.std()
    if std < 1e-10:
        return 0.0
    z = (prices[-1] - mu) / std
    return _clip(z / 3.0)


def volume_zscore(state: MarketState) -> float:
    """
    Z-score of recent volume relative to baseline.
    High positive = volume surge (regime event likely).
    """
    vols_long  = state.vwap_vol.window(300)
    vols_short = state.vwap_vol.window(30)
    if len(vols_long) < 10 or len(vols_short) == 0:
        return 0.0
    mu   = vols_long.mean()
    std  = vols_long.std()
    curr = vols_short.mean()
    if std < 1e-10:
        return 0.0
    z = (curr - mu) / std
    return _clip(z / 3.0)


def return_autocorrelation(state: MarketState) -> float:
    """
    Lag-1 autocorrelation of 15-second returns over the last 5 minutes.
    Positive → trending; negative → mean-reverting.
    """
    prices = state.price.window(300)
    if len(prices) < 20:
        return 0.0
    returns = np.diff(np.log(prices + 1e-10))
    if len(returns) < 4:
        return 0.0
    try:
        corr, _ = pearsonr(returns[:-1], returns[1:])
        return _clip(float(corr))
    except Exception:
        return 0.0


def directional_ratio(state: MarketState) -> float:
    """
    Net displacement / total distance over ZSCORE_WINDOW seconds.
    +1 = perfectly trending up; -1 = perfectly trending down; 0 = choppy.
    """
    prices = state.price.window(ZSCORE_WINDOW)
    if len(prices) < 5:
        return 0.0
    net_displacement = prices[-1] - prices[0]
    total_distance   = np.sum(np.abs(np.diff(prices)))
    if total_distance < 1e-10:
        return 0.0
    ratio = net_displacement / total_distance  # [-1, 1]
    return _clip(ratio)


def hurst_exponent_proxy(state: MarketState) -> float:
    """
    Simplified Hurst exponent proxy using variance scaling.
    H > 0.5 → trending; H < 0.5 → mean-reverting.
    Normalised: H=0.5 → 0, H=1 → +1, H=0 → -1.
    """
    prices = state.price.window(300)
    if len(prices) < 20:
        return 0.0
    log_p = np.log(prices + 1e-10)
    n = len(log_p)
    half = n // 2
    if half < 5:
        return 0.0
    var_full = np.var(np.diff(log_p))
    var_half = np.var(np.diff(log_p[:half]))
    if var_half < 1e-20:
        return 0.0
    # For Brownian: var_full ≈ 2 * var_half (H=0.5)
    ratio = var_full / (2.0 * var_half + 1e-20)
    # ratio ≈ 1 at H=0.5; log2(ratio) ≈ 2H-1
    h_proxy = np.log2(ratio + 1e-10) / 2.0 + 0.5
    return _clip((h_proxy - 0.5) * 4.0)  # centre around 0


def regime_score(state: MarketState) -> float:
    """
    Composite regime score: trending (+1) vs choppy/mean-reverting (-1).
    Combines directional_ratio and autocorrelation.
    """
    dr   = directional_ratio(state)
    ac   = return_autocorrelation(state)
    score = (dr + ac) / 2.0
    return _clip(score)


def compute(state: MarketState) -> list[float]:
    """Returns 9 quant signals."""
    return [
        realized_vol_short(state),
        realized_vol_long(state),
        vol_ratio(state),
        price_zscore(state),
        volume_zscore(state),
        return_autocorrelation(state),
        directional_ratio(state),
        hurst_exponent_proxy(state),
        regime_score(state),
    ]
