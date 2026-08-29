"""
Technical indicator signals derived from price and volume history.
Signals 5–13 in the feature vector (9 signals).
"""

import numpy as np
from data.binance_ws import MarketState
from config import (
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL_PERIOD,
    BB_PERIOD, BB_STD, EMA_FAST, EMA_SLOW,
    EMA_MEDIUM_FAST, EMA_MEDIUM_SLOW,
)


def _ema(prices: np.ndarray, period: int) -> float:
    if len(prices) < period:
        return float(prices[-1]) if len(prices) > 0 else 0.0
    k = 2.0 / (period + 1)
    ema = prices[0]
    for p in prices[1:]:
        ema = p * k + ema * (1 - k)
    return ema


def _rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    gains  = deltas[deltas > 0].mean() if (deltas > 0).any() else 1e-10
    losses = -deltas[deltas < 0].mean() if (deltas < 0).any() else 1e-10
    rs     = gains / losses
    return 100.0 - 100.0 / (1.0 + rs)


def _sma(arr: np.ndarray, period: int) -> float:
    if len(arr) < period:
        return float(arr.mean()) if len(arr) > 0 else 0.0
    return float(arr[-period:].mean())


def _clip(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ── Per-signal functions ────────────────────────────────────────────────────────

def rsi_signal(closes: np.ndarray) -> float:
    """RSI normalised to [-1, 1]: 0→−1 (oversold), 100→+1 (overbought)."""
    rsi = _rsi(closes, RSI_PERIOD)
    return _clip((rsi - 50.0) / 50.0)


def macd_histogram(closes: np.ndarray) -> float:
    """
    True MACD histogram (MACD line − signal line), normalised by price std dev.
    Positive = bullish momentum divergence; negative = bearish.
    Range [-1, 1].
    """
    if len(closes) < MACD_SLOW + MACD_SIGNAL_PERIOD:
        return 0.0
    # Build MACD line history from MACD_SLOW onward so the signal EMA is meaningful
    macd_values = []
    for i in range(MACD_SLOW, len(closes) + 1):
        sub = closes[:i]
        macd_values.append(_ema(sub, MACD_FAST) - _ema(sub, MACD_SLOW))
    macd_arr = np.array(macd_values)
    if len(macd_arr) < MACD_SIGNAL_PERIOD:
        return 0.0
    signal_line = _ema(macd_arr, MACD_SIGNAL_PERIOD)
    histogram   = float(macd_arr[-1] - signal_line)
    std = closes[-20:].std() if len(closes) >= 20 else 1.0
    return _clip(histogram / (std + 1e-10))


def ema_fast_slow_cross(closes: np.ndarray) -> float:
    """EMA(5) vs EMA(13) cross: positive = fast above slow. Normalised."""
    if len(closes) < EMA_SLOW:
        return 0.0
    fast = _ema(closes, EMA_FAST)
    slow = _ema(closes, EMA_SLOW)
    rel  = (fast - slow) / (slow + 1e-10)
    return _clip(rel * 200.0)  # amplify the small fractional difference


def ema_medium_cross(closes: np.ndarray) -> float:
    """EMA(9) vs EMA(21) cross: positive = fast above slow. Normalised."""
    if len(closes) < EMA_MEDIUM_SLOW:
        return 0.0
    fast = _ema(closes, EMA_MEDIUM_FAST)
    slow = _ema(closes, EMA_MEDIUM_SLOW)
    rel  = (fast - slow) / (slow + 1e-10)
    return _clip(rel * 200.0)


def bollinger_position(closes: np.ndarray) -> float:
    """
    Where current price sits in BB range: 0→lower band, 1→upper band.
    Normalised to [-1, 1]: 0.5 → 0.0 (midpoint = neutral).
    """
    if len(closes) < BB_PERIOD:
        return 0.0
    window = closes[-BB_PERIOD:]
    mid    = window.mean()
    std    = window.std()
    upper  = mid + BB_STD * std
    lower  = mid - BB_STD * std
    band_width = upper - lower
    if band_width < 1e-10:
        return 0.0
    pos = (closes[-1] - lower) / band_width  # [0, 1]
    return _clip((pos - 0.5) * 2.0)          # [-1, 1]


def bollinger_width(closes: np.ndarray) -> float:
    """
    BB width normalised by mid price — proxy for short-term volatility.
    Range [0, 1] mapped to [-1, 1] (wide = high volatility = +1).
    """
    if len(closes) < BB_PERIOD:
        return 0.0
    window = closes[-BB_PERIOD:]
    mid    = window.mean()
    std    = window.std()
    width  = (BB_STD * 2 * std) / (mid + 1e-10)  # fractional range
    return _clip(width * 50.0 - 1.0)  # typical width ~0.02–0.04


def _vwap_deviation(state: MarketState, window_s: int) -> float:
    """VWAP deviation over a configurable window. Range [-1, 1]."""
    pv = state.vwap_price_vol.window(window_s)
    v  = state.vwap_vol.window(window_s)
    if len(pv) == 0 or v.sum() < 1e-10:
        return 0.0
    vwap  = pv.sum() / v.sum()
    price = state.mid
    if not price or not vwap:
        return 0.0
    dev = (price - vwap) / vwap
    return _clip(dev * 500.0)


def _price_momentum(state: MarketState, fetch_s: int) -> float:
    """Price return from oldest price in fetch_s window to now. Range [-1, 1]."""
    prices = state.price.window(fetch_s)
    if len(prices) < 3:
        return 0.0
    current = prices[-1]
    past    = prices[0]
    ret     = (current - past) / (past + 1e-10)
    std     = prices.std()
    if std < 1e-10:
        return _clip(ret * 2000.0)
    return _clip(ret / (std / current + 1e-10) * 0.1)


def stochastic_k(klines: list, period: int = 14) -> float:
    """
    Stochastic %K: where did the close land within the recent high-low range?
    +1 = closed at the top (overbought), -1 = closed at the bottom (oversold).
    Complementary to RSI — uses high/low data RSI ignores.
    """
    if len(klines) < period:
        return 0.0
    recent  = klines[-period:]
    lowest  = min(c["low"]  for c in recent)
    highest = max(c["high"] for c in recent)
    close   = klines[-1]["close"]
    rng = highest - lowest
    if rng < 1e-10:
        return 0.0
    k = (close - lowest) / rng   # [0, 1]
    return _clip((k - 0.5) * 2.0)


# Keep original names as convenience wrappers
def vwap_deviation(state: MarketState) -> float:
    return _vwap_deviation(state, 300)

def price_momentum_60s(state: MarketState) -> float:
    return _price_momentum(state, 90)

def price_momentum_15s(state: MarketState) -> float:
    return _price_momentum(state, 20)


def compute(state: MarketState, interval: str = "5m") -> list[float]:
    """
    Returns 9 technical signals.
    For 15m markets, uses 5m candles and longer momentum/VWAP windows
    — signals that are meaningful at a 15-minute prediction horizon.
    """
    if interval == "15m" and len(state.klines_5m) >= 10:
        # 5m candles give cleaner indicator readings for a 15m prediction
        closes  = np.array([c["close"] for c in state.klines_5m[-50:]], dtype=np.float64)
        vwap_s  = 900   # 15-min VWAP window
        mom_l_s = 360   # ~5-min lookback for "long" momentum
        mom_s_s = 90    # ~60s lookback for "short" momentum
    else:
        if state.klines_1m:
            closes = np.array([c["close"] for c in state.klines_1m[-50:]], dtype=np.float64)
            if len(closes) < 5:
                closes = state.price.last(500)
        else:
            closes = state.price.last(500)
        vwap_s  = 300   # 5-min VWAP window
        mom_l_s = 90    # ~60s lookback
        mom_s_s = 20    # ~15s lookback

    if len(closes) == 0:
        return [0.0] * 9

    return [
        rsi_signal(closes),
        macd_histogram(closes),
        ema_fast_slow_cross(closes),
        ema_medium_cross(closes),
        bollinger_position(closes),
        bollinger_width(closes),
        _vwap_deviation(state, vwap_s),
        _price_momentum(state, mom_l_s),
        _price_momentum(state, mom_s_s),
    ]
