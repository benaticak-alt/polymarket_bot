"""
Anchored VWAP (AVWAP) deviation features.

AVWAP is a volume-weighted average price measured from a chosen *anchor* bar
forward (rather than over a fixed lookback). It acts as a fair-value magnet:
price tends to revert toward it, and a crossing flips its role between support
and resistance. As a standalone strategy it is a proven loser intraday, but it
is a cheap, orthogonal *context* feature for the ensemble — it answers "how
stretched is price from the level real money was actually committed at."

Two anchors are exposed:
  1. Day-open    — VWAP since the most recent UTC midnight (intraday fair value).
                   Buffer depth (~16h of 5m candles) bounds how far back we reach;
                   when it doesn't span midnight this is a session VWAP, still a
                   valid magnet.
  2. Liq-cascade — VWAP since the single largest liquidation event currently in
                   the buffer. That is the price level where forced flow cleared,
                   a smart-money fill magnet. Asset-agnostic: anchoring on the
                   *largest* recent event needs no absolute USD threshold (works
                   for BTC and for thinner SOL/XRP alike). Self-regulating too —
                   a very recent cascade yields AVWAP≈price≈neutral deviation.

Each feature is the z-scored deviation (price - AVWAP) / sigma, where sigma is
the volume-weighted stdev of typical price around the AVWAP, then divided by
Z_SCALE so ~3 sigma maps to the [-1, 1] rails. Returns 0.0 (neutral) whenever
there isn't enough data to anchor.
"""

import time
from data.binance_ws import MarketState
from utils.ring_buffer import LiquidationBuffer

# 3 sigma of stretch saturates the feature.
Z_SCALE = 3.0
# Need at least this many anchored candles before a deviation is meaningful.
_MIN_CANDLES = 3


def _utc_midnight(ref_ts: float) -> float:
    """UTC midnight on or before ref_ts, derived from ref_ts itself (not the
    wall clock) to stay consistent with the candle timestamps and dodge any
    local-clock drift."""
    tm = time.gmtime(ref_ts)
    secs_into_day = tm.tm_hour * 3600 + tm.tm_min * 60 + tm.tm_sec
    return ref_ts - secs_into_day


def _avwap_zdev(candles: list[dict], anchor_ts: float,
                current_price: float, z_scale: float) -> float:
    """Volume-weighted z-scored deviation of current_price from the AVWAP
    anchored at anchor_ts. Neutral (0.0) without enough volume/candles."""
    pv = 0.0
    vol = 0.0
    tps: list[float] = []
    vols: list[float] = []
    for c in candles:
        if c["ts"] < anchor_ts:
            continue
        v = c["volume"]
        if v <= 0:
            continue
        tp = (c["high"] + c["low"] + c["close"]) / 3.0
        pv += tp * v
        vol += v
        tps.append(tp)
        vols.append(v)

    if vol <= 0 or len(tps) < _MIN_CANDLES:
        return 0.0

    avwap = pv / vol
    var = sum(v * (tp - avwap) ** 2 for tp, v in zip(tps, vols)) / vol
    sigma = var ** 0.5
    if sigma <= 0:
        return 0.0

    z = (current_price - avwap) / sigma
    return max(-1.0, min(1.0, z / z_scale))


def _last_cascade_ts(liq_buffer: LiquidationBuffer) -> float | None:
    """Timestamp of the single largest liquidation event in the buffer, or None
    if the buffer is empty."""
    times = liq_buffer._times
    if not times:
        return None
    best_ts: float | None = None
    best_usd = 0.0
    for t, b, s in zip(times, liq_buffer._buy_usd, liq_buffer._sell_usd):
        usd = b + s
        if usd > best_usd:
            best_usd = usd
            best_ts = t
    return best_ts


def compute(market_state: MarketState, current_price: float,
            *, z_scale: float = Z_SCALE) -> list[float]:
    """Returns [avwap_day_dev, avwap_liq_dev], each z-scored and clipped to
    [-1, 1]. Both default to 0.0 (neutral) when data is insufficient."""
    candles = market_state.klines_5m or market_state.klines_1m
    if not candles or current_price <= 0:
        return [0.0, 0.0]

    ref_ts = candles[-1]["ts"]

    day_dev = _avwap_zdev(candles, _utc_midnight(ref_ts), current_price, z_scale)

    cascade_ts = _last_cascade_ts(market_state.liq_buffer)
    if cascade_ts is None:
        liq_dev = 0.0
    else:
        # 1m candles give a finer anchor than the 5m series.
        fine = market_state.klines_1m or candles
        liq_dev = _avwap_zdev(fine, cascade_ts, current_price, z_scale)

    return [day_dev, liq_dev]
