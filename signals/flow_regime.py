"""
Replacement signals for the dead oracle (23–26) and Kalshi (48–49) slots.

Oracle slot (4 signals) — derivatives / volatility / micro regime:
  vrp           variance risk premium: Deribit implied vol − realized vol
  oi_price_div  open-interest × price-direction interaction (conviction vs unwind)
  price_accel   second-order price move (acceleration), vol-scaled
  vol_skew      upside vs downside semivariance asymmetry

Kalshi slot (2 signals) — order-flow / candle persistence:
  cvd_slope     is aggressor (taker) flow accelerating up or down?
  candle_run    signed length of the current consecutive-candle run

All values are clipped to [-1, 1]. Each returns 0.0 when its inputs are
unavailable so the model reads it as no-information rather than a false cue.
"""

import math
import numpy as np
from data.binance_ws import MarketState
from data.coinalyze_api import CoinAlyzeState
from data.deribit_iv import DeribitIVState
from signals.barrier import _vol_per_sec

SECONDS_PER_YEAR = 365 * 24 * 3600


def _clip(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ── Oracle-slot replacements (4) ────────────────────────────────────────────────

def vrp(state: MarketState, deribit: DeribitIVState | None) -> float:
    """Variance risk premium = implied (DVOL) − realized, both annualised.

    Positive → options price more vol than realized (calm / mean-revert regime).
    Negative → realized outrunning implied (vol underpriced / breakout regime).
    """
    if deribit is None or deribit.last_updated <= 0:
        return 0.0
    sigma_sec = _vol_per_sec(state, 300.0)
    if sigma_sec <= 0:
        return 0.0
    realized = sigma_sec * math.sqrt(SECONDS_PER_YEAR)
    implied  = deribit.dvol_raw / 100.0
    return _clip((implied - realized) / 0.3)


def oi_price_div(state: MarketState, ca: CoinAlyzeState | None) -> float:
    """Open-interest change combined with price direction.

    OI rising with the move → conviction (sign follows price, magnitude grows).
    OI falling → unwind / weakening trend (sign flips → fade).
    """
    if ca is None or ca.last_updated <= 0 or ca.is_stale:
        return 0.0
    prices = state.price.window(120)
    if len(prices) < 5:
        return 0.0
    ret = float(np.log(prices[-1] / prices[0])) if prices[0] > 0 else 0.0
    if abs(ret) < 1e-5:
        return 0.0
    price_dir = 1.0 if ret > 0 else -1.0
    oi_mag    = _clip(ca.oi_change_pct / 0.03)
    return _clip(price_dir * oi_mag)


def price_accel(state: MarketState) -> float:
    """Price acceleration: change in the short-window return (2nd derivative)."""
    p = state.price.window(60)
    if len(p) < 9:
        return 0.0
    k = len(p) // 3
    early = p[:k].mean()
    mid   = p[k:2 * k].mean()
    late  = p[2 * k:].mean()
    if early <= 0 or mid <= 0:
        return 0.0
    r1 = (mid - early) / early
    r2 = (late - mid) / mid
    return _clip((r2 - r1) / 0.0008)


def vol_skew(state: MarketState) -> float:
    """Upside vs downside semivariance over the last 5 minutes.

    +1 → upside moves carry the variance; -1 → downside moves dominate.
    """
    p = state.price.window(300)
    if len(p) < 12:
        return 0.0
    r = np.diff(np.log(p + 1e-10))
    up = float(np.sum(r[r > 0] ** 2))
    dn = float(np.sum(r[r < 0] ** 2))
    if up + dn < 1e-18:
        return 0.0
    return _clip((up - dn) / (up + dn))


def compute_oracle_replacement(
    state: MarketState,
    ca: CoinAlyzeState | None,
    deribit: DeribitIVState | None,
) -> list[float]:
    """Returns 4 signals for the former oracle slot (indices 23–26)."""
    return [
        vrp(state, deribit),
        oi_price_div(state, ca),
        price_accel(state),
        vol_skew(state),
    ]


# ── Kalshi-slot replacements (2) ────────────────────────────────────────────────

def cvd_slope(state: MarketState) -> float:
    """Is taker (aggressor) flow accelerating? Compares the net signed volume of
    the most recent 30s against the prior 30s.

    +1 → buy pressure building; -1 → sell pressure building. Distinct from the
    flow-imbalance *level* signals — this captures the *change* in order flow.
    """
    tf = state.trade_flow
    if not tf._times:
        return 0.0
    times = list(tf._times)
    vols  = list(tf._signed_vol)
    now   = times[-1]
    half1 = 0.0  # [now-60, now-30)
    half2 = 0.0  # [now-30, now]
    for t, v in zip(times, vols):
        if t >= now - 30:
            half2 += v
        elif t >= now - 60:
            half1 += v
    denom = abs(half1) + abs(half2)
    if denom < 1e-12:
        return 0.0
    return _clip((half2 - half1) / denom)


def candle_run(state: MarketState, interval: str = "5m") -> float:
    """Signed length of the current consecutive same-direction candle run.

    +0.75 → three up candles in a row; -1.0 → four or more down candles.
    Captures short-term momentum persistence (distinct from RSI/MACD level).
    """
    klines = state.klines_5m if interval == "15m" else state.klines_1m
    closes = [c["close"] for c in klines][-10:]
    if len(closes) < 3:
        return 0.0
    dirs = [
        1 if closes[i] > closes[i - 1] else (-1 if closes[i] < closes[i - 1] else 0)
        for i in range(1, len(closes))
    ]
    last = dirs[-1]
    if last == 0:
        return 0.0
    run = 0
    for d in reversed(dirs):
        if d == last:
            run += 1
        else:
            break
    return _clip(last * min(run / 4.0, 1.0))


def compute_kalshi_replacement(state: MarketState, interval: str = "5m") -> list[float]:
    """Returns 2 signals for the former Kalshi slot (indices 48–49)."""
    return [
        cvd_slope(state),
        candle_run(state, interval),
    ]
