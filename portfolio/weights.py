"""Alternative target-weight schemes and small, capped tilts.

Everything here is an *option* to test in the backtest, not a default. The
README lesson applies: a scheme earns its place only if it beats plain fixed
weights after costs on data it wasn't tuned on.
"""

from typing import Callable

import numpy as np
import pandas as pd

# A tilt function sees price history up to and including today and returns
# asset -> additive weight change (e.g. +0.02). It must not look at future rows.
TiltFn = Callable[[pd.DataFrame], dict[str, float]]


def inverse_vol(history: pd.DataFrame, assets: list[str], total: float,
                lookback: int = 63) -> dict[str, float]:
    """Weights proportional to 1/realized-vol over `lookback` rows, summing to `total`.

    Lower-volatility assets get more weight, so each contributes more equal risk
    (naive risk parity; ignores correlations). Falls back to equal weight if
    there's not enough history.
    """
    rets = history[assets].pct_change().iloc[-lookback:].dropna()
    if len(rets) < max(10, lookback // 3):
        return {a: total / len(assets) for a in assets}
    vol = rets.std().replace(0, np.nan)
    inv = (1 / vol).fillna(0)
    if inv.sum() == 0:
        return {a: total / len(assets) for a in assets}
    return {a: float(total * inv[a] / inv.sum()) for a in assets}


def apply_tilts(targets: dict[str, float], tilts: dict[str, float],
                max_tilt: float = 0.05) -> dict[str, float]:
    """Add clipped tilts to targets, then rescale so the invested total is unchanged.

    The cash target is never touched by a tilt; only the mix among assets moves.
    """
    total = sum(targets.values())
    raw = {a: max(0.0, t + float(np.clip(tilts.get(a, 0.0), -max_tilt, max_tilt)))
           for a, t in targets.items()}
    s = sum(raw.values())
    if s == 0:
        return dict(targets)
    return {a: total * w / s for a, w in raw.items()}


def trend_tilt(size: float = 0.03, window: int = 200) -> TiltFn:
    """Example tilt: +size to assets above their `window`-row moving average,
    -size to those below. Unvalidated. It's here to exercise the hook and to be
    beaten by fixed weights in the backtest, not as a strategy recommendation."""
    def fn(history: pd.DataFrame) -> dict[str, float]:
        if len(history) < window:
            return {}
        ma = history.iloc[-window:].mean()
        last = history.iloc[-1]
        return {a: (size if last[a] > ma[a] else -size) for a in history.columns}
    return fn
