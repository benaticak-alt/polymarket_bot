"""
Microstructure signals derived from order book and trade flow.
Signals 0–4 in the feature vector.
"""

import numpy as np
from data.binance_ws import MarketState
from config import OBI_LEVELS_SHALLOW, OBI_LEVELS_DEEP, TRADE_FLOW_WINDOW_S, TRADE_FLOW_WINDOW_L


def _clip(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def obi_shallow(state: MarketState) -> float:
    """Order book imbalance using top OBI_LEVELS_SHALLOW levels. Range [-1, 1]."""
    ob = state.order_book
    bid = ob.bid_volume(OBI_LEVELS_SHALLOW)
    ask = ob.ask_volume(OBI_LEVELS_SHALLOW)
    total = bid + ask
    return (bid - ask) / total if total > 0 else 0.0


def obi_deep(state: MarketState) -> float:
    """Order book imbalance using top OBI_LEVELS_DEEP levels. Range [-1, 1]."""
    ob = state.order_book
    bid = ob.bid_volume(OBI_LEVELS_DEEP)
    ask = ob.ask_volume(OBI_LEVELS_DEEP)
    total = bid + ask
    return (bid - ask) / total if total > 0 else 0.0


def trade_flow_short(state: MarketState) -> float:
    """Signed trade flow imbalance over TRADE_FLOW_WINDOW_S seconds. Range [-1, 1]."""
    return _clip(state.trade_flow.imbalance(TRADE_FLOW_WINDOW_S))


def trade_flow_long(state: MarketState) -> float:
    """Signed trade flow imbalance over TRADE_FLOW_WINDOW_L seconds. Range [-1, 1]."""
    return _clip(state.trade_flow.imbalance(TRADE_FLOW_WINDOW_L))


def spread_ratio(state: MarketState) -> float:
    """
    Bid-ask spread normalised by mid price.
    Inverted: narrow spread → +1 (liquid), wide spread → -1 (illiquid).
    Range [-1, 1].
    """
    ob  = state.order_book
    spd = ob.spread()
    mid = ob.mid_price()
    if not spd or not mid or mid == 0:
        return 0.0
    raw = spd / mid   # typically 0.0001–0.005
    # map to [-1, 1]: 0→+1, 0.005+→-1
    normalised = 1.0 - (raw / 0.003) * 2.0
    return _clip(normalised)


def compute(state: MarketState) -> list[float]:
    """Returns 5 microstructure signals."""
    return [
        obi_shallow(state),
        obi_deep(state),
        trade_flow_short(state),
        trade_flow_long(state),
        spread_ratio(state),
    ]
