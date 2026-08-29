"""
Tracks running bankroll and peak bankroll for drawdown-brake position sizing.
State is persisted in bankroll.json so it survives restarts.
"""

import json
import tempfile
import os
from config import BANKROLL_STATE_PATH, STARTING_BANKROLL, POLYMARKET_FEE
from utils.logger import get_logger

log = get_logger("bankroll")


def load() -> dict:
    try:
        if BANKROLL_STATE_PATH.exists():
            with open(BANKROLL_STATE_PATH) as f:
                return json.load(f)
    except Exception as e:
        log.warning(f"Could not load bankroll state: {e} — using defaults.")
    return {"bankroll": STARTING_BANKROLL, "peak": STARTING_BANKROLL}


def save(state: dict) -> None:
    try:
        BANKROLL_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = BANKROLL_STATE_PATH.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, BANKROLL_STATE_PATH)
    except Exception as e:
        log.warning(f"Could not save bankroll state: {e}")


def update(stake_usdc: float, market_price: float, won: bool) -> float:
    """Apply a settled trade's P&L to the bankroll. Returns the exact realized
    delta (new bankroll − old), so callers can record it as the trade's P&L."""
    state = load()
    before = state["bankroll"]
    if won:
        profit = stake_usdc * (1.0 - market_price) / market_price
        state["bankroll"] += profit * (1.0 - POLYMARKET_FEE)
    else:
        state["bankroll"] -= stake_usdc
    state["bankroll"] = max(state["bankroll"], 0.01)
    state["peak"] = max(state["peak"], state["bankroll"])
    save(state)
    dd = (state["peak"] - state["bankroll"]) / state["peak"]
    log.info(
        f"Bankroll: ${state['bankroll']:.2f}  peak=${state['peak']:.2f}  "
        f"drawdown={dd:.1%}  dd_mult={drawdown_multiplier(state):.2f}"
    )
    return state["bankroll"] - before


def apply_exit(stake_usdc: float, entry_price: float, exit_price: float) -> float:
    """Book P&L from selling a position early (before settlement). Returns the
    exact realized delta (new bankroll − old).

    Shares bought = stake/entry_price; selling at exit_price yields
    stake*(exit_price/entry_price - 1) profit. Fee applies to positive profit
    only, mirroring settlement accounting.
    """
    state = load()
    if entry_price <= 0:
        return 0.0
    before = state["bankroll"]
    profit = stake_usdc * (exit_price / entry_price - 1.0)
    if profit > 0:
        profit *= (1.0 - POLYMARKET_FEE)
    state["bankroll"] = max(state["bankroll"] + profit, 0.01)
    state["peak"] = max(state["peak"], state["bankroll"])
    save(state)
    dd = (state["peak"] - state["bankroll"]) / state["peak"]
    log.info(
        f"Bankroll (early exit @ {exit_price:.3f}): ${state['bankroll']:.2f}  "
        f"peak=${state['peak']:.2f}  drawdown={dd:.1%}  "
        f"dd_mult={drawdown_multiplier(state):.2f}"
    )
    return state["bankroll"] - before


def drawdown_multiplier(state: dict) -> float:
    """
    Stake multiplier based on current drawdown from peak.
      0–5%  drawdown → 1.00×
      10%   drawdown → 0.85×
      20%   drawdown → 0.55×
      30%+  drawdown → 0.25×
    """
    peak = state.get("peak", STARTING_BANKROLL)
    bankroll = state.get("bankroll", STARTING_BANKROLL)
    if peak <= 0:
        return 1.0
    dd = (peak - bankroll) / peak
    if dd <= 0.05:
        return 1.0
    fraction = min((dd - 0.05) / 0.25, 1.0)
    return 1.0 - fraction * 0.75
