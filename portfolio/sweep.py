"""Satellite → core sweep: move bot bankroll above SATELLITE_CAP_USD into the core.

Caps how much rides on the trading edge, and banks its winnings into holdings
that don't depend on that edge persisting.

Paper books only. Applying a sweep debits bankroll.json and credits the
portfolio ledger. With real money the USDC withdrawal from Polymarket and the
deposit at your broker are manual, so --apply refuses when DRY_RUN is off:
editing bankroll.json then would desync it from the on-chain balance.
"""

from dataclasses import dataclass

import bankroll_state
from config import BASE_DIR, DRY_RUN, SATELLITE_CAP_USD, SATELLITE_SWEEP_MIN_USD
from portfolio.ledger import Ledger


@dataclass
class SweepPlan:
    bankroll: float
    cap: float
    amount: float      # 0 = nothing to sweep

    @property
    def reason(self) -> str:
        if self.amount > 0:
            return f"bankroll ${self.bankroll:,.2f} is ${self.amount:,.2f} over cap ${self.cap:,.2f}"
        return f"bankroll ${self.bankroll:,.2f} not far enough over cap ${self.cap:,.2f}"


def plan_sweep(bankroll: float, cap: float = SATELLITE_CAP_USD,
               min_sweep: float = SATELLITE_SWEEP_MIN_USD) -> SweepPlan:
    excess = bankroll - cap
    return SweepPlan(bankroll, cap, round(excess, 2) if excess >= min_sweep else 0.0)


def apply_sweep(sp: SweepPlan, ledger: Ledger) -> None:
    if sp.amount <= 0:
        return
    if not DRY_RUN:
        raise RuntimeError("DRY_RUN is off: withdraw from Polymarket and deposit at your "
                           "broker manually, then record it with `python -m portfolio deposit`.")
    if (BASE_DIR / "bot.lock").exists():
        raise RuntimeError("bot.lock present: the bot may be running and writing bankroll.json. "
                           "Stop it (or delete a stale lock) before sweeping.")
    state = bankroll_state.load()
    if abs(state["bankroll"] - sp.bankroll) > 0.01:
        raise RuntimeError("bankroll changed since the sweep was planned; re-run it")
    state["bankroll"] -= sp.amount
    # Lower the peak too. A withdrawal isn't a loss and mustn't trip the bot's
    # drawdown brake (bankroll_state.drawdown_multiplier).
    state["peak"] = max(state["peak"] - sp.amount, state["bankroll"])
    bankroll_state.save(state)
    ledger.deposit(sp.amount, source="sweep")
