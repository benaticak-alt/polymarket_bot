"""Order execution. Only the paper broker exists.

A live broker is deliberately not implemented: wiring real money to a brokerage
API is your call and your credentials. To add one, implement the `Broker`
protocol below and keep PaperBroker as the default (mirrors DRY_RUN in the bot).
"""

from typing import Protocol

from portfolio.ledger import Ledger
from portfolio.rebalancer import Order, Plan
from portfolio.settings import PortfolioSettings


class Broker(Protocol):
    def execute(self, plan: Plan, ledger: Ledger, prices: dict[str, float],
                when: str | None = None) -> list[Order]:
        """Execute plan.orders, update the ledger with actual fills, return the fills."""
        ...


class PaperBroker:
    """Fills every order at the given price, charging cost_bps + slippage_bps.

    Sells execute before buys so their proceeds fund the buys.
    """

    def __init__(self, settings: PortfolioSettings):
        self.c = settings.total_cost_rate

    def execute(self, plan: Plan, ledger: Ledger, prices: dict[str, float],
                when: str | None = None) -> list[Order]:
        fills = []
        for o in sorted(plan.orders, key=lambda o: o.side != "sell"):
            units = o.units if o.side == "buy" else -o.units
            cost = o.notional * self.c
            ledger.apply_fill(o.asset, units, prices[o.asset], cost, when)
            fills.append(o)
        return fills
