"""Drift-band rebalancer. Pure: takes holdings + prices + policy, returns orders.

Two modes, chosen automatically:
  full    — some asset is outside its band: trade every asset back to target.
  deploy  — nothing breached but there's idle cash above the cash target (a
            deposit or a sweep landed): buy-only, filling the most underweight
            assets first. This is the cheapest kind of rebalancing — it moves
            weights toward target without selling anything.
Neither mode fires otherwise. Doing nothing is the default outcome.
"""

from dataclasses import dataclass, field

from portfolio.ledger import Ledger
from portfolio.settings import PortfolioSettings


@dataclass
class Order:
    asset: str
    side: str          # "buy" | "sell"
    notional: float    # USD, before costs
    units: float       # always positive
    est_cost: float


@dataclass
class DriftRow:
    asset: str
    weight: float
    target: float
    band: float

    @property
    def deviation(self) -> float:
        return self.weight - self.target

    @property
    def breached(self) -> bool:
        return abs(self.deviation) > self.band + 1e-12


@dataclass
class Plan:
    mode: str                       # "full" | "deploy" | "none"
    value: float
    drift: list[DriftRow] = field(default_factory=list)
    orders: list[Order] = field(default_factory=list)

    @property
    def turnover(self) -> float:
        return sum(o.notional for o in self.orders)


def drift_report(ledger: Ledger, prices: dict[str, float],
                 s: PortfolioSettings, targets: dict[str, float] | None = None) -> list[DriftRow]:
    targets = targets if targets is not None else s.targets
    w = ledger.weights(prices)
    assets = sorted(set(targets) | set(w))
    return [DriftRow(a, w.get(a, 0.0), targets.get(a, 0.0), s.band(targets.get(a, 0.0)))
            for a in assets]


def plan(ledger: Ledger, prices: dict[str, float], s: PortfolioSettings,
         targets: dict[str, float] | None = None) -> Plan:
    """`targets` overrides s.targets (used by tilts / inverse-vol weighting)."""
    targets = targets if targets is not None else s.targets
    V = ledger.value(prices)
    drift = drift_report(ledger, prices, s, targets)
    if V <= 0:
        return Plan("none", V, drift)

    cash_target = max(0.0, 1.0 - sum(targets.values()))
    c = s.total_cost_rate
    current = {r.asset: r.weight * V for r in drift}

    if any(r.breached for r in drift):
        orders = _full(ledger, prices, s, targets, current, V, cash_target, c)
        mode = "full"
    else:
        budget = ledger.cash - cash_target * V
        orders = _deploy(prices, s, targets, current, V, budget, c)
        mode = "deploy"
    return Plan(mode if orders else "none", V, drift, orders)


def _full(ledger, prices, s, targets, current, V, cash_target, c) -> list[Order]:
    orders = []
    proceeds = 0.0
    if s.allow_sells:
        for a, cur in current.items():
            excess = cur - targets.get(a, 0.0) * V
            if excess >= s.min_trade_usd:
                # Selling the whole position: use exact units so no dust is left.
                units = (ledger.positions.get(a, 0.0) if targets.get(a, 0.0) == 0
                         else excess / prices[a])
                notional = units * prices[a]
                orders.append(Order(a, "sell", notional, units, notional * c))
                proceeds += notional * (1 - c)

    budget = ledger.cash + proceeds - cash_target * V
    wants = {a: t * V - current.get(a, 0.0) for a, t in targets.items()}
    orders += _buys(prices, s, {a: w for a, w in wants.items() if w > 0}, budget, c)
    return orders


def _deploy(prices, s, targets, current, V, budget, c) -> list[Order]:
    if budget * (1 - c) < s.min_trade_usd or not targets:
        return []
    spendable = budget / (1 + c)
    shortfall = {a: max(0.0, t * V - current.get(a, 0.0)) for a, t in targets.items()}
    gap = sum(shortfall.values())
    if gap >= spendable:
        wants = {a: spendable * sf / gap for a, sf in shortfall.items()}
    else:
        # Fill every shortfall, then spread the rest pro rata to target.
        rest = spendable - gap
        tsum = sum(targets.values())
        wants = {a: shortfall[a] + rest * t / tsum for a, t in targets.items()}
    return _buys(prices, s, wants, budget, c)


def _buys(prices, s, wants: dict[str, float], budget: float, c: float) -> list[Order]:
    """Turn desired buy notionals into orders that fit inside `budget` incl. costs."""
    wants = {a: w for a, w in wants.items() if w > 0}
    if budget <= 0 or not wants:
        return []
    need = sum(wants.values()) * (1 + c)
    scale = min(1.0, budget / need)
    orders = []
    for a in sorted(wants):
        notional = wants[a] * scale
        if notional >= s.min_trade_usd:
            orders.append(Order(a, "buy", notional, notional / prices[a], notional * c))
    return orders
