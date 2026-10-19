"""Paper holdings ledger: cash + units per asset, with an append-only event log.

Persisted atomically (tmp file + os.replace), same pattern as bankroll_state.py.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class Ledger:
    cash: float = 0.0
    positions: dict[str, float] = field(default_factory=dict)   # asset -> units
    contributed: float = 0.0     # external money in (deposits + sweeps), for P&L
    costs_paid: float = 0.0
    events: list[dict] = field(default_factory=list)

    # ── valuation ────────────────────────────────────────────────────────────
    def value(self, prices: dict[str, float]) -> float:
        return self.cash + sum(u * _price(prices, a) for a, u in self.positions.items() if u)

    def weights(self, prices: dict[str, float]) -> dict[str, float]:
        v = self.value(prices)
        if v <= 0:
            return {}
        return {a: u * _price(prices, a) / v for a, u in self.positions.items() if u}

    # ── mutations ────────────────────────────────────────────────────────────
    def deposit(self, amount: float, source: str = "deposit", when: str | None = None) -> None:
        if amount <= 0:
            raise ValueError("deposit amount must be positive")
        self.cash += amount
        self.contributed += amount
        self._log(when, type=source, amount=round(amount, 2))

    def apply_fill(self, asset: str, units: float, price: float, cost: float,
                   when: str | None = None) -> None:
        """units > 0 buys, units < 0 sells. Cost is always paid in cash."""
        held = self.positions.get(asset, 0.0)
        if units < 0 and -units > held + 1e-9:
            raise ValueError(f"cannot sell {-units:.6f} {asset}; only {held:.6f} held")
        notional = units * price
        if notional + cost > self.cash + 1e-6:
            raise ValueError(f"insufficient cash for {asset}: need {notional + cost:.2f}, "
                             f"have {self.cash:.2f}")
        self.cash -= notional + cost
        self.costs_paid += cost
        new = held + units
        if abs(new) < 1e-12:
            self.positions.pop(asset, None)
        else:
            self.positions[asset] = new
        self._log(when, type="fill", asset=asset, units=units, price=price,
                  notional=round(notional, 2), cost=round(cost, 4))

    def _log(self, when: str | None, **event) -> None:
        event["at"] = when or datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.events.append(event)

    # ── persistence ──────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {"cash": self.cash, "positions": self.positions,
                "contributed": self.contributed, "costs_paid": self.costs_paid,
                "events": self.events}

    @classmethod
    def from_dict(cls, d: dict) -> "Ledger":
        return cls(cash=d.get("cash", 0.0), positions=dict(d.get("positions", {})),
                   contributed=d.get("contributed", 0.0), costs_paid=d.get("costs_paid", 0.0),
                   events=list(d.get("events", [])))


def load(path: Path) -> Ledger:
    if not path.exists():
        return Ledger()
    with open(path) as f:
        return Ledger.from_dict(json.load(f))


def save(ledger: Ledger, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(ledger.to_dict(), f, indent=2)
    os.replace(tmp, path)


def _price(prices: dict[str, float], asset: str) -> float:
    try:
        p = prices[asset]
    except KeyError:
        raise KeyError(f"no price for held asset {asset!r}") from None
    if p <= 0:
        raise ValueError(f"non-positive price for {asset}: {p}")
    return p
