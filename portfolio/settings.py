"""Portfolio policy: target weights, drift bands, and cost assumptions."""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PortfolioSettings:
    # asset -> target weight. Must sum to <= 1; the remainder is the cash target.
    targets: dict[str, float]
    # Drift band per asset is min(abs_band, rel_band * target): the "5/25" rule.
    # A 60% asset rebalances at ±5pp; a 5% asset at ±1.25pp.
    abs_band: float = 0.05
    rel_band: float = 0.25
    min_trade_usd: float = 25.0      # skip orders smaller than this
    cost_bps: float = 5.0            # commission/fee, per side
    slippage_bps: float = 5.0        # assumed fill slippage, per side
    allow_sells: bool = True         # False = rebalance with new cash only (tax-friendlier)
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        for a, w in self.targets.items():
            if w < 0:
                raise ValueError(f"target for {a} is negative ({w})")
        total = sum(self.targets.values())
        if total > 1.0 + 1e-9:
            raise ValueError(f"targets sum to {total:.4f} > 1")
        if self.abs_band < 0 or self.rel_band < 0:
            raise ValueError("bands must be non-negative")

    @property
    def cash_target(self) -> float:
        return max(0.0, 1.0 - sum(self.targets.values()))

    @property
    def total_cost_rate(self) -> float:
        return (self.cost_bps + self.slippage_bps) / 1e4

    def band(self, target: float) -> float:
        return min(self.abs_band, self.rel_band * target)


def load(path: Path) -> PortfolioSettings:
    with open(path) as f:
        raw = json.load(f)
    known = {k: raw[k] for k in (
        "abs_band", "rel_band", "min_trade_usd", "cost_bps", "slippage_bps", "allow_sells"
    ) if k in raw}
    extra = {k: v for k, v in raw.items() if k not in known and k != "targets"}
    return PortfolioSettings(targets={k: float(v) for k, v in raw["targets"].items()},
                             extra=extra, **known)
