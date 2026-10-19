"""Backtest rebalancing policies against each other on a daily price table.

Every strategy gets identical prices, contributions and cost assumptions, so the
comparison is apples to apples. Buy-and-hold is always included as the benchmark
every other policy has to beat after costs.

Returns are time-weighted (contributions removed), so a policy can't look good
just because money was added to it.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio.broker import PaperBroker
from portfolio.ledger import Ledger
from portfolio.rebalancer import Order, Plan, plan
from portfolio.settings import PortfolioSettings
from portfolio.weights import TiltFn, apply_tilts, inverse_vol

TRADING_DAYS = 252


@dataclass
class Strategy:
    name: str
    kind: str = "bands"          # "bands" | "calendar" | "buy_hold"
    weighting: str = "fixed"     # "fixed" | "inverse_vol"
    tilt: TiltFn | None = None
    max_tilt: float = 0.05


@dataclass
class Result:
    name: str
    final_value: float
    contributed: float
    cagr: float
    vol: float
    sharpe: float
    max_drawdown: float
    turnover: float              # traded notional / avg value, per year
    costs: float
    n_trades: int
    equity: pd.Series


def run(prices: pd.DataFrame, s: PortfolioSettings, strategy: Strategy,
        initial: float = 10_000.0, contribution: float = 500.0,
        contribute_every: int = 21, check_every: int = 5) -> Result:
    """Simulate one strategy. `prices`: DatetimeIndex rows, one column per asset."""
    assets = list(s.targets)
    missing = set(assets) - set(prices.columns)
    if missing:
        raise ValueError(f"price table lacks target assets: {sorted(missing)}")
    prices = prices[assets].dropna()
    if len(prices) < 2:
        raise ValueError("need at least 2 rows of prices")

    ledger = Ledger()
    broker = PaperBroker(s)
    invested = sum(s.targets.values())
    values, flows, traded = [], [], 0.0
    n_trades = 0

    for i, (ts, row) in enumerate(prices.iterrows()):
        px = row.to_dict()
        when = str(ts.date()) if hasattr(ts, "date") else str(ts)
        flow = 0.0
        if i == 0:
            flow = initial
        elif contribution > 0 and i % contribute_every == 0:
            flow = contribution
        if flow:
            ledger.deposit(flow, when=when)

        if flow or i % check_every == 0:
            p = _decide(ledger, px, s, strategy, prices.iloc[: i + 1], assets, invested)
            if p.orders:
                broker.execute(p, ledger, px, when)
                traded += p.turnover
                n_trades += len(p.orders)

        values.append(ledger.value(px))
        flows.append(flow)

    equity = pd.Series(values, index=prices.index, name=strategy.name)
    return _metrics(strategy.name, equity, pd.Series(flows, index=prices.index),
                    ledger, traded, n_trades)


def _decide(ledger, px, s, strategy, history, assets, invested) -> Plan:
    targets = dict(s.targets)
    if strategy.weighting == "inverse_vol":
        targets = inverse_vol(history, assets, invested)
    if strategy.tilt is not None:
        targets = apply_tilts(targets, strategy.tilt(history), strategy.max_tilt)

    if strategy.kind == "buy_hold":
        return _buy_hold(ledger, px, s, targets)
    if strategy.kind == "calendar":
        # Zero-width bands: any drift at a check triggers a full rebalance.
        cal = PortfolioSettings(targets, abs_band=0.0, rel_band=0.0,
                                min_trade_usd=s.min_trade_usd, cost_bps=s.cost_bps,
                                slippage_bps=s.slippage_bps, allow_sells=s.allow_sells)
        return plan(ledger, px, cal, targets)
    return plan(ledger, px, s, targets)


def _buy_hold(ledger, px, s, targets) -> Plan:
    """Deploy new cash at target weights; never sell, never steer toward target."""
    V = ledger.value(px)
    budget = ledger.cash - s.cash_target * V
    c = s.total_cost_rate
    tsum = sum(targets.values())
    if budget * (1 - c) < s.min_trade_usd or tsum == 0:
        return Plan("none", V)
    spend = budget / (1 + c)
    orders = [Order(a, "buy", spend * t / tsum, spend * t / tsum / px[a], spend * t / tsum * c)
              for a, t in targets.items() if spend * t / tsum >= s.min_trade_usd]
    return Plan("deploy" if orders else "none", V, [], orders)


def _metrics(name, equity: pd.Series, flows: pd.Series, ledger: Ledger,
             traded: float, n_trades: int) -> Result:
    # Time-weighted daily return: today's flow arrives before today's close.
    prev = equity.shift(1).fillna(0) + flows
    rets = (equity / prev.replace(0, np.nan) - 1).iloc[1:].fillna(0)
    years = max(len(rets) / TRADING_DAYS, 1e-9)
    growth = float((1 + rets).prod())
    cagr = growth ** (1 / years) - 1
    vol = float(rets.std() * np.sqrt(TRADING_DAYS))
    sharpe = float(rets.mean() * TRADING_DAYS / vol) if vol > 0 else 0.0
    idx = (1 + rets).cumprod()
    max_dd = float((idx / idx.cummax() - 1).min()) if len(idx) else 0.0
    turnover = traded / max(float(equity.mean()), 1e-9) / years
    return Result(name, float(equity.iloc[-1]), ledger.contributed, cagr, vol, sharpe,
                  max_dd, turnover, ledger.costs_paid, n_trades, equity)


def compare(prices: pd.DataFrame, s: PortfolioSettings, strategies: list[Strategy],
            **kw) -> pd.DataFrame:
    rows = []
    for st in strategies:
        r = run(prices, s, st, **kw)
        rows.append({"strategy": r.name, "final": r.final_value, "contributed": r.contributed,
                     "CAGR": r.cagr, "vol": r.vol, "sharpe": r.sharpe,
                     "max_dd": r.max_drawdown, "turnover/yr": r.turnover,
                     "costs": r.costs, "trades": r.n_trades})
    return pd.DataFrame(rows).set_index("strategy")


def default_strategies() -> list[Strategy]:
    from portfolio.weights import trend_tilt
    return [
        Strategy("buy_hold", kind="buy_hold"),
        Strategy("calendar_weekly", kind="calendar"),
        Strategy("bands_5/25"),
        Strategy("bands_inverse_vol", weighting="inverse_vol"),
        Strategy("bands_trend_tilt", tilt=trend_tilt()),
    ]


def synthetic_prices(assets: list[str], days: int = TRADING_DAYS * 10,
                     seed: int = 7) -> pd.DataFrame:
    """Correlated geometric Brownian motion. For exercising the code ONLY.
    Results on synthetic data say nothing about real markets."""
    rng = np.random.default_rng(seed)
    n = len(assets)
    mu = rng.uniform(0.02, 0.09, n) / TRADING_DAYS
    sig = rng.uniform(0.05, 0.22, n) / np.sqrt(TRADING_DAYS)
    corr = np.full((n, n), 0.2) + np.eye(n) * 0.8
    shocks = rng.multivariate_normal(np.zeros(n), corr, size=days) * sig
    paths = 100 * np.exp(np.cumsum(mu - sig ** 2 / 2 + shocks, axis=0))
    idx = pd.bdate_range("2016-01-04", periods=days)
    return pd.DataFrame(paths, index=idx, columns=assets)


def load_prices_csv(path) -> pd.DataFrame:
    """Wide CSV: a `date` column plus one close-price column per asset.
    Use adjusted closes (dividends reinvested) or total returns will be understated."""
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    return df.apply(pd.to_numeric, errors="coerce")
