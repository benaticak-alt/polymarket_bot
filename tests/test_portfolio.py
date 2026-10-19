import pandas as pd
import pytest

from portfolio import backtest, sweep
from portfolio.broker import PaperBroker
from portfolio.ledger import Ledger
from portfolio.rebalancer import plan
from portfolio.settings import PortfolioSettings
from portfolio.weights import apply_tilts, inverse_vol

PX = {"A": 100.0, "B": 50.0}


def settings(**kw):
    base = dict(targets={"A": 0.6, "B": 0.4}, cost_bps=0, slippage_bps=0, min_trade_usd=1)
    base.update(kw)
    return PortfolioSettings(**base)


def holding(a_val, b_val, cash=0.0):
    return Ledger(cash=cash, positions={"A": a_val / PX["A"], "B": b_val / PX["B"]})


# ── settings ─────────────────────────────────────────────────────────────────
def test_targets_over_one_rejected():
    with pytest.raises(ValueError):
        PortfolioSettings(targets={"A": 0.7, "B": 0.4})


def test_band_is_five_twenty_five():
    s = PortfolioSettings(targets={"A": 0.6, "B": 0.04})
    assert s.band(0.6) == pytest.approx(0.05)    # absolute 5pp binds
    assert s.band(0.04) == pytest.approx(0.01)   # 25% relative binds


# ── rebalancer ───────────────────────────────────────────────────────────────
def test_inside_band_does_nothing():
    p = plan(holding(630, 370), PX, settings())
    assert p.mode == "none" and not p.orders


def test_breach_rebalances_to_target():
    s, led = settings(), holding(700, 300)
    p = plan(led, PX, s)
    assert p.mode == "full"
    PaperBroker(s).execute(p, led, PX)
    w = led.weights(PX)
    assert w["A"] == pytest.approx(0.6) and w["B"] == pytest.approx(0.4)


def test_cash_deploys_to_underweight_first_without_selling():
    led = holding(600, 360, cash=40)     # B 36% vs 40%: inside band, cash is idle
    p = plan(led, PX, settings())
    assert p.mode == "deploy"
    assert all(o.side == "buy" for o in p.orders)
    assert {o.asset for o in p.orders} == {"B"}
    assert sum(o.notional for o in p.orders) == pytest.approx(40)


def test_no_sells_mode_only_buys():
    led = holding(700, 300, cash=50)
    p = plan(led, PX, settings(allow_sells=False))
    assert all(o.side == "buy" for o in p.orders)


def test_costs_never_overdraw_cash():
    s = settings(cost_bps=50, slippage_bps=50)
    led = Ledger(cash=1000)
    p = plan(led, PX, s)
    PaperBroker(s).execute(p, led, PX)
    assert led.cash >= -1e-6
    assert led.costs_paid > 0


def test_zero_target_asset_is_fully_sold():
    led = Ledger(positions={"A": 6, "B": 8, "OLD": 3})
    px = {**PX, "OLD": 10.0}
    s = settings()
    PaperBroker(s).execute(plan(led, px, s), led, px)
    assert "OLD" not in led.positions


def test_min_trade_filters_dust():
    p = plan(holding(700, 300), PX, settings(min_trade_usd=10_000))
    assert not p.orders


# ── ledger ───────────────────────────────────────────────────────────────────
def test_ledger_rejects_oversell_and_overspend():
    led = Ledger(cash=10, positions={"A": 1})
    with pytest.raises(ValueError):
        led.apply_fill("A", -2, 100, 0)
    with pytest.raises(ValueError):
        led.apply_fill("B", 1, 50, 0)


# ── weights ──────────────────────────────────────────────────────────────────
def test_tilts_are_capped_and_preserve_invested_total():
    t = apply_tilts({"A": 0.6, "B": 0.3}, {"A": 0.5, "B": -0.5}, max_tilt=0.05)
    assert sum(t.values()) == pytest.approx(0.9)
    assert t["A"] == pytest.approx(0.9 * 0.65 / 0.9)


def test_inverse_vol_favors_calmer_asset():
    prices = backtest.synthetic_prices(["A", "B"], days=300, seed=1)
    w = inverse_vol(prices, ["A", "B"], 1.0)
    vol = prices.pct_change().iloc[-63:].std()
    calmer = vol.idxmin()
    assert w[calmer] > 0.5 and sum(w.values()) == pytest.approx(1.0)


# ── backtest ─────────────────────────────────────────────────────────────────
def test_backtest_runs_and_tracks_contributions():
    s = settings(cost_bps=5, slippage_bps=5, min_trade_usd=25)
    prices = backtest.synthetic_prices(["A", "B"], days=500)
    table = backtest.compare(prices, s, backtest.default_strategies(),
                             initial=1000, contribution=100)
    assert set(table.index) >= {"buy_hold", "bands_5/25", "calendar_weekly"}
    assert (table["contributed"] == table["contributed"].iloc[0]).all()
    # Bands trade less than rebalancing every week.
    assert table.loc["bands_5/25", "trades"] < table.loc["calendar_weekly", "trades"]


def test_flat_prices_give_zero_return_despite_contributions():
    idx = pd.bdate_range("2020-01-01", periods=100)
    prices = pd.DataFrame({"A": 100.0, "B": 50.0}, index=idx)
    r = backtest.run(prices, settings(), backtest.Strategy("bh", kind="buy_hold"),
                     initial=1000, contribution=100)
    assert r.cagr == pytest.approx(0, abs=1e-9)
    assert r.final_value == pytest.approx(r.contributed)


# ── sweep ────────────────────────────────────────────────────────────────────
def test_sweep_plan_respects_cap_and_minimum():
    assert sweep.plan_sweep(1200, cap=1000, min_sweep=50).amount == 200
    assert sweep.plan_sweep(1030, cap=1000, min_sweep=50).amount == 0
    assert sweep.plan_sweep(800, cap=1000, min_sweep=50).amount == 0


def test_apply_sweep_moves_money_and_lowers_peak(monkeypatch, tmp_path):
    state = {"bankroll": 1500.0, "peak": 1600.0}
    monkeypatch.setattr(sweep.bankroll_state, "load", lambda: dict(state))
    monkeypatch.setattr(sweep.bankroll_state, "save", lambda s: state.update(s))
    monkeypatch.setattr(sweep, "DRY_RUN", True)
    monkeypatch.setattr(sweep, "BASE_DIR", tmp_path)
    led = Ledger()
    sweep.apply_sweep(sweep.plan_sweep(1500, cap=1000, min_sweep=50), led)
    assert state == {"bankroll": 1000.0, "peak": 1100.0}   # drawdown unchanged
    assert led.cash == 500 and led.contributed == 500


def test_apply_sweep_refuses_when_live(monkeypatch):
    monkeypatch.setattr(sweep, "DRY_RUN", False)
    with pytest.raises(RuntimeError):
        sweep.apply_sweep(sweep.SweepPlan(1500, 1000, 500), Ledger())
