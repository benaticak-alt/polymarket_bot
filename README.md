# Polymarket Multi-Asset Up/Down Trading System

An async Python trading system for Polymarket's short-horizon crypto "up or down" prediction markets. It ingests live exchange and derivatives data, computes a 58-feature signal vector, feeds it to an online-retrained gradient-boosting model, sizes bets with fractional Kelly against an expected-value threshold, and executes against the Polymarket CLOB.

BTC, ETH, SOL and XRP, on 5-minute and 15-minute settlement windows.

`DRY_RUN=true` by default. It does not place real orders unless you explicitly turn that off.

---

## The part worth reading first

The most useful thing I did on this project was discover that half of my own results were fake.

The system ran profitably in paper trading for weeks. In June 2026 I audited where the fills were actually coming from, by tagging every trade with the source of the order book it priced against. The result:

- Roughly **half of all 15-minute trades** — and 82% of those in the 0.50 price bucket — were not pricing against the live CLOB at all. The CLOB book empties out near settlement, so by our T-180s entry the client was silently falling back to Polymarket's frozen Gamma seed price.
- Those fallback trades had won **84 out of 84**. A 100% win rate is not an edge, it is a bug. There was no liquidity to fill against, so the "fills" were fictional.
- They had inflated the paper bankroll by about **53%**.
- Filtering to trades that priced against a real CLOB book, the measured win rate was **~68%** over that sample — smaller, slower, and actually fillable.

The fix is `REQUIRE_CLOB_BOOK` (default on, `config.py`). When a market only offers a Gamma-seeded book, the decision is still computed and logged for model training, but **no order is placed and no capital is committed**. Halving your own headline number is not fun. Shipping a system whose backtest cannot be reproduced with real money is worse.

This pattern — instrument the thing, distrust the good result, find the leak — is most of what the rest of the codebase is.

Two more findings in the same spirit, both still true and both visible in the config:

- **The 15m model is effectively a three-feature model.** Permutation importance concentrates almost entirely on `barrier_z`, `pm_yes_mid` and `futures_basis`. The other 55 features contribute nearly nothing. They are still computed and still fed in, because pruning them resets a mature model's learned weights for no measured gain — but I am not going to claim a 58-signal edge when the honest answer is three.
- **The 5m interval is structurally negative-EV.** Fee drag and spread dominate at that horizon regardless of model quality, in both directions — the "buy the cheap underdog" version fails too. 5m windows are still logged for training, but order placement on them is disabled.

---

## How it works

```
Binance WS ─┐
Coinalyze   │
Deribit IV  ├─▶ signals/aggregator.py ──▶ model/ensemble.py ──▶ execution/ev_calculator.py ──▶ execution/trader.py
News/Reddit │      58-feature vector        P(up), calibrated        EV + Kelly sizing           CLOB order
Polymarket ─┘
                                                    ▲                                                │
                                                    └──────── model/trainer.py ◀─────────────────────┘
                                                              settlement outcome → replay buffer
```

**Data layer** (`data/`) — async clients for Binance futures and spot websockets, Polymarket CLOB and Gamma, Polymarket's real-time data service, Coinalyze (funding, open interest, liquidations), Deribit (implied vol), Chainlink Data Streams, plus Fear & Greed, Reddit and news sentiment. Each maintains its own state object and reconnects independently, so one dead feed degrades the vector rather than halting the bot.

**Signal layer** (`signals/`) — 58 features: order-book imbalance at three depths, trade flow, RSI/MACD/EMA/Bollinger/VWAP, realized vol and vol ratios, z-scores, autocorrelation, Hurst exponent, regime classification, variance risk premium, funding rate, OI change, liquidation imbalance, anchored VWAP deviation, spot-to-Polymarket lag, BTC lead-lag, and the barrier pair below.

**The barrier signal** (`signals/barrier.py`) is the one that matters. At T-3min the settlement question reduces to: will price still be on the same side of the window-open reference when the window closes? Modeling that as a driftless random walk gives an analytic fair value — `barrier_z` (displacement in units of expected remaining-window volatility) and `barrier_prob` (`2·Φ(z) − 1`). Handing the model the random-walk baseline means the other features only have to learn the *residual* drift edge, rather than rediscovering first-passage probability from scratch. This is why it dominates the importance ranking.

**Model** (`model/ensemble.py`) — `HistGradientBoostingClassifier`, one per interval. Gradient boosting replaced an MLP that degraded under `partial_fit`; this does a full refit on a 2000-sample replay buffer every 25 new labeled outcomes, with a 50-sample floor before it will predict at all. Probabilities are calibrated (isotonic, sigmoid fallback), hyperparameters auto-tune by `TimeSeriesSplit` grid search every 5th retrain once 200+ samples exist, and permutation importance is logged so feature decay is visible rather than assumed.

**Labels** come from `model/trainer.py`, which holds each placed trade as a `PendingTrade` until its market settles, then writes the realized outcome back into the replay buffer. The model learns from settled reality, not from its own predictions.

**Execution** (`execution/`) — `ev_calculator.py` computes `EV = P(win)·profit_if_win − P(lose)·stake` net of Polymarket's 2% winnings fee, then applies the gates: EV ≥ 0.50 on 15m, model conviction ≥ 0.60, entry price ≤ 0.60 (no paying up for near-certainties whose downside is asymmetric). Surviving trades size with fractional Kelly — 0.35 Kelly, capped at 15% of bankroll. `trader.py` handles the entry window: watch from T-185s, execute at T-180s, hard abort after T-170s.

The 15m EV threshold was raised from 0.30 to 0.50 after the thin-EV bucket was measured losing money on its own. Most of the constants in `config.py` have a similar story attached in a comment.

---

## Risk controls

Most of these exist because something went wrong first.

| Control | What it does |
|---|---|
| `REQUIRE_CLOB_BOOK` | No order without a real order book. See above. |
| Single-instance lock | `bot.lock` via `msvcrt`/`fcntl`. Two instances writing the same CSVs corrupted them with null bytes on 2026-08-11. OS-released, so no stale-lock problem. |
| Exposure tracker (`exposure.py`) | Process-wide committed-capital pool. Without it every simultaneous bet sizes off the full bankroll and the account is silently over-leveraged. |
| Fractional Kelly | 0.35 Kelly, hard cap 15% of bankroll per bet. |
| Correlation haircut | Sizing is reduced for simultaneous positions in correlated assets. |
| Conviction floor | Below `MODEL_CONVICTION_FLOOR` the model does not bet, regardless of EV. |
| Early exit | Closes at 0.92 rather than riding to settlement for the last 8 cents. |
| Live-test budget | Real-money mode is capped at $2/order, $5 max order, $10 total, 5 orders, BTC only, 15m only — a bounded trial, not a switch to production. |
| Circuit breaker | Pauses on a losing streak. Currently disabled by design: it blocks exactly the high-conviction bets whose win rate is being measured. Re-enable for capital protection once accuracy is validated. |

---

## Backtesting

`backtest.py` pulls 30 days of 1-minute Binance candles, reconstructs signal snapshots at every 5m and 15m window boundary, trains on the first 70% of windows and evaluates on the last 30%.

It is deliberately handicapped: there is no historical order book, no historical Polymarket price, and no historical Coinalyze data, so microstructure features are proxied from candles and the unavailable feeds are held neutral. **A backtest number from this harness is a sanity check, not a forecast** — the live system's best features are precisely the ones the backtest cannot reconstruct. Treating the two as comparable is how you end up believing an 84-for-84 win rate.

```bash
python backtest.py --days 60 --interval 15m --plot
```

---

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env      # fill in your own keys
python main.py
```

`DRY_RUN=true` is the default and is where you should stay. Everything is configured through `.env` and `config.py`; every behavioral change listed above is reversible by environment variable, because a trading system you cannot roll back in one line is a trading system you cannot safely tune.

---

## Known limitations

Stated plainly, because you would find them anyway:

- **Sample sizes are small.** Prediction-market settlement is the rate limiter — a 15m window produces at most 96 labeled outcomes per asset per day, and only a fraction are tradeable. Several parameters here were tuned on a few hundred trades. Some of that fit is noise.
- **The edge shrinks as samples grow.** Every strategy variant I have tested has seen its measured edge decay with more data. The 68% figure above is from one measurement window and should be assumed optimistic.
- **The feature set is largely inert.** Three signals do the work; see above.
- **No live-money validation at scale.** Real-money mode has only ever run under the bounded live-test caps.
- **Windows-first.** The instance lock branches on platform and the scheduling helpers assume Windows.

---

## Layout

```
main.py                 async event loop, wires everything, instance lock
config.py               all tunable parameters, env-overridable
data/                   exchange, derivatives, oracle and sentiment clients
signals/                58-feature vector; barrier.py is the important one
model/                  gradient-boosting ensemble + online outcome trainer
execution/              EV calculation, Kelly sizing, CLOB order placement
backtest.py             offline harness (candle-only, handicapped)
paper_trading.py        paper log + settlement accounting
exposure.py             process-wide open-exposure tracker
```

---

Built as a personal research project. Nothing here is financial advice, and the measured results above are historical, small-sample, and not a claim about future performance.
