# Core + Satellite Plan

The trading bot is the **satellite**: a capped, high-risk sleeve whose edge has to keep proving itself. The **core** is a rules-based portfolio whose returns come from holding assets, not from predicting them. Bot bankroll above a cap gets swept into the core, so winnings get banked in something that doesn't depend on the edge lasting.

Everything here runs on paper. There's no broker connection, and adding one is the last step, not the first.

## Status

| Phase | What | Status |
|---|---|---|
| 1 | Policy, ledger, drift-band rebalancer, paper broker | **Done** (`portfolio/`) |
| 2 | Bot → core sweep with cap, DRY_RUN and bot.lock guards | **Done** (`portfolio/sweep.py`) |
| 3 | Backtester: buy-and-hold vs calendar vs bands vs inverse-vol vs tilt, after costs | **Done** (`portfolio/backtest.py`), run on synthetic data only so far |
| 4 | Real historical data + honest evaluation | **Next: needs you** |
| 5 | Scheduled paper operation | Not started |
| 6 | Tax-lot tracking | Not started |
| 7 | Live broker adapter | Not started, deliberately last |

## Using it now

```bash
python -m portfolio backtest --synthetic          # exercise the code
python -m portfolio deposit 1000                  # paper cash in
python -m portfolio plan --prices prices.csv      # what would it trade?
python -m portfolio plan --prices prices.csv --apply
python -m portfolio status --prices prices.csv
python -m portfolio sweep                         # bot bankroll over SATELLITE_CAP_USD?
python -m pytest tests/test_portfolio.py
```

Policy lives in `portfolio_targets.json` (gitignored). Copy `portfolio_targets.example.json`; its assets and weights are placeholders, not a recommendation. The sweep cap is `SATELLITE_CAP_USD` / `SATELLITE_SWEEP_MIN_USD` in `.env`.

## How the rebalancer decides

- **Band per asset** = `min(abs_band, rel_band × target)` (the 5/25 rule). A 60% asset trades at ±5pp; a 5% asset at ±1.25pp.
- **Any asset outside its band** → full rebalance to target. Sells go first and fund the buys.
- **All inside, idle cash present** (deposit or sweep) → buy-only, most-underweight first. It moves toward target without selling.
- **Otherwise nothing happens.** Orders under `min_trade_usd` are dropped.
- `allow_sells: false` turns it into cash-flow-only rebalancing (slower, avoids realizing gains).

## Phase 4: what to do next (your decisions)

1. **Pick the assets and weights.** This is the one decision the code can't make, and I'm not a licensed advisor. The usual inputs are time horizon, how big a drawdown you'd actually sit through, and whether this money is needed soon.
2. **Get adjusted-close history** for those assets, as a wide CSV with a `date` column and one column per asset name in your targets file. It needs to be *adjusted* (dividends reinvested), or bond/equity returns will be understated. Aim for 15+ years so it spans at least one bad bear market.
3. **Run `python -m portfolio backtest --prices that.csv`** and apply the same suspicion the README applies to the bot:
   - Default to fixed weights with bands. Inverse-vol or a tilt only replaces it if it wins after costs **and** keeps winning on a period you held out (fit on the first half, judge on the second).
   - On synthetic data inverse-vol "won" by ~2.5pp CAGR. That comes from the random parameters I drew, not a finding. Expect a real test to shrink or reverse it.
   - The example trend tilt traded ~5× more than bands and lost after costs on synthetic data. That's the expected result, and it's there to demonstrate the hook.
4. **Decide the satellite cap.** Current bot bankroll is about **$76** against a $1,000 default cap, so the sweep won't fire for a while. That's fine: the cap limits risk going forward, and the bot doesn't have to fund the core. Deposits can come from outside.

## Later phases

- **5. Scheduled paper operation.** A weekly job: pull prices → `sweep` → `plan --apply` → log. Run it for a few months and compare the paper ledger with what you'd have done by hand.
- **6. Tax lots.** The ledger tracks units, not lots. Before any live use in a taxable account, add per-lot cost basis, prefer selling highest-basis lots, and add a harvest-losses check. Skip this for tax-advantaged accounts.
- **7. Live broker.** Implement the `Broker` protocol in `portfolio/broker.py` for a brokerage API, behind a separate `PORTFOLIO_LIVE=true` flag with PaperBroker as the default, mirroring `DRY_RUN`. Start with the same small-budget, capped-orders pattern as `live_test.py`. The sweep stays manual for real money: withdraw from Polymarket, deposit at the broker, record it with `deposit`.

## Known limits

- Fills in the paper broker and backtest happen at the day's close with a flat bps cost. That's fine for liquid index funds at a few trades a year, and wrong for anything thin.
- Inverse-vol ignores correlations (naive risk parity).
- Cash earns 0% in the backtest, which slightly penalises holding a cash target.
- No FX, no fractional-share limits, no dividends as separate cash flows (use adjusted prices instead).
