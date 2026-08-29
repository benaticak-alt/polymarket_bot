# Go-Live Checklist (paper → real money)

Don't apply the config flips until you're actually going live — re-enabling the
safety gates mid-accuracy-phase would block the high-conviction bets you're
currently measuring.

## Gate 0 — Earn the right to go live (the part that takes calendar time)
- [ ] ~100-200 **clean-book** 15m bets accumulated since the 2026-06-05 fallback fix
      (started ~28 bets / 12.9h on 06-06; ≈3-7 days to reach a trustworthy sample).
- [ ] 15m win rate holds (was 89% clean) — not a hot streak.
- [ ] Chainlink shadow 15m flip rate stays low (~3%). If it climbs >8%, stop and
      reconsider settling on `chainlink` before going live.

## Gate 1 — Config flips (config.py)
- [ ] L196 `STREAK_PAUSE_WR`            0.0   → 0.40   (circuit breaker back on)
- [ ] L171 `SKIP_LOW_LIQUIDITY_HOURS`   False → True   (block 2-6am ET thin book)
- [ ] L172 `SKIP_ON_IV_SPIKE`           False → True
- [ ] L173 `SKIP_ON_MODEL_DISAGREEMENT` False → True
- [ ] (optional) L165 `CONFLUENCE_MIN_AGREE` 1 → 2  (require a 2nd source to agree)

## Gate 2 — Stake sizing decision (yours to make)
- [ ] Set `MAX_POSITION_USDC` to match the real capital you actually deposit.
- [ ] Decide flat cap vs. % compounding (`KELLY_MAX_PCT`, currently 0.15 = 15%/trade).
      Note: the $500 cap currently pins every bet flat; % only compounds if the cap
      is lifted above `KELLY_MAX_PCT * bankroll`.

## Gate 3 — Prove the live execution path (NEVER run yet — DRY_RUN has always been true)
- [ ] `.env`: `DRY_RUN=false` with `MAX_POSITION_USDC=5` for the FIRST day.
- [ ] Confirm a real order: places on Polymarket CLOB, wallet signs, fills, settles.
- [ ] Confirm bankroll.json accounting matches the real fill (price + 2% fee).
- [ ] Watch for slippage / partial fills — paper assumed frictionless $500 fills at
      the best ask; real 15m books are thin. This is the real unknown.

## Gate 4 — Scale up
- [ ] Only after a clean day at $5, raise `MAX_POSITION_USDC` toward target size.
- [ ] Keep watching the circuit breaker + drawdown brake actually trigger as intended.

## Notes
- 15m is the engine (+EV, barrier-driven, ~79%); 5m is structurally -EV but kept live.
- Settlement flipped to `chainlink` on 2026-06-06 — Polymarket resolves on the
  Chainlink BTC/USD Data Stream (our configured feed is that exact stream). This
  recomputes barrier_z off Chainlink, so the Gate-0 clean-sample counter restarts
  from the flip; re-accumulate ~100-200 clean 15m bets on the new basis.
- Fallback fix (data/polymarket_api.py): missing order book now SKIPS instead of
  inventing a 0.50 price. A 0.50 that still appears is a real even-money market.
