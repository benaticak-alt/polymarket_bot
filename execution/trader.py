"""
Trade executor: manages the T-10s execution window for each active market.
Coordinates signal computation → model inference → EV check → order placement.
"""

import asyncio
import time
import datetime as _dt
from data.binance_ws import MarketState
from data.oracle_feed import OracleState
from data.coinalyze_api import CoinAlyzeState
from data.polymarket_api import PolymarketClient, PolymarketMarket, OrderBookState
from data.fear_greed import FearGreedState
from data.reddit_sentiment import RedditSentimentState
from data.marketmind_api import MarketMindState
from data.deribit_iv import DeribitIVState
from data.news_sentiment import NewsSentimentState
from data.binance_spot import SpotState
from data.kalshi_api import KalshiState
from signals.aggregator import SignalAggregator, SIGNAL_NAMES
from signals.regime import detect_regime, threshold_multiplier
from model.ensemble import SignalEnsemble
from model.trainer import OutcomeTracker, PendingTrade
from paper_trading import PaperTradingLog, PaperTrade
from execution.ev_calculator import compute_ev, TradeDecision
from config import (
    ENTRY_WINDOW_START_S, ENTRY_WINDOW_EXEC_S, ENTRY_HARD_ABORT_S, DRY_RUN,
    EV_THRESHOLD_5M, EV_THRESHOLD_15M, CONFLUENCE_MIN_AGREE,
    SKIP_LOW_LIQUIDITY_HOURS, SKIP_ON_IV_SPIKE, SKIP_ON_MODEL_DISAGREEMENT,
    CHAINLINK_SETTLEMENT_SOURCE, TRADE_5M, USE_MAKER_ORDERS, MAKER_TICK,
    ARB_SUM_THRESHOLD, REQUIRE_CLOB_BOOK, CL_PRICE_MAX_AGE_S,
    SHADOW_EARLY_ENTRY_ENABLED, SHADOW_ENTRY_TIMES,
    SHADOW_DYNAMIC_ENABLED, SHADOW_DYNAMIC_INTERVAL_S,
    SHADOW_DYNAMIC_START_S, SHADOW_DYNAMIC_END_S, SHADOW_NOCONV_ENABLED, SKIP_HOURS_UTC,
    LIVE_TEST_MODE, LIVE_TEST_STAKE_USDC, LIVE_MAX_ORDER_USDC,
    LIVE_TEST_ASSET, LIVE_TEST_INTERVAL,
)
from utils.logger import get_logger

# Confluence gate signal indices, resolved at import so a reorder breaks loudly.
_IDX_CVD_SLOPE   = SIGNAL_NAMES.index("cvd_slope")
_IDX_CANDLE_RUN  = SIGNAL_NAMES.index("candle_run")
_IDX_OBI_SHALLOW = SIGNAL_NAMES.index("obi_shallow")

log = get_logger("trader")


class MarketExecutor:
    """
    Manages the execution lifecycle for a single active Polymarket market.
    Runs as an asyncio task.
    """

    def __init__(
        self,
        market: PolymarketMarket,
        market_state: MarketState,
        oracle_state: OracleState,
        ca_state: CoinAlyzeState,
        pm_client: PolymarketClient,
        aggregator: SignalAggregator,
        ensemble: SignalEnsemble,
        tracker: OutcomeTracker,
        paper_log: "PaperTradingLog",
        fg_state: "FearGreedState | None" = None,
        reddit_state: "RedditSentimentState | None" = None,
        mm_state: "MarketMindState | None" = None,
        deribit_state: "DeribitIVState | None" = None,
        news_state: "NewsSentimentState | None" = None,
        spot_state: "SpotState | None" = None,
        kalshi_state: "KalshiState | None" = None,
        ensemble_other: "SignalEnsemble | None" = None,
        cl_state: "ChainlinkAssetState | None" = None,
        btc_market_state: "MarketState | None" = None,
        asset_name: str = "btc",
    ):
        self._asset_name   = asset_name
        self.market        = market
        self.market_state  = market_state
        self.oracle_state  = oracle_state
        self.ca_state      = ca_state
        self.pm_client     = pm_client
        self.aggregator    = aggregator
        self.ensemble      = ensemble
        self.ensemble_other = ensemble_other
        self.tracker       = tracker
        self.paper_log     = paper_log
        self.fg_state      = fg_state
        self.reddit_state  = reddit_state
        self.mm_state      = mm_state
        self.deribit_state = deribit_state
        self.news_state    = news_state
        self.spot_state    = spot_state
        self.kalshi_state  = kalshi_state
        self.cl_state      = cl_state
        self.btc_market_state = btc_market_state
        self._traded              = False
        self._shadow_fired: set   = set()    # fixed shadow times already logged
        self._dyn_logged          = False    # dynamic shadow catch already logged
        self._dyn_last            = 0.0      # last dynamic-check wall time
        self._noconv_done         = False    # no-conviction shadow already logged
        self._window_entry_price: float = 0.0
        self._window_entry_price_cl: float = 0.0

    async def run(self) -> None:
        """
        Poll at 1Hz until the execution window opens, then act at T-10s.
        """
        market_id = self.market.condition_id[:12]

        while True:
            now    = time.time()
            t_left = self.market.end_ts - now  # seconds until settlement

            # Record BTC price at window start (matches Polymarket's resolution logic)
            if t_left <= self.market.window_step and self._window_entry_price == 0.0:
                self._window_entry_price = self.market_state.mid
                # Capture the Chainlink Data Streams reference at the same instant —
                # this is the line Polymarket actually settles up/down against.
                # Skip if the observation is stale (RTDS outage) — a frozen entry
                # corrupts both the barrier override and the settlement label.
                if (self.cl_state and self.cl_state.price > 0
                        and now - self.cl_state.last_update <= CL_PRICE_MAX_AGE_S):
                    self._window_entry_price_cl = self.cl_state.price

            # ── Shadow entry probes (15m only, measurement — no capital) ─────────
            is15    = self.market.interval == "15m"
            shadows = SHADOW_EARLY_ENTRY_ENABLED and is15
            dyn_on  = shadows and SHADOW_DYNAMIC_ENABLED
            if shadows:
                # Fixed snapshots at each configured time (T600, T330, …).
                for T in SHADOW_ENTRY_TIMES:
                    if T not in self._shadow_fired and t_left <= T:
                        self._shadow_fired.add(T)
                        await self._shadow_evaluate(f"T{int(T)}")
                # Dynamic: catch the first moment a bettable, gate-passing decision
                # appears — including AFTER the normal T-180s entry.
                if (dyn_on and not self._dyn_logged
                        and SHADOW_DYNAMIC_END_S < t_left <= SHADOW_DYNAMIC_START_S
                        and now - self._dyn_last >= SHADOW_DYNAMIC_INTERVAL_S):
                    self._dyn_last = now
                    if await self._shadow_evaluate("dyn", placeable_only=True):
                        self._dyn_logged = True
                # No-conviction probe at the real entry: same T-180s decision, but
                # with the low-conviction gate removed (min_conviction=0).
                if (SHADOW_NOCONV_ENABLED and not self._noconv_done
                        and t_left <= ENTRY_WINDOW_EXEC_S):
                    self._noconv_done = True
                    await self._shadow_evaluate("noconv", min_conviction=0.0)

            # ── Real execution at T-180s (once) ──────────────────────────────────
            if t_left < ENTRY_WINDOW_EXEC_S and not self._traded:
                await self._execute()
                self._traded = True   # don't break yet — dynamic shadow may watch on

            # ── Exit: real trade resolved AND dynamic watch finished ─────────────
            real_done = self._traded or t_left < ENTRY_HARD_ABORT_S
            dyn_done  = (not dyn_on) or self._dyn_logged or t_left <= SHADOW_DYNAMIC_END_S
            if real_done and dyn_done:
                break

            if t_left < ENTRY_WINDOW_START_S and not self._traded:
                # In the watch window pre-entry — poll at 250ms for precision
                await asyncio.sleep(0.25)
            else:
                # Not in window yet, or post-entry dynamic watching — 1s is fine
                await asyncio.sleep(1.0)

    async def _execute(self) -> None:
        market_id = self.market.condition_id[:12]
        t_left    = self.market.end_ts - time.time()

        log.info(f"[{market_id}] T{t_left:+.1f}s — entering execution window.")

        # ── Staleness guard — abort if Binance feed is dead ────────────────────
        data_age = time.time() - self.market_state.last_update if self.market_state.last_update > 0 else float("inf")
        if data_age > 5.0:
            age_str = f"{data_age:.1f}s" if data_age < 9999 else "never received"
            log.warning(
                f"[{market_id}] Market data stale ({age_str}) — skipping trade."
            )
            self._traded = True
            return

        # ── Compute signals ────────────────────────────────────────────────────
        pm_book   = self.pm_client.get_order_book(
            self.market.token_id_yes,
            self.market.token_id_no,
            fallback_bid=self.market.best_bid,
            fallback_ask=self.market.best_ask,
        )
        # Polymarket book depth imbalance (separate CLOB call — once per execution)
        pm_obi = self.pm_client.get_pm_obi(self.market.token_id_yes)

        # Intra-market arb check (log-only): buying BOTH sides costs yes_ask+no_ask
        # and pays exactly $1 at settlement. A sum under the threshold is riskless
        # edge — measuring frequency/size before deciding to act on it.
        if pm_book.yes_ask > 0 and pm_book.no_ask > 0:
            book_sum = pm_book.yes_ask + pm_book.no_ask
            if book_sum < ARB_SUM_THRESHOLD:
                log.info(
                    f"[ARB] [{market_id}] {self.market.interval} YES+NO = "
                    f"{book_sum:.4f} < {ARB_SUM_THRESHOLD} — riskless edge "
                    f"{(1.0 - book_sum):.4f}/share (log-only)"
                )

        # In Chainlink-settlement mode, compute barrier_z against the Data Streams
        # price (reference + current) so the dominant 15m feature is measured on the
        # same source Polymarket resolves on. Falls back to Binance if no CL price.
        barrier_ref      = self._window_entry_price
        barrier_override = 0.0
        if (CHAINLINK_SETTLEMENT_SOURCE == "chainlink" and self.cl_state
                and self.cl_state.price > 0 and self._window_entry_price_cl > 0
                and time.time() - self.cl_state.last_update <= CL_PRICE_MAX_AGE_S):
            barrier_ref      = self._window_entry_price_cl
            barrier_override = self.cl_state.price

        signal_vec = self.aggregator.compute(
            self.market_state, self.oracle_state, self.ca_state, pm_book,
            fg_state=self.fg_state, reddit_state=self.reddit_state,
            mm_state=self.mm_state, interval=self.market.interval,
            deribit_state=self.deribit_state, news_state=self.news_state,
            spot_state=self.spot_state, kalshi_state=self.kalshi_state,
            reference_price=barrier_ref,
            seconds_remaining=self.market.end_ts - time.time(),
            barrier_price_override=barrier_override,
            pm_obi=pm_obi,
            btc_market_state=self.btc_market_state,
        )

        # ── Model inference ────────────────────────────────────────────────────
        model_prob = self.ensemble.predict(signal_vec)

        # ── Dynamic EV threshold (regime + recent performance) ─────────────────
        regime    = detect_regime(self.market_state, self.deribit_state)
        recent_wr = self.paper_log.recent_win_rate if self.paper_log else 0.5
        mult      = threshold_multiplier(regime, recent_wr)
        base_thr  = EV_THRESHOLD_5M if self.market.interval == "5m" else EV_THRESHOLD_15M
        ev_thr    = base_thr * mult

        # ── EV filter ─────────────────────────────────────────────────────────
        decision = compute_ev(model_prob, self.market, pm_book, self.market.interval,
                              ev_threshold_override=ev_thr)

        # ── Untrained-model gate ───────────────────────────────────────────────
        # An unfitted model returns a flat p=0.5 from predict(). compute_ev scores
        # that as huge +EV on any market priced away from 0.5 (the cheap side looks
        # like a bargain), so without this guard the bot bets a no-information
        # default against an informed market and bleeds money. Still log the
        # decision so the outcome trains the model — just never place an order.
        if not self.ensemble.is_trained:
            decision.should_trade = False
            decision.stake_usdc   = 0.0
            decision.reason       = "model_untrained"

        log.info(
            f"[{market_id}] p(up)={model_prob:.3f} | "
            f"EV={decision.ev_fraction:.3f} | "
            f"thr={ev_thr:.3f} (regime={regime}, wr={recent_wr:.0%}) | "
            f"dir={decision.direction} | "
            f"trade={decision.should_trade} | "
            f"reason={decision.reason}"
        )

        # ── Order gating ────────────────────────────────────────────────────────
        # Every decision is logged below for training regardless of these gates —
        # they block ONLY whether a bet is placed, so the model still learns from
        # low-liquidity hours, IV spikes, and disagreement windows.
        low_liq = False
        if SKIP_LOW_LIQUIDITY_HOURS:
            try:
                from zoneinfo import ZoneInfo
                _et_hour = _dt.datetime.now(ZoneInfo("America/New_York")).hour
            except Exception:
                _et_hour = (_dt.datetime.now(_dt.timezone.utc).hour - 4) % 24
            low_liq = 2 <= _et_hour < 6

        iv_spike = bool(
            SKIP_ON_IV_SPIKE and self.deribit_state
            and self.deribit_state.last_updated > 0
            and self.deribit_state.iv_score > 0.6
        )

        disagree = False
        if (SKIP_ON_MODEL_DISAGREEMENT and self.ensemble_other
                and self.ensemble_other.is_trained):
            other_prob = self.ensemble_other.predict(signal_vec)
            if (abs(model_prob - 0.5) > 0.15 and abs(other_prob - 0.5) > 0.15
                    and (model_prob > 0.5) != (other_prob > 0.5)):
                disagree = True

        # Live-test gating: only one asset + 15m, and stop at the budget/order cap.
        lt_active   = LIVE_TEST_MODE and not DRY_RUN
        lt_scope_ok = (self._asset_name == LIVE_TEST_ASSET
                       and self.market.interval == LIVE_TEST_INTERVAL)
        lt_can, lt_reason = True, ""
        if lt_active and lt_scope_ok:
            import live_test
            lt_can, lt_reason = live_test.can_place(LIVE_TEST_STAKE_USDC)

        block_reason = ""
        if not decision.should_trade:
            block_reason = decision.reason
        elif self.market.interval == "5m" and not TRADE_5M:
            # 5m is structurally -EV (high win rate, but pays favorite prices).
            # Keep logging decisions so the model still trains, but commit no
            # capital — lets 15m compound off the full bankroll/exposure budget.
            block_reason = "5m_disabled"
        elif REQUIRE_CLOB_BOOK and not pm_book.is_clob:
            # No real resting book — only the frozen Gamma seed. Those "fills" are
            # fictional (no liquidity); log for training but commit no capital.
            block_reason = "no_clob_book"
        elif lt_active and not lt_scope_ok:
            block_reason = "live_test_scope"   # not the test asset/interval
        elif lt_active and not lt_can:
            block_reason = f"live_test_{lt_reason}"   # budget or order cap hit
        elif low_liq:
            block_reason = "low_liquidity_hours"
        elif _dt.datetime.now(_dt.timezone.utc).hour in SKIP_HOURS_UTC:
            block_reason = "skip_hours_utc"
        elif disagree:
            block_reason = "model_disagreement"
        elif iv_spike:
            block_reason = "iv_spike"
        elif self.paper_log and self.paper_log.is_paused:
            block_reason = "circuit_breaker"
            log.info(f"[{market_id}] Circuit breaker active — logging only, no order placed.")
        else:
            # Confluence gate: require 2+ independent sources to agree on direction.
            expecting_up   = model_prob > 0.5
            agrees, checked = 1, 1   # model is baseline

            # MarketMind (skip if never polled or NEUTRAL)
            if self.mm_state and self.mm_state.last_updated > 0:
                mm_dir = (
                    self.mm_state.direction_5m if self.market.interval == "5m"
                    else self.mm_state.direction_15m
                )
                if mm_dir != 0.0:
                    checked += 1
                    if (expecting_up and mm_dir > 0) or (not expecting_up and mm_dir < 0):
                        agrees += 1

            # Order-flow trend — is aggressor (taker) flow accelerating our way?
            cvd_slope = float(signal_vec[_IDX_CVD_SLOPE])
            if abs(cvd_slope) > 0.15:
                checked += 1
                if (expecting_up and cvd_slope > 0) or (not expecting_up and cvd_slope < 0):
                    agrees += 1

            # Candle-run persistence
            candle_run = float(signal_vec[_IDX_CANDLE_RUN])
            if abs(candle_run) > 0.4:
                checked += 1
                if (expecting_up and candle_run > 0) or (not expecting_up and candle_run < 0):
                    agrees += 1

            # Order book imbalance
            obi = float(signal_vec[_IDX_OBI_SHALLOW])
            if abs(obi) > 0.30:
                checked += 1
                if (expecting_up and obi > 0) or (not expecting_up and obi < 0):
                    agrees += 1

            if checked >= 2 and agrees < CONFLUENCE_MIN_AGREE:
                block_reason = "confluence"
                log.info(
                    f"[{market_id}] Confluence gate blocked "
                    f"({agrees}/{checked} agree | p={model_prob:.3f} "
                    f"cvd_slope={cvd_slope:.3f} candle_run={candle_run:.3f})"
                )

        # Live-test mode: replace the (paper-bankroll-anchored) Kelly stake with a
        # flat tiny REAL stake for the in-scope order, before it's logged or placed.
        if lt_active and not block_reason and decision.should_trade and decision.direction in ("YES", "NO"):
            decision.stake_usdc = round(min(LIVE_TEST_STAKE_USDC, LIVE_MAX_ORDER_USDC), 2)

        # Log every decision to paper trading so the model can train on all outcomes
        pt = None
        if self.paper_log and self.market_state.mid > 0 and decision.direction in ("YES", "NO"):
            pt = PaperTrade(
                timestamp=time.time(),
                market_id=self.market.condition_id,
                interval=self.market.interval,
                question=self.market.question,
                direction=decision.direction,
                model_prob=decision.model_prob,
                ev_fraction=decision.ev_fraction,
                stake_usdc=decision.stake_usdc,
                market_price=decision.market_price,
                btc_price_entry=self._window_entry_price or self.market_state.mid,
                chainlink_price_entry=self._window_entry_price_cl,
                signal_vec=signal_vec,
                settlement_ts=self.market.end_ts,
                train_count=self.ensemble._train_count,
                block_reason=block_reason,
                market=self.market,
                book_src=("clob" if pm_book.is_clob else "gamma"),
                book_spread=pm_book.spread_yes(),
            )
            self.paper_log.record_decision(pt)

        if not decision.should_trade or block_reason:
            self._traded = True
            return

        # ── Check we're still in time ──────────────────────────────────────────
        if time.time() > self.market.end_ts - ENTRY_HARD_ABORT_S:
            log.warning(f"[{market_id}] Slipped past hard abort — skipping.")
            self._traded = True
            return

        # ── Place order ────────────────────────────────────────────────────────
        # Maker path (live only): rest a GTC limit one tick above our side's bid to
        # earn the spread instead of crossing it. Falls back to a taker market order
        # under DRY_RUN or when disabled. T-3min entry leaves time for a maker fill.
        btc_entry_price = self.market_state.mid
        if USE_MAKER_ORDERS and not DRY_RUN:
            our_bid = pm_book.yes_bid if decision.direction == "YES" else pm_book.no_bid
            # Improve the bid by a tick but never cross above our evaluated price.
            limit_price = min(decision.market_price, round(our_bid + MAKER_TICK, 2))
            result = self.pm_client.place_limit_order(
                token_id=decision.token_id,
                side="BUY",
                amount_usdc=decision.stake_usdc,
                limit_price=limit_price,
                dry_run=DRY_RUN,
            )
        else:
            result = self.pm_client.place_market_order(
                token_id=decision.token_id,
                side="BUY",
                amount_usdc=decision.stake_usdc,
                dry_run=DRY_RUN,
            )
        self._traded = True

        if result.get("status") in ("ok", "dry_run"):
            if pt is not None:
                pt.order_placed = True
            import exposure
            exposure.add(decision.stake_usdc)
            import correlation
            correlation.add(self.market.condition_id, decision.direction)
            pending = PendingTrade(
                market_id=self.market.condition_id,
                token_id=decision.token_id,
                direction=decision.direction,
                entry_signal_vec=signal_vec,
                entry_price_btc=self._window_entry_price or btc_entry_price,
                entry_ts=time.time(),
                settlement_ts=self.market.end_ts,
                usdc_staked=decision.stake_usdc,
            )
            self.tracker.record_trade(pending)

            # Count the order against the live-test budget/order caps.
            if lt_active and result.get("status") == "ok":
                import live_test
                live_test.record(decision.stake_usdc)

            tag = "[DRY]" if DRY_RUN else "[LIVE]"
            book_src = "clob" if pm_book.is_clob else "gamma"
            log.info(
                f"{tag} [{market_id}] PLACED {decision.direction} "
                f"${decision.stake_usdc:.2f} @ {decision.market_price:.4f} "
                f"| book={book_src} | EV={decision.ev_fraction:.3f}"
            )
            # Surface the raw exchange response so the actual fill price can be
            # compared against the expected entry — the whole point of the probe.
            if result.get("status") == "ok":
                log.info(f"[LIVE FILL] [{market_id}] expected≈{decision.market_price:.4f} "
                         f"({book_src}) | exchange response: {result.get('response')}")
        else:
            log.error(f"[{market_id}] Order failed: {result}")

    async def _shadow_evaluate(self, kind: str, placeable_only: bool = False,
                               min_conviction: float | None = None) -> bool:
        """Run the signal→model→EV path and log a shadow row (kind = T600/T330/dyn).
        Places no order and never trains. Returns whether the decision would clear
        all gates. If placeable_only, only records (and only returns True) when it
        would — used by the dynamic probe to catch the first bettable moment.
        Skips the pm_obi book fetch (≈0 importance) to keep the extra HTTP low."""
        try:
            data_age = time.time() - self.market_state.last_update if self.market_state.last_update > 0 else float("inf")
            if data_age > 5.0:
                return False

            pm_book = self.pm_client.get_order_book(
                self.market.token_id_yes, self.market.token_id_no,
                fallback_bid=self.market.best_bid, fallback_ask=self.market.best_ask,
            )
            barrier_ref      = self._window_entry_price
            barrier_override = 0.0
            if (CHAINLINK_SETTLEMENT_SOURCE == "chainlink" and self.cl_state
                    and self.cl_state.price > 0 and self._window_entry_price_cl > 0
                    and time.time() - self.cl_state.last_update <= CL_PRICE_MAX_AGE_S):
                barrier_ref      = self._window_entry_price_cl
                barrier_override = self.cl_state.price

            signal_vec = self.aggregator.compute(
                self.market_state, self.oracle_state, self.ca_state, pm_book,
                fg_state=self.fg_state, reddit_state=self.reddit_state,
                mm_state=self.mm_state, interval="15m",
                deribit_state=self.deribit_state, news_state=self.news_state,
                spot_state=self.spot_state, kalshi_state=self.kalshi_state,
                reference_price=barrier_ref,
                seconds_remaining=self.market.end_ts - time.time(),
                barrier_price_override=barrier_override,
                pm_obi=0.0, btc_market_state=self.btc_market_state,
            )
            if not self.ensemble.is_trained:
                return False
            model_prob = self.ensemble.predict(signal_vec)
            decision   = compute_ev(model_prob, self.market, pm_book, "15m",
                                    conviction_floor_override=min_conviction)

            if placeable_only and not decision.should_trade:
                return False

            import shadow_entry
            shadow_entry.record(
                market_id=self.market.condition_id, asset=self._asset_name, kind=kind,
                seconds_left=self.market.end_ts - time.time(),
                price=decision.market_price, direction=decision.direction,
                model_prob=decision.model_prob, ev_fraction=decision.ev_fraction,
                would_place=decision.should_trade,
                book_src=("clob" if pm_book.is_clob else "gamma"),
            )
            return bool(decision.should_trade)
        except Exception as e:
            log.debug(f"[{self.market.condition_id[:12]}] shadow eval failed: {e}")
            return False


class TradingOrchestrator:
    """
    Top-level orchestrator: discovers markets, spawns executors,
    and manages the settlement tracking loop.
    """

    def __init__(
        self,
        market_state: MarketState,
        oracle_state: OracleState,
        ca_state: CoinAlyzeState,
        pm_client: PolymarketClient,
        aggregator: SignalAggregator,
        ensemble: SignalEnsemble,
        tracker: OutcomeTracker,
        paper_log: "PaperTradingLog",
        fg_state: "FearGreedState | None" = None,
        reddit_state: "RedditSentimentState | None" = None,
        mm_state: "MarketMindState | None" = None,
        deribit_state: "DeribitIVState | None" = None,
        news_state: "NewsSentimentState | None" = None,
        spot_state: "SpotState | None" = None,
        kalshi_state: "KalshiState | None" = None,
        ensemble_5m: "SignalEnsemble | None" = None,
        ensemble_15m: "SignalEnsemble | None" = None,
        tracker_5m: "OutcomeTracker | None" = None,
        tracker_15m: "OutcomeTracker | None" = None,
        paper_log_5m: "PaperTradingLog | None" = None,
        paper_log_15m: "PaperTradingLog | None" = None,
        asset_name: str = "btc",
        cl_state: "ChainlinkAssetState | None" = None,
        btc_market_state: "MarketState | None" = None,
    ):
        self._asset_name   = asset_name
        self.cl_state      = cl_state
        self.btc_market_state = btc_market_state
        self.market_state  = market_state
        self.oracle_state  = oracle_state
        self.ca_state      = ca_state
        self.pm_client     = pm_client
        self.aggregator    = aggregator
        self.ensemble      = ensemble       # fallback if per-interval not set
        self.tracker       = tracker
        self.paper_log     = paper_log
        self.fg_state      = fg_state
        self.reddit_state  = reddit_state
        self.mm_state      = mm_state
        self.deribit_state = deribit_state
        self.news_state    = news_state
        self.spot_state    = spot_state
        self.kalshi_state  = kalshi_state
        self._ensemble_5m  = ensemble_5m  or ensemble
        self._ensemble_15m = ensemble_15m or ensemble
        self._tracker_5m   = tracker_5m   or tracker
        self._tracker_15m  = tracker_15m  or tracker
        self._paper_log_5m  = paper_log_5m  or paper_log
        self._paper_log_15m = paper_log_15m or paper_log
        self._active_ids: set[str] = set()

    async def market_discovery_loop(self) -> None:
        """Refresh active markets every 30 seconds, with exponential backoff on errors."""
        backoff = 30  # seconds; doubles on each consecutive error, caps at 300
        asset_upper = self._asset_name.upper()
        while True:
            try:
                markets = self.pm_client.fetch_active_markets(self._asset_name)
                if backoff > 30:
                    log.info(f"[{asset_upper}] Polymarket API recovered — resuming normal discovery.")
                backoff = 30  # reset on success
                log.info(f"[{asset_upper}] Discovered {len(markets)} active markets.")

                for market in markets:
                    if market.condition_id in self._active_ids:
                        continue
                    t_left = market.end_ts - time.time()
                    # Only spawn executor if > 20s remaining
                    if t_left < 20:
                        continue

                    self._active_ids.add(market.condition_id)
                    is_15m = market.interval == "15m"
                    executor = MarketExecutor(
                        market=market,
                        market_state=self.market_state,
                        oracle_state=self.oracle_state,
                        ca_state=self.ca_state,
                        pm_client=self.pm_client,
                        aggregator=self.aggregator,
                        ensemble=self._ensemble_15m        if is_15m else self._ensemble_5m,
                        ensemble_other=self._ensemble_5m   if is_15m else self._ensemble_15m,
                        tracker=self._tracker_15m          if is_15m else self._tracker_5m,
                        paper_log=self._paper_log_15m      if is_15m else self._paper_log_5m,
                        fg_state=self.fg_state,
                        reddit_state=self.reddit_state,
                        mm_state=self.mm_state,
                        deribit_state=self.deribit_state,
                        news_state=self.news_state,
                        spot_state=self.spot_state,
                        kalshi_state=self.kalshi_state,
                        cl_state=self.cl_state,
                        btc_market_state=self.btc_market_state,
                        asset_name=self._asset_name,
                    )
                    task = asyncio.create_task(executor.run(), name=f"exec_{market.condition_id[:8]}")
                    task.add_done_callback(
                        lambda t: log.error(f"Executor crashed: {t.exception()}", exc_info=t.exception())
                        if not t.cancelled() and t.exception() else None
                    )
                    log.info(
                        f"Spawned executor: {market.question[:60]} | "
                        f"interval={market.interval} | T-{t_left:.0f}s"
                    )
            except Exception as e:
                log.warning(
                    f"Market discovery error (retrying in {backoff}s): {e}"
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 300)  # double each failure, cap at 5 min
                continue

            await asyncio.sleep(30)
