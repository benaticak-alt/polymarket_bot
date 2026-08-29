"""
Polymarket Multi-Asset Up/Down Trading System
Entry point — wires all components and runs the async event loop.
"""

import asyncio
import signal
import sys
from utils.logger import get_logger
from data.binance_ws import BinanceStreamClient, BinanceUSTradeClient, MarketState
from data.oracle_feed import OracleState
from data.polymarket_api import PolymarketClient
from data.coinalyze_api import CoinAlyzeClient, CoinAlyzeState
from data.fear_greed import FearGreedClient, FearGreedState
from data.reddit_sentiment import RedditSentimentClient, RedditSentimentState
from data.marketmind_api import MarketMindClient, MarketMindState
from data.deribit_iv import DeribitIVClient, DeribitIVState
from data.news_sentiment import NewsSentimentClient, NewsSentimentState
from data.binance_spot import BinanceSpotClient, SpotState
from data.kalshi_api import KalshiState
from data.kalshi_trade import KalshiTradeClient
from data.chainlink_stream import ChainlinkStreamClient, ChainlinkAssetState
from data.polymarket_rtds import PolymarketRTDSClient
from execution.arb_executor import arb_loop
from signals.aggregator import SignalAggregator
from model.ensemble import SignalEnsemble
from model.trainer import OutcomeTracker
from execution.trader import TradingOrchestrator
from paper_trading import PaperTradingLog
from config import (
    DRY_RUN, TRADING_ASSETS, SLIM_5M_FEATURES,
    CHAINLINK_STREAMS_ENABLED, CHAINLINK_SETTLEMENT_SOURCE, CHAINLINK_DS_FEED_IDS,
    CHAINLINK_DS_API_KEY, CHAINLINK_DS_API_SECRET, CHAINLINK_PRICE_SOURCE,
)

log = get_logger("main")


async def main() -> None:
    log.info("=" * 60)
    log.info("Polymarket Multi-Asset Trading System starting...")
    log.info(f"DRY_RUN = {DRY_RUN}")
    log.info(f"Assets: {[a['name'].upper() for a in TRADING_ASSETS]}")
    log.info("=" * 60)

    # Reset the live-test budget counters at the start of each run (bounded trial).
    from config import LIVE_TEST_MODE
    if LIVE_TEST_MODE and not DRY_RUN:
        import live_test
        live_test.reset()
        log.warning("LIVE TEST MODE ON — real orders, capped. See live_test_status.json.")

    # ── Shared external API states (crypto-wide, not asset-specific) ───────────
    fg_state      = FearGreedState()
    reddit_state  = RedditSentimentState()
    mm_state      = MarketMindState()
    deribit_state = DeribitIVState()
    news_state    = NewsSentimentState()
    kalshi_state  = KalshiState()
    kalshi_trader = KalshiTradeClient()

    # ── Shared clients ─────────────────────────────────────────────────────────
    pm_client      = PolymarketClient()
    aggregator     = SignalAggregator()
    fg_client      = FearGreedClient(fg_state)
    reddit_client  = RedditSentimentClient(reddit_state)
    mm_client      = MarketMindClient(mm_state)
    deribit_client = DeribitIVClient(deribit_state)
    news_client    = NewsSentimentClient(news_state)

    # kalshi_state / oracle_state kept for signature compatibility but no longer
    # fed into the vector: their slots were replaced (Kalshi API dead, Polygon
    # RPCs flaky), so no KalshiClient / OracleFeedClient poller is started.

    # ── BTC-specific shared states ─────────────────────────────────────────────
    oracle_state  = OracleState()
    ca_state      = CoinAlyzeState()
    ca_client     = CoinAlyzeClient(ca_state)

    # ── Graceful shutdown ──────────────────────────────────────────────────────
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _shutdown(sig_name: str) -> None:
        log.info(f"Received {sig_name} — shutting down...")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda s=sig.name: _shutdown(s))
        except NotImplementedError:
            pass

    # ── Per-asset components ───────────────────────────────────────────────────
    tasks = [
        asyncio.create_task(fg_client.run(),            name="fear_greed"),
        asyncio.create_task(reddit_client.run(),        name="reddit_sentiment"),
        asyncio.create_task(mm_client.run(),            name="marketmind"),
        asyncio.create_task(deribit_client.run(),       name="deribit_iv"),
        asyncio.create_task(news_client.run(),          name="news_sentiment"),
        asyncio.create_task(ca_client.run(),            name="coinalyze"),
        asyncio.create_task(arb_loop(pm_client, kalshi_trader), name="arb_executor"),
        asyncio.create_task(stop_event.wait(),          name="stop_sentinel"),
    ]

    # ── Chainlink Data Streams (settlement-truth price source) ──────────────────
    # One client streams all asset feeds; each asset's state is the price source
    # Polymarket actually settles on. Disabled unless explicitly enabled AND creds
    # present — otherwise cl_states stays empty and the bot runs exactly as before.
    cl_states: dict[str, ChainlinkAssetState] = {}
    if CHAINLINK_STREAMS_ENABLED and CHAINLINK_PRICE_SOURCE == "rtds":
        # Polymarket's free re-broadcast of the Chainlink settlement prices —
        # no key needed, unaffected by Chainlink's July 2026 paid-tier change.
        symbols: dict[str, ChainlinkAssetState] = {}
        for a in TRADING_ASSETS:
            st = ChainlinkAssetState()
            cl_states[a["name"]] = st
            symbols[f"{a['name']}/usd"] = st
        rtds_client = PolymarketRTDSClient(symbols)
        tasks.append(asyncio.create_task(rtds_client.run(), name="chainlink_rtds"))
        log.info(
            f"Chainlink via Polymarket RTDS ON — settle_source='{CHAINLINK_SETTLEMENT_SOURCE}', "
            f"assets={list(cl_states)}"
        )
    elif CHAINLINK_STREAMS_ENABLED and CHAINLINK_DS_API_KEY and CHAINLINK_DS_API_SECRET:
        feeds: dict[str, ChainlinkAssetState] = {}
        for a in TRADING_ASSETS:
            fid = CHAINLINK_DS_FEED_IDS.get(a["name"])
            if fid:
                st = ChainlinkAssetState()
                cl_states[a["name"]] = st
                feeds[fid] = st
        if feeds:
            cl_client = ChainlinkStreamClient(feeds)
            tasks.append(asyncio.create_task(cl_client.run(), name="chainlink_stream"))
            log.info(
                f"Chainlink Data Streams ON — settle_source='{CHAINLINK_SETTLEMENT_SOURCE}', "
                f"feeds={list(cl_states)}"
            )
        else:
            log.warning("Chainlink enabled but no feed IDs matched trading assets.")
    elif CHAINLINK_STREAMS_ENABLED:
        log.warning("CHAINLINK_STREAMS_ENABLED set but API key/secret missing — Data Streams off.")

    # BTC's market state, shared with the alt models for the btc_lead feature
    # (BTC leads alts at minute horizons). BTC is first in TRADING_ASSETS, so it's
    # populated before the alts are wired. BTC itself receives None (feature = 0).
    btc_mstate: MarketState | None = None

    for asset_cfg in TRADING_ASSETS:
        asset_name = asset_cfg["name"]   # "btc", "eth", …
        symbol     = asset_cfg["symbol"] # "BTCUSDT", "ETHUSDT", …

        # btc keeps "5m"/"15m" names so existing model weights / replay buffers load cleanly
        suffix = "" if asset_name == "btc" else f"_{asset_name}"

        # Per-asset market data streams
        mstate      = MarketState()
        if asset_name == "btc":
            btc_mstate = mstate
        spot_state  = SpotState()
        binance     = BinanceStreamClient(mstate, symbol=symbol)
        binance_us  = BinanceUSTradeClient(mstate, symbol=symbol)
        spot_client = BinanceSpotClient(spot_state, mstate, symbol=symbol)

        # Oracle & CoinAlyze are BTC-specific; non-BTC assets receive a null state
        ora = oracle_state if asset_name == "btc" else OracleState()
        ca  = ca_state     if asset_name == "btc" else CoinAlyzeState()

        # Chainlink Data Streams state for this asset (None unless enabled).
        # The price fn returns 0.0 when the last observation is older than
        # CL_PRICE_MAX_AGE_S — settlement then falls back to Binance rather than
        # settling against a frozen price from an RTDS outage.
        import time as _time
        from config import CL_PRICE_MAX_AGE_S
        cl_state    = cl_states.get(asset_name)
        cl_price_fn = (
            (lambda c=cl_state: c.price
             if _time.time() - c.last_update <= CL_PRICE_MAX_AGE_S else 0.0)
            if cl_state else None
        )

        # Per-asset models. 5m is pruned to a slim feature set (A/B showed its
        # non-price signals are net-harmful noise); 15m keeps the full set.
        ens_5m  = SignalEnsemble(name=f"5m{suffix}", keep_features=SLIM_5M_FEATURES)
        ens_15m = SignalEnsemble(name=f"15m{suffix}")
        trk_5m  = OutcomeTracker(ens_5m)
        trk_15m = OutcomeTracker(ens_15m)
        log_5m  = PaperTradingLog(ens_5m,  name=f"5m{suffix}")
        log_15m = PaperTradingLog(ens_15m, name=f"15m{suffix}")

        orchestrator = TradingOrchestrator(
            market_state=mstate,
            oracle_state=ora,
            ca_state=ca,
            pm_client=pm_client,
            aggregator=aggregator,
            ensemble=ens_5m,
            tracker=trk_5m,
            paper_log=log_5m,
            fg_state=fg_state,
            reddit_state=reddit_state,
            mm_state=mm_state,
            deribit_state=deribit_state,
            news_state=news_state,
            spot_state=spot_state,
            kalshi_state=kalshi_state,
            ensemble_5m=ens_5m,
            ensemble_15m=ens_15m,
            tracker_5m=trk_5m,
            tracker_15m=trk_15m,
            paper_log_5m=log_5m,
            paper_log_15m=log_15m,
            asset_name=asset_name,
            cl_state=cl_state,
            btc_market_state=None if asset_name == "btc" else btc_mstate,
        )

        # Capture loop variables in default args to avoid closure bug
        tasks.extend([
            asyncio.create_task(binance.run(),    name=f"binance_ws_{asset_name}"),
            asyncio.create_task(binance_us.run(), name=f"binance_us_{asset_name}"),
            asyncio.create_task(spot_client.run(), name=f"spot_{asset_name}"),
            asyncio.create_task(orchestrator.market_discovery_loop(), name=f"discovery_{asset_name}"),
            asyncio.create_task(
                trk_5m.settlement_loop(lambda s=mstate: s.mid),
                name=f"settle_trk_5m_{asset_name}",
            ),
            asyncio.create_task(
                trk_15m.settlement_loop(lambda s=mstate: s.mid),
                name=f"settle_trk_15m_{asset_name}",
            ),
            asyncio.create_task(
                log_5m.settlement_loop(
                    lambda s=mstate: s.mid,
                    chainlink_price_fn=cl_price_fn,
                    settle_source=CHAINLINK_SETTLEMENT_SOURCE,
                    pm_client=pm_client,
                ),
                name=f"settle_paper_5m_{asset_name}",
            ),
            asyncio.create_task(
                log_15m.settlement_loop(
                    lambda s=mstate: s.mid,
                    chainlink_price_fn=cl_price_fn,
                    settle_source=CHAINLINK_SETTLEMENT_SOURCE,
                    pm_client=pm_client,
                ),
                name=f"settle_paper_15m_{asset_name}",
            ),
        ])
        log.info(f"[{asset_name.upper()}] Components wired — stream={symbol}.")

    log.info("All tasks launched. System running.")

    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    for t in done:
        if t.get_name() != "stop_sentinel" and (exc := t.exception()):
            log.error(f"Task '{t.get_name()}' crashed: {exc}", exc_info=exc)

    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)

    log.info("Shutdown complete.")


def _acquire_single_instance_lock():
    """Refuse to start if another bot instance is already running. Two instances
    writing the same CSV/state files at once corrupts them (null-byte damage seen
    2026-08-11). The OS releases this lock automatically if the process dies, so
    there's no stale-lock problem. Returns the held file handle (keep it alive)."""
    import os
    from config import BASE_DIR
    lock_path = BASE_DIR / "bot.lock"
    fh = open(lock_path, "w")
    try:
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("\n*** Another bot instance is already running — refusing to start. ***")
        print("*** Close the other one first, or check with: tasklist | findstr python ***\n")
        sys.exit(1)
    fh.write(str(os.getpid())); fh.flush()
    return fh


if __name__ == "__main__":
    _lock = _acquire_single_instance_lock()   # held for process lifetime
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
