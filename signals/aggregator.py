"""
Signal aggregator: combines all signal modules into a single normalised
feature vector for the ensemble.

Feature vector layout (58 signals):
  [0–4]   Microstructure  (5 signals)
  [5–13]  Technical       (9 signals)
  [14–22] Quantitative    (9 signals)
  [23–26] Flow/regime     (4 signals: vrp, oi_price_div, price_accel, vol_skew)
  [27–30] CoinAlyze       (4 signals)
  [31–33] Market context  (3 signals: pm_yes_mid, time_sin, time_cos)
  [34–35] Fear & Greed    (2 signals: fg_score, fg_change)
  [36–37] Reddit BART     (2 signals: reddit_sentiment, reddit_change)
  [38–39] MarketMind      (2 signals: mm_direction, mm_confidence)
  [40–41] Deribit IV      (2 signals: deribit_iv_score, deribit_iv_change)
  [42–43] News BART       (2 signals: news_sentiment, news_change)
  [44–45] Futures basis   (2 signals: futures_basis, basis_change)
  [46–47] Liquidations    (2 signals: liq_net, liq_intensity)
  [48–49] Flow persist    (2 signals: cvd_slope, candle_run)
  [50]    Deep OBI        (1 signal:  obi_deep_50)
  [51–52] Barrier         (2 signals: barrier_z, barrier_prob)
  [53–54] Anchored VWAP   (2 signals: avwap_day_dev, avwap_liq_dev)
  [55]    Spot-PM lag      (1 signal:  spot_pm_lag)
  [56]    PM book OBI      (1 signal:  pm_obi)
  [57]    BTC lead-lag     (1 signal:  btc_lead — alts only, 0 for BTC)
"""

import math
import time
import numpy as np
from data.binance_ws import MarketState
from data.oracle_feed import OracleState
from data.polymarket_api import OrderBookState
from data.coinalyze_api import CoinAlyzeState
from data.fear_greed import FearGreedState
from data.reddit_sentiment import RedditSentimentState
from data.marketmind_api import MarketMindState
from data.deribit_iv import DeribitIVState
from data.news_sentiment import NewsSentimentState
from data.binance_spot import SpotState
from data.kalshi_api import KalshiState
from signals import microstructure, technical, quant, coinalyze_signals, barrier, flow_regime, avwap, lag, lead_lag
from signals.technical import stochastic_k
from utils.logger import get_logger
from config import SIGNAL_COUNT, LIQ_WINDOW_S, LIQ_SCALE_USD, OBI_LEVELS_VERY_DEEP, BASE_DIR

log = get_logger("aggregator")

# ── Per-signal usage tracking ────────────────────────────────────────────────
# Counts how often each signal is non-zero at decision time. Complements the
# model's importance dump: importance says which signals the model *uses*; usage
# says which are even *firing* (a frequently-zero signal = a dead/stale feed).
# Persisted to signal_usage.json for the dashboard / ad-hoc inspection.
_USAGE_PATH      = BASE_DIR / "signal_usage.json"
_USAGE_LOG_EVERY = 250
_usage_counts    = np.zeros(SIGNAL_COUNT, dtype=np.int64)
_usage_total     = 0


def _record_usage(arr: np.ndarray) -> None:
    global _usage_counts, _usage_total
    _usage_counts += (np.abs(arr) > 1e-6)
    _usage_total += 1
    if _usage_total % _USAGE_LOG_EVERY == 0:
        pct = _usage_counts / _usage_total
        try:
            import json
            with open(_USAGE_PATH, "w") as f:
                json.dump({"total": _usage_total,
                           "usage": {SIGNAL_NAMES[i]: round(float(pct[i]), 3)
                                     for i in range(SIGNAL_COUNT)}}, f, indent=2)
        except Exception:
            pass
        dead = int((pct < 0.01).sum())
        log.info(f"[USAGE] {_usage_total} computes | {SIGNAL_COUNT - dead} signals "
                 f"firing, {dead} effectively dead (<1%) — see signal_usage.json")

SIGNAL_NAMES = [
    # Microstructure
    "obi_shallow", "obi_deep", "flow_short", "flow_long", "spread_ratio",
    # Technical
    "rsi", "macd_hist", "ema_fast_slow", "ema_mid", "bb_pos", "bb_width",
    "vwap_dev", "mom_60s", "mom_15s",
    # Quantitative
    "rv_short", "rv_long", "vol_ratio", "price_z", "vol_z",
    "autocorr", "dir_ratio", "hurst", "regime",
    # Flow / regime (replaced dead oracle slot)
    "vrp", "oi_price_div", "price_accel", "vol_skew",
    # CoinAlyze
    "ca_funding_rate", "ca_oi_change", "ca_liq_imbalance", "stoch_k",
    # Market context
    "pm_yes_mid", "time_sin", "time_cos",
    # Fear & Greed
    "fg_score", "fg_change",
    # Reddit BART sentiment
    "reddit_sentiment", "reddit_change",
    # MarketMind website prediction
    "mm_direction", "mm_confidence",
    # Deribit implied volatility index
    "deribit_iv_score", "deribit_iv_change",
    # Crypto news BART sentiment
    "news_sentiment", "news_change",
    # Futures basis vs spot (contango / backwardation)
    "futures_basis", "basis_change",
    # Real-time BTC liquidation pressure (Binance forceOrder)
    "liq_net", "liq_intensity",
    # Order-flow trend + candle-run persistence (replaced dead Kalshi slot)
    "cvd_slope", "candle_run",
    # Deep order book imbalance (50 levels via futures REST)
    "obi_deep_50",
    # First-passage barrier: displacement from window-start reference price
    "barrier_z", "barrier_prob",
    # Anchored VWAP deviation: day-open magnet, last liquidation-cascade magnet
    "avwap_day_dev", "avwap_liq_dev",
    # Spot move not yet reflected in the Polymarket price (latency-arb lag)
    "spot_pm_lag",
    # Polymarket book depth imbalance (resting-order lean, 5 levels)
    "pm_obi",
    # BTC's recent z-scaled move — lead-lag info for the alt models (0 for BTC)
    "btc_lead",
]

assert len(SIGNAL_NAMES) == SIGNAL_COUNT, (
    f"SIGNAL_NAMES length {len(SIGNAL_NAMES)} != SIGNAL_COUNT {SIGNAL_COUNT}"
)


class SignalAggregator:

    def compute(
        self,
        market_state: MarketState,
        oracle_state: OracleState,
        ca_state: CoinAlyzeState | None = None,
        pm_book: OrderBookState | None = None,
        fg_state: FearGreedState | None = None,
        reddit_state: RedditSentimentState | None = None,
        mm_state: MarketMindState | None = None,
        interval: str = "5m",
        deribit_state: DeribitIVState | None = None,
        news_state: NewsSentimentState | None = None,
        spot_state: SpotState | None = None,
        kalshi_state: KalshiState | None = None,
        reference_price: float = 0.0,
        seconds_remaining: float = 0.0,
        barrier_price_override: float = 0.0,
        pm_obi: float = 0.0,
        btc_market_state: MarketState | None = None,
    ) -> np.ndarray:
        """
        Compute and return a normalised float32 array of shape (SIGNAL_COUNT,).
        All values are in [-1, 1].
        """
        try:
            micro  = list(microstructure.compute(market_state))
            tech   = technical.compute(market_state, interval=interval)
            quant_ = quant.compute(market_state)

            # For 15m markets, replace 30s/60s trade flow with 5min/15min windows.
            # Short-window flow is noise at a 15-minute horizon; sustained pressure matters.
            if interval == "15m":
                micro[2] = float(max(-1.0, min(1.0, market_state.trade_flow.imbalance(300))))
                micro[3] = float(max(-1.0, min(1.0, market_state.trade_flow.imbalance(900))))
            # Former oracle slot — Polygon RPCs unreliable; replaced with
            # derivatives/vol/micro regime signals computed from live feeds.
            oracle = flow_regime.compute_oracle_replacement(
                market_state, ca_state, deribit_state
            )
            ca     = coinalyze_signals.compute(ca_state) if ca_state else [0.0] * 4
            # Replace unavailable ca_long_short (index 3) with Stochastic %K
            klines_for_stoch = market_state.klines_5m if interval == "15m" else market_state.klines_1m
            ca[3] = stochastic_k(klines_for_stoch, period=14)

            # Polymarket consensus: centered implied prob [-1, 1]
            pm_yes_mid = (pm_book.yes_implied_prob() - 0.5) * 2.0 if pm_book else 0.0

            # Time-of-day (UTC) as cyclical features
            t         = time.gmtime()
            utc_frac  = (t.tm_hour + t.tm_min / 60.0) / 24.0
            time_sin  = math.sin(2 * math.pi * utc_frac)
            time_cos  = math.cos(2 * math.pi * utc_frac)

            context = [pm_yes_mid, time_sin, time_cos]

            # Fear & Greed signals — only use if the API has actually responded at least once
            fg_live = fg_state is not None and fg_state.last_updated > 0
            fg = [
                float(fg_state.score)  if fg_live else 0.0,
                float(fg_state.change) if fg_live else 0.0,
            ]

            # Reddit BART sentiment signals — only use after first successful scrape
            reddit_live = reddit_state is not None and reddit_state.last_updated > 0
            reddit = [
                float(reddit_state.score)  if reddit_live else 0.0,
                float(reddit_state.change) if reddit_live else 0.0,
            ]

            # MarketMind website prediction signals (timeframe-aware)
            mm_live = mm_state is not None and mm_state.last_updated > 0
            if mm_live:
                if interval == "15m":
                    mm = [mm_state.direction_15m, mm_state.confidence_15m]
                else:
                    mm = [mm_state.direction_5m, mm_state.confidence_5m]
            else:
                mm = [0.0, 0.0]

            # Deribit implied volatility signals — only use after first successful fetch
            dv_live = deribit_state is not None and deribit_state.last_updated > 0
            dv = [
                float(deribit_state.iv_score)  if dv_live else 0.0,
                float(deribit_state.iv_change) if dv_live else 0.0,
            ]

            # Crypto news BART sentiment signals — only use after first successful scrape
            news_live = news_state is not None and news_state.last_updated > 0
            news = [
                float(news_state.score)  if news_live else 0.0,
                float(news_state.change) if news_live else 0.0,
            ]

            # Futures basis vs spot (contango = positive, backwardation = negative)
            basis_live = spot_state is not None and spot_state.last_updated > 0
            basis = [
                float(spot_state.basis)        if basis_live else 0.0,
                float(spot_state.basis_change) if basis_live else 0.0,
            ]

            # Real-time liquidation pressure from Binance forceOrder stream
            buy_liq, sell_liq = market_state.liq_buffer.window(LIQ_WINDOW_S)
            total_liq = buy_liq + sell_liq + 1e-9
            liq_net       = float((buy_liq - sell_liq) / total_liq)   # [-1, 1]
            liq_intensity = float(min(1.0, math.log1p(total_liq) / math.log1p(LIQ_SCALE_USD)))
            liq = [liq_net, liq_intensity]

            # Former Kalshi slot — Kalshi API dead (domain moved, prices relocated
            # to orderbook endpoint). Replaced with order-flow trend + candle run.
            flow_persist = flow_regime.compute_kalshi_replacement(market_state, interval)

            # Deep order book imbalance at 50 levels (from REST snapshot)
            if spot_state and spot_state.deep_book.bids:
                ob50  = spot_state.deep_book
                bid50 = ob50.bid_volume(OBI_LEVELS_VERY_DEEP)
                ask50 = ob50.ask_volume(OBI_LEVELS_VERY_DEEP)
                tot50 = bid50 + ask50
                obi50 = float((bid50 - ask50) / tot50) if tot50 > 0 else 0.0
            else:
                obi50 = 0.0

            # First-passage barrier signal — displacement from the window-start
            # reference price, normalized by expected remaining-window volatility.
            barrier_sig = barrier.compute(
                market_state, reference_price, seconds_remaining,
                current_override=barrier_price_override,
            )

            # Anchored VWAP deviations. Use the Binance last price for consistency
            # with the (Binance) klines the AVWAP is built from; fall back to the
            # window reference, then the last candle close.
            cur_px = market_state.price.last_value() or reference_price
            if not cur_px and market_state.klines_1m:
                cur_px = market_state.klines_1m[-1]["close"]
            avwap_sig = avwap.compute(market_state, float(cur_px or 0.0))

            # Spot-vs-Polymarket lag: recent spot move PM hasn't yet priced in.
            lag_sig = lag.compute(market_state, pm_book)

            # Polymarket book depth imbalance (passed in — needs its own CLOB call)
            pm_obi_sig = [float(max(-1.0, min(1.0, pm_obi)))]

            # BTC's recent move as lead-lag info for the alt models (None for BTC)
            btc_lead_sig = lead_lag.compute(btc_market_state)

            vec = (
                micro + tech + quant_ + oracle + ca + context
                + fg + reddit + mm + dv + news
                + basis + liq + flow_persist + [obi50] + barrier_sig
                + avwap_sig + lag_sig + pm_obi_sig + btc_lead_sig
            )
            assert len(vec) == SIGNAL_COUNT, f"Signal count mismatch: {len(vec)}"

            arr = np.array(vec, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
            arr = np.clip(arr, -1.0, 1.0)

            nonzero = int((arr != 0).sum())
            if nonzero < SIGNAL_COUNT // 2:
                log.warning(
                    f"Only {nonzero}/{SIGNAL_COUNT} signals non-zero — "
                    f"klines_1m={len(market_state.klines_1m)} "
                    f"price_buf={len(market_state.price)} "
                    f"trade_flow_buf={len(market_state.trade_flow._times)}"
                )
            _record_usage(arr)
            return arr

        except Exception as e:
            log.error(f"Signal computation error: {e}", exc_info=True)
            return np.zeros(SIGNAL_COUNT, dtype=np.float32)

    def describe(self, vec: np.ndarray) -> dict[str, float]:
        """Human-readable dict of signal name → value."""
        return {name: round(float(v), 4) for name, v in zip(SIGNAL_NAMES, vec)}
