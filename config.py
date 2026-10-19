from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent

# ── CoinAlyze ─────────────────────────────────────────────────────────────────
COINALYZE_API_KEY = os.getenv("COINALYZE_API_KEY", "")

# ── Binance ────────────────────────────────────────────────────────────────────
BINANCE_STREAM_BASE  = "wss://fstream.binance.com/ws"
BINANCE_FAPI_WS      = "wss://ws-fapi.binance.com/ws-fapi/v1"
SYMBOL               = "BTCUSDT"
SYMBOL_LOWER         = SYMBOL.lower()

BINANCE_STREAMS = [
    f"{SYMBOL_LOWER}@aggTrade",
    f"{SYMBOL_LOWER}@depth20@100ms",
    f"{SYMBOL_LOWER}@kline_1m",
    f"{SYMBOL_LOWER}@kline_5m",
    f"{SYMBOL_LOWER}@bookTicker",
]

# ── Trading assets ─────────────────────────────────────────────────────────────
# Add entries here to trade additional Polymarket crypto up/down markets.
# Each asset gets its own Binance stream, ensemble, and paper log.
# "pm_slug_prefix" must match the Polymarket slug format: {prefix}-updown-{5m|15m}-{ts}
TRADING_ASSETS = [
    {"name": "btc", "symbol": "BTCUSDT", "pm_slug_prefix": "btc"},
    {"name": "eth", "symbol": "ETHUSDT", "pm_slug_prefix": "eth"},
    {"name": "sol", "symbol": "SOLUSDT", "pm_slug_prefix": "sol"},
    {"name": "xrp", "symbol": "XRPUSDT", "pm_slug_prefix": "xrp"},
]

# ── Polymarket ─────────────────────────────────────────────────────────────────
POLYMARKET_CLOB_URL   = "https://clob.polymarket.com"
POLYMARKET_GAMMA_URL  = "https://gamma-api.polymarket.com"
PM_API_KEY            = os.getenv("PM_API_KEY", "")
PM_API_SECRET         = os.getenv("PM_API_SECRET", "")
PM_API_PASSPHRASE     = os.getenv("PM_API_PASSPHRASE", "")
PM_PRIVATE_KEY        = os.getenv("PM_PRIVATE_KEY", "")
PM_CHAIN_ID           = 137  # Polygon mainnet

# ── Chainlink Oracle (Polygon) ─────────────────────────────────────────────────
POLYGON_RPC_URL              = os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com")
CHAINLINK_BTC_USD_POLYGON    = "0xc907E116054Ad103354f2D350FD2514433D57F6d"
CHAINLINK_AGGREGATOR_ABI     = [
    {
        "inputs": [],
        "name": "latestRoundData",
        "outputs": [
            {"name": "roundId",         "type": "uint80"},
            {"name": "answer",          "type": "int256"},
            {"name": "startedAt",       "type": "uint256"},
            {"name": "updatedAt",       "type": "uint256"},
            {"name": "answeredInRound", "type": "uint80"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "stateMutability": "view",
        "type": "function",
    },
]

# ── Chainlink Data Streams (settlement-truth price source) ──────────────────────
# Polymarket resolves its 5m/15m crypto up/down markets on Chainlink Data Streams
# (BTC/USD, ETH/USD) + Chainlink Automation — NOT on Binance. The bot currently
# labels WIN/LOSS and computes barrier_z off Binance futures mid, a subtly
# different price; that gap can flip the occasional settlement label and feed the
# model wrong ground truth. These let the bot settle and reference off the same
# source Polymarket actually resolves on.
#
# Data Streams uses HMAC auth (a separate key/secret from the Polymarket keys).
CHAINLINK_DS_REST_URL   = os.getenv("CHAINLINK_DS_REST_URL", "https://api.dataengine.chain.link")
CHAINLINK_DS_WS_URL     = os.getenv("CHAINLINK_DS_WS_URL",   "wss://ws.dataengine.chain.link")
CHAINLINK_DS_API_KEY    = os.getenv("CHAINLINK_DS_API_KEY", "")
CHAINLINK_DS_API_SECRET = os.getenv("CHAINLINK_DS_API_SECRET", "")

# Crypto (v3 schema) Data Streams feed IDs. VERIFY against the current Chainlink
# Data Streams feed list before going live — feed IDs are occasionally rotated.
CHAINLINK_DS_FEED_IDS = {
    "btc": os.getenv("CHAINLINK_DS_FEED_BTC", "0x00039d9e45394f473ab1f050a1b963e6b05351e52d71e507509ada0c95ed75b8"),
    "eth": os.getenv("CHAINLINK_DS_FEED_ETH", "0x000362205e10b3a147d02792eccee483dca6c7b44ecce7012cb8c6e0b68b3ae9"),
    # SOL/USD + XRP/USD RefPrice-DS-Premium-Global-003 — verified present on the
    # account and cross-checked against Chainlink's published feed IDs (2026-06-06).
    "sol": os.getenv("CHAINLINK_DS_FEED_SOL", "0x0003b778d3f6b2ac4991302b89cb313f99a42467d6c9c5f96f57c29c0d2bc24f"),
    "xrp": os.getenv("CHAINLINK_DS_FEED_XRP", "0x0003c16c6aed42294f5cb4741f6e59ba2d728f0eae2eb9e6d3f555808c59fc45"),
}

# Master switch — the client only starts when this is True AND creds are present.
CHAINLINK_STREAMS_ENABLED = os.getenv("CHAINLINK_STREAMS_ENABLED", "false").lower() == "true"

# Where the Chainlink settlement-truth prices come from:
#   "rtds"        — Polymarket's free Real-Time Data Socket re-broadcast of the
#                   Chainlink prices (no key, no auth; survives Chainlink's
#                   July 2026 paid-tier change). Default.
#   "datastreams" — direct Chainlink Data Streams (needs CHAINLINK_DS_API_KEY/
#                   SECRET; paid after July 2026 unless on a sponsored key).
CHAINLINK_PRICE_SOURCE  = os.getenv("CHAINLINK_PRICE_SOURCE", "rtds").lower()
POLYMARKET_RTDS_WS_URL  = os.getenv("POLYMARKET_RTDS_WS_URL", "wss://ws-live-data.polymarket.com")

# Max age of the last Chainlink observation before the price is treated as
# missing (settlement then falls back to Binance instead of using a frozen
# price). Added 2026-07-08 after RTDS outages froze cl prices at settlement
# ~15-20% of the time on 07-07/08, corrupting labels.
CL_PRICE_MAX_AGE_S = float(os.getenv("CL_PRICE_MAX_AGE_S", "30"))

# Phase gate for the settlement/barrier price source:
#   "binance"   — Phase 1: settle + barrier_z on Binance (unchanged), log Chainlink
#                 divergence to chainlink_shadow.csv. Measure before cutting over.
#   "chainlink" — Phase 2: settle + barrier_z on Chainlink Data Streams (the source
#                 Polymarket resolves on), log Binance as the shadow comparison.
CHAINLINK_SETTLEMENT_SOURCE = os.getenv("CHAINLINK_SETTLEMENT_SOURCE", "binance").lower()

# ── Signal Hyperparameters ─────────────────────────────────────────────────────
RSI_PERIOD             = 14
MACD_FAST              = 12
MACD_SLOW              = 26
MACD_SIGNAL_PERIOD     = 9
BB_PERIOD              = 20
BB_STD                 = 2.0
EMA_FAST               = 5
EMA_SLOW               = 13
EMA_MEDIUM_FAST        = 9
EMA_MEDIUM_SLOW        = 21
OBI_LEVELS_SHALLOW     = 5
OBI_LEVELS_DEEP        = 20
OBI_LEVELS_VERY_DEEP   = 50
TRADE_FLOW_WINDOW_S    = 30    # seconds for short-window flow
TRADE_FLOW_WINDOW_L    = 60    # seconds for long-window flow
VOL_WINDOW_SHORT       = 30    # seconds
VOL_WINDOW_LONG        = 300   # seconds (5 min)
ZSCORE_WINDOW          = 300   # seconds
AUTOCORR_LAGS          = 1
LIQ_WINDOW_S           = 60        # seconds of liquidation history to aggregate
LIQ_SCALE_USD          = 5_000_000 # $5M per window = max signal

# Total number of signals fed to the model
SIGNAL_COUNT = 58  # 44 + futures_basis(2) + liq(2) + flow_persist(2) + obi_deep_50(1) + barrier(2) + avwap(2) + lag(1) + pm_obi(1) + btc_lead(1)

# Spot-vs-Polymarket lag feature: recent spot move (z-scaled over this window)
# minus PM's current lean. Captures latency-arb (spot leads, PM lags).
LAG_WINDOW_S = float(os.getenv("LAG_WINDOW_S", "30"))
LAG_Z_SCALE  = float(os.getenv("LAG_Z_SCALE", "3.0"))

# BTC→altcoin lead-lag feature: BTC's recent z-scaled move, fed to the ETH/SOL/
# XRP models (BTC leads alts at minute horizons; BTC's own models get 0.0 —
# their mom_* signals already cover it).
BTC_LEAD_WINDOW_S = float(os.getenv("BTC_LEAD_WINDOW_S", "60"))

# Intra-market arbitrage logger: flag windows where buying BOTH sides costs less
# than the guaranteed $1 settlement payout. Log-only for now (no orders) — just
# measuring how often the riskless edge appears and how big it is.
ARB_SUM_THRESHOLD = float(os.getenv("ARB_SUM_THRESHOLD", "0.99"))

# ── Model ──────────────────────────────────────────────────────────────────────
REPLAY_BUFFER_SIZE = 2000
RETRAIN_INTERVAL   = 25        # retrain every N new labeled outcomes
MIN_TRAIN_SAMPLES  = 50        # separate 5m/15m models — each sees fewer trades
MODEL_WEIGHTS_PATH = BASE_DIR / "model" / "weights" / "ensemble.weights.h5"

# ── 5m feature pruning ──────────────────────────────────────────────────────────
# Walk-forward A/B on the live buffers showed the 5m model's 50 non-price signals
# are net-harmful: they drag a strong market-price signal (OOF AUC 0.79) down to
# 0.74. Restricting the 5m models to price + barrier + futures-basis recovers most
# of the gap (AUC 0.73->0.76), lowers Brier, and yields MORE high-conviction bets
# at equal win rate. The ensemble zeroes every other column, so the input vector
# shape is unchanged. 15m is left on the full set — it's barrier-driven and already
# the profitable interval. Set to None to disable pruning.
SLIM_5M_FEATURES = ["pm_yes_mid", "barrier_z", "futures_basis"]

# ── Execution ──────────────────────────────────────────────────────────────────
ENTRY_WINDOW_START_S  = 185    # seconds before close: start watching
ENTRY_WINDOW_EXEC_S   = 180    # seconds before close: execute (T-3min)
ENTRY_HARD_ABORT_S    = 170    # never enter after this

# Shadow early-entry probe (15m only): logs what the model WOULD decide at points
# EARLIER than the real T-180s entry — to test whether earlier entry surfaces more
# bettable prices AND profitable calls (the market is less "decided" earlier).
# Pure measurement: no capital, no training. Writes shadow_early_entry.csv with a
# "kind" column (T600 / T330 / dyn). Windows are 900s, so 600s≈33%, 330s≈63% in.
SHADOW_EARLY_ENTRY_ENABLED = os.getenv("SHADOW_EARLY_ENTRY_ENABLED", "true").lower() in ("1","true","yes")
SHADOW_ENTRY_TIMES = [float(x) for x in os.getenv("SHADOW_ENTRY_TIMES", "600,330").split(",") if x.strip()]

# Dynamic shadow: instead of a fixed time, poll every INTERVAL seconds from
# START down to END and log the FIRST moment the model's decision would clear all
# gates — catching a transient dip into the bettable range, even AFTER the normal
# T-180s entry (END < 180 keeps watching past it). One "dyn" row per market.
SHADOW_DYNAMIC_ENABLED    = os.getenv("SHADOW_DYNAMIC_ENABLED", "true").lower() in ("1","true","yes")
SHADOW_DYNAMIC_INTERVAL_S = float(os.getenv("SHADOW_DYNAMIC_INTERVAL_S", "45"))
SHADOW_DYNAMIC_START_S    = float(os.getenv("SHADOW_DYNAMIC_START_S", "600"))
SHADOW_DYNAMIC_END_S      = float(os.getenv("SHADOW_DYNAMIC_END_S", "30"))

# No-conviction shadow (kind="noconv"): at the real T-180s entry, log what the
# decision would be with the low-conviction gate REMOVED — to test whether the
# bets currently cancelled for low model confidence would actually have won.
SHADOW_NOCONV_ENABLED     = os.getenv("SHADOW_NOCONV_ENABLED", "true").lower() in ("1","true","yes")

EV_THRESHOLD_5M       = 0.05   # 5% minimum expected value for 5m markets
EV_THRESHOLD_15M      = float(os.getenv("EV_THRESHOLD_15M", "0.50"))
                               # Raised 0.30→0.50 (2026-07-17) on clean clob-only
                               # data: the 0.27-0.50 EV bucket LOSES money
                               # (-0.04/$1, n=195) while 0.50+ is positive
                               # (+0.06 to +5.8/$1). Cuts ~half the trades — the
                               # half that collectively bled.
POLYMARKET_FEE        = 0.02   # 2% fee on winnings (deducted from profit, not stake)

MAX_POSITION_USDC     = float(os.getenv("MAX_POSITION_USDC", "500"))
MIN_POSITION_USDC     = 1.0
KELLY_FRACTION        = 0.35   # fractional Kelly for position sizing
KELLY_MAX_PCT         = 0.15   # max bet as fraction of current bankroll (auto-scaling)
MAX_OPEN_EXPOSURE_PCT = float(os.getenv("MAX_OPEN_EXPOSURE_PCT", "1.0"))
                               # total capital across all simultaneously-open bets
                               # can't exceed this × bankroll. 1.0 = can't risk more
                               # than the bankroll at once (real-account constraint);
                               # lower it to keep a cash buffer.
STARTING_BANKROLL     = float(os.getenv("STARTING_BANKROLL", "500"))
BANKROLL_STATE_PATH   = BASE_DIR / "bankroll.json"

# ── Confluence gate ────────────────────────────────────────────────────────────
# Minimum number of agreeing sources (model counts as 1) required to place a bet.
# The gate only activates when 2+ sources had a reading. Set to 1 to disable it
# (the model's vote alone always satisfies the requirement).
CONFLUENCE_MIN_AGREE = 1

# ── Pre-trade filters (order-only; decisions are always logged for training) ─────
# All default OFF for the accuracy-first phase — the conviction floor is the primary
# quality gate, and in paper mode there's no real fill risk, so these only reduce the
# data we collect. They block ONLY the order, never training. Re-enable for live.
SKIP_LOW_LIQUIDITY_HOURS   = False  # block 2-6am ET (thin liquidity / live-fill risk)
# Comma-separated UTC hours to block orders in (e.g. "12,13,14,15,16,17").
# Clean-data finding 2026-07-17: 12-18 UTC ran breakeven (-0.03/$1, n=95) vs
# 00-06 UTC at +1.01/$1 — but one month of hour-slices is thin; OFF by default,
# re-check when the post-reset sample doubles before enabling.
SKIP_HOURS_UTC = {int(h) for h in os.getenv("SKIP_HOURS_UTC", "").split(",") if h.strip().isdigit()}
SKIP_ON_IV_SPIKE           = False  # block when Deribit DVOL spikes (iv_score > 0.6)
SKIP_ON_MODEL_DISAGREEMENT = False  # block when 5m/15m models strongly disagree

# ── Model conviction floor ───────────────────────────────────────────────────────
# Minimum model probability (for the chosen direction) required to place a bet.
# History: 0.70→0.50 (2026-08-16, on noconv-shadow), then →0.60 (2026-08-28) after
# 12 days of REAL placed data settled the question the shadow's tiny sample couldn't:
#   0.60-0.70 conv → 60% wr, +0.95/$1, +$10k (n=25)  ← keep
#   0.50-0.60 conv → 19% wr, -0.32/$1, -$2.2k (n=16) ← cut (shadow's n=7 @57% was noise)
# So 0.60 is the real quality edge. Env-tunable.
MODEL_CONVICTION_FLOOR = float(os.getenv("MODEL_CONVICTION_FLOOR", "0.60"))

# 5m needs a higher bar: backtest showed the 0.70-0.80 band is a coin flip (model
# overconfident there, 48% WR, net loser), while 0.80+ clears 74% WR and flips 5m
# positive. 15m keeps the 0.70 floor where its edge already lives.
MODEL_CONVICTION_FLOOR_5M = 0.80

# Block 5m bets on the cheap (<0.5) side. These are against-the-model longshots that
# EV picks for their fat payout but lose ~27% of the time — a persistent 5m bleeder.
BLOCK_5M_CONTRARIAN = True

# 15m entry-price ceiling. Bets at/above this lose money on two independent
# ~100-trade clean samples (-0.07 and -0.089 per $1): favorites where the market
# already priced the move. Added 2026-08-07; blocks ~20% of 15m volume (the
# negative fifth). 1.0 disables. Reversible via env.
MAX_ENTRY_PRICE_15M = float(os.getenv("MAX_ENTRY_PRICE_15M", "0.60"))

# 15m toss-up band block. Prices in [lo, hi) are near coin-flips where the model
# has NO directional edge (every directional feature pruned to 0; only barrier_z
# — a lagging read — survives). Backtest 2026-08-07: cutting 0.40-0.55 barely
# dents return (-8%) but nearly halves max drawdown (24%->15%); concentrates
# capital on the contrarian bets that are +105% of P&L. Set lo>=hi to disable.
BLOCK_15M_TOSSUP_LO = float(os.getenv("BLOCK_15M_TOSSUP_LO", "0.40"))
BLOCK_15M_TOSSUP_HI = float(os.getenv("BLOCK_15M_TOSSUP_HI", "0.55"))

# Master switch for placing 5m orders. Measured -EV on both sides (favorites
# -3%/-9% realized P&L over 400+ trades; underdogs worse), so capital is parked
# in 15m to compound. When False, 5m decisions are still logged (model keeps
# training) but no 5m order is placed and no 5m capital/exposure is committed.
TRADE_5M = os.getenv("TRADE_5M", "false").lower() in ("1", "true", "yes")

# Correlation-aware sizing. BTC/ETH/SOL/XRP up/down bets in the same window are
# effectively one macro bet, so full Kelly over-bets when several lean the same
# way. A new bet is scaled by 1/(1 + rho*n) where n = open same-direction bets.
# rho=0 disables the haircut; rho=1 would split a single bet n ways. 0.5 reflects
# strong-but-imperfect intraday crypto correlation. Reversible via env.
CORR_HAIRCUT_RHO = float(os.getenv("CORR_HAIRCUT_RHO", "0.5"))

# Early exit for 15m: once an open position's own side can be sold at/above this
# price on the Polymarket book, close it before settlement. Locks in most of the
# win and cuts the variance of holding a near-certain bet to a 0/1 resolution —
# the peak→trough bankroll swings the user wants smoothed. The model still trains
# on the true settlement outcome; only the bankroll P&L is realized at exit.
# Only 15m (the profit engine). Reversible via env.
EARLY_EXIT_ENABLED  = os.getenv("EARLY_EXIT_ENABLED", "true").lower() in ("1", "true", "yes")
EARLY_EXIT_PRICE    = float(os.getenv("EARLY_EXIT_PRICE", "0.92"))
                      # Raised 0.85→0.92 (2026-07-17): on clean data, exiting at
                      # ~0.87 avg net-cost $5.1k of $35.9k (14%) vs holding —
                      # 119/128 exits would have won anyway, and the 9 saved
                      # losses didn't cover the give-up. 0.92 keeps the tail
                      # de-risk while surrendering ~half as much per win.
EARLY_EXIT_INTERVAL = "15m"

# Maker/limit orders. When live (not DRY_RUN) and enabled, place a resting GTC
# limit BUY that improves the bid by one tick instead of a taker market order —
# earns the maker side of the spread (better effective price, no taker fee) and
# avoids crossing a wide book. Trade-off: a limit can go unfilled if the book
# runs away in a fast window, so the order is posted at T-3min (ENTRY_WINDOW_EXEC_S)
# to leave fill time. Dormant under DRY_RUN. Reversible via env.
USE_MAKER_ORDERS = os.getenv("USE_MAKER_ORDERS", "false").lower() in ("1", "true", "yes")
MAKER_TICK       = float(os.getenv("MAKER_TICK", "0.01"))  # price improvement step

# Require a REAL CLOB order book to place. Measured 2026-06-15..17: ~half of 15m
# trades (82% of the 0.50 bucket) fell back to the frozen Gamma seed because the
# CLOB book empties out by our T-180s entry — and those gamma "fills" won 100%
# (84/84), i.e. they're fictional (no liquidity to fill against), inflating the
# paper bankroll by ~53%. When True, decisions on a gamma-only book are still
# LOGGED for training but no order/capital is committed. The real, fillable edge
# is the clob trades (68% win). Essential before any live test. Reversible via env.
REQUIRE_CLOB_BOOK = os.getenv("REQUIRE_CLOB_BOOK", "true").lower() in ("1", "true", "yes")

# ── Circuit breaker ────────────────────────────────────────────────────────────
# Disabled (0.0) for the accuracy-first phase: pausing on a losing streak blocks the
# high-conviction bets whose win rate we're trying to measure. Set back to ~0.40 to
# re-enable capital protection once accuracy is validated.
STREAK_PAUSE_WR      = 0.0    # pause betting when recent win rate drops below this (0 = never)
STREAK_RESUME_WR     = 0.50   # resume only after win rate recovers above this
STREAK_WINDOW        = 10     # number of recent active trades to evaluate

# ── Live test mode ───────────────────────────────────────────────────────────────
# A tightly-capped real-money probe to answer "is the 91%-at-0.50 edge real or
# look-ahead?" Only meaningful when DRY_RUN=false. Independent of the paper engine:
# it bypasses the (paper-bankroll-anchored) Kelly sizing and uses a flat tiny stake,
# restricts to ONE asset + 15m, and stops after a hard order/budget cap. All env-
# tunable; OFF by default. The DRY_RUN=true switch remains the instant kill.
LIVE_TEST_MODE        = os.getenv("LIVE_TEST_MODE", "false").lower() in ("1", "true", "yes")
# Defaults are the conservative SMOKE TEST (5 orders / $10). Bump via env once a
# first run looks clean, e.g. LIVE_TEST_MAX_ORDERS=25 LIVE_TEST_BUDGET_USDC=50.
LIVE_TEST_STAKE_USDC  = float(os.getenv("LIVE_TEST_STAKE_USDC", "2"))    # flat per-order stake
LIVE_MAX_ORDER_USDC   = float(os.getenv("LIVE_MAX_ORDER_USDC", "5"))     # hard reject above this (any live order)
LIVE_TEST_BUDGET_USDC = float(os.getenv("LIVE_TEST_BUDGET_USDC", "10"))  # stop once this much is deployed
LIVE_TEST_MAX_ORDERS  = int(os.getenv("LIVE_TEST_MAX_ORDERS", "5"))      # stop after this many orders
LIVE_TEST_ASSET       = os.getenv("LIVE_TEST_ASSET", "btc").lower()      # only trade this asset
LIVE_TEST_INTERVAL    = "15m"                                            # 15m only (the engine)
LIVE_TEST_STATE_PATH  = BASE_DIR / "live_test_status.json"               # bot→dashboard status file

# ── Core portfolio (portfolio/) ────────────────────────────────────────────────
# Rules-based core holdings that the trading bot (the "satellite") sweeps excess
# bankroll into. Paper-only: there is no broker connection; see PORTFOLIO_PLAN.md.
# Targets fall back to the tracked example file if your own copy doesn't exist.
PORTFOLIO_TARGETS_PATH  = BASE_DIR / "portfolio_targets.json"
PORTFOLIO_EXAMPLE_PATH  = BASE_DIR / "portfolio_targets.example.json"
PORTFOLIO_STATE_PATH    = BASE_DIR / "portfolio_state.json"
SATELLITE_CAP_USD       = float(os.getenv("SATELLITE_CAP_USD", "1000"))
                               # bot bankroll above this is eligible to sweep to core
SATELLITE_SWEEP_MIN_USD = float(os.getenv("SATELLITE_SWEEP_MIN_USD", "50"))
                               # don't bother sweeping less than this

# ── System ─────────────────────────────────────────────────────────────────────
DRY_RUN   = os.getenv("DRY_RUN", "true").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
