"""
Backtester for the Polymarket BTC signal ensemble.

Downloads 30 days of 1-minute Binance candles, simulates T-10s signal
snapshots at every 5m and 15m window boundary, trains the model on the
first 70% of windows, and evaluates on the remaining 30%.

Signals available in backtest (no historical oracle/Polymarket/CoinAlyze):
  - All technical signals    (RSI, MACD, EMA, BB, VWAP, momentum)
  - All quantitative signals (vol, z-score, autocorr, Hurst, regime)
  - Microstructure proxied from candle data (limited — no live order book)
  - Oracle/CoinAlyze set to 0 (neutral) for the backtest

Run:
  python backtest.py
  python backtest.py --days 60 --interval 5m
  python backtest.py --days 30 --interval 15m --plot
"""

import argparse
import time
import requests
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime, timezone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, classification_report,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# ── Argument parsing ───────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Backtest the signal ensemble.")
parser.add_argument("--days",     type=int,   default=30,   help="Days of history to fetch")
parser.add_argument("--interval", type=str,   default="5m", choices=["5m", "15m"])
# Note: Polymarket has 5m and 15m BTC up/down markets (not 10m)
parser.add_argument("--plot",     action="store_true",       help="Show equity curve plot")
parser.add_argument("--ev-threshold", type=float, default=0.04)
args = parser.parse_args()

INTERVAL_MINUTES = 5 if args.interval == "5m" else 15
print(f"\n{'='*60}")
print(f"Backtest: BTC {args.interval} windows | {args.days} days | EV≥{args.ev_threshold:.0%}")
print(f"{'='*60}\n")

# ── 1. Download historical 1m candles from Binance ─────────────────────────────

def _fetch_from_binance(base: str, symbol: str, interval: str,
                        start_ms: int, end_ms: int) -> list:
    all_rows = []
    while start_ms < end_ms:
        resp = requests.get(base, params={
            "symbol":    symbol,
            "interval":  interval,
            "startTime": start_ms,
            "endTime":   end_ms,
            "limit":     1000,
        }, timeout=15)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            break
        all_rows.extend(rows)
        start_ms = rows[-1][0] + 1
        time.sleep(0.15)
    return all_rows


def _fetch_from_bybit(symbol: str, interval: str,
                      start_ms: int, end_ms: int) -> list:
    """Bybit V5 kline — returns same column layout as Binance."""
    bybit_interval = {"1m": "1", "5m": "5", "15m": "15"}[interval]
    base     = "https://api.bybit.com/v5/market/kline"
    all_rows = []
    cursor_end = end_ms

    while cursor_end > start_ms:
        resp = requests.get(base, params={
            "category": "linear",
            "symbol":   symbol,
            "interval": bybit_interval,
            "start":    start_ms,
            "end":      cursor_end,
            "limit":    1000,
        }, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        rows = data.get("result", {}).get("list", [])
        if not rows:
            break
        # Bybit returns [startTime, open, high, low, close, volume, turnover]
        # newest first — reverse to oldest-first
        rows = list(reversed(rows))
        for r in rows:
            ts = int(r[0])
            if ts < start_ms:
                continue
            # Map to Binance format: [open_time, open, high, low, close, volume, ...]
            all_rows.append([
                ts,           # open_time
                r[1],         # open
                r[2],         # high
                r[3],         # low
                r[4],         # close
                r[5],         # volume
                ts + 59999,   # close_time (approx)
                r[6],         # quote vol (turnover)
                0,            # trades (unavailable)
                str(float(r[5]) * 0.5),  # taker_buy_base (approx 50%)
                str(float(r[6]) * 0.5),  # taker_buy_quote
                "0",
            ])
        cursor_end = int(rows[0][0]) - 1
        time.sleep(0.15)

    all_rows.sort(key=lambda r: r[0])
    return all_rows


def fetch_klines(symbol: str, interval: str, days: int) -> pd.DataFrame:
    end_ms   = int(time.time() * 1000)
    start_ms = end_ms - days * 86_400_000

    sources = [
        ("Binance US",  lambda: _fetch_from_binance("https://api.binance.us/api/v3/klines", symbol, interval, start_ms, end_ms)),
        ("Binance",     lambda: _fetch_from_binance("https://api.binance.com/api/v3/klines", symbol, interval, start_ms, end_ms)),
        ("Bybit",       lambda: _fetch_from_bybit(symbol, interval, start_ms, end_ms)),
    ]

    all_rows = []
    for name, fn in sources:
        try:
            print(f"Downloading {days}d of {interval} candles from {name}...")
            all_rows = fn()
            if all_rows:
                print(f"  Success via {name} ({len(all_rows)} candles)")
                break
        except Exception as e:
            print(f"  {name} failed: {e} — trying next source...")

    if not all_rows:
        raise RuntimeError("All data sources failed. Check your internet connection.")

    df = pd.DataFrame(all_rows, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_vol","trades","taker_buy_base",
        "taker_buy_quote","ignore"
    ])
    for col in ["open","high","low","close","volume","taker_buy_base","taker_buy_quote"]:
        df[col] = df[col].astype(float)
    df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.sort_values("open_time").reset_index(drop=True)
    print(f"  Downloaded {len(df)} candles ({df['open_time'].iloc[0].date()} → {df['open_time'].iloc[-1].date()})")
    return df


# Fetch 1m candles for signal computation, window candles for outcomes
df_1m  = fetch_klines("BTCUSDT", "1m",           args.days)
df_win = fetch_klines("BTCUSDT", args.interval,   args.days)

# ── 2. Signal computation from candle history ──────────────────────────────────

def ema(series: np.ndarray, period: int) -> float:
    if len(series) < period:
        return float(series[-1]) if len(series) > 0 else 0.0
    k, e = 2.0 / (period + 1), series[0]
    for p in series[1:]:
        e = p * k + e * (1 - k)
    return e

def rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    d = np.diff(closes[-(period+1):])
    g = d[d > 0].mean() if (d > 0).any() else 1e-10
    l = -d[d < 0].mean() if (d < 0).any() else 1e-10
    return 100.0 - 100.0 / (1.0 + g / l)

def clip(v, lo=-1.0, hi=1.0):
    return max(lo, min(hi, v))

def compute_signals(closes: np.ndarray, volumes: np.ndarray,
                    buy_vol: np.ndarray, sell_vol: np.ndarray) -> np.ndarray:
    """
    Compute 31-element signal vector from candle history.
    Oracle (4), CoinAlyze (4), and live microstructure (5) are set to 0.
    Technical (9) and Quant (9) are fully computed.
    """
    if len(closes) < 30:
        return np.zeros(31, dtype=np.float32)

    c  = closes
    rv = volumes
    bv = buy_vol
    sv = sell_vol

    # ── Microstructure proxies (from candle taker volume) ─────────────────────
    # OBI proxy: taker buy vs sell volume ratio over last 5 candles
    bv5   = bv[-5:].sum()
    sv5   = sv[-5:].sum()
    tot5  = bv5 + sv5
    obi_s = (bv5 - sv5) / tot5 if tot5 > 0 else 0.0

    bv20  = bv[-20:].sum()
    sv20  = sv[-20:].sum()
    tot20 = bv20 + sv20
    obi_d = (bv20 - sv20) / tot20 if tot20 > 0 else 0.0

    flow_s = clip(obi_s)
    flow_l = clip(obi_d)
    spread = 0.0   # no live spread in backtest

    micro = [obi_s, obi_d, flow_s, flow_l, spread]

    # ── Technical ─────────────────────────────────────────────────────────────
    rsi_v  = clip((rsi(c) - 50.0) / 50.0)

    # MACD histogram proxy
    fast_e = ema(c, 12); slow_e = ema(c, 26)
    macd_v = fast_e - slow_e
    std20  = c[-20:].std() if len(c) >= 20 else 1.0
    macd_h = clip(macd_v / (std20 + 1e-10))

    ema5   = ema(c, 5);  ema13 = ema(c, 13)
    ema9   = ema(c, 9);  ema21 = ema(c, 21)
    cross1 = clip((ema5  - ema13) / (ema13 + 1e-10) * 200)
    cross2 = clip((ema9  - ema21) / (ema21 + 1e-10) * 200)

    bb_w   = c[-20:] if len(c) >= 20 else c
    bb_mid = bb_w.mean(); bb_std = bb_w.std()
    bb_up  = bb_mid + 2 * bb_std; bb_lo = bb_mid - 2 * bb_std
    bb_pos = clip(((c[-1] - bb_lo) / (bb_up - bb_lo + 1e-10) - 0.5) * 2)
    bb_wid = clip((bb_up - bb_lo) / (bb_mid + 1e-10) * 50 - 1)

    # VWAP deviation (session = last 300 candles)
    w = min(300, len(c))
    pv_sum = (c[-w:] * rv[-w:]).sum(); v_sum = rv[-w:].sum()
    vwap   = pv_sum / v_sum if v_sum > 0 else c[-1]
    vwap_d = clip((c[-1] - vwap) / vwap * 500)

    # Momentum
    mom60 = clip((c[-1] - c[-60]) / (c[-60] + 1e-10) * 200) if len(c) >= 60 else 0.0
    mom15 = clip((c[-1] - c[-15]) / (c[-15] + 1e-10) * 2000) if len(c) >= 15 else 0.0

    tech = [rsi_v, macd_h, cross1, cross2, bb_pos, bb_wid, vwap_d, mom60, mom15]

    # ── Quantitative ──────────────────────────────────────────────────────────
    def realized_vol(prices, window):
        if len(prices) < window + 1:
            return 0.0
        rets = np.diff(np.log(prices[-window:] + 1e-10))
        return rets.std() * np.sqrt(86400 / 60)  # daily vol proxy (1m candles)

    rv_s   = clip(realized_vol(c, 30) * 5.0 - 1.0)
    rv_l   = clip(realized_vol(c, 300) * 5.0 - 1.0)
    rvs_n  = realized_vol(c, 30); rvl_n = realized_vol(c, 300)
    vr     = clip((rvs_n / (rvl_n + 1e-12) - 1.0) * 2.0)

    mu_z   = c[-300:].mean() if len(c) >= 300 else c.mean()
    std_z  = c[-300:].std()  if len(c) >= 300 else c.std()
    pz     = clip((c[-1] - mu_z) / (std_z + 1e-10) / 3.0)

    # Volume z-score
    v_mu   = rv[-300:].mean() if len(rv) >= 300 else rv.mean()
    v_std  = rv[-300:].std()  if len(rv) >= 300 else rv.std()
    vz     = clip((rv[-30:].mean() - v_mu) / (v_std + 1e-10) / 3.0)

    # Autocorrelation
    rets   = np.diff(np.log(c[-50:] + 1e-10)) if len(c) >= 50 else np.array([0.0])
    if len(rets) >= 4:
        try:
            from scipy.stats import pearsonr
            ac, _ = pearsonr(rets[:-1], rets[1:])
            ac = clip(float(ac))
        except Exception:
            ac = 0.0
    else:
        ac = 0.0

    # Directional ratio
    p_w    = c[-300:] if len(c) >= 300 else c
    net_d  = p_w[-1] - p_w[0]
    tot_d  = np.abs(np.diff(p_w)).sum()
    dr     = clip(net_d / (tot_d + 1e-10))

    # Hurst proxy
    if len(c) >= 20:
        lp  = np.log(c[-20:] + 1e-10)
        vf  = np.var(np.diff(lp))
        vh  = np.var(np.diff(lp[:10]))
        r   = vf / (2.0 * vh + 1e-20)
        hp  = clip((np.log2(r + 1e-10) / 2.0 + 0.5 - 0.5) * 4.0)
    else:
        hp = 0.0

    regime = clip((dr + ac) / 2.0)

    quant = [rv_s, rv_l, vr, pz, vz, ac, dr, hp, regime]

    # ── Oracle + CoinAlyze → 0 (not available historically) ──────────────────
    oracle    = [0.0, 0.0, 0.0, 0.0]
    coinalyze = [0.0, 0.0, 0.0, 0.0]

    vec = micro + tech + quant + oracle + coinalyze
    arr = np.array(vec, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(arr, -1.0, 1.0)


# ── 3. Build dataset: one row per window ───────────────────────────────────────

print(f"\nBuilding signal snapshots for {len(df_win)} {args.interval} windows...")

closes_1m  = df_1m["close"].values
opens_1m   = df_1m["open"].values
vols_1m    = df_1m["volume"].values
buy_vol_1m = df_1m["taker_buy_base"].values
sell_vol_1m = vols_1m - buy_vol_1m
times_1m   = df_1m["open_time"].values   # numpy datetime64

X_list, y_list, meta = [], [], []

# Need enough 1m candle history before first window
WARMUP_CANDLES = 350

for i, row in df_win.iterrows():
    if i < 2:
        continue  # need next candle for outcome

    # Outcome: did BTC close higher than it opened this window?
    outcome = 1 if float(row["close"]) > float(row["open"]) else 0

    # Find 1m candles available at T-10s (just before window close)
    window_close = row["close_time"]
    mask = times_1m < window_close.to_datetime64()
    idx  = np.where(mask)[0]

    if len(idx) < WARMUP_CANDLES:
        continue

    # Use candles up to T-10s snapshot
    end_idx = idx[-1] + 1
    c   = closes_1m[max(0, end_idx - 500):end_idx]
    v   = vols_1m[max(0, end_idx - 500):end_idx]
    bv  = buy_vol_1m[max(0, end_idx - 500):end_idx]
    sv  = sell_vol_1m[max(0, end_idx - 500):end_idx]

    sig = compute_signals(c, v, bv, sv)
    X_list.append(sig)
    y_list.append(outcome)
    meta.append({
        "time":  row["open_time"],
        "open":  float(row["open"]),
        "close": float(row["close"]),
    })

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int32)
print(f"  {len(X)} usable windows | base rate UP={y.mean():.2%}")

# ── 4. Train/test split (70/30, time-ordered) ──────────────────────────────────

split = int(len(X) * 0.70)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
meta_test = meta[split:]

print(f"  Train: {len(X_train)} windows | Test: {len(X_test)} windows")

# ── 5. Train model ─────────────────────────────────────────────────────────────

print("\nTraining MLP ensemble on training set...")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    learning_rate="adaptive",
    learning_rate_init=1e-3,
    max_iter=200,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=15,
    verbose=False,
)
model.fit(X_train_s, y_train)
print("  Training complete.")

# ── 6. Evaluate ────────────────────────────────────────────────────────────────

probs  = model.predict_proba(X_test_s)[:, 1]
preds  = (probs >= 0.5).astype(int)

acc    = accuracy_score(y_test, preds)
prec   = precision_score(y_test, preds, zero_division=0)
rec    = recall_score(y_test, preds, zero_division=0)
auc    = roc_auc_score(y_test, probs)

print(f"\n{'─'*50}")
print(f"  OVERALL ACCURACY   : {acc:.2%}")
print(f"  PRECISION (UP)     : {prec:.2%}")
print(f"  RECALL (UP)        : {rec:.2%}")
print(f"  ROC-AUC            : {auc:.4f}")
print(f"{'─'*50}")
print("\nDetailed report:")
print(classification_report(y_test, preds, target_names=["DOWN","UP"]))

# ── 7. EV-filtered backtest ────────────────────────────────────────────────────

# Simulate: only trade when model confidence exceeds EV threshold
# Assume Polymarket fair price = 0.50 (50/50 market as baseline)
# EV = P(win) * 0.5 - P(lose) * 0.5  → trade if |prob - 0.5| > threshold

print(f"{'─'*50}")
print(f"EV-filtered simulation (threshold={args.ev_threshold:.0%}):")

trades      = []
correct     = 0
total_ev    = 0.0
stake       = 1.0  # normalised stake per trade
bankroll    = [100.0]
current     = 100.0

for i, (prob, actual) in enumerate(zip(probs, y_test)):
    edge = abs(prob - 0.5)
    if edge < args.ev_threshold:
        bankroll.append(current)
        continue

    direction_up = prob >= 0.5
    won = (direction_up and actual == 1) or (not direction_up and actual == 0)

    # Simple EV: pay 0.5, win 0.5 (50c market)
    payout = stake * 0.9 if won else -stake  # 10% vig
    current += payout
    bankroll.append(current)

    ev_this = (prob * 0.45) - ((1 - prob) * 0.5) if direction_up else \
              ((1 - prob) * 0.45) - (prob * 0.5)
    total_ev += ev_this

    trades.append({
        "won": won,
        "prob": prob,
        "actual": actual,
        "ev": ev_this,
    })
    correct += int(won)

n_trades = len(trades)
if n_trades > 0:
    win_rate   = correct / n_trades
    avg_ev     = total_ev / n_trades
    final_bank = current
    roi        = (final_bank - 100.0) / 100.0

    print(f"  Trades taken       : {n_trades} / {len(probs)} ({n_trades/len(probs):.1%} of windows)")
    print(f"  Win rate           : {win_rate:.2%}")
    print(f"  Avg EV per trade   : {avg_ev:.4f}")
    print(f"  Final bankroll     : ${final_bank:.2f}  (started $100)")
    print(f"  ROI                : {roi:+.2%}")
else:
    print(f"  No trades passed EV threshold of {args.ev_threshold:.0%}")

# ── 8. Signal importance ───────────────────────────────────────────────────────

SIGNAL_NAMES = [
    "obi_shallow","obi_deep","flow_short","flow_long","spread_ratio",
    "rsi","macd_hist","ema_fast_slow","ema_mid","bb_pos","bb_width",
    "vwap_dev","mom_60s","mom_15s",
    "rv_short","rv_long","vol_ratio","price_z","vol_z",
    "autocorr","dir_ratio","hurst","regime",
    "oracle_vs_spot","oracle_lag","oracle_mom","exchange_div",
    "ca_funding_rate","ca_oi_change","ca_liq_imbalance","ca_long_short",
]

w          = model.coefs_[0]
importance = np.linalg.norm(w, axis=1)
importance /= importance.sum()
ranked     = sorted(zip(SIGNAL_NAMES, importance), key=lambda x: -x[1])

print(f"\n{'─'*50}")
print("Top 10 signal importances (from first-layer weights):")
for name, score in ranked[:10]:
    bar = "█" * int(score * 200)
    print(f"  {name:<25} {score:.4f}  {bar}")

print(f"\n{'─'*50}")
print("NOTE: Oracle and CoinAlyze signals were zeroed in backtest.")
print("Live performance will differ as those signals activate.")
print(f"{'─'*50}\n")

# ── 9. Optional plot ───────────────────────────────────────────────────────────

if args.plot:
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        axes[0].plot(bankroll, linewidth=1)
        axes[0].axhline(100, color="gray", linestyle="--", linewidth=0.8)
        axes[0].set_title(f"Equity Curve — BTC {args.interval} | EV≥{args.ev_threshold:.0%}")
        axes[0].set_ylabel("Bankroll ($)")
        axes[0].set_xlabel("Window index")
        axes[0].grid(True, alpha=0.3)

        names  = [n for n, _ in ranked[:15]]
        scores = [s for _, s in ranked[:15]]
        axes[1].barh(names[::-1], scores[::-1])
        axes[1].set_title("Signal Importance (top 15)")
        axes[1].set_xlabel("Normalised weight norm")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("backtest_results.png", dpi=150)
        print("Plot saved to backtest_results.png")
        plt.show()
    except ImportError:
        print("matplotlib not installed — skipping plot. Run: pip install matplotlib")
