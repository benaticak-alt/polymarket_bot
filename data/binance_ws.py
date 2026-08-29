"""
Binance Futures WebSocket client.
Subscribes to aggTrade, depth20, kline_1m, kline_5m, bookTicker streams
and populates shared ring buffers used by all signal modules.
"""

import asyncio
import json
import time
import requests
import websockets
from utils.ring_buffer import TimeSeriesBuffer, OrderBookSnapshot, TradeFlowBuffer, LiquidationBuffer
from utils.logger import get_logger
from config import BINANCE_STREAM_BASE, SYMBOL_LOWER, BINANCE_STREAMS

KLINES_REST_URL = "https://api.binance.us/api/v3/klines"

log = get_logger("binance_ws")


class MarketState:
    """Shared, mutable market state populated by the WebSocket feed."""

    def __init__(self):
        # Price & trade flow
        self.price          = TimeSeriesBuffer(maxlen=20_000)
        self.trade_flow     = TradeFlowBuffer(maxlen=10_000)

        # Order book
        self.order_book     = OrderBookSnapshot()

        # Klines (close prices keyed by interval)
        self.klines_1m: list[dict] = []   # rolling list of closed 1m candles
        self.klines_5m: list[dict] = []

        # Best bid/ask from bookTicker
        self.best_bid: float = 0.0
        self.best_ask: float = 0.0

        # Volume (for VWAP)
        self.vwap_price_vol = TimeSeriesBuffer(maxlen=20_000)  # price*vol
        self.vwap_vol       = TimeSeriesBuffer(maxlen=20_000)  # raw vol

        # Real-time liquidation events (from forceOrder stream)
        self.liq_buffer = LiquidationBuffer(maxlen=5_000)

        self.last_update: float = 0.0

    @property
    def mid(self) -> float:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2.0
        return self.price.last_value() or 0.0


class BinanceStreamClient:
    """
    Multiplexed Binance Futures stream client.
    Reconnects automatically on disconnect.
    """

    def __init__(self, state: MarketState, symbol: str = "BTCUSDT"):
        self.state   = state
        self._symbol = symbol.lower()
        self._running = False

    def _build_url(self) -> str:
        s = self._symbol
        streams = "/".join([
            f"{s}@aggTrade",
            f"{s}@depth20@100ms",
            f"{s}@kline_1m",
            f"{s}@kline_5m",
            f"{s}@bookTicker",
            f"{s}@forceOrder",
        ])
        return f"wss://fstream.binance.com/stream?streams={streams}"

    async def _handle_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        # Combined stream wraps in {"stream": ..., "data": ...}
        if "stream" in msg:
            stream = msg["stream"]
            data   = msg["data"]
        else:
            stream = msg.get("e", "")
            data   = msg

        event = data.get("e", stream)

        # Any arriving message proves the feed is live — update heartbeat
        # so the staleness guard passes even before the first aggTrade event.
        self.state.last_update = time.time()

        try:
            if event == "aggTrade":
                self._handle_trade(data)
            elif event == "depthUpdate":
                self._handle_depth(data)
            elif event == "kline":
                self._handle_kline(data)
            elif event == "bookTicker":
                self._handle_book_ticker(data)
            elif event == "forceOrder":
                self._handle_liquidation(data)
        except Exception as e:
            log.warning(f"Handler error for event '{event}': {e}", exc_info=True)

    def _handle_trade(self, d: dict) -> None:
        price = float(d["p"])
        qty   = float(d["q"])
        ts    = d["T"] / 1000.0  # trade time in seconds
        self.state.price.append(price, ts)
        self.state.trade_flow.append(qty, d["m"], ts)  # m=True: buyer is maker → sell aggression
        self.state.vwap_price_vol.append(price * qty, ts)
        self.state.vwap_vol.append(qty, ts)
        self.state.last_update = ts

    def _handle_depth(self, d: dict) -> None:
        self.state.order_book.update(d.get("b", []), d.get("a", []))

    def _handle_kline(self, d: dict) -> None:
        k = d["k"]
        candle = {
            "open":   float(k["o"]),
            "high":   float(k["h"]),
            "low":    float(k["l"]),
            "close":  float(k["c"]),
            "volume": float(k["v"]),
            "closed": k["x"],
            "ts":     k["T"] / 1000.0,
        }
        interval = k["i"]
        if interval == "1m":
            if candle["closed"]:
                if self.state.klines_1m and not self.state.klines_1m[-1]["closed"]:
                    self.state.klines_1m[-1] = candle
                else:
                    self.state.klines_1m.append(candle)
                if len(self.state.klines_1m) > 500:
                    self.state.klines_1m.pop(0)
            else:
                # Keep the current open candle at the tail so signals always have
                # a "live" close price even before the minute closes.
                if self.state.klines_1m and not self.state.klines_1m[-1]["closed"]:
                    self.state.klines_1m[-1] = candle
                else:
                    self.state.klines_1m.append(candle)
        elif interval == "5m":
            if candle["closed"]:
                if self.state.klines_5m and not self.state.klines_5m[-1]["closed"]:
                    self.state.klines_5m[-1] = candle
                else:
                    self.state.klines_5m.append(candle)
                if len(self.state.klines_5m) > 200:
                    self.state.klines_5m.pop(0)
            else:
                if self.state.klines_5m and not self.state.klines_5m[-1]["closed"]:
                    self.state.klines_5m[-1] = candle
                else:
                    self.state.klines_5m.append(candle)

    def _handle_book_ticker(self, d: dict) -> None:
        bid = float(d.get("b", 0))
        ask = float(d.get("a", 0))
        self.state.best_bid = bid
        self.state.best_ask = ask
        # Fall back to bookTicker mid price only when no aggTrade data has
        # arrived recently — either because the US trade client hasn't connected
        # yet, or because it disconnected. 30s grace handles brief reconnects.
        if bid > 0 and ask > 0:
            last_trade_ts = self.state.trade_flow._times[-1] if self.state.trade_flow._times else 0
            if (time.time() - last_trade_ts) > 30:
                self.state.price.append((bid + ask) / 2.0)

    def _handle_liquidation(self, d: dict) -> None:
        o = d.get("o", d)  # event wraps order in "o" key
        side  = o.get("S", "")           # "BUY" = short liq, "SELL" = long liq
        qty   = float(o.get("q", 0))     # quantity in BTC
        price = float(o.get("ap", 0)) or float(o.get("p", 0))  # avg fill price
        ts    = o.get("T", 0) / 1000.0 if o.get("T") else None
        if side and qty > 0 and price > 0:
            usd_value = qty * price
            self.state.liq_buffer.append(side, usd_value, ts)

    def _fetch_historical_klines(self, interval: str, limit: int = 50) -> list[dict]:
        """Fetch closed historical klines via REST to warm up TA signals at startup."""
        try:
            resp = requests.get(
                KLINES_REST_URL,
                params={"symbol": self._symbol.upper(), "interval": interval, "limit": limit},
                timeout=10,
            )
            resp.raise_for_status()
            candles = []
            for row in resp.json():
                candles.append({
                    "open":   float(row[1]),
                    "high":   float(row[2]),
                    "low":    float(row[3]),
                    "close":  float(row[4]),
                    "volume": float(row[5]),
                    "closed": True,
                    "ts":     float(row[6]) / 1000.0,
                })
            return candles
        except Exception as e:
            log.warning(f"Historical klines fetch failed ({interval}): {e}")
            return []

    async def _init_klines(self) -> None:
        """Pre-populate klines from REST so TA signals work immediately on startup."""
        loop = asyncio.get_event_loop()
        klines_1m = await loop.run_in_executor(None, lambda: self._fetch_historical_klines("1m", 60))
        klines_5m = await loop.run_in_executor(None, lambda: self._fetch_historical_klines("5m", 60))
        if klines_1m:
            self.state.klines_1m.extend(klines_1m)
            log.info(f"Pre-loaded {len(klines_1m)} historical 1m candles.")
        if klines_5m:
            self.state.klines_5m.extend(klines_5m)
            log.info(f"Pre-loaded {len(klines_5m)} historical 5m candles.")

    async def run(self) -> None:
        self._running = True
        await self._init_klines()
        url     = self._build_url()
        backoff = 3  # seconds; doubles on each failure, capped at 60s
        while self._running:
            conn_t0 = time.time()
            try:
                log.info(f"Connecting to Binance stream: {url[:80]}...")
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    log.info("Binance stream connected.")
                    backoff = 3  # reset on successful connection
                    conn_t0 = time.time()
                    msg_count = 0
                    async for raw in ws:
                        await self._handle_message(raw)
                        msg_count += 1
                        if msg_count == 1:
                            log.info("Binance stream: first message received — data flowing.")
            except (websockets.ConnectionClosed, OSError) as e:
                # A socket that was up a while then closed is the server cycling the
                # connection (routine) — log quietly. A fast drop is a real problem.
                msg = f"Binance WS disconnected ({e}), reconnecting in {backoff}s..."
                (log.debug if time.time() - conn_t0 >= 30 else log.warning)(msg)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
            except Exception as e:
                log.error(f"Unexpected Binance WS error: {e}", exc_info=True)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    def stop(self) -> None:
        self._running = False


class BinanceUSTradeClient:
    """
    Connects to Binance US aggTrade stream (wss://stream.binance.us:9443).
    Accessible to US users where fstream.binance.com aggTrade is geo-blocked.
    Fills the trade_flow, price, and VWAP buffers — recovering flow_short,
    flow_long, and vwap_dev signals that are dead without directional trade data.
    """

    def __init__(self, state: MarketState, symbol: str = "BTCUSDT"):
        self.state   = state
        self._symbol = symbol.lower()
        self._running = False

    async def run(self) -> None:
        self._running = True
        url = f"wss://stream.binance.us:9443/stream?streams={self._symbol}@aggTrade"
        backoff = 3
        while self._running:
            conn_t0 = time.time()
            try:
                log.info("Connecting to Binance US aggTrade stream...")
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    log.info("Binance US aggTrade stream connected — trade flow data active.")
                    backoff = 3
                    conn_t0 = time.time()
                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        data = msg.get("data", msg)
                        if data.get("e") == "aggTrade":
                            price = float(data["p"])
                            qty   = float(data["q"])
                            ts    = data["T"] / 1000.0
                            self.state.price.append(price, ts)
                            self.state.trade_flow.append(qty, data["m"], ts)
                            self.state.vwap_price_vol.append(price * qty, ts)
                            self.state.vwap_vol.append(qty, ts)
            except (websockets.ConnectionClosed, OSError) as e:
                msg = f"Binance US aggTrade disconnected ({e}), reconnecting in {backoff}s..."
                (log.debug if time.time() - conn_t0 >= 30 else log.warning)(msg)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
            except Exception as e:
                log.error(f"Unexpected Binance US aggTrade error: {e}", exc_info=True)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    def stop(self) -> None:
        self._running = False
