"""Polymarket Real-Time Data Socket client (Chainlink price re-broadcast).

Polymarket re-broadcasts the Chainlink prices its crypto up/down markets settle
on through a free, no-auth websocket (topic `crypto_prices_chainlink`, assets
btc/eth/sol/xrp vs usd). This replaces a direct (paid after July 2026) Chainlink
Data Streams subscription: same settlement-truth prices, sourced from
Polymarket's own view of them, with a trivial JSON payload instead of
ABI-encoded signed reports.

Protocol (per docs.polymarket.com/developers/RTDS and the official TS client):
  - subscribe: {"action":"subscribe","subscriptions":[{"topic":
    "crypto_prices_chainlink","type":"*","filters":""}]}  (empty filters = all)
  - keepalive: plain-text "ping" every 5 s (server replies "pong")
  - update:    {"topic":...,"type":"update","payload":
                {"symbol":"btc/usd","timestamp":<ms>,"value":<price>}}

Populates the same ChainlinkAssetState objects as the Data Streams client, so
nothing downstream changes. RTDS carries no bid/ask — the bot only ever reads
`.price`, so they're set to 0.
"""

import asyncio
import json
import time

import websockets

from data.chainlink_stream import ChainlinkAssetState
from utils.logger import get_logger
from config import POLYMARKET_RTDS_WS_URL

log = get_logger("polymarket_rtds")

_PING_INTERVAL_S = 5.0

_SUBSCRIBE_MSG = json.dumps({
    "action": "subscribe",
    "subscriptions": [
        {"topic": "crypto_prices_chainlink", "type": "*", "filters": ""}
    ],
})


class PolymarketRTDSClient:
    """Streams Chainlink prices from Polymarket's RTDS and updates shared states.

    symbols: maps an RTDS symbol ("btc/usd") → the ChainlinkAssetState it feeds.
    Reconnects automatically; backoff behavior mirrors the other stream clients.
    """

    def __init__(self, symbols: dict[str, ChainlinkAssetState]) -> None:
        self._symbols = {sym.lower(): st for sym, st in symbols.items()}
        self._running = False

    def _handle_message(self, raw: str) -> None:
        # Keepalive replies ("pong") and acks aren't JSON payloads — skip fast.
        if "payload" not in raw:
            return
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        payloads = msg.get("payload")
        if isinstance(payloads, dict):
            payloads = [payloads]
        if not isinstance(payloads, list):
            return
        for p in payloads:
            try:
                symbol = str(p.get("symbol", "")).lower()
                state = self._symbols.get(symbol)
                if state is None:
                    continue
                value = float(p.get("value", 0.0))
                ts_ms = float(p.get("timestamp", 0.0))
                if value > 0:
                    state._set(value, 0.0, 0.0, ts_ms / 1000.0)
            except (TypeError, ValueError):
                continue

    async def _ping_loop(self, ws) -> None:
        while True:
            await asyncio.sleep(_PING_INTERVAL_S)
            await ws.send("ping")

    async def run(self) -> None:
        self._running = True
        backoff = 3
        connected_once = False
        while self._running:
            conn_t0 = time.time()
            ping_task = None
            try:
                (log.debug if connected_once else log.info)(
                    f"Connecting to Polymarket RTDS: {POLYMARKET_RTDS_WS_URL}"
                )
                async with websockets.connect(
                    POLYMARKET_RTDS_WS_URL,
                    ping_interval=20, ping_timeout=10, close_timeout=5,
                ) as ws:
                    await ws.send(_SUBSCRIBE_MSG)
                    ping_task = asyncio.create_task(self._ping_loop(ws))
                    if not connected_once:
                        log.info(
                            "Polymarket RTDS connected — Chainlink prices for "
                            f"{sorted(self._symbols)}."
                        )
                        connected_once = True
                    backoff = 3
                    conn_t0 = time.time()
                    first = True
                    # Watchdog recv: a half-open socket ("no close frame") can
                    # hang a bare recv forever while prices silently freeze —
                    # observed 2026-07-07/08 corrupting settlement labels. If
                    # nothing (not even a pong) arrives in 30s, force reconnect.
                    while True:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=30)
                        except asyncio.TimeoutError:
                            log.warning("RTDS silent for 30s — forcing reconnect.")
                            break
                        if isinstance(raw, bytes):
                            raw = raw.decode("utf-8", errors="ignore")
                        self._handle_message(raw)
                        if first and "payload" in raw:
                            log.debug("Polymarket RTDS: first price received.")
                            first = False
            except (websockets.ConnectionClosed, OSError) as e:
                msg = f"RTDS disconnected ({e}), reconnecting in {backoff}s…"
                (log.debug if time.time() - conn_t0 >= 30 else log.warning)(msg)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
            except Exception as e:
                log.error(f"Unexpected RTDS error: {e}", exc_info=True)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
            finally:
                if ping_task is not None:
                    ping_task.cancel()

    def stop(self) -> None:
        self._running = False
