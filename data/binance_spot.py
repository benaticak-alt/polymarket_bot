"""
Binance spot price + deep futures order book poller.
No auth required. Polls REST every 5 seconds.

Provides:
  - spot_price: BTC/USDT spot mid price
  - basis / basis_change: (futures - spot) / spot, normalised to [-1, 1]
  - deep_book: 50-level futures order book for wide OBI signal
"""

import asyncio
import time
import requests
from dataclasses import dataclass, field
from utils.ring_buffer import OrderBookSnapshot
from utils.logger import get_logger

log = get_logger("binance_spot")

POLL_INTERVAL     = 5.0
SPOT_URL          = "https://api.binance.us/api/v3/ticker/bookTicker"
SPOT_DEPTH_URL    = "https://api.binance.us/api/v3/depth"
SYMBOL            = "BTCUSDT"
BASIS_NORM        = 0.005   # ±0.5% basis → ±1.0 normalised


@dataclass
class SpotState:
    spot_price:   float = 0.0
    basis:        float = 0.0    # normalised futures-spot spread
    basis_change: float = 0.0    # change vs prior poll
    deep_book:    OrderBookSnapshot = field(default_factory=OrderBookSnapshot)
    last_updated: float = 0.0


class BinanceSpotClient:

    def __init__(self, state: SpotState, market_state, symbol: str = "BTCUSDT"):
        self.state        = state
        self.market_state = market_state
        self._symbol      = symbol.upper()
        self._prev_basis  = 0.0
        self._session     = requests.Session()
        self._session.headers.update({"User-Agent": "polymarket-bot/1.0"})

    async def run(self) -> None:
        while True:
            try:
                await asyncio.get_event_loop().run_in_executor(None, self._fetch)
            except Exception as e:
                log.warning(f"Spot poll error: {e}")
            await asyncio.sleep(POLL_INTERVAL)

    def _fetch(self) -> None:
        self._fetch_spot_price()
        self._fetch_deep_book()
        self.state.last_updated = time.time()

    def _fetch_spot_price(self) -> None:
        try:
            resp = self._session.get(SPOT_URL, params={"symbol": self._symbol}, timeout=5)
            resp.raise_for_status()
            d = resp.json()
            bid = float(d["bidPrice"])
            ask = float(d["askPrice"])
            self.state.spot_price = (bid + ask) / 2.0

            futures_mid = self.market_state.mid
            if self.state.spot_price > 0 and futures_mid > 0:
                raw = (futures_mid - self.state.spot_price) / self.state.spot_price
                basis = float(max(-1.0, min(1.0, raw / BASIS_NORM)))
                self.state.basis_change = basis - self._prev_basis
                self._prev_basis        = basis
                self.state.basis        = basis
        except Exception as e:
            log.debug(f"Spot price error: {e}")

    def _fetch_deep_book(self) -> None:
        try:
            resp = self._session.get(
                SPOT_DEPTH_URL,
                params={"symbol": self._symbol, "limit": 50},
                timeout=5,
            )
            resp.raise_for_status()
            d = resp.json()
            self.state.deep_book.update(d.get("bids", []), d.get("asks", []))
        except Exception as e:
            log.debug(f"Deep book error: {e}")
