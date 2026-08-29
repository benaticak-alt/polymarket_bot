"""
Kalshi cross-platform BTC direction signal.
Fetches the nearest open BTC market from Kalshi and uses its
implied probability as an independent directional confirmation signal.
No API key required for read-only access. Polls every 15 seconds.
"""
import asyncio
import time
import requests
from dataclasses import dataclass
from utils.logger import get_logger

log = get_logger("kalshi")

POLL_INTERVAL   = 15.0
KALSHI_API_BASE = "https://trading-api.kalshi.com/trade-api/v2"
BTC_SERIES      = ["KXBTC", "KXBTCD", "BTCUSD", "KXBTCR"]


@dataclass
class KalshiState:
    yes_prob:      float = 0.5   # Kalshi implied P(BTC up) from nearest open market
    kalshi_signal: float = 0.0   # (yes_prob - 0.5) * 2, range [-1, 1]
    last_updated:  float = 0.0


class KalshiClient:

    def __init__(self, state: KalshiState):
        self.state    = state
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "polymarket-bot/1.0",
            "Accept": "application/json",
        })

    async def run(self) -> None:
        while True:
            try:
                await asyncio.get_event_loop().run_in_executor(None, self._fetch)
            except Exception as e:
                log.warning(f"Kalshi poll error: {e}")
            await asyncio.sleep(POLL_INTERVAL)

    def _fetch(self) -> None:
        markets = self._get_btc_markets()
        if not markets:
            log.debug("No open Kalshi BTC markets found — using neutral signal.")
            return

        prices = [self._extract_yes_price(m) for m in markets]
        prices = [p for p in prices if 0.01 < p < 0.99]
        if not prices:
            return

        # Use the market whose yes price is closest to 0.5 (most active/contested)
        nearest = min(markets, key=lambda m: abs(self._extract_yes_price(m) - 0.5))
        yes_price = self._extract_yes_price(nearest)
        if not (0.01 < yes_price < 0.99):
            return

        self.state.yes_prob      = yes_price
        self.state.kalshi_signal = float(max(-1.0, min(1.0, (yes_price - 0.5) * 2)))
        self.state.last_updated  = time.time()

        log.debug(
            f"Kalshi BTC: yes={yes_price:.3f} "
            f"signal={self.state.kalshi_signal:+.3f} "
            f"({nearest.get('ticker', '?')})"
        )

    def _get_btc_markets(self) -> list[dict]:
        now = time.time()
        for series in BTC_SERIES:
            try:
                resp = self._session.get(
                    f"{KALSHI_API_BASE}/markets",
                    params={"series_ticker": series, "status": "open", "limit": 20},
                    timeout=5,
                )
                if resp.status_code == 200:
                    markets = resp.json().get("markets", [])
                    future  = [m for m in markets if _close_ts(m) > now + 60]
                    if future:
                        return future
            except Exception:
                continue
        return []

    def _extract_yes_price(self, market: dict) -> float:
        bid  = market.get("yes_bid",   0) or 0
        ask  = market.get("yes_ask",   0) or 0
        last = market.get("last_price", 0) or 0
        if bid and ask:
            mid_cents = (bid + ask) / 2.0
        elif last:
            mid_cents = float(last)
        else:
            return 0.0
        return float(max(0.0, min(100.0, mid_cents))) / 100.0


def _close_ts(market: dict) -> float:
    close = market.get("close_time", "") or ""
    if not close:
        return 0.0
    try:
        from datetime import datetime, timezone
        return datetime.fromisoformat(close.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0
