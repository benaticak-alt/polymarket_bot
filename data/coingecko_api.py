"""
CoinGecko Pro API client.
Provides supplementary market context signals:
  - BTC dominance trend
  - Global market sentiment (fear/greed proxy via volume/cap ratio)
  - BTC volume anomaly vs 30-day average
  - Price change velocity across timeframes
Polled every 60 seconds (respects Pro rate limits).
"""

import asyncio
import time
import requests
from dataclasses import dataclass, field
from utils.logger import get_logger
from config import COINGECKO_API_KEY

log = get_logger("coingecko")

BASE_URL  = "https://pro-api.coingecko.com/api/v3"
HEADERS   = {"x-cg-pro-api-key": COINGECKO_API_KEY}
POLL_INTERVAL = 60.0  # seconds


@dataclass
class CoinGeckoState:
    # BTC-specific
    btc_price_usd: float        = 0.0
    btc_volume_24h: float       = 0.0
    btc_vol_change_pct: float   = 0.0   # vs prior period
    btc_price_change_1h: float  = 0.0   # % change
    btc_price_change_24h: float = 0.0
    btc_market_cap: float       = 0.0

    # Global market
    btc_dominance: float        = 0.0   # % of total crypto market cap
    total_market_cap: float     = 0.0
    total_volume_24h: float     = 0.0

    # Derived
    vol_to_cap_ratio: float     = 0.0   # volume / market_cap — sentiment proxy
    dominance_delta: float      = 0.0   # change in BTC dominance since last poll

    # Historical for trend
    _dominance_history: list[float] = field(default_factory=list)
    _vol_history: list[float]       = field(default_factory=list)

    last_updated: float = 0.0

    @property
    def age_seconds(self) -> float:
        return time.time() - self.last_updated if self.last_updated else 9999.0

    @property
    def is_stale(self) -> bool:
        return self.age_seconds > 180.0  # data older than 3 min is stale


class CoinGeckoClient:

    def __init__(self, state: CoinGeckoState):
        self.state    = state
        self._session = requests.Session()
        self._session.headers.update(HEADERS)
        self._running = False

    async def run(self) -> None:
        self._running = True
        # Initial fetch immediately
        await asyncio.get_event_loop().run_in_executor(None, self._fetch_all)
        while self._running:
            await asyncio.sleep(POLL_INTERVAL)
            await asyncio.get_event_loop().run_in_executor(None, self._fetch_all)

    def _fetch_all(self) -> None:
        self._fetch_btc_data()
        self._fetch_global_data()
        self.state.last_updated = time.time()

    def _fetch_btc_data(self) -> None:
        try:
            resp = self._session.get(
                f"{BASE_URL}/coins/bitcoin",
                params={
                    "localization":   "false",
                    "tickers":        "false",
                    "market_data":    "true",
                    "community_data": "false",
                    "developer_data": "false",
                },
                timeout=10,
            )
            resp.raise_for_status()
            d  = resp.json()
            md = d.get("market_data", {})

            self.state.btc_price_usd       = md.get("current_price",    {}).get("usd", 0)
            self.state.btc_volume_24h      = md.get("total_volume",     {}).get("usd", 0)
            self.state.btc_market_cap      = md.get("market_cap",       {}).get("usd", 0)
            self.state.btc_price_change_1h  = md.get("price_change_percentage_1h_in_currency",  {}).get("usd", 0)
            self.state.btc_price_change_24h = md.get("price_change_percentage_24h_in_currency", {}).get("usd", 0)

            # Volume anomaly: compare to 7-day rolling average
            self.state._vol_history.append(self.state.btc_volume_24h)
            if len(self.state._vol_history) > 168:  # 168 hours = 7 days at 1/hr
                self.state._vol_history.pop(0)
            if len(self.state._vol_history) >= 2:
                avg_vol = sum(self.state._vol_history[:-1]) / len(self.state._vol_history[:-1])
                self.state.btc_vol_change_pct = (
                    (self.state.btc_volume_24h - avg_vol) / avg_vol
                    if avg_vol > 0 else 0.0
                )

            # Vol/cap ratio
            if self.state.btc_market_cap > 0:
                self.state.vol_to_cap_ratio = self.state.btc_volume_24h / self.state.btc_market_cap

            log.debug(
                f"BTC ${self.state.btc_price_usd:,.0f} | "
                f"vol_change={self.state.btc_vol_change_pct:+.1%} | "
                f"1h={self.state.btc_price_change_1h:+.2f}%"
            )
        except Exception as e:
            log.warning(f"CoinGecko BTC fetch error: {e}")

    def _fetch_global_data(self) -> None:
        try:
            resp = self._session.get(f"{BASE_URL}/global", timeout=10)
            resp.raise_for_status()
            gd = resp.json().get("data", {})

            prev_dominance          = self.state.btc_dominance
            self.state.btc_dominance    = gd.get("market_cap_percentage", {}).get("btc", 0)
            self.state.total_market_cap = gd.get("total_market_cap",      {}).get("usd", 0)
            self.state.total_volume_24h = gd.get("total_volume",          {}).get("usd", 0)

            # Dominance delta (trend signal)
            if prev_dominance > 0:
                self.state.dominance_delta = self.state.btc_dominance - prev_dominance

            self.state._dominance_history.append(self.state.btc_dominance)
            if len(self.state._dominance_history) > 24:
                self.state._dominance_history.pop(0)

            log.debug(
                f"BTC dominance={self.state.btc_dominance:.1f}% "
                f"delta={self.state.dominance_delta:+.3f}"
            )
        except Exception as e:
            log.warning(f"CoinGecko global fetch error: {e}")

    def stop(self) -> None:
        self._running = False
