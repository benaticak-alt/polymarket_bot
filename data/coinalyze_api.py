"""
CoinAlyze API client.
Provides derivatives market context signals:
  - Funding rate (perpetual futures)
  - Open interest + OI change
  - Liquidation imbalance (long vs short liquidations)
  - Long/short ratio
Polled every 60 seconds.
"""

import asyncio
import time
import requests
from dataclasses import dataclass, field
from utils.logger import get_logger
from config import COINALYZE_API_KEY

log = get_logger("coinalyze")

BASE_URL      = "https://api.coinalyze.net/v1"
SYMBOL        = "BTCUSDT_PERP.A"   # aggregated perpetual across exchanges
POLL_INTERVAL = 60.0


@dataclass
class CoinAlyzeState:
    # Funding rate (hourly, annualised by exchange)
    funding_rate: float         = 0.0   # e.g. 0.0001 = 0.01% per 8h

    # Open interest
    open_interest: float        = 0.0   # current OI in USD
    oi_change_pct: float        = 0.0   # % change vs prior poll

    # Liquidations (last hour)
    liq_long_usd: float         = 0.0   # long liquidations (USD)
    liq_short_usd: float        = 0.0   # short liquidations (USD)

    # Long/short ratio
    long_ratio: float           = 0.5   # fraction of accounts long
    short_ratio: float          = 0.5

    # Internal history
    _oi_history: list[float]    = field(default_factory=list)

    last_updated: float         = 0.0

    @property
    def age_seconds(self) -> float:
        return time.time() - self.last_updated if self.last_updated else 9999.0

    @property
    def is_stale(self) -> bool:
        return self.age_seconds > 180.0

    @property
    def liq_imbalance(self) -> float:
        """Signed liquidation imbalance: positive = more longs liquidated (bearish)."""
        total = self.liq_long_usd + self.liq_short_usd
        if total == 0:
            return 0.0
        return (self.liq_long_usd - self.liq_short_usd) / total


class CoinAlyzeClient:

    def __init__(self, state: CoinAlyzeState):
        self.state    = state
        self._session = requests.Session()
        self._session.headers.update({
            "api_key": COINALYZE_API_KEY,
        })
        self._running = False

    async def run(self) -> None:
        self._running = True
        await asyncio.get_event_loop().run_in_executor(None, self._fetch_all)
        while self._running:
            await asyncio.sleep(POLL_INTERVAL)
            await asyncio.get_event_loop().run_in_executor(None, self._fetch_all)

    def _fetch_all(self) -> None:
        self._fetch_funding_rate()
        self._fetch_open_interest()
        self._fetch_liquidations()
        self._fetch_long_short_ratio()
        self.state.last_updated = time.time()
        log.debug(
            f"CoinAlyze | funding={self.state.funding_rate:.6f} | "
            f"OI_chg={self.state.oi_change_pct:+.2%} | "
            f"liq_imb={self.state.liq_imbalance:+.3f} | "
            f"L/S={self.state.long_ratio:.2f}/{self.state.short_ratio:.2f}"
        )

    def _fetch_funding_rate(self) -> None:
        try:
            resp = self._session.get(
                f"{BASE_URL}/funding-rate",
                params={"symbols": SYMBOL},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and data:
                # API returns percentage (0.01 = 0.01% per 8h); convert to decimal
                self.state.funding_rate = float(data[0].get("value", 0)) / 100
            elif isinstance(data, dict):
                history = data.get("history", [])
                if history:
                    self.state.funding_rate = float(history[-1].get("value", 0)) / 100
        except Exception as e:
            log.warning(f"Funding rate fetch error: {e}")

    def _fetch_open_interest(self) -> None:
        try:
            resp = self._session.get(
                f"{BASE_URL}/open-interest",
                params={"symbols": SYMBOL},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            oi = 0.0
            if isinstance(data, list) and data:
                oi = float(data[0].get("value", 0))
            elif isinstance(data, dict):
                history = data.get("history", [])
                if history:
                    oi = float(history[-1].get("value", 0))

            # Compute OI change
            if self.state.open_interest > 0:
                self.state.oi_change_pct = (oi - self.state.open_interest) / self.state.open_interest

            self.state._oi_history.append(oi)
            if len(self.state._oi_history) > 60:
                self.state._oi_history.pop(0)

            self.state.open_interest = oi
        except Exception as e:
            log.warning(f"Open interest fetch error: {e}")

    def _fetch_liquidations(self) -> None:
        try:
            now = int(time.time())
            resp = self._session.get(
                f"{BASE_URL}/liquidation-history",
                params={"symbols": SYMBOL, "interval": "1hour", "from": now - 3600, "to": now},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and data:
                history = data[0].get("history", [])
                if history:
                    entry = history[-1]
                    # API uses "l" = long liquidations, "s" = short liquidations
                    self.state.liq_long_usd  = float(entry.get("l", entry.get("long",  0)))
                    self.state.liq_short_usd = float(entry.get("s", entry.get("short", 0)))
        except Exception as e:
            log.debug(f"Liquidations unavailable: {e} — signals defaulting to 0")

    def _fetch_long_short_ratio(self) -> None:
        # Not available on current CoinAlyze plan — signal stays neutral (0.5/0.5)
        pass

    def stop(self) -> None:
        self._running = False
