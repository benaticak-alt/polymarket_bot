"""
MarketMind BTC signal — polls the user's own prediction website.
Returns direction + confidence for 5m and 15m timeframes.
Polls every 60 seconds and caches; executor reads from state at T-10s.
"""

import asyncio
import time
import requests
from dataclasses import dataclass
from utils.logger import get_logger

log = get_logger("marketmind")

MARKETMIND_URL  = "https://marketmind-server-4gb8.vercel.app/api/btc-signal"
POLL_INTERVAL   = 60   # seconds


@dataclass
class MarketMindState:
    direction_5m:   float = 0.0   # -1 DOWN, 0 NEUTRAL, 1 UP
    confidence_5m:  float = 0.0   # 0–1
    direction_15m:  float = 0.0
    confidence_15m: float = 0.0
    last_updated:   float = 0.0


class MarketMindClient:

    def __init__(self, state: MarketMindState):
        self.state = state

    async def run(self) -> None:
        while True:
            await asyncio.gather(
                asyncio.to_thread(self._poll_one, "5m"),
                asyncio.to_thread(self._poll_one, "15m"),
            )
            await asyncio.sleep(POLL_INTERVAL)

    def _poll_one(self, tf: str) -> None:
        try:
            r = requests.get(
                MARKETMIND_URL,
                params={"timeframe": tf},
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()

            raw_dir    = data.get("direction", "NEUTRAL")
            direction  = 1.0 if raw_dir == "UP" else (-1.0 if raw_dir == "DOWN" else 0.0)
            confidence = float(data.get("confidence", 50)) / 100.0

            if tf == "5m":
                self.state.direction_5m  = direction
                self.state.confidence_5m = confidence
            else:
                self.state.direction_15m  = direction
                self.state.confidence_15m = confidence

            self.state.last_updated = time.time()
            log.info(f"MarketMind {tf}: {raw_dir} conf={confidence:.2f}")
        except Exception as e:
            log.warning(f"MarketMind poll error ({tf}): {e}")
