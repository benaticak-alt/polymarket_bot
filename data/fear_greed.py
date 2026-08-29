"""
Crypto Fear & Greed Index from alternative.me — free, no API key.
Polls every 15 minutes (index only updates daily).
"""

import asyncio
import time
import requests
from dataclasses import dataclass
from utils.logger import get_logger

log = get_logger("fear_greed")

POLL_INTERVAL = 900  # 15 minutes


@dataclass
class FearGreedState:
    score: float  = 0.0   # normalised [-1, 1]; 0 = neutral (50/100)
    change: float = 0.0   # day-over-day change, normalised
    raw: int      = 50    # raw 0–100 value
    last_updated: float = 0.0


class FearGreedClient:

    def __init__(self, state: FearGreedState):
        self.state = state

    async def run(self) -> None:
        while True:
            self._poll()
            await asyncio.sleep(POLL_INTERVAL)

    def _poll(self) -> None:
        try:
            r = requests.get(
                "https://api.alternative.me/fng/",
                params={"limit": 2},
                timeout=10,
            )
            r.raise_for_status()
            data = r.json().get("data", [])
            if not data:
                return

            current_raw = int(data[0]["value"])
            prev_raw    = int(data[1]["value"]) if len(data) > 1 else current_raw

            self.state.raw          = current_raw
            self.state.score        = (current_raw - 50) / 50.0
            self.state.change       = (current_raw - prev_raw) / 50.0
            self.state.last_updated = time.time()

            log.info(
                f"Fear & Greed: {current_raw}/100 "
                f"({data[0].get('value_classification', '')}) "
                f"day_change={current_raw - prev_raw:+d}"
            )
        except Exception as e:
            log.warning(f"Fear & Greed poll error: {e}")
