"""
Deribit Volatility Index (DVOL) for BTC — free public API, no account needed.
DVOL is BTC's equivalent of VIX: 30-day implied volatility from options prices.
High DVOL = market expects big moves. Low DVOL = calm market.
Polls every 5 minutes (index updates every hour but intraday moves matter).
"""

import asyncio
import time
import requests
from dataclasses import dataclass
from utils.logger import get_logger

log = get_logger("deribit_iv")

POLL_INTERVAL = 300   # 5 minutes
DVOL_URL      = "https://www.deribit.com/api/v2/public/get_volatility_index_data"


@dataclass
class DeribitIVState:
    iv_score:  float = 0.0   # normalised [-1, 1]; 0 = neutral (~80 DVOL)
    iv_change: float = 0.0   # change from previous reading, normalised
    dvol_raw:  float = 80.0  # raw DVOL value (typical range 40–150)
    last_updated: float = 0.0


class DeribitIVClient:

    def __init__(self, state: DeribitIVState):
        self.state       = state
        self._prev_dvol  = 80.0

    async def run(self) -> None:
        while True:
            self._poll()
            await asyncio.sleep(POLL_INTERVAL)

    def _poll(self) -> None:
        try:
            now_ms   = int(time.time() * 1000)
            start_ms = now_ms - 2 * 3600 * 1000   # 2 hours back → 2 hourly candles
            r = requests.get(
                DVOL_URL,
                params={
                    "currency":        "BTC",
                    "resolution":      "3600",
                    "start_timestamp": start_ms,
                    "end_timestamp":   now_ms,
                },
                timeout=10,
            )
            r.raise_for_status()
            data = r.json().get("result", {}).get("data", [])
            if not data:
                return

            # Each entry: [timestamp_ms, open, high, low, close]
            current_dvol = float(data[-1][4])   # latest close
            prev_dvol    = float(data[-2][4]) if len(data) > 1 else self._prev_dvol

            # Normalise: centre at 80 (typical baseline), scale so ±40 maps to ±1
            self.state.iv_score   = (current_dvol - 80.0) / 40.0
            self.state.iv_change  = (current_dvol - prev_dvol) / 40.0
            self.state.dvol_raw   = current_dvol
            self.state.last_updated = time.time()
            self._prev_dvol       = current_dvol

            log.info(
                f"Deribit DVOL: {current_dvol:.1f} "
                f"(Δ{current_dvol - prev_dvol:+.1f}) "
                f"norm={self.state.iv_score:+.3f}"
            )
        except Exception as e:
            log.warning(f"Deribit IV poll error: {e}")
