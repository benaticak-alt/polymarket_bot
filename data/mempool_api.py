"""
Mempool.space on-chain activity signals.
No API key required. Polls every 30 seconds.

High fee rates → strong demand for block space → whale/institutional activity.
High mempool vsize → congestion / sustained flow.
Both signals are useful as BTC market-regime indicators.
"""

import asyncio
import time
import requests
from dataclasses import dataclass
from utils.logger import get_logger

log = get_logger("mempool")

POLL_INTERVAL  = 30.0
FEES_URL       = "https://mempool.space/api/v1/fees/recommended"
MEMPOOL_URL    = "https://mempool.space/api/mempool"

FEE_BASELINE   = 15.0          # sat/vB quiet market baseline
VSIZE_BASELINE = 50_000_000    # 50 MB = moderate congestion baseline


@dataclass
class MempoolState:
    fee_rate_fast:  float = 15.0   # sat/vB for next-block inclusion
    mempool_vsize:  int   = 0      # total pending bytes
    fee_norm:       float = 0.0    # normalised fee: 0 at baseline, ±1 at extremes
    congestion:     float = 0.0    # normalised vsize: 0 at baseline, ±1 at extremes
    last_updated:   float = 0.0


class MempoolClient:

    def __init__(self, state: MempoolState):
        self.state    = state
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "polymarket-bot/1.0"})

    async def run(self) -> None:
        while True:
            try:
                await asyncio.get_event_loop().run_in_executor(None, self._fetch)
            except Exception as e:
                log.warning(f"Mempool poll error: {e}")
            await asyncio.sleep(POLL_INTERVAL)

    def _fetch(self) -> None:
        self._fetch_fees()
        self._fetch_mempool()
        self.state.last_updated = time.time()
        log.debug(
            f"Mempool | fee={self.state.fee_rate_fast:.1f} sat/vB "
            f"(norm={self.state.fee_norm:+.3f}) | "
            f"vsize={self.state.mempool_vsize / 1e6:.1f}MB "
            f"(congestion={self.state.congestion:+.3f})"
        )

    def _fetch_fees(self) -> None:
        try:
            resp = self._session.get(FEES_URL, timeout=5)
            resp.raise_for_status()
            fees = resp.json()
            self.state.fee_rate_fast = float(fees.get("fastestFee", FEE_BASELINE))
            # Normalise: (fee - baseline) / baseline → 0 at rest, negative below, positive above
            self.state.fee_norm = float(max(-1.0, min(1.0,
                (self.state.fee_rate_fast - FEE_BASELINE) / FEE_BASELINE
            )))
        except Exception as e:
            log.debug(f"Fee rate fetch error: {e}")

    def _fetch_mempool(self) -> None:
        try:
            resp = self._session.get(MEMPOOL_URL, timeout=5)
            resp.raise_for_status()
            mem = resp.json()
            self.state.mempool_vsize = int(mem.get("vsize", 0))
            self.state.congestion = float(max(-1.0, min(1.0,
                (self.state.mempool_vsize - VSIZE_BASELINE) / VSIZE_BASELINE
            )))
        except Exception as e:
            log.debug(f"Mempool vsize fetch error: {e}")
