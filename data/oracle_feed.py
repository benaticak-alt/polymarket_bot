"""
Chainlink BTC/USD oracle feed reader (Polygon mainnet).
Polls the on-chain aggregator and exposes price + lag metrics.
"""

import asyncio
import time
from utils.logger import get_logger
from config import (
    POLYGON_RPC_URL,
    CHAINLINK_BTC_USD_POLYGON,
    CHAINLINK_AGGREGATOR_ABI,
)

log = get_logger("oracle_feed")

POLL_INTERVAL = 10.0  # seconds between oracle polls (Chainlink updates every 20-60 min)


class OracleState:
    """Shared oracle state updated by OracleFeedClient."""

    def __init__(self):
        self.price: float       = 0.0
        self.updated_at: float  = 0.0   # on-chain updatedAt timestamp
        self.fetched_at: float  = 0.0   # local time of last successful poll
        self.round_id: int      = 0
        self.decimals: int      = 8

    @property
    def lag_seconds(self) -> float:
        """Seconds since the oracle last published a new price."""
        if not self.updated_at:
            return 9999.0
        return time.time() - self.updated_at

    @property
    def fetch_age(self) -> float:
        """Seconds since we last successfully polled the oracle."""
        if not self.fetched_at:
            return 9999.0
        return time.time() - self.fetched_at

    def divergence_from(self, spot_price: float) -> float:
        """Signed fractional divergence: (oracle - spot) / spot."""
        if not self.price or not spot_price:
            return 0.0
        return (self.price - spot_price) / spot_price


class OracleFeedClient:

    def __init__(self, state: OracleState):
        self.state   = state
        self._w3     = None
        self._contract = None
        self._running  = False
        self._init_web3()

    def _init_web3(self) -> None:
        try:
            from web3 import Web3
            # Try multiple public Polygon RPCs
            rpcs = [
                POLYGON_RPC_URL,
                "https://rpc.ankr.com/polygon",
                "https://polygon-bor-rpc.publicnode.com",
                "https://polygon.llamarpc.com",
                "https://polygon.drpc.org",
                "https://1rpc.io/matic",
                "https://polygon-rpc.com",
                "https://rpc-mainnet.matic.network",
                "https://rpc-mainnet.maticvigil.com",
            ]
            for rpc in rpcs:
                try:
                    w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 8}))
                    if w3.is_connected():
                        self._w3 = w3
                        log.info(f"Web3 connected via {rpc}")
                        break
                except Exception:
                    continue

            if not self._w3:
                log.warning("All Polygon RPCs failed — oracle feed disabled.")
                return

            self._contract = self._w3.eth.contract(
                address=Web3.to_checksum_address(CHAINLINK_BTC_USD_POLYGON),
                abi=CHAINLINK_AGGREGATOR_ABI,
            )
            self.state.decimals = 8  # BTC/USD feed always uses 8 decimals
            log.info(f"Chainlink oracle connected via {rpc}")
        except ImportError:
            log.warning("web3 not installed — oracle feed disabled.")
        except Exception as e:
            log.error(f"Oracle init failed: {e}")

    async def run(self) -> None:
        self._running = True
        while self._running:
            self._poll()
            await asyncio.sleep(POLL_INTERVAL)

    def _poll(self) -> None:
        if not self._contract:
            return
        try:
            round_data = self._contract.functions.latestRoundData().call()
            round_id, answer, started_at, updated_at, answered_in_round = round_data
            price = answer / (10 ** self.state.decimals)

            if round_id != self.state.round_id:
                log.debug(f"Oracle updated: ${price:,.2f}  lag={time.time()-updated_at:.1f}s")

            self.state.price      = price
            self.state.updated_at = float(updated_at)
            self.state.fetched_at = time.time()
            self.state.round_id   = round_id
        except Exception as e:
            log.debug(f"Oracle poll error: {e}")

    def stop(self) -> None:
        self._running = False
