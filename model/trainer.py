"""
Online trainer: records trade outcomes after settlement and feeds them
to the SignalEnsemble for incremental learning.
"""

import time
import asyncio
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from model.ensemble import SignalEnsemble
from signals.aggregator import SIGNAL_NAMES
from utils.logger import get_logger

log = get_logger("trainer")


@dataclass
class PendingTrade:
    """A placed trade waiting for settlement to compute its outcome."""
    market_id: str
    token_id: str
    direction: str          # "YES" or "NO"
    entry_signal_vec: np.ndarray
    entry_price_btc: float
    entry_ts: float
    settlement_ts: float    # expected settlement time
    usdc_staked: float
    settled: bool = False
    btc_went_up: Optional[bool] = None
    _exit_snapshot: float = field(default=0.0, repr=False, compare=False)


class OutcomeTracker:
    """
    Tracks pending trades and resolves outcomes after settlement.
    On resolution, pushes (signal_vec, outcome) into the ensemble.
    """

    def __init__(self, ensemble: SignalEnsemble):
        self._ensemble = ensemble
        self._pending: list[PendingTrade] = []
        self._last_logged_n: int = 0

    def record_trade(self, trade: PendingTrade) -> None:
        self._pending.append(trade)
        log.info(
            f"Tracking trade: {trade.market_id} | {trade.direction} | "
            f"settles in {trade.settlement_ts - time.time():.0f}s"
        )

    async def settlement_loop(self, get_btc_price_fn) -> None:
        """
        Background coroutine that checks settled trades and records outcomes.
        get_btc_price_fn: callable → float (current BTC price)
        """
        while True:
            now = time.time()
            still_pending = []

            for trade in self._pending:
                if trade.settled:
                    continue
                # Snapshot price at the moment of settlement
                if trade._exit_snapshot == 0.0 and now >= trade.settlement_ts:
                    trade._exit_snapshot = get_btc_price_fn()

                # 5s grace period then resolve
                if now < trade.settlement_ts + 5:
                    still_pending.append(trade)
                    continue

                # Resolve outcome using snapshot price
                current_price = trade._exit_snapshot or get_btc_price_fn()
                btc_went_up   = current_price > trade.entry_price_btc
                trade.settled = True
                trade.btc_went_up = btc_went_up

                outcome_str = "UP" if btc_went_up else "DOWN"
                correct = (trade.direction == "YES") == btc_went_up
                log.info(
                    f"Settlement: {trade.market_id} | BTC {outcome_str} | "
                    f"{'CORRECT' if correct else 'WRONG'} | "
                    f"${trade.usdc_staked:.2f} staked"
                )

                # Note: add_outcome is handled by PaperTradingLog which covers
                # all decisions. OutcomeTracker only does outcome logging here.

            self._pending = still_pending

            # Log importance when sample count crosses a new multiple of 100
            n = self._ensemble.n_samples
            if n > 0 and n % 100 == 0 and n != self._last_logged_n:
                self._ensemble.log_importance(SIGNAL_NAMES)
                self._last_logged_n = n

            await asyncio.sleep(5)

    @property
    def pending_count(self) -> int:
        return len([t for t in self._pending if not t.settled])
