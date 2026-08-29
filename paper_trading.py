"""
Paper trading log and outcome tracker.
Writes every decision and outcome to papertrade_log.csv.
Feeds settled outcomes into the ensemble for online learning.
"""

import asyncio
import csv
import time
from collections import deque
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional
import numpy as np

from model.ensemble import SignalEnsemble
from signals.aggregator import SIGNAL_NAMES
from config import (
    STREAK_PAUSE_WR, STREAK_RESUME_WR, STREAK_WINDOW,
    EARLY_EXIT_ENABLED, EARLY_EXIT_PRICE, EARLY_EXIT_INTERVAL,
)
from utils.logger import get_logger

log = get_logger("paper_trading")

LOG_DIR = Path(__file__).parent

FIELDNAMES = [
    "timestamp", "market_id", "interval", "question",
    "direction", "model_prob", "ev_fraction", "stake_usdc",
    "market_price", "btc_price_entry", "btc_price_exit",
    "outcome",       # "WIN" / "LOSS" / "PENDING"
    "correct",       # 1 / 0 / ""
    "train_count",   # model retrain count at time of trade
    "order_placed",  # 1 if order was actually sent, 0 if blocked/filtered
    "block_reason",  # why no order was placed (confluence/circuit_breaker/etc), "" if placed
    "realized_pnl",  # exact bankroll delta booked for this trade ("" if no order)
    "exit_price",    # close price: settlement 1.0/0.0, or the early-exit sell price
    "book_src",      # "clob" = real resting book at entry, "gamma" = seed fallback
    "book_spread",   # yes_ask - yes_bid at entry (tight = real liquid book)
]


@dataclass
class PaperTrade:
    timestamp:       float
    market_id:       str
    interval:        str
    question:        str
    direction:       str        # "YES" or "NO"
    model_prob:      float
    ev_fraction:     float
    stake_usdc:      float
    market_price:    float
    btc_price_entry: float
    btc_price_exit:  float = 0.0
    outcome:         str = "PENDING"
    correct:         int = -1
    train_count:     int = 0
    order_placed:    bool = False
    block_reason:    str = ""
    # Exact bankroll delta booked at settlement/early-exit, and the close price
    # (1.0/0.0 at settlement, or the sell price for an early exit). Make the CSV a
    # self-contained ledger so a bankroll audit needs no log-matching.
    realized_pnl:    float | None = None
    exit_price:      float | None = None
    # Entry book provenance: whether a real CLOB book backed the price, and its
    # spread. Lets us measure how often we trade on the Gamma seed vs a live book.
    book_src:        str = ""
    book_spread:     float | None = None
    signal_vec:      Optional[np.ndarray] = field(default=None, repr=False)
    settlement_ts:   float = 0.0
    # Chainlink Data Streams window-open price (settlement-truth source). 0.0 when
    # the Data Streams client is disabled or had no price yet at window open.
    chainlink_price_entry: float = 0.0
    _exit_snapshot:    float = field(default=0.0, repr=False, compare=False)
    _cl_exit_snapshot: float = field(default=0.0, repr=False, compare=False)
    # Polymarket market handle (token ids + fallback prices) for early-exit book
    # polling. Not serialized. None for blocked/unplaced decisions.
    market:            object = field(default=None, repr=False, compare=False)
    # Set once the position is sold early; the settlement pass then skips re-booking
    # bankroll/exposure (already realized) but still trains on the true outcome.
    _early_exited:     bool = field(default=False, repr=False, compare=False)


class PaperTradingLog:

    def __init__(self, ensemble: SignalEnsemble, name: str = "ensemble"):
        self._ensemble      = ensemble
        self._name          = name
        # name is "5m"/"15m" (btc) or "5m_eth"/"15m_eth" — asset is the suffix.
        parts = name.split("_", 1)
        self._asset         = parts[1] if len(parts) > 1 else "btc"
        self._log_path      = LOG_DIR / f"papertrade_log_{name}.csv"
        self._pending:  list[PaperTrade] = []
        self._settled:  list[PaperTrade] = []
        self._recent_wins:  deque[bool]  = deque(maxlen=20)
        self._paused:   bool = False   # circuit breaker state
        self._stat_tick: int = 0
        self._ensure_csv()
        self._migrate_csv()

    def _ensure_csv(self) -> None:
        if not self._log_path.exists():
            with open(self._log_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    def _migrate_csv(self) -> None:
        """Upgrade a pre-existing CSV to the current header (adds new columns as
        blanks) so appends/rewrites stay aligned. Idempotent — skips if already
        current. Historical rows keep blank realized_pnl/exit_price; the audit
        script reconstructs those from the logs, and all new rows are exact."""
        try:
            with open(self._log_path, "r", newline="") as f:
                header = next(csv.reader(f), None)
            if header is None:
                return
            missing = [c for c in FIELDNAMES if c not in header]
            if not missing:
                return
            with open(self._log_path, "r", newline="") as f:
                rows = list(csv.DictReader(f))
            with open(self._log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
                writer.writeheader()
                for row in rows:
                    for c in missing:
                        row.setdefault(c, "")
                    writer.writerow(row)
            log.info(f"[{self._name}] Migrated CSV — added columns: {', '.join(missing)}.")
        except Exception as e:
            log.warning(f"[{self._name}] CSV migration skipped: {e}")

    def record_decision(self, trade: PaperTrade) -> None:
        """Call this when a paper trade decision is made (T-10s)."""
        self._pending.append(trade)
        self._write_row(trade)
        log.info(
            f"[PAPER] {trade.interval} | {trade.direction} | "
            f"prob={trade.model_prob:.3f} | EV={trade.ev_fraction:.3f} | "
            f"${trade.stake_usdc:.2f}"
        )

    def _write_row(self, trade: PaperTrade) -> None:
        row = {
            "timestamp":       trade.timestamp,
            "market_id":       trade.market_id[:20],
            "interval":        trade.interval,
            "question":        trade.question[:60],
            "direction":       trade.direction,
            "model_prob":      round(trade.model_prob, 4),
            "ev_fraction":     round(trade.ev_fraction, 4),
            "stake_usdc":      trade.stake_usdc,
            "market_price":    round(trade.market_price, 4),
            "btc_price_entry": round(trade.btc_price_entry, 2),
            "btc_price_exit":  round(trade.btc_price_exit, 2),
            "outcome":         trade.outcome,
            "correct":         trade.correct if trade.correct >= 0 else "",
            "train_count":     trade.train_count,
            "order_placed":    1 if trade.order_placed else 0,
            "block_reason":    trade.block_reason,
            "realized_pnl":    "" if trade.realized_pnl is None else round(trade.realized_pnl, 4),
            "exit_price":      "" if trade.exit_price is None else round(trade.exit_price, 4),
            "book_src":        trade.book_src,
            "book_spread":     "" if trade.book_spread is None else round(trade.book_spread, 4),
        }
        with open(self._log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writerow(row)

    def _update_row(self, trade: PaperTrade) -> None:
        try:
            rows = []
            with open(self._log_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["market_id"] == trade.market_id[:20] and row["outcome"] == "PENDING":
                        row["btc_price_exit"] = round(trade.btc_price_exit, 2)
                        row["outcome"]        = trade.outcome
                        row["correct"]        = trade.correct if trade.correct >= 0 else ""
                        # order_placed is only known after the order step, which
                        # runs after the decision row was first written.
                        row["order_placed"]   = 1 if trade.order_placed else 0
                        if trade.realized_pnl is not None:
                            row["realized_pnl"] = round(trade.realized_pnl, 4)
                        if trade.exit_price is not None:
                            row["exit_price"] = round(trade.exit_price, 4)
                    rows.append(row)
            with open(self._log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
        except Exception as e:
            log.warning(f"Could not update CSV row: {e}")

    def _try_early_exit(self, trade: PaperTrade, pm_client) -> None:
        """If the open position's own side can be sold at/above EARLY_EXIT_PRICE,
        realize the win now and release its capital. Trains on true outcome later."""
        try:
            mk = trade.market
            book = pm_client.get_order_book(
                mk.token_id_yes, mk.token_id_no,
                fallback_bid=mk.best_bid, fallback_ask=mk.best_ask,
            )
            # Price we could SELL our held side at (hit the bid).
            our_bid = book.yes_bid if trade.direction == "YES" else book.no_bid
            if our_bid <= 0 or our_bid < EARLY_EXIT_PRICE:
                return

            import bankroll_state, exposure, correlation
            trade.realized_pnl = bankroll_state.apply_exit(
                trade.stake_usdc, trade.market_price, our_bid
            )
            trade.exit_price = our_bid
            exposure.release(trade.stake_usdc)
            correlation.release(trade.market_id)
            trade._early_exited = True
            # Persist the early-exit P&L/price to the row now (settlement later
            # skips re-booking). market_id makes the event unambiguously matchable.
            self._update_row(trade)
            log.info(
                f"[EARLY EXIT] {trade.market_id[:20]} | {trade.interval} | "
                f"{trade.direction} | sold @ {our_bid:.3f} "
                f"(entry {trade.market_price:.3f}) | ${trade.stake_usdc:.2f} staked"
            )
        except Exception as e:
            log.debug(f"Early-exit check failed: {e}")

    async def settlement_loop(
        self,
        get_btc_price_fn,
        *,
        chainlink_price_fn=None,
        settle_source: str = "binance",
        pm_client=None,
    ) -> None:
        """
        Background loop: resolve pending trades after settlement,
        update CSV, and feed outcomes to the ensemble for training.

        get_btc_price_fn   — Binance futures mid (always passed).
        chainlink_price_fn — Chainlink Data Streams price, or None when disabled.
        settle_source      — "chainlink" trains the model on the Chainlink-settled
                             label (entry+exit both Chainlink); "binance" keeps the
                             Binance label. The other source is always logged to the
                             shadow CSV for divergence measurement.
        pm_client          — Polymarket client for early-exit book polling. None
                             disables early exit (e.g. the OutcomeTracker loops).
        """
        use_cl = chainlink_price_fn is not None
        while True:
            now          = time.time()
            still_pending = []

            for trade in self._pending:
                # ── Early exit: sell a near-certain open 15m position before it
                # resolves, to lock in the win and cut hold-to-settlement variance.
                # The trade stays pending so the model still trains on the true
                # outcome; only the bankroll P&L is realized here.
                if (EARLY_EXIT_ENABLED and pm_client is not None
                        and not trade._early_exited
                        and trade.order_placed and trade.stake_usdc > 0
                        and trade.interval == EARLY_EXIT_INTERVAL
                        and trade.market is not None
                        and now < trade.settlement_ts):
                    self._try_early_exit(trade, pm_client)

                # Snapshot price at the moment of settlement (not 15s later)
                if trade._exit_snapshot == 0.0 and now >= trade.settlement_ts:
                    trade._exit_snapshot = get_btc_price_fn()
                    if use_cl and trade._cl_exit_snapshot == 0.0:
                        trade._cl_exit_snapshot = chainlink_price_fn()

                # Give 5s grace for snapshot to land, then resolve
                if now < trade.settlement_ts + 5:
                    still_pending.append(trade)
                    continue

                # Resolve. Pick a consistent (entry, exit) pair from one source so
                # the trained label can't be corrupted by mixing price sources.
                binance_exit = trade._exit_snapshot or get_btc_price_fn()
                cl_exit      = trade._cl_exit_snapshot or (chainlink_price_fn() if use_cl else 0.0)

                if settle_source == "chainlink" and trade.chainlink_price_entry and cl_exit:
                    exit_price  = cl_exit
                    btc_went_up = cl_exit > trade.chainlink_price_entry
                else:
                    exit_price  = binance_exit
                    btc_went_up = binance_exit > trade.btc_price_entry

                won = (
                    (trade.direction == "YES" and btc_went_up) or
                    (trade.direction == "NO"  and not btc_went_up)
                )

                # Feed the true up/down to the early-entry shadow (no-op if this
                # market wasn't shadowed, e.g. 5m).
                try:
                    import shadow_entry
                    shadow_entry.settle(trade.market_id, btc_went_up)
                except Exception:
                    pass

                # Shadow: record both sources + whether the label would flip.
                if use_cl:
                    import chainlink_shadow
                    chainlink_shadow.record(
                        asset=self._asset,
                        interval=trade.interval,
                        direction=trade.direction,
                        binance_entry=trade.btc_price_entry,
                        binance_exit=binance_exit,
                        chainlink_entry=trade.chainlink_price_entry,
                        chainlink_exit=cl_exit,
                    )

                trade.btc_price_exit = exit_price
                trade.outcome        = "WIN" if won else "LOSS"
                trade.correct        = 1 if won else 0

                # Update running bankroll for drawdown-brake sizing, BEFORE writing
                # the row so the exact realized P&L is persisted in the CSV.
                # Only for placed orders; skip if already realized at early exit
                # (bankroll/exposure/correlation were booked when we sold).
                if trade.order_placed and trade.stake_usdc > 0 and not trade._early_exited:
                    import bankroll_state
                    trade.realized_pnl = bankroll_state.update(
                        trade.stake_usdc, trade.market_price, won
                    )
                    trade.exit_price = 1.0 if won else 0.0
                    import exposure
                    exposure.release(trade.stake_usdc)
                    import correlation
                    correlation.release(trade.market_id)

                self._settled.append(trade)
                # Only track active trades (EV>0) in the circuit breaker window
                if trade.ev_fraction > 0:
                    self._recent_wins.append(won)
                    self._update_circuit_breaker()
                self._update_row(trade)

                log.info(
                    f"[PAPER SETTLED] {trade.interval} | {trade.direction} | "
                    f"{trade.outcome} | BTC {trade.btc_price_entry:.0f}→{exit_price:.0f}"
                )

                # Feed to ensemble
                if trade.signal_vec is not None:
                    self._ensemble.add_outcome(trade.signal_vec, btc_went_up)

            self._pending = still_pending
            self._stat_tick += 1
            if self._stat_tick % 6 == 0:   # every ~60s
                self._log_stats()
            await asyncio.sleep(10)

    def _log_stats(self) -> None:
        if not self._settled:
            return
        total  = len(self._settled)
        wins   = sum(1 for t in self._settled if t.correct == 1)
        recent = self._settled[-20:]
        r_wins = sum(1 for t in recent if t.correct == 1)
        log.info(
            f"[PAPER STATS] Total={total} | "
            f"Overall={wins/total:.1%} | "
            f"Last20={r_wins/len(recent):.1%} | "
            f"Model_retrains={self._ensemble._train_count}"
        )

    def _update_circuit_breaker(self) -> None:
        if len(self._recent_wins) < STREAK_WINDOW:
            return
        recent_wr = sum(list(self._recent_wins)[-STREAK_WINDOW:]) / STREAK_WINDOW
        if not self._paused and recent_wr < STREAK_PAUSE_WR:
            self._paused = True
            log.warning(
                f"[CIRCUIT BREAKER] Pausing bets — recent win rate {recent_wr:.0%} "
                f"< {STREAK_PAUSE_WR:.0%} threshold. Logging only until recovery."
            )
        elif self._paused and recent_wr >= STREAK_RESUME_WR:
            self._paused = False
            log.info(
                f"[CIRCUIT BREAKER] Resuming bets — win rate recovered to {recent_wr:.0%}."
            )

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def total_trades(self) -> int:
        return len(self._settled)

    @property
    def win_rate(self) -> float:
        if not self._settled:
            return 0.0
        return sum(1 for t in self._settled if t.correct == 1) / len(self._settled)

    @property
    def recent_win_rate(self) -> float:
        """Rolling win rate over last 20 settled trades. Returns 0.5 until 10 trades settled."""
        if len(self._recent_wins) < 10:
            return 0.5
        return sum(self._recent_wins) / len(self._recent_wins)
