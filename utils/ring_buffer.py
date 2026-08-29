from collections import deque
import numpy as np
import time


class TimeSeriesBuffer:
    """Circular buffer keyed by timestamp for time-windowed signal computation."""

    def __init__(self, maxlen: int = 10_000):
        self._times: deque[float] = deque(maxlen=maxlen)
        self._values: deque[float] = deque(maxlen=maxlen)

    def append(self, value: float, ts: float | None = None) -> None:
        self._times.append(ts if ts is not None else time.time())
        self._values.append(value)

    def window(self, seconds: float) -> np.ndarray:
        if not self._times:
            return np.array([], dtype=np.float64)
        # Use the most-recent buffer timestamp as reference, not time.time().
        # Avoids empty windows caused by Windows clock drift vs Binance exchange time.
        cutoff = self._times[-1] - seconds
        vals = [v for t, v in zip(self._times, self._values) if t >= cutoff]
        return np.array(vals, dtype=np.float64)

    def last(self, n: int = 1) -> np.ndarray:
        vals = list(self._values)[-n:]
        return np.array(vals, dtype=np.float64)

    def last_value(self) -> float | None:
        return self._values[-1] if self._values else None

    def __len__(self) -> int:
        return len(self._values)


class OrderBookSnapshot:
    """Stores the latest L2 order book state."""

    def __init__(self):
        self.bids: list[tuple[float, float]] = []  # (price, qty)
        self.asks: list[tuple[float, float]] = []
        self.ts: float = 0.0

    def update(self, bids: list, asks: list) -> None:
        self.bids = [(float(p), float(q)) for p, q in bids]
        self.asks = [(float(p), float(q)) for p, q in asks]
        self.ts = time.time()

    def mid_price(self) -> float | None:
        if not self.bids or not self.asks:
            return None
        return (self.bids[0][0] + self.asks[0][0]) / 2.0

    def spread(self) -> float | None:
        if not self.bids or not self.asks:
            return None
        return self.asks[0][0] - self.bids[0][0]

    def bid_volume(self, levels: int = 5) -> float:
        return sum(q for _, q in self.bids[:levels])

    def ask_volume(self, levels: int = 5) -> float:
        return sum(q for _, q in self.asks[:levels])


class LiquidationBuffer:
    """Tracks real-time BTC liquidation events (from Binance forceOrder stream) by side."""

    def __init__(self, maxlen: int = 5_000):
        self._times    = deque(maxlen=maxlen)
        self._buy_usd  = deque(maxlen=maxlen)   # short liquidations → forced buy
        self._sell_usd = deque(maxlen=maxlen)   # long liquidations → forced sell

    def append(self, side: str, usd_value: float, ts: float | None = None) -> None:
        t = ts if ts is not None else time.time()
        self._times.append(t)
        if side == "BUY":
            self._buy_usd.append(usd_value)
            self._sell_usd.append(0.0)
        else:
            self._buy_usd.append(0.0)
            self._sell_usd.append(usd_value)

    def window(self, seconds: float) -> tuple[float, float]:
        """Returns (buy_vol_usd, sell_vol_usd) in the last `seconds`."""
        if not self._times:
            return 0.0, 0.0
        cutoff = self._times[-1] - seconds
        buy  = sum(v for t, v in zip(self._times, self._buy_usd)  if t >= cutoff)
        sell = sum(v for t, v in zip(self._times, self._sell_usd) if t >= cutoff)
        return buy, sell


class TradeFlowBuffer:
    """Tracks signed trade volume (buy = positive, sell = negative)."""

    def __init__(self, maxlen: int = 5_000):
        self._times: deque[float] = deque(maxlen=maxlen)
        self._signed_vol: deque[float] = deque(maxlen=maxlen)

    def append(self, qty: float, is_buyer_maker: bool, ts: float | None = None) -> None:
        signed = -qty if is_buyer_maker else qty  # buyer_maker=True means sell aggression
        self._times.append(ts if ts is not None else time.time())
        self._signed_vol.append(signed)

    def imbalance(self, seconds: float) -> float:
        if not self._times:
            return 0.0
        cutoff = self._times[-1] - seconds
        vols = [v for t, v in zip(self._times, self._signed_vol) if t >= cutoff]
        if not vols:
            return 0.0
        total = sum(abs(v) for v in vols)
        return sum(vols) / total if total > 0 else 0.0
