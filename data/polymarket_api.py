"""
Polymarket CLOB API client.
Handles market discovery, order book polling, and trade execution.
"""

import asyncio
import time
import requests
from dataclasses import dataclass
from typing import Optional
from utils.logger import get_logger
from config import (
    POLYMARKET_CLOB_URL, POLYMARKET_GAMMA_URL,
    PM_API_KEY, PM_API_SECRET, PM_API_PASSPHRASE,
    PM_PRIVATE_KEY, PM_CHAIN_ID, DRY_RUN, LIVE_MAX_ORDER_USDC,
)

log = get_logger("polymarket_api")


@dataclass
class PolymarketMarket:
    condition_id: str
    question: str
    end_date_iso: str
    end_ts: float            # Unix timestamp of settlement
    token_id_yes: str        # CLOB token for YES outcome
    token_id_no: str         # CLOB token for NO outcome
    interval: str            # "5m" or "15m"
    best_bid: float = 0.0    # Gamma API best bid (YES token); 0 = no real quote
    best_ask: float = 0.0    # Gamma API best ask (YES token); 0 = no real quote
    window_step: int = 300   # seconds in this market's window (300=5m, 900=15m)
    active: bool = True


@dataclass
class OrderBookState:
    yes_bid: float = 0.0
    yes_ask: float = 0.0
    no_bid: float  = 0.0
    no_ask: float  = 0.0
    yes_bid_size: float = 0.0
    yes_ask_size: float = 0.0
    ts: float = 0.0
    is_clob: bool = False   # True if a live CLOB book was found; False = Gamma seed only

    def yes_implied_prob(self) -> float:
        if self.yes_bid and self.yes_ask:
            return (self.yes_bid + self.yes_ask) / 2.0
        return 0.5

    def spread_yes(self) -> float:
        return self.yes_ask - self.yes_bid


class PolymarketClient:

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "POLY_API_KEY":        PM_API_KEY,
            "POLY_API_SECRET":     PM_API_SECRET,
            "POLY_API_PASSPHRASE": PM_API_PASSPHRASE,
        })
        self._clob_client = None
        self._init_clob_client()

    def _init_clob_client(self) -> None:
        if not PM_PRIVATE_KEY or DRY_RUN:
            log.info("DRY_RUN or no private key — trade execution disabled.")
            return
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds
            creds = ApiCreds(
                api_key=PM_API_KEY,
                api_secret=PM_API_SECRET,
                api_passphrase=PM_API_PASSPHRASE,
            )
            self._clob_client = ClobClient(
                host=POLYMARKET_CLOB_URL,
                chain_id=PM_CHAIN_ID,
                private_key=PM_PRIVATE_KEY,
                creds=creds,
            )
            log.info("CLOB client initialized.")
        except ImportError:
            log.warning("py-clob-client not installed — trade execution disabled.")
        except Exception as e:
            log.error(f"CLOB client init failed: {e}")

    # ── Market Discovery ───────────────────────────────────────────────────────

    def _generate_slugs(self, asset: str = "btc") -> list[tuple[str, str]]:
        """
        Generate upcoming up/down market slugs from current time.
        Pattern: {asset}-updown-{interval}-{unix_settlement_ts}
        Returns list of (slug, interval) tuples for the next ~10 windows.
        """
        now    = int(time.time())
        slugs  = []

        for interval, step in [("5m", 300), ("15m", 900)]:
            # Include current window + next 8 (executor handles timing cutoff)
            current_boundary = (now // step) * step + step
            for i in range(9):
                ts   = current_boundary + i * step
                slug = f"{asset}-updown-{interval}-{ts}"
                slugs.append((slug, interval))

        return slugs

    def fetch_active_markets(self, asset: str = "btc") -> list[PolymarketMarket]:
        """Discover upcoming up/down markets for any asset (btc, eth, sol…)."""
        import json as _json

        markets  = []
        slugs    = self._generate_slugs(asset)
        seen_ids: set[str] = set()

        for slug, interval in slugs:
            try:
                resp = self._session.get(
                    f"{POLYMARKET_GAMMA_URL}/markets",
                    params={"slug": slug},
                    timeout=8,
                )
                if resp.status_code == 404:
                    continue
                resp.raise_for_status()

                data  = resp.json()
                items = data if isinstance(data, list) else [data]

                for mkt in items:
                    if not isinstance(mkt, dict):
                        continue

                    raw_tokens = mkt.get("clobTokenIds", [])
                    if isinstance(raw_tokens, str):
                        try:
                            raw_tokens = _json.loads(raw_tokens)
                        except Exception:
                            raw_tokens = []

                    yes_token, no_token = None, None
                    if isinstance(raw_tokens, list) and len(raw_tokens) >= 2:
                        if isinstance(raw_tokens[0], dict):
                            yes_token = next(
                                (t.get("token_id") for t in raw_tokens
                                 if "yes" in str(t.get("outcome", "")).lower()), None
                            )
                            no_token = next(
                                (t.get("token_id") for t in raw_tokens
                                 if "no" in str(t.get("outcome", "")).lower()), None
                            )
                        else:
                            yes_token = str(raw_tokens[0])
                            no_token  = str(raw_tokens[1])

                    if not yes_token or not no_token:
                        continue

                    end_iso = (
                        mkt.get("endDate") or mkt.get("end_date_iso") or
                        mkt.get("endDateIso") or ""
                    )
                    end_ts = self._parse_end_ts(end_iso)
                    if not end_ts:
                        slug_ts = slug.rsplit("-", 1)[-1]
                        end_ts  = float(slug_ts) if slug_ts.isdigit() else 0.0

                    if end_ts and end_ts < time.time():
                        continue

                    condition_id = (
                        mkt.get("conditionId") or mkt.get("condition_id") or slug
                    )
                    if condition_id in seen_ids:
                        continue
                    seen_ids.add(condition_id)

                    step = 900 if interval == "15m" else 300
                    markets.append(PolymarketMarket(
                        condition_id=condition_id,
                        question=mkt.get("question", slug),
                        end_date_iso=end_iso,
                        end_ts=end_ts,
                        token_id_yes=yes_token,
                        token_id_no=no_token,
                        interval=interval,
                        best_bid=float(mkt.get("bestBid") or 0.0),
                        best_ask=float(mkt.get("bestAsk") or 0.0),
                        window_step=step,
                    ))
            except Exception as e:
                log.debug(f"Slug {slug}: {e}")

        return markets

    def fetch_active_btc_markets(self) -> list[PolymarketMarket]:
        """Backward-compatible alias — use fetch_active_markets('btc') directly."""
        return self.fetch_active_markets("btc")

    def _detect_interval(self, question: str) -> Optional[str]:
        if "15m" in question or "15-m" in question:
            return "15m"
        if "5m" in question or "5-m" in question:
            return "5m"
        return None

    def _parse_end_ts(self, iso: str) -> float:
        if not iso:
            return 0.0
        try:
            from datetime import datetime, timezone
            iso_clean = iso.replace("Z", "+00:00")
            dt = datetime.fromisoformat(iso_clean)
            return dt.timestamp()
        except Exception:
            return 0.0

    # ── Order Book ─────────────────────────────────────────────────────────────

    def get_order_book(
        self,
        token_id_yes: str,
        token_id_no: str,
        fallback_bid: float = 0.0,
        fallback_ask: float = 0.0,
    ) -> OrderBookState:
        state = OrderBookState(ts=time.time())

        # Seed with Gamma prices — always available and reliable
        if fallback_bid > 0 and fallback_ask > 0:
            state.yes_bid = fallback_bid
            state.yes_ask = fallback_ask
            state.no_bid  = round(1.0 - fallback_ask, 6)
            state.no_ask  = round(1.0 - fallback_bid, 6)

        # Try CLOB — use live prices when orders exist, keep Gamma fallback otherwise.
        # Gamma fallback is still tradeable via Polymarket's instant-buy at T-10s
        # even when the CLOB book has no resting limit orders.
        try:
            resp = self._session.get(
                f"{POLYMARKET_CLOB_URL}/book",
                params={"token_id": token_id_yes},
                timeout=5,
            )
            resp.raise_for_status()
            book  = resp.json()
            bids  = book.get("bids", [])
            asks  = book.get("asks", [])
            if bids and asks:
                # CLOB returns bids ascending and asks descending, so the best
                # bid is the highest-priced bid and the best ask is the
                # lowest-priced ask — never index [0] blindly.
                best_bid = max(bids, key=lambda b: float(b["price"]))
                best_ask = min(asks, key=lambda a: float(a["price"]))
                clob_bid = float(best_bid["price"])
                clob_ask = float(best_ask["price"])
                state.yes_bid      = clob_bid
                state.yes_ask      = clob_ask
                state.yes_bid_size = float(best_bid["size"])
                state.yes_ask_size = float(best_ask["size"])
                state.no_bid       = round(1.0 - clob_ask, 6)
                state.no_ask       = round(1.0 - clob_bid, 6)
                state.is_clob      = True   # real resting orders, not just the Gamma seed
        except Exception as e:
            log.debug(f"CLOB order book unavailable for {token_id_yes[:12]}...: {e}")

        # Skip only if market has fully resolved (nothing left to trade).
        if state.yes_ask <= 0.05 or state.yes_ask >= 0.95:
            log.debug(
                f"Market effectively resolved (yes_ask={state.yes_ask:.3f}) — skipping."
            )
            state.yes_bid = state.yes_ask = state.no_bid = state.no_ask = 0.0

        return state

    def get_pm_obi(self, token_id_yes: str) -> float:
        """Order book imbalance on Polymarket YES token: in [-1, 1]."""
        try:
            resp = self._session.get(
                f"{POLYMARKET_CLOB_URL}/book",
                params={"token_id": token_id_yes},
                timeout=5,
            )
            resp.raise_for_status()
            book = resp.json()
            # Top-of-book = highest bids and lowest asks (CLOB lists them at the
            # far ends, not index 0), so sort before taking the nearest 5 levels.
            top_bids = sorted(book.get("bids", []), key=lambda b: float(b["price"]), reverse=True)[:5]
            top_asks = sorted(book.get("asks", []), key=lambda a: float(a["price"]))[:5]
            bid_vol = sum(float(b["size"]) for b in top_bids)
            ask_vol = sum(float(a["size"]) for a in top_asks)
            total = bid_vol + ask_vol
            return (bid_vol - ask_vol) / total if total > 0 else 0.0
        except Exception:
            return 0.0

    # ── Trade Execution ────────────────────────────────────────────────────────

    def place_market_order(
        self,
        token_id: str,
        side: str,           # "BUY" or "SELL"
        amount_usdc: float,
        dry_run: bool = True,
    ) -> dict:
        if dry_run or DRY_RUN:
            log.info(f"[DRY RUN] {side} ${amount_usdc:.2f} on token {token_id[:12]}...")
            return {"status": "dry_run", "token_id": token_id, "side": side, "amount": amount_usdc}

        # Hard safety cap: never submit a real order above LIVE_MAX_ORDER_USDC,
        # regardless of what sizing requested. Last line of defense.
        if amount_usdc > LIVE_MAX_ORDER_USDC:
            log.error(f"BLOCKED: order ${amount_usdc:.2f} exceeds LIVE_MAX_ORDER_USDC "
                      f"${LIVE_MAX_ORDER_USDC:.2f} — not submitting.")
            return {"status": "error", "reason": "exceeds_live_cap"}

        if not self._clob_client:
            log.error("CLOB client not initialized — cannot place order.")
            return {"status": "error", "reason": "no_client"}

        try:
            from py_clob_client.clob_types import MarketOrderArgs, BUY, SELL
            order_args = MarketOrderArgs(
                token_id=token_id,
                amount=amount_usdc,
            )
            signed = self._clob_client.create_market_order(order_args)
            resp   = self._clob_client.post_order(signed)
            log.info(f"Order placed: {resp}")
            return {"status": "ok", "response": resp}
        except Exception as e:
            log.error(f"Order placement failed: {e}")
            return {"status": "error", "reason": str(e)}

    def place_limit_order(
        self,
        token_id: str,
        side: str,            # "BUY" or "SELL"
        amount_usdc: float,
        limit_price: float,   # price per share in (0, 1)
        dry_run: bool = True,
    ) -> dict:
        """Post a resting GTC limit order (maker). size = amount_usdc / limit_price."""
        if limit_price <= 0 or limit_price >= 1:
            return {"status": "error", "reason": f"bad_limit_price {limit_price}"}

        size = round(amount_usdc / limit_price, 2)
        if dry_run or DRY_RUN:
            log.info(
                f"[DRY RUN] LIMIT {side} {size} @ {limit_price:.3f} "
                f"(${amount_usdc:.2f}) on token {token_id[:12]}..."
            )
            return {"status": "dry_run", "token_id": token_id, "side": side,
                    "price": limit_price, "size": size}

        # Hard safety cap (last line of defense), same as the market path.
        if amount_usdc > LIVE_MAX_ORDER_USDC:
            log.error(f"BLOCKED: limit order ${amount_usdc:.2f} exceeds "
                      f"LIVE_MAX_ORDER_USDC ${LIVE_MAX_ORDER_USDC:.2f} — not submitting.")
            return {"status": "error", "reason": "exceeds_live_cap"}

        if not self._clob_client:
            log.error("CLOB client not initialized — cannot place order.")
            return {"status": "error", "reason": "no_client"}

        try:
            from py_clob_client.clob_types import OrderArgs, OrderType, BUY, SELL
            order_args = OrderArgs(
                token_id=token_id,
                price=limit_price,
                size=size,
                side=BUY if side == "BUY" else SELL,
            )
            signed = self._clob_client.create_order(order_args)
            resp   = self._clob_client.post_order(signed, OrderType.GTC)
            log.info(f"Limit order placed: {resp}")
            return {"status": "ok", "response": resp}
        except Exception as e:
            log.error(f"Limit order placement failed: {e}")
            return {"status": "error", "reason": str(e)}
