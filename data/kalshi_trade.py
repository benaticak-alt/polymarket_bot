"""
Kalshi authenticated trading client.
Handles login, token refresh, and order placement.

Requires KALSHI_EMAIL and KALSHI_PASSWORD in .env
(or KALSHI_API_KEY for key-based auth if Kalshi enables it for your account).
"""

import os, time, uuid, requests
from utils.logger import get_logger

log = get_logger("kalshi_trade")

KALSHI_API    = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_EMAIL  = os.getenv("KALSHI_EMAIL", "")
KALSHI_PASS   = os.getenv("KALSHI_PASSWORD", "")
TOKEN_TTL     = 3600   # re-login after 1 hour


class KalshiTradeClient:
    """
    Authenticated Kalshi client for placing market orders.
    Thread-safe for use inside asyncio via run_in_executor.
    """

    def __init__(self):
        self._session   = requests.Session()
        self._session.headers.update({
            "User-Agent": "polymarket-bot/1.0",
            "Accept":     "application/json",
            "Content-Type": "application/json",
        })
        self._token     = ""
        self._token_ts  = 0.0
        self._enabled   = bool(KALSHI_EMAIL and KALSHI_PASS)

        if not self._enabled:
            log.warning(
                "KALSHI_EMAIL / KALSHI_PASSWORD not set — "
                "Kalshi trading disabled. Add to .env to enable."
            )

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    # ── Auth ───────────────────────────────────────────────────────────────────

    def _ensure_auth(self) -> bool:
        if not self._enabled:
            return False
        if self._token and (time.time() - self._token_ts) < TOKEN_TTL:
            return True
        return self._login()

    def _login(self) -> bool:
        try:
            resp = self._session.post(
                f"{KALSHI_API}/log_in",
                json={"email": KALSHI_EMAIL, "password": KALSHI_PASS},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                token = data.get("token") or data.get("access_token", "")
                if token:
                    self._token    = token
                    self._token_ts = time.time()
                    self._session.headers["Authorization"] = f"Bearer {token}"
                    log.info("Kalshi: logged in successfully.")
                    return True
            log.warning(f"Kalshi login failed: {resp.status_code} {resp.text[:200]}")
            return False
        except Exception as e:
            log.warning(f"Kalshi login error: {e}")
            return False

    # ── Order placement ────────────────────────────────────────────────────────

    def place_market_order(
        self,
        ticker: str,
        side: str,          # "yes" or "no"
        amount_dollars: float,
        yes_price: float,   # current ask price (0.0–1.0), used to compute count
        dry_run: bool = True,
    ) -> dict:
        """
        Place a market buy order on Kalshi.

        ticker:         Market ticker e.g. "KXRECSSNBER-2026DEC-T"
        side:           "yes" or "no"
        amount_dollars: How many dollars to spend (approximate)
        yes_price:      Current ask price for YES (to estimate contract count)
        dry_run:        If True, simulate only — no real order placed

        Returns dict with 'status' key: 'ok', 'dry_run', or 'error'.
        """
        if amount_dollars < 1.0:
            return {"status": "error", "reason": "amount_too_small"}

        # Compute contract count: each contract pays $1 at settlement
        price = yes_price if side == "yes" else (1.0 - yes_price)
        price = max(0.01, min(0.99, price))
        count = max(1, int(amount_dollars / price))

        if dry_run:
            log.info(
                f"[KALSHI DRY] Would BUY {count} {side.upper()} contracts "
                f"on {ticker} @ ${price:.2f} (~${count * price:.2f})"
            )
            return {
                "status":  "dry_run",
                "ticker":  ticker,
                "side":    side,
                "count":   count,
                "est_cost": round(count * price, 2),
            }

        if not self._ensure_auth():
            return {"status": "error", "reason": "auth_failed"}

        order_id = str(uuid.uuid4())
        payload  = {
            "action":           "buy",
            "client_order_id":  order_id,
            "count":            count,
            "side":             side,
            "ticker":           ticker,
            "type":             "market",
        }

        try:
            resp = self._session.post(
                f"{KALSHI_API}/portfolio/orders",
                json=payload,
                timeout=10,
            )
            if resp.status_code in (200, 201):
                data = resp.json()
                order = data.get("order", data)
                log.info(
                    f"[KALSHI] Placed {side.upper()} order on {ticker}: "
                    f"{count} contracts @ ${price:.2f} "
                    f"| order_id={order.get('id', order_id)}"
                )
                return {"status": "ok", "order": order, "ticker": ticker, "side": side}
            else:
                log.error(f"[KALSHI] Order failed {resp.status_code}: {resp.text[:300]}")
                return {"status": "error", "reason": resp.text[:200]}
        except Exception as e:
            log.error(f"[KALSHI] Order exception: {e}")
            return {"status": "error", "reason": str(e)}

    def get_balance(self) -> float:
        """Return available balance in dollars. Returns -1 on failure."""
        if not self._ensure_auth():
            return -1.0
        try:
            resp = self._session.get(f"{KALSHI_API}/portfolio/balance", timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                cents = data.get("balance", data.get("available_balance", 0))
                return float(cents) / 100.0  # Kalshi returns cents
        except Exception:
            pass
        return -1.0
