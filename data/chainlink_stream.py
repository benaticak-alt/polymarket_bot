"""
Chainlink Data Streams client.

Polymarket settles its 5m/15m crypto up/down markets on Chainlink Data Streams
(BTC/USD, ETH/USD). This client streams the same signed low-latency price reports
so the bot can settle WIN/LOSS labels and compute barrier_z against the exact
price source Polymarket resolves on — instead of the Binance futures mid it uses
today.

Data Streams delivers ABI-encoded, signed reports over a WebSocket. Each report
is unwrapped to its inner blob and decoded with the v3 (crypto) schema to recover
the benchmark price (18 decimals). Auth is HMAC-SHA256 over a canonical string of
method/path/body-hash/clientId/timestamp, regenerated per connection.

Disabled by default — only runs when CHAINLINK_STREAMS_ENABLED is set and the
API key/secret are present. When off, nothing here touches the trading path.
"""

import asyncio
import hashlib
import hmac
import json
import time

import requests
import websockets
from eth_abi import decode as abi_decode

from utils.ring_buffer import TimeSeriesBuffer
from utils.logger import get_logger
from config import (
    CHAINLINK_DS_REST_URL,
    CHAINLINK_DS_WS_URL,
    CHAINLINK_DS_API_KEY,
    CHAINLINK_DS_API_SECRET,
)

log = get_logger("chainlink_stream")

# Empty-body SHA-256, precomputed (Data Streams GET requests carry no body).
_EMPTY_BODY_SHA256 = hashlib.sha256(b"").hexdigest()

# v3 (crypto) report-blob schema — the inner `reportBlob` decodes to these fields.
# price/bid/ask are int192 with 18 decimals.
_V3_SCHEMA = [
    "bytes32",  # feedId
    "uint32",   # validFromTimestamp
    "uint32",   # observationsTimestamp
    "uint192",  # nativeFee
    "uint192",  # linkFee
    "uint32",   # expiresAt
    "int192",   # price (benchmark)
    "int192",   # bid
    "int192",   # ask
]

# fullReport envelope: (reportContext, reportBlob, rs, ss, rawVs)
_ENVELOPE_SCHEMA = ["bytes32[3]", "bytes", "bytes32[]", "bytes32[]", "bytes32"]


class ChainlinkAssetState:
    """Latest Chainlink Data Streams price for one asset (e.g. BTC/USD)."""

    def __init__(self) -> None:
        self.price: float = 0.0
        self.bid: float = 0.0
        self.ask: float = 0.0
        self.observations_ts: float = 0.0   # report's observation timestamp (s)
        self.last_update: float = 0.0        # local wall clock of last update (s)
        self.history = TimeSeriesBuffer(maxlen=20_000)

    def _set(self, price: float, bid: float, ask: float, obs_ts: float) -> None:
        self.price = price
        self.bid = bid
        self.ask = ask
        self.observations_ts = obs_ts
        self.last_update = time.time()
        if price > 0:
            self.history.append(price, obs_ts or self.last_update)


def _auth_headers(method: str, path: str) -> dict:
    """HMAC-SHA256 auth headers for a Data Streams request.

    The signed string is: "{METHOD} {PATH} {sha256(body)} {clientId} {ts_ms}".
    PATH must be the exact request path (including query string) the server sees.
    """
    ts = int(time.time() * 1000)
    to_sign = f"{method} {path} {_EMPTY_BODY_SHA256} {CHAINLINK_DS_API_KEY} {ts}"
    sig = hmac.new(
        CHAINLINK_DS_API_SECRET.encode(),
        to_sign.encode(),
        hashlib.sha256,
    ).hexdigest()
    return {
        "Authorization": CHAINLINK_DS_API_KEY,
        "X-Authorization-Timestamp": str(ts),
        "X-Authorization-Signature-SHA256": sig,
    }


def _decode_full_report(full_report_hex: str) -> tuple[float, float, float, float] | None:
    """Decode a Data Streams `fullReport` hex blob → (price, bid, ask, obs_ts).

    Returns None if the blob can't be decoded as a v3 crypto report.
    """
    try:
        raw = bytes.fromhex(full_report_hex[2:] if full_report_hex.startswith("0x") else full_report_hex)
        _ctx, report_blob, _rs, _ss, _vs = abi_decode(_ENVELOPE_SCHEMA, raw)
        (
            _feed_id, _valid_from, obs_ts,
            _native_fee, _link_fee, _expires_at,
            price, bid, ask,
        ) = abi_decode(_V3_SCHEMA, report_blob)
        return (price / 1e18, bid / 1e18, ask / 1e18, float(obs_ts))
    except Exception as e:
        log.debug(f"report decode failed: {e}")
        return None


class ChainlinkStreamClient:
    """Streams Data Streams reports for a set of feeds and updates shared states.

    feeds: maps a lowercased feed ID → the ChainlinkAssetState it populates.
    Reconnects automatically; identical backoff behavior to the Binance client.
    """

    def __init__(self, feeds: dict[str, ChainlinkAssetState]) -> None:
        # Normalize feed IDs to lowercase for routing match against report feedID.
        self._feeds = {fid.lower(): st for fid, st in feeds.items()}
        self._running = False

    def _ws_path(self) -> str:
        feed_ids = ",".join(self._feeds.keys())
        return f"/api/v1/ws?feedIDs={feed_ids}"

    def _route(self, report: dict) -> None:
        full = report.get("fullReport")
        feed_id = (report.get("feedID") or report.get("feedId") or "").lower()
        if not full or feed_id not in self._feeds:
            return
        decoded = _decode_full_report(full)
        if decoded is None:
            return
        price, bid, ask, obs_ts = decoded
        self._feeds[feed_id]._set(price, bid, ask, obs_ts)

    def _handle_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        # Reports may arrive wrapped in {"report": {...}} or bare.
        report = msg.get("report", msg)
        if isinstance(report, list):
            for r in report:
                self._route(r)
        elif isinstance(report, dict):
            self._route(report)

    def _warmup(self) -> None:
        """Best-effort REST fetch of each feed's latest report so a price exists
        immediately at startup, before the first WS push lands."""
        for feed_id, state in self._feeds.items():
            path = f"/api/v1/reports/latest?feedID={feed_id}"
            try:
                resp = requests.get(
                    CHAINLINK_DS_REST_URL + path,
                    headers=_auth_headers("GET", path),
                    timeout=10,
                )
                resp.raise_for_status()
                report = resp.json().get("report", {})
                decoded = _decode_full_report(report.get("fullReport", ""))
                if decoded:
                    price, bid, ask, obs_ts = decoded
                    state._set(price, bid, ask, obs_ts)
                    log.info(f"Chainlink warmup {feed_id[:18]}…: ${price:,.2f}")
            except Exception as e:
                log.warning(f"Chainlink REST warmup failed for {feed_id[:18]}…: {e}")

    async def run(self) -> None:
        if not (CHAINLINK_DS_API_KEY and CHAINLINK_DS_API_SECRET):
            log.warning("Chainlink Data Streams creds missing — client not started.")
            return
        self._running = True

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._warmup)

        backoff = 3
        connected_once = False
        while self._running:
            conn_t0 = time.time()
            path = self._ws_path()
            url = CHAINLINK_DS_WS_URL + path
            headers = _auth_headers("GET", path)
            try:
                # First connect logs at INFO; routine reconnects are quiet.
                (log.debug if connected_once else log.info)(
                    f"Connecting to Chainlink Data Streams: {path[:60]}…"
                )
                # websockets renamed extra_headers→additional_headers in v13; support both.
                try:
                    cm = websockets.connect(
                        url, extra_headers=headers,
                        ping_interval=20, ping_timeout=10, close_timeout=5,
                    )
                except TypeError:
                    cm = websockets.connect(
                        url, additional_headers=headers,
                        ping_interval=20, ping_timeout=10, close_timeout=5,
                    )
                async with cm as ws:
                    if not connected_once:
                        log.info("Chainlink Data Streams connected.")
                        connected_once = True
                    backoff = 3
                    conn_t0 = time.time()
                    first = True
                    async for raw in ws:
                        self._handle_message(raw)
                        if first:
                            log.debug("Chainlink Data Streams: first report received.")
                            first = False
            except (websockets.ConnectionClosed, OSError) as e:
                # Up a while then closed = routine server cycling → quiet.
                msg = f"Chainlink WS disconnected ({e}), reconnecting in {backoff}s…"
                (log.debug if time.time() - conn_t0 >= 30 else log.warning)(msg)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
            except Exception as e:
                log.error(f"Unexpected Chainlink WS error: {e}", exc_info=True)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    def stop(self) -> None:
        self._running = False
