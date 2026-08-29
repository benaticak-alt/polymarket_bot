"""
One-shot Chainlink Data Streams credential check.

Run this AFTER putting your creds in .env:

    CHAINLINK_DS_API_KEY=<the candlestick UUID>
    CHAINLINK_DS_API_SECRET=<the password>

Then:  python chainlink_check.py

It fetches the latest BTC/USD report over REST, prints the decoded price, and
checks your system clock skew against Chainlink's server (Data Streams rejects
any request whose timestamp is >5s off). On an auth failure it tells you which
credential mapping to try next. Nothing here touches the bot or trading state.
"""

import sys
import time
from email.utils import parsedate_to_datetime

import requests

from config import (
    CHAINLINK_DS_REST_URL,
    CHAINLINK_DS_API_KEY,
    CHAINLINK_DS_API_SECRET,
    CHAINLINK_DS_FEED_IDS,
)
from data.chainlink_stream import _auth_headers, _decode_full_report


def main() -> int:
    if not CHAINLINK_DS_API_KEY or not CHAINLINK_DS_API_SECRET:
        print("FAIL: CHAINLINK_DS_API_KEY / CHAINLINK_DS_API_SECRET not set in .env.")
        print("      Put the UUID in API_KEY and the password in API_SECRET, then rerun.")
        return 1

    print(f"API key : {CHAINLINK_DS_API_KEY[:8]}…  (len={len(CHAINLINK_DS_API_KEY)})")
    print(f"Secret  : set, len={len(CHAINLINK_DS_API_SECRET)}")
    print(f"REST    : {CHAINLINK_DS_REST_URL}")

    feed_id = CHAINLINK_DS_FEED_IDS["btc"]
    path = f"/api/v1/reports/latest?feedID={feed_id}"

    try:
        resp = requests.get(
            CHAINLINK_DS_REST_URL + path,
            headers=_auth_headers("GET", path),
            timeout=15,
        )
    except Exception as e:
        print(f"\nFAIL: request error — {e}")
        print("      Check CHAINLINK_DS_REST_URL is correct / reachable.")
        return 1

    # Clock skew vs server Date header (Data Streams allows max 5s).
    server_date = resp.headers.get("Date")
    if server_date:
        try:
            skew = time.time() - parsedate_to_datetime(server_date).timestamp()
            flag = "  <-- TOO HIGH, fix Windows time sync!" if abs(skew) > 4 else ""
            print(f"Clock   : local is {skew:+.1f}s vs server{flag}")
        except Exception:
            pass

    print(f"\nHTTP {resp.status_code}")

    if resp.status_code == 200:
        report = resp.json().get("report", {})
        decoded = _decode_full_report(report.get("fullReport", ""))
        if decoded:
            price, bid, ask, obs_ts = decoded
            age = time.time() - obs_ts if obs_ts else float("nan")
            print(f"OK: BTC/USD = ${price:,.2f}  (bid {bid:,.2f} / ask {ask:,.2f})")
            print(f"    report age: {age:.1f}s")
            print("\nCredentials WORK — key=UUID, secret=password is correct.")
            print("Set CHAINLINK_STREAMS_ENABLED=true and restart the bot.")
            return 0
        print("Auth OK but could not decode the report blob — paste the output to me.")
        print(report)
        return 1

    if resp.status_code in (401, 403):
        print("Auth REJECTED. Most likely the key/secret are swapped or mislabeled.")
        print("Try, in order:")
        print("  1. Swap them: API_KEY=<password>, API_SECRET=<candlestick UUID>")
        print("  2. If you also have a 'username', try API_KEY=<username>.")
        print("  3. Re-check the portal for a value literally labeled 'secret'.")
        print(f"\nServer said: {resp.text[:300]}")
        return 1

    if resp.status_code == 404:
        print("Auth OK, but this feed ID isn't on your account. Listing your feeds…")
        _list_feeds()
        return 1

    print(f"Unexpected response: {resp.text[:300]}")
    return 1


def _list_feeds() -> None:
    """Auth works — fetch /api/v1/feeds to discover the account's real feed IDs."""
    path = "/api/v1/feeds"
    try:
        resp = requests.get(
            CHAINLINK_DS_REST_URL + path,
            headers=_auth_headers("GET", path),
            timeout=15,
        )
    except Exception as e:
        print(f"  feed list request error — {e}")
        return
    if resp.status_code != 200:
        print(f"  HTTP {resp.status_code} listing feeds: {resp.text[:300]}")
        return
    feeds = resp.json().get("feeds", resp.json())
    print(f"\n  {len(feeds)} feed(s) available — fetching each latest price to identify them:")
    for f in feeds:
        fid = f.get("feedID") or f.get("feedId") or f
        price = _latest_price(fid)
        label = ""
        if price > 50_000:
            label = "  <-- likely BTC/USD"
        elif 1_000 < price < 10_000:
            label = "  <-- likely ETH/USD"
        price_str = f"${price:,.2f}" if price else "(no price)"
        print(f"    {fid}  {price_str}{label}")
    print("\n  Confirm which is BTC and which is ETH from the prices — I'll set them in config.")


def _latest_price(feed_id: str) -> float:
    path = f"/api/v1/reports/latest?feedID={feed_id}"
    try:
        resp = requests.get(
            CHAINLINK_DS_REST_URL + path,
            headers=_auth_headers("GET", path),
            timeout=15,
        )
        if resp.status_code != 200:
            return 0.0
        decoded = _decode_full_report(resp.json().get("report", {}).get("fullReport", ""))
        return decoded[0] if decoded else 0.0
    except Exception:
        return 0.0


if __name__ == "__main__":
    sys.exit(main())
