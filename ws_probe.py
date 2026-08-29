"""One-shot Chainlink Data Streams WS probe — diagnoses the reconnect loop.

Connects exactly like the bot's client, then reports: did the handshake succeed,
how many report frames arrive in 30s, and the exact close code/reason when the
socket drops. Read-only — touches nothing in the trading path.

    python ws_probe.py
"""

import asyncio
import time

import websockets

from config import CHAINLINK_DS_WS_URL, CHAINLINK_DS_FEED_IDS
from data.chainlink_stream import _auth_headers, _decode_full_report
import json


async def main() -> None:
    feed_ids = ",".join(f.lower() for f in CHAINLINK_DS_FEED_IDS.values())
    path = f"/api/v1/ws?feedIDs={feed_ids}"
    url = CHAINLINK_DS_WS_URL + path
    headers = _auth_headers("GET", path)
    print(f"Connecting: {url[:70]}…")

    try:
        try:
            cm = websockets.connect(url, extra_headers=headers,
                                    ping_interval=20, ping_timeout=10)
        except TypeError:
            cm = websockets.connect(url, additional_headers=headers,
                                    ping_interval=20, ping_timeout=10)
        async with cm as ws:
            print("HANDSHAKE OK — connected.")
            frames = 0
            t0 = time.time()
            try:
                while time.time() - t0 < 30:
                    raw = await asyncio.wait_for(ws.recv(), timeout=30 - (time.time() - t0))
                    frames += 1
                    if frames <= 3:
                        msg = json.loads(raw)
                        rep = msg.get("report", msg)
                        full = rep.get("fullReport") if isinstance(rep, dict) else None
                        dec = _decode_full_report(full) if full else None
                        price = f"${dec[0]:,.2f}" if dec else "(undecodable)"
                        print(f"  frame {frames}: price {price}")
            except asyncio.TimeoutError:
                pass
            print(f"\n{frames} frames in {time.time()-t0:.0f}s "
                  f"({frames/max(time.time()-t0,1):.1f}/s)")
            if frames == 0:
                print("  *** Connected but ZERO reports — server isn't pushing. ***")
    except Exception as e:
        print(f"\nClosed/failed: {type(e).__name__}: {e!r}")
        code = getattr(e, "code", None)
        reason = getattr(e, "reason", None)
        if code is not None:
            print(f"  close code={code}  reason={reason!r}")


if __name__ == "__main__":
    asyncio.run(main())
