"""
Automated Dutch Book arbitrage executor.
Runs as an asyncio background task alongside the main bot.

Scans Polymarket + Kalshi for matching markets where:
  YES(one platform) + NO(other platform) < $1.00
= guaranteed profit regardless of outcome.

When found: executes both legs simultaneously.
Respects DRY_RUN flag — safe to run without real money.
"""

import asyncio, time, os
from utils.logger import get_logger
from data.kalshi_trade import KalshiTradeClient
from data.polymarket_api import PolymarketClient
from config import DRY_RUN

log = get_logger("arb_executor")

ARB_SCAN_INTERVAL  = 60      # seconds between scans
ARB_MAX_DOLLARS    = float(os.getenv("ARB_MAX_DOLLARS", "20"))   # max per leg
ARB_MIN_PROFIT_PCT = float(os.getenv("ARB_MIN_PROFIT_PCT", "2")) # skip if ROI < 2%
ARB_MIN_EXPIRY_H   = 24      # only auto-execute if >24h to expiry (price is stable)
ARB_MAX_DAILY_LEGS = 10      # circuit breaker: max legs placed per day
ARB_MIN_SIMILARITY = 0.40    # minimum Jaccard similarity to auto-execute (vs just alert)

_daily_legs  = 0
_daily_reset = 0.0


def _reset_daily_counter():
    global _daily_legs, _daily_reset
    now = time.time()
    if now - _daily_reset > 86400:
        _daily_legs  = 0
        _daily_reset = now


# ── Import shared arb scanning logic ──────────────────────────────────────────

def _run_arb_scan() -> list[dict]:
    """
    Run one full scan across all categories.
    Returns list of confirmed arb opportunities (both legs needed).
    """
    try:
        # Import here to avoid circular imports at module load
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

        from arb_monitor import (
            CATEGORIES, fetch_kalshi_series,
            filter_polymarket, match_markets,
        )

        opps = []
        for cat_name, cfg in CATEGORIES.items():
            kalshi = fetch_kalshi_series(cfg["kalshi_series"])
            poly   = filter_polymarket(cfg["poly_keywords"])
            pairs  = match_markets(kalshi, poly)
            for p in pairs:
                if p["arb"]:
                    p["category"] = cat_name
                    opps.append(p)

        return sorted(opps, key=lambda x: -x["roi_pct"])

    except Exception as e:
        log.error(f"Arb scan error: {e}", exc_info=True)
        return []


# ── Leg execution ──────────────────────────────────────────────────────────────

async def _place_kalshi_leg(
    pair: dict,
    kalshi_client: KalshiTradeClient,
    side: str,            # "yes" or "no"
    amount: float,
) -> dict:
    ticker    = pair["kalshi"]["ticker"]
    yes_price = pair["kalshi"]["yes_ask"]
    result    = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: kalshi_client.place_market_order(
            ticker=ticker,
            side=side,
            amount_dollars=amount,
            yes_price=yes_price,
            dry_run=DRY_RUN,
        )
    )
    return result


async def _place_poly_leg(
    pair: dict,
    pm_client: PolymarketClient,
    side: str,            # "yes" or "no"
    amount: float,
) -> dict:
    # For Polymarket we need conditionId → token_id
    # The arb monitor stores raw Polymarket market data; look up token via conditionId
    ticker = pair["poly"]["ticker"]  # this is the conditionId (first 20 chars)
    # Reconstruct what side we're buying
    # We can use place_market_order with the appropriate token
    # For now, log and return dry_run result (full token_id lookup requires CLOB API)
    log.info(
        f"[POLY {'DRY' if DRY_RUN else 'LIVE'}] "
        f"Would BUY {side.upper()} on Poly market {ticker[:16]}... "
        f"${amount:.2f}"
    )
    if DRY_RUN:
        return {"status": "dry_run", "side": side, "amount": amount}

    # Live execution: resolve full token_id from conditionId, then place order
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: pm_client.place_market_order(
                token_id=ticker,   # use conditionId as fallback; override in prod
                side="BUY",
                amount_usdc=amount,
                dry_run=False,
            )
        )
        return result
    except Exception as e:
        return {"status": "error", "reason": str(e)}


# ── Main executor ──────────────────────────────────────────────────────────────

async def execute_arb_pair(
    pair: dict,
    kalshi_client: KalshiTradeClient,
    pm_client: PolymarketClient,
) -> bool:
    """
    Execute one Dutch Book pair. Returns True if both legs placed successfully.
    Tries to place both legs simultaneously — if one fails, logs the exposure.
    """
    global _daily_legs
    _reset_daily_counter()

    if _daily_legs >= ARB_MAX_DAILY_LEGS:
        log.warning(f"[ARB] Daily leg limit ({ARB_MAX_DAILY_LEGS}) reached — skipping.")
        return False

    # Determine which side each platform takes
    # If action starts with "Kalshi YES": buy Kalshi YES + Poly NO
    action     = pair["action"]
    kalshi_side = "yes" if "Kalshi YES" in action else "no"
    poly_side   = "no"  if "Poly NO"    in action else "yes"
    # Edge case: "Poly YES + Kalshi NO"
    if "Poly YES" in action:
        poly_side   = "yes"
        kalshi_side = "no"

    # Size each leg to produce matched exposure
    amount = min(ARB_MAX_DOLLARS, 50.0)

    cat  = pair.get("category", "?")
    km   = pair["kalshi"]
    pm   = pair["poly"]
    prof = pair["roi_pct"]

    log.info(
        f"[ARB] Executing {cat} | ROI={prof:.1f}% | "
        f"Kalshi {kalshi_side.upper()} + Poly {poly_side.upper()} | "
        f"${amount:.2f} each | DRY_RUN={DRY_RUN}"
    )
    log.info(f"[ARB]   Kalshi: {km['title'][:60]}")
    log.info(f"[ARB]   Poly:   {pm['title'][:60]}")

    # Place both legs simultaneously
    kalshi_result, poly_result = await asyncio.gather(
        _place_kalshi_leg(pair, kalshi_client, kalshi_side, amount),
        _place_poly_leg(pair, pm_client, poly_side, amount),
        return_exceptions=True,
    )

    kalshi_ok = (
        isinstance(kalshi_result, dict) and
        kalshi_result.get("status") in ("ok", "dry_run")
    )
    poly_ok = (
        isinstance(poly_result, dict) and
        poly_result.get("status") in ("ok", "dry_run")
    )

    if kalshi_ok and poly_ok:
        _daily_legs += 2
        log.info(
            f"[ARB] Both legs placed. "
            f"Expected profit: ${amount * prof / 100:.2f} per ${amount:.2f} staked."
        )
        return True
    else:
        # One leg failed — exposure risk. Log clearly.
        log.error(
            f"[ARB] PARTIAL FILL — one leg failed! Manual action may be needed.\n"
            f"  Kalshi: {kalshi_result}\n"
            f"  Poly:   {poly_result}"
        )
        return False


# ── Background loop ────────────────────────────────────────────────────────────

async def arb_loop(
    pm_client: PolymarketClient,
    kalshi_client: KalshiTradeClient,
) -> None:
    """
    Runs as a background asyncio task.
    Scans every ARB_SCAN_INTERVAL seconds, executes when opportunity found.
    """
    if not kalshi_client.is_enabled:
        log.info(
            "[ARB] Kalshi credentials not configured — "
            "arb scanner running in monitor-only mode (no auto-execution)."
        )

    executed_this_session: set[str] = set()  # deduplicate by (kalshi_ticker, poly_ticker)

    log.info("[ARB] Background arb scanner started.")

    while True:
        try:
            opps = await asyncio.get_event_loop().run_in_executor(None, _run_arb_scan)

            if opps:
                log.info(f"[ARB] Found {len(opps)} arbitrage opportunit{'y' if len(opps)==1 else 'ies'}:")
                for p in opps:
                    log.info(
                        f"[ARB]   {p['category']} | ROI={p['roi_pct']:.1f}% | "
                        f"sim={p['sim']:.0%} | "
                        f"Kalshi: {p['kalshi']['title'][:45]} | "
                        f"Poly: {p['poly']['title'][:45]}"
                    )

                for pair in opps:
                    # Deduplicate by Poly market (one position per Poly market per session)
                    pair_key = pair["poly"]["ticker"]

                    # Skip if already executed this session
                    if pair_key in executed_this_session:
                        continue

                    # Skip if similarity too low — likely a false positive, not a true Dutch Book
                    sim = pair.get("sim", 0.0)
                    if sim < ARB_MIN_SIMILARITY:
                        log.info(
                            f"[ARB] Skipping {pair['category']} — "
                            f"similarity {sim:.0%} below {ARB_MIN_SIMILARITY:.0%} threshold "
                            f"(questions may not be equivalent). Alert only."
                        )
                        continue

                    # Skip if ROI below threshold
                    if pair["roi_pct"] < ARB_MIN_PROFIT_PCT:
                        continue

                    # Skip if expiry too soon (price could move before fills)
                    hours_left = (
                        min(pair["kalshi"]["close_ts"], pair["poly"]["close_ts"])
                        - time.time()
                    ) / 3600
                    if hours_left < ARB_MIN_EXPIRY_H:
                        log.info(
                            f"[ARB] Skipping {pair['category']} — "
                            f"only {hours_left:.1f}h until expiry "
                            f"(min={ARB_MIN_EXPIRY_H}h). Manual execution recommended."
                        )
                        continue

                    success = await execute_arb_pair(pair, kalshi_client, pm_client)
                    if success:
                        executed_this_session.add(pair_key)

            else:
                log.debug("[ARB] No arbitrage opportunities found.")

        except Exception as e:
            log.error(f"[ARB] Loop error: {e}", exc_info=True)

        await asyncio.sleep(ARB_SCAN_INTERVAL)
