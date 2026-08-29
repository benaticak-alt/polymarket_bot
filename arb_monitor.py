"""
Dutch Book Arbitrage Monitor - Polymarket x Kalshi (All Categories)
Scans for overlapping markets where YES(one side) + NO(other) < $1.00
= guaranteed profit regardless of outcome.

Run: python arb_monitor.py
Refreshes every 60 seconds.
"""

import sys, os, re, time, math, requests
from datetime import datetime, timezone
from collections import defaultdict

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

KALSHI_API  = "https://api.elections.kalshi.com/trade-api/v2"
POLY_GAMMA  = "https://gamma-api.polymarket.com"

REFRESH          = 60      # seconds between full scans
SIMILARITY_MIN   = 0.30   # Jaccard threshold to consider two questions "same event"
DATE_WINDOW_DAYS = 14     # settlement dates within N days = potential match
MIN_PROFIT       = 0.005  # alert if guaranteed profit >= 0.5 cents per $1 staked
MAX_POLY_MARKETS = 1000   # how many Polymarket markets to scan

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "polymarket-bot/1.0", "Accept": "application/json"})

# ── Categories: kalshi series + polymarket keyword filters ─────────────────────

CATEGORIES = {
    "Fed/Rates": {
        "kalshi_series": ["KXFEDDECISION", "KXFED", "KXRATE"],
        "poly_keywords": ["fed ", "fomc", "rate cut", "rate hike", "interest rate",
                          "federal reserve", "basis point"],
    },
    "NBA": {
        # Kalshi uses "Pro Basketball Finals" not "NBA Finals"
        "kalshi_series": ["KXNBA"],
        "poly_keywords": ["nba", "pro basketball", "basketball finals",
                          "thunder", "celtics", "knicks", "pacers",
                          "lakers", "warriors", "spurs", "nuggets", "heat"],
    },
    "NFL": {
        "kalshi_series": ["KXNFLWINS", "KXNFL", "KXSUPERBOWL"],
        "poly_keywords": ["nfl", "super bowl", "football",
                          "chiefs", "eagles", "ravens", "49ers", "bills",
                          "cowboys", "packers", "patriots"],
    },
    "MLB": {
        "kalshi_series": ["KXMLB", "KXWORLDSERIES", "KXMLBDEBUT"],
        "poly_keywords": ["mlb", "baseball", "world series", "yankees",
                          "dodgers", "mets", "cubs", "red sox"],
    },
    "Elections": {
        "kalshi_series": ["KXHOUSERACE", "KXMIDTERMMOV", "KXPRESPARTY",
                          "KXVOTEPRIMARY", "KXPRIMARYPLACE", "KXPRIMARYMOV",
                          "CONTROLS", "CONTROLH", "GOVPARTYVT",
                          "KXHOUSEWINSTATE", "KXDSENATESEATS"],
        "poly_keywords": ["election", "senate", "governor", "congress",
                          "ballot", "primary", "runoff", "democrat win",
                          "republican win", "house seat"],
    },
    "Recession/Econ": {
        "kalshi_series": ["KXRECSSNBER", "KXU3MAX", "KXGOVTCUTS"],
        "poly_keywords": ["recession", "gdp", "unemployment", "job",
                          "payroll", "economic"],
    },
    "Geopolitics": {
        "kalshi_series": ["KXG7LEADEROUT", "KXAFRICALEADEROUT", "KXNEXTISRAELPM",
                          "KXNEXTROMANIAPM", "KXXISUCCESSOR"],
        "poly_keywords": ["war", "ceasefire", "sanctions", "nato",
                          "prime minister", "resign", "coup", "invade",
                          "peace deal", "conflict"],
    },
    "Tech/AI": {
        "kalshi_series": ["KXOAIANTH", "KXDEELRIP", "KXRAMPBREX"],
        "poly_keywords": ["ipo", "openai", "anthropic", "gpt",
                          "ai regulation", "tech layoff", "startup"],
    },
    "Crypto": {
        "kalshi_series": ["KXBTC", "KXETH"],
        "poly_keywords": ["bitcoin above", "bitcoin below", "btc price",
                          "ethereum above"],
    },
}

STOP_WORDS = {
    "the", "a", "an", "will", "be", "in", "on", "at", "by", "or", "and",
    "of", "to", "for", "that", "this", "with", "is", "are", "was", "were",
    "who", "what", "when", "how", "if", "not", "no", "yes", "do", "does",
    "have", "has", "had", "can", "would", "could", "should", "may", "might",
    "get", "got", "been", "after", "before", "during", "win", "wins", "won",
}


# ── Text similarity ────────────────────────────────────────────────────────────

def keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOP_WORDS and len(w) > 2}


def jaccard(a: str, b: str) -> float:
    ka, kb = keywords(a), keywords(b)
    if not ka or not kb:
        return 0.0
    inter = len(ka & kb)
    union = len(ka | kb)
    return inter / union


# ── Datetime helpers ───────────────────────────────────────────────────────────

def parse_dt(s: str) -> float:
    if not s:
        return 0.0
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def fmt_date(ts: float) -> str:
    if not ts:
        return "?"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def fmt_countdown(ts: float) -> str:
    secs = ts - time.time()
    if secs < 0:
        return "expired"
    d, r  = divmod(int(secs), 86400)
    h, r2 = divmod(r, 3600)
    m     = r2 // 60
    if d:
        return f"{d}d {h}h"
    if h:
        return f"{h}h {m}m"
    return f"{m}m"


# ── Kalshi fetcher ─────────────────────────────────────────────────────────────

def fetch_kalshi_series(series_list: list[str]) -> list[dict]:
    now    = time.time()
    result = []
    seen   = set()

    for series in series_list:
        try:
            resp = SESSION.get(
                f"{KALSHI_API}/markets",
                params={"series_ticker": series, "status": "open", "limit": 100},
                timeout=10,
            )
            if resp.status_code != 200:
                continue
            for m in resp.json().get("markets", []):
                ticker = m.get("ticker", "")
                if ticker in seen:
                    continue
                seen.add(ticker)

                close_ts = parse_dt(m.get("close_time", ""))
                if close_ts < now:
                    continue

                title = m.get("title", "") or m.get("subtitle", "") or ticker
                yes_ask = float(m.get("yes_ask_dollars") or 0)
                yes_bid = float(m.get("yes_bid_dollars") or 0)
                no_ask  = float(m.get("no_ask_dollars")  or 0)
                no_bid  = float(m.get("no_bid_dollars")  or 0)

                if yes_ask <= 0 and no_ask <= 0:
                    continue

                result.append({
                    "source":   "kalshi",
                    "series":   series,
                    "ticker":   ticker,
                    "title":    title,
                    "close_ts": close_ts,
                    "yes_ask":  yes_ask,
                    "yes_bid":  yes_bid,
                    "no_ask":   no_ask,
                    "no_bid":   no_bid,
                    "yes_mid":  (yes_ask + yes_bid) / 2 if yes_bid else yes_ask,
                    "no_mid":   (no_ask  + no_bid)  / 2 if no_bid  else no_ask,
                })
        except Exception:
            continue

    return result


# ── Polymarket fetcher ─────────────────────────────────────────────────────────

_poly_cache: list[dict] = []
_poly_cache_ts: float   = 0.0
POLY_CACHE_TTL = 120  # seconds


def fetch_polymarket_all() -> list[dict]:
    global _poly_cache, _poly_cache_ts
    if time.time() - _poly_cache_ts < POLY_CACHE_TTL and _poly_cache:
        return _poly_cache

    result = []
    now    = time.time()
    offset = 0

    while len(result) < MAX_POLY_MARKETS:
        try:
            resp = SESSION.get(
                f"{POLY_GAMMA}/markets",
                params={"active": "true", "closed": "false",
                        "limit": 100, "offset": offset},
                timeout=10,
            )
            resp.raise_for_status()
            data    = resp.json()
            markets = data if isinstance(data, list) else data.get("markets", data.get("data", []))
            if not markets:
                break

            for m in markets:
                close_ts = parse_dt(m.get("endDateIso") or m.get("endDate") or "")
                if close_ts < now:
                    continue

                question = (m.get("question") or m.get("title") or "").strip()
                if not question:
                    continue

                op = m.get("outcomePrices")
                yes_ask = no_ask = 0.0
                if op:
                    try:
                        if isinstance(op, str):
                            import json as _j; op = _j.loads(op)
                        yes_ask = float(op[0])
                        no_ask  = float(op[1])
                    except Exception:
                        pass

                if yes_ask <= 0 and no_ask <= 0:
                    # Try bestBid/bestAsk
                    yes_ask = float(m.get("bestAsk") or 0)
                    no_ask  = 1.0 - yes_ask if yes_ask else 0.0

                if yes_ask <= 0:
                    continue

                result.append({
                    "source":   "polymarket",
                    "ticker":   (m.get("conditionId") or m.get("id") or "")[:20],
                    "title":    question,
                    "close_ts": close_ts,
                    "yes_ask":  yes_ask,
                    "yes_bid":  yes_ask,
                    "no_ask":   no_ask,
                    "no_bid":   no_ask,
                    "yes_mid":  yes_ask,
                    "no_mid":   no_ask,
                    "volume":   float(m.get("volumeNum") or m.get("volume") or 0),
                })

            offset += 100
            if len(markets) < 100:
                break

        except Exception:
            break

    _poly_cache    = result
    _poly_cache_ts = time.time()
    return result


def filter_polymarket(keywords_list: list[str]) -> list[dict]:
    all_markets = fetch_polymarket_all()
    if not keywords_list:
        return all_markets
    result = []
    for m in all_markets:
        title_lower = m["title"].lower()
        if any(kw in title_lower for kw in keywords_list):
            result.append(m)
    return result


# ── Matching ───────────────────────────────────────────────────────────────────

def match_markets(kalshi: list[dict], poly: list[dict]) -> list[dict]:
    """
    Find pairs where questions are similar and settlement dates are close.
    Returns list of matched pairs with similarity score and Dutch Book info.
    """
    matched = []
    day_s   = 86400

    for km in kalshi:
        for pm in poly:
            # Settlement dates must be within window
            if abs(km["close_ts"] - pm["close_ts"]) > DATE_WINDOW_DAYS * day_s:
                continue

            sim = jaccard(km["title"], pm["title"])
            if sim < SIMILARITY_MIN:
                continue

            # Dutch Book calculations (using ask prices — worst case for buyer)
            cost_a = km["yes_ask"] + pm["no_ask"]   # buy Kalshi YES + Poly NO
            cost_b = pm["yes_ask"] + km["no_ask"]   # buy Poly YES + Kalshi NO

            profit_a = 1.0 - cost_a
            profit_b = 1.0 - cost_b
            best_profit  = max(profit_a, profit_b)
            best_cost    = cost_a if profit_a >= profit_b else cost_b
            best_action  = (
                f"Kalshi YES ${km['yes_ask']:.2f} + Poly NO ${pm['no_ask']:.2f}"
                if profit_a >= profit_b else
                f"Poly YES ${pm['yes_ask']:.2f} + Kalshi NO ${km['no_ask']:.2f}"
            )

            # Also compute mid-price gap (how much they disagree)
            prob_gap = abs(km["yes_mid"] - pm["yes_mid"])

            matched.append({
                "kalshi":       km,
                "poly":         pm,
                "sim":          sim,
                "cost":         best_cost,
                "profit":       best_profit,
                "roi_pct":      best_profit / best_cost * 100 if best_cost > 0 else 0,
                "action":       best_action,
                "arb":          best_profit > MIN_PROFIT,
                "prob_gap":     prob_gap,
            })

    # Deduplicate: keep only the best Kalshi match per Poly market
    # (avoids executing the same Poly position multiple times for different Kalshi strikes)
    best_per_poly: dict[str, dict] = {}
    for pair in matched:
        key = pair["poly"]["ticker"]
        if key not in best_per_poly or pair["roi_pct"] > best_per_poly[key]["roi_pct"]:
            best_per_poly[key] = pair
    matched = list(best_per_poly.values())

    return sorted(matched, key=lambda x: (-x["arb"], -x["sim"]))


# ── Main scan ──────────────────────────────────────────────────────────────────

def run_scan() -> dict:
    """Run one full scan across all categories. Returns dict of results."""
    results = {}

    for cat_name, cfg in CATEGORIES.items():
        kalshi = fetch_kalshi_series(cfg["kalshi_series"])
        poly   = filter_polymarket(cfg["poly_keywords"])
        pairs  = match_markets(kalshi, poly)

        results[cat_name] = {
            "kalshi_count": len(kalshi),
            "poly_count":   len(poly),
            "pairs":        pairs,
            "arb_count":    sum(1 for p in pairs if p["arb"]),
        }

    return results


# ── Display ────────────────────────────────────────────────────────────────────

def display(results: dict, scan_num: int) -> None:
    os.system("cls" if os.name == "nt" else "clear")
    W = 70

    print("=" * W)
    print("  DUTCH BOOK ARB MONITOR  -  Polymarket x Kalshi (All Categories)")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}  |  Scan #{scan_num}")
    print("=" * W)

    all_pairs  = [p for cat in results.values() for p in cat["pairs"]]
    arb_pairs  = [p for p in all_pairs if p["arb"]]
    total_k    = sum(c["kalshi_count"] for c in results.values())
    total_p    = len(fetch_polymarket_all())

    print(f"\n  Markets scanned: {total_k} Kalshi + {total_p} Polymarket")
    print(f"  Matched pairs:   {len(all_pairs)}")

    # ── Arbitrage opportunities ────────────────────────────────────────────────
    if arb_pairs:
        print(f"\n{'*'*W}")
        print(f"  *** {len(arb_pairs)} ARBITRAGE OPPORTUNIT{'Y' if len(arb_pairs)==1 else 'IES'} FOUND ***")
        print(f"{'*'*W}")
        for i, p in enumerate(sorted(arb_pairs, key=lambda x: -x["roi_pct"]), 1):
            print(f"\n  #{i}  {p['action']}")
            print(f"       Cost: ${p['cost']:.4f}  Guaranteed win: $1.00")
            print(f"       PROFIT: ${p['profit']:.4f}  ({p['roi_pct']:.1f}% ROI)")
            print(f"       Kalshi: {p['kalshi']['title'][:60]}")
            print(f"       Poly:   {p['poly']['title'][:60]}")
            km_exp = fmt_countdown(p['kalshi']['close_ts'])
            pm_exp = fmt_countdown(p['poly']['close_ts'])
            print(f"       Expires: Kalshi {km_exp}  |  Poly {pm_exp}")
            print(f"       Similarity: {p['sim']:.0%}  [ACT FAST]")
    else:
        print(f"\n  No arbitrage right now.")

    # ── Category breakdown ─────────────────────────────────────────────────────
    print(f"\n  BY CATEGORY")
    print(f"  {'Category':<18} {'Kalshi':<8} {'Poly':<8} {'Pairs':<8} {'Best match / Status'}")
    print(f"  {'-'*65}")

    for cat_name, res in results.items():
        pairs = res["pairs"]
        if pairs:
            best = pairs[0]
            sim_str  = f"{best['sim']:.0%} sim"
            gap_str  = f"gap={best['prob_gap']:.0%}"
            roi_str  = f"ROI={best['roi_pct']:+.1f}%" if best["arb"] else f"cost={best['cost']:.2f}"
            status   = f"{sim_str} | {gap_str} | {roi_str}"
        else:
            status = "no matches found"

        arb_flag = " ***" if res["arb_count"] > 0 else ""
        print(
            f"  {cat_name:<18} {res['kalshi_count']:<8} {res['poly_count']:<8} "
            f"{len(pairs):<8} {status}{arb_flag}"
        )

    # ── Best non-arb matched pairs (top 10 by similarity) ─────────────────────
    non_arb = [p for p in all_pairs if not p["arb"]]
    non_arb_sorted = sorted(non_arb, key=lambda x: (-x["sim"], -x["prob_gap"]))[:10]

    if non_arb_sorted:
        print(f"\n  CLOSEST MATCHED PAIRS  (same question, different prices = directional edge)")
        print(f"  {'Sim':<6} {'Gap':<6} {'K-YES':<8} {'P-YES':<8} {'Action when gap > 20%'}")
        print(f"  {'-'*65}")
        for p in non_arb_sorted:
            km, pm = p["kalshi"], p["poly"]
            # Who to bet with: bet YES on whichever platform has it cheaper
            if km["yes_mid"] < pm["yes_mid"]:
                edge = f"Buy Kalshi YES (${km['yes_mid']:.2f} vs Poly ${pm['yes_mid']:.2f})"
            else:
                edge = f"Buy Poly YES (${pm['yes_mid']:.2f} vs Kalshi ${km['yes_mid']:.2f})"
            print(
                f"  {p['sim']:.0%}   {p['prob_gap']:.0%}   "
                f"${km['yes_mid']:.2f}    ${pm['yes_mid']:.2f}    {edge[:35]}"
            )
            print(f"         K: {km['title'][:60]}")
            print(f"         P: {pm['title'][:60]}")
            print()

    print(f"\n  HOW IT WORKS:")
    print(f"  When YES(Kalshi) + NO(Poly) < $1.00 for the same question ->")
    print(f"  buy both for guaranteed $1.00 payout regardless of outcome.")
    print(f"  Even without arb, large price gaps show which platform to bet on.")
    print(f"\n  Refreshing every {REFRESH}s -- Ctrl+C to stop")
    print("=" * W)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting multi-category arb monitor...")
    print("Fetching Polymarket market catalog (1000 markets)...")

    scan_num   = 0
    total_opps = 0
    best_roi   = 0.0

    try:
        while True:
            scan_num += 1
            results   = run_scan()
            arb_found = sum(c["arb_count"] for c in results.values())
            total_opps += arb_found
            if arb_found:
                for cat in results.values():
                    for p in cat["pairs"]:
                        if p["arb"]:
                            best_roi = max(best_roi, p["roi_pct"])

            display(results, scan_num)

            if arb_found:
                print(f"\n  *** ARB FOUND at {time.strftime('%H:%M:%S')} ***")
                print(f"  Check display above -- ACT WITHIN MINUTES.\n")

            time.sleep(REFRESH)

    except KeyboardInterrupt:
        print(f"\nStopped. Scans: {scan_num} | Total opps: {total_opps} | Best ROI: {best_roi:.1f}%")
