"""Find the actual BTC 5m/10m markets on Polymarket."""
import requests, json, os, time
from dotenv import load_dotenv
load_dotenv()

GAMMA = "https://gamma-api.polymarket.com"
CLOB  = "https://clob.polymarket.com"

session = requests.Session()
session.headers.update({
    "POLY_API_KEY":        os.getenv("PM_API_KEY", ""),
    "POLY_API_SECRET":     os.getenv("PM_API_SECRET", ""),
    "POLY_API_PASSPHRASE": os.getenv("PM_API_PASSPHRASE", ""),
})

# ── 1. Gamma: paginate through ALL markets looking for BTC minute markets ──────
print("=== Scanning ALL Gamma markets (paginated) ===")
btc_minute_markets = []
offset = 0
limit  = 100
pages  = 0

while pages < 20:  # scan up to 2000 markets
    try:
        r = session.get(f"{GAMMA}/markets", params={
            "active": "true", "closed": "false",
            "limit": limit, "offset": offset
        }, timeout=15)
        data  = r.json()
        items = data if isinstance(data, list) else data.get("markets", [])
        if not items:
            break
        for item in items:
            q = item.get("question", "").lower()
            if any(kw in q for kw in ["btc", "bitcoin"]):
                btc_minute_markets.append(item)
        offset += limit
        pages  += 1
        time.sleep(0.2)
    except Exception as e:
        print(f"Error at offset {offset}: {e}")
        break

print(f"Scanned {offset} markets, found {len(btc_minute_markets)} BTC markets:")
for m in btc_minute_markets:
    print(f"  -> {m.get('question','')}")
    print(f"     end={m.get('end_date_iso') or m.get('endDate','')} | id={m.get('condition_id','')[:16]}...")
    print(f"     tokens={m.get('tokens','none')}")

# ── 2. Try Gamma events endpoint ───────────────────────────────────────────────
print("\n=== Gamma /events endpoint ===")
try:
    r = session.get(f"{GAMMA}/events", params={"active": "true", "closed": "false", "limit": 100}, timeout=15)
    print(f"Status: {r.status_code}")
    data  = r.json()
    items = data if isinstance(data, list) else data.get("events", [])
    print(f"Total events: {len(items)}")
    btc_events = [i for i in items if any(kw in i.get("title","").lower() for kw in ["btc","bitcoin","crypto"])]
    print(f"BTC events: {len(btc_events)}")
    for e in btc_events[:10]:
        print(f"  -> {e.get('title','')} | slug={e.get('slug','')}")
        for m in e.get("markets", [])[:3]:
            print(f"       market: {m.get('question','')}")
except Exception as e:
    print(f"Error: {e}")

# ── 3. CLOB with accepting_orders only and next_cursor pagination ──────────────
print("\n=== CLOB — accepting orders, paginated ===")
btc_clob = []
cursor = ""
pages  = 0
while pages < 10:
    try:
        params = {"limit": 100, "accepting_orders": "true"}
        if cursor:
            params["next_cursor"] = cursor
        r    = session.get(f"{CLOB}/markets", params=params, timeout=15)
        data = r.json()
        items  = data.get("data", [])
        cursor = data.get("next_cursor", "")
        for i in items:
            q = i.get("question","").lower()
            if any(kw in q for kw in ["btc","bitcoin","crypto","minute","5m","10m"]):
                btc_clob.append(i)
        if not cursor or not items:
            break
        pages += 1
        time.sleep(0.2)
    except Exception as e:
        print(f"CLOB error: {e}")
        break

print(f"BTC/minute markets in CLOB: {len(btc_clob)}")
for i in btc_clob[:10]:
    print(f"  -> {i.get('question','')} | end={i.get('end_date_iso','')} | accepting={i.get('accepting_orders')}")
    print(f"     clobTokenIds={i.get('clobTokenIds','')}")

# ── 4. Try known BTC market slugs directly ────────────────────────────────────
print("\n=== Direct slug lookup ===")
slugs = [
    "will-btc-go-up-in-5-minutes",
    "will-btc-go-up-in-10-minutes",
    "btc-5-minute",
    "btc-10-minute",
    "bitcoin-5m",
    "bitcoin-10m",
    "btc-up-5min",
]
for slug in slugs:
    try:
        r = session.get(f"{GAMMA}/markets", params={"slug": slug}, timeout=5)
        data = r.json()
        items = data if isinstance(data, list) else data.get("markets", [])
        if items:
            print(f"  FOUND slug={slug}: {items[0].get('question','')}")
        else:
            print(f"  not found: {slug}")
    except Exception as e:
        print(f"  error {slug}: {e}")
