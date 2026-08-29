"""Test slug generation and API responses in real time."""
import time, requests, os
from dotenv import load_dotenv
load_dotenv()

GAMMA = "https://gamma-api.polymarket.com"
session = requests.Session()
session.headers.update({
    "POLY_API_KEY":        os.getenv("PM_API_KEY", ""),
    "POLY_API_SECRET":     os.getenv("PM_API_SECRET", ""),
    "POLY_API_PASSPHRASE": os.getenv("PM_API_PASSPHRASE", ""),
})

now = int(time.time())
print(f"Current Unix time: {now}")
print(f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(now))}\n")

# Generate slugs the same way the bot does
slugs = []
for interval, step in [("5m", 300), ("15m", 900)]:
    boundary = ((now // step) + 1) * step
    for i in range(5):
        ts = boundary + i * step
        slugs.append((f"btc-updown-{interval}-{ts}", interval, ts))

print("Generated slugs to test:")
for slug, interval, ts in slugs:
    t = time.strftime('%H:%M:%S UTC', time.gmtime(ts))
    print(f"  {slug}  (settles {t})")

print("\nQuerying Gamma API for each slug...\n")
for slug, interval, ts in slugs:
    try:
        # Try /events endpoint
        r = session.get(f"{GAMMA}/events", params={"slug": slug}, timeout=8)
        print(f"[{slug}]")
        print(f"  /events  status={r.status_code}  body={r.text[:150]}")

        # Try /markets endpoint
        r2 = session.get(f"{GAMMA}/markets", params={"slug": slug}, timeout=8)
        print(f"  /markets status={r2.status_code}  body={r2.text[:150]}")
        print()
        time.sleep(0.3)
    except Exception as e:
        print(f"  ERROR: {e}\n")
