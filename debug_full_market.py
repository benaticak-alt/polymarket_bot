"""Print the full JSON of one market so we can see every field name."""
import time, requests, json, os
from dotenv import load_dotenv
load_dotenv()

session = requests.Session()
session.headers.update({
    "POLY_API_KEY":        os.getenv("PM_API_KEY", ""),
    "POLY_API_SECRET":     os.getenv("PM_API_SECRET", ""),
    "POLY_API_PASSPHRASE": os.getenv("PM_API_PASSPHRASE", ""),
})

now  = int(time.time())
step = 300
slug = f"btc-updown-5m-{((now // step) + 1) * step}"

print(f"Fetching: {slug}\n")

r = requests.get(f"https://gamma-api.polymarket.com/markets", params={"slug": slug}, timeout=10)
print(json.dumps(r.json(), indent=2))
