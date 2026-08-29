"""
Crypto news sentiment via RSS feeds + CryptoPanic + BART MNLI classifier.
Sources: CoinDesk, Cointelegraph, Decrypt, Bitcoin Magazine, CryptoPanic.
No API key required.
Polls every 60 seconds — fast enough to catch major breaking news within one candle.
"""

import asyncio
import time
from dataclasses import dataclass
from utils.logger import get_logger
from data.bart_classifier import get_classifier

log = get_logger("news_sentiment")

NEWS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
    "https://bitcoinmagazine.com/feed",
    "https://www.newsbtc.com/feed/",
    "https://cryptoslate.com/feed/",
]
POLL_INTERVAL   = 60     # 1 minute — catch breaking news within one 5m candle
MAX_ARTICLES    = 10     # articles per RSS source
LABELS          = ["bullish on Bitcoin", "bearish on Bitcoin"]


@dataclass
class NewsSentimentState:
    score:        float = 0.0   # -1 (bearish) to 1 (bullish)
    change:       float = 0.0   # change from previous poll
    article_count: int  = 0
    last_updated: float = 0.0


class NewsSentimentClient:

    def __init__(self, state: NewsSentimentState):
        self.state       = state
        self._prev_score = 0.0
        self._classifier = None
        self._classifier = get_classifier()

    async def run(self) -> None:
        while True:
            if self._classifier:
                await asyncio.get_event_loop().run_in_executor(None, self._poll)
            await asyncio.sleep(POLL_INTERVAL)

    def _poll(self) -> None:
        try:
            import feedparser
            headlines = []

            # RSS feeds
            for feed_url in NEWS_FEEDS:
                try:
                    feed = feedparser.parse(feed_url)
                    for entry in feed.entries[:MAX_ARTICLES]:
                        title = entry.get("title", "").strip()
                        if title and len(title) > 10:
                            headlines.append(title)
                except Exception as e:
                    log.debug(f"RSS fetch error for {feed_url}: {e}")

            if not headlines:
                log.debug("No news articles fetched.")
                return

            results = self._classifier(headlines, LABELS, multi_label=False)

            scores = []
            for r in results:
                label_score = dict(zip(r["labels"], r["scores"]))
                bullish = label_score.get("bullish on Bitcoin", 0.5)
                bearish = label_score.get("bearish on Bitcoin", 0.5)
                scores.append(bullish - bearish)

            new_score = float(sum(scores) / len(scores)) if scores else 0.0

            self.state.change        = new_score - self._prev_score
            self._prev_score         = new_score
            self.state.score         = new_score
            self.state.article_count = len(headlines)
            self.state.last_updated  = time.time()

            log.info(
                f"News sentiment: {new_score:+.3f} "
                f"(Δ{self.state.change:+.3f}, {len(headlines)} articles classified)"
            )
        except ImportError:
            log.warning("feedparser not installed — run: pip install feedparser")
        except Exception as e:
            log.warning(f"News sentiment poll error: {e}")
