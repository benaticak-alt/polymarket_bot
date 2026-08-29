"""
Reddit crypto sentiment via public RSS feeds + BART MNLI zero-shot classification.
No Reddit API key required.

Requires: pip install feedparser transformers torch
First run downloads ~1.6 GB (facebook/bart-large-mnli). Cached after that.
"""

import asyncio
import time
from dataclasses import dataclass
from utils.logger import get_logger
from data.bart_classifier import get_classifier

log = get_logger("reddit_sentiment")

REDDIT_FEEDS = [
    "https://www.reddit.com/r/Bitcoin/hot/.rss",
    "https://www.reddit.com/r/CryptoCurrency/hot/.rss",
    "https://www.reddit.com/r/btc/hot/.rss",
]
POLL_INTERVAL = 300    # 5 minutes
MAX_POSTS     = 15     # posts per subreddit
LABELS        = ["bullish on Bitcoin", "bearish on Bitcoin"]


@dataclass
class RedditSentimentState:
    score: float       = 0.0   # -1 (bearish) to 1 (bullish)
    change: float      = 0.0   # change from previous poll
    post_count: int    = 0
    last_updated: float = 0.0


class RedditSentimentClient:

    def __init__(self, state: RedditSentimentState):
        self.state       = state
        self._prev_score = 0.0
        self._classifier = None
        self._load_model()

    def _load_model(self) -> None:
        self._classifier = get_classifier()

    async def run(self) -> None:
        while True:
            if self._classifier:
                # Run blocking inference in a thread pool so it doesn't block the loop
                await asyncio.get_event_loop().run_in_executor(None, self._poll)
            await asyncio.sleep(POLL_INTERVAL)

    def _poll(self) -> None:
        try:
            import feedparser
            titles = []
            for feed_url in REDDIT_FEEDS:
                try:
                    feed = feedparser.parse(feed_url)
                    for entry in feed.entries[:MAX_POSTS]:
                        title = entry.get("title", "").strip()
                        if title and len(title) > 5:
                            titles.append(title)
                except Exception as e:
                    log.debug(f"RSS fetch error for {feed_url}: {e}")

            if not titles:
                log.debug("No Reddit posts fetched.")
                return

            results = self._classifier(titles, LABELS, multi_label=False)

            scores = []
            for r in results:
                label_score = dict(zip(r["labels"], r["scores"]))
                bullish = label_score.get("bullish on Bitcoin", 0.5)
                bearish = label_score.get("bearish on Bitcoin", 0.5)
                scores.append(bullish - bearish)

            new_score = float(sum(scores) / len(scores)) if scores else 0.0

            self.state.change       = new_score - self._prev_score
            self._prev_score        = new_score
            self.state.score        = new_score
            self.state.post_count   = len(titles)
            self.state.last_updated = time.time()

            log.info(
                f"Reddit sentiment: {new_score:+.3f} "
                f"(Δ{self.state.change:+.3f}, {len(titles)} posts classified)"
            )
        except Exception as e:
            log.warning(f"Reddit sentiment poll error: {e}")
