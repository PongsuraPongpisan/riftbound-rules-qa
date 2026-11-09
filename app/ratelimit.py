"""In-memory rate limits for the public app: a per-IP limit per minute and a global daily cap.

Kept in memory on purpose (one small server, no database). Limits reset when the server restarts.
"""
import time
from collections import defaultdict, deque
from datetime import datetime, timezone


class RateLimiter:
    def __init__(self, per_minute: int, per_day: int):
        self.per_minute = per_minute
        self.per_day = per_day
        self.recent: dict[str, deque] = defaultdict(deque)
        self.day = ""
        self.day_count = 0

    def check(self, client_id: str, now: float | None = None) -> str | None:
        """Return None if the request may go ahead, otherwise a message explaining the limit."""
        now = time.time() if now is None else now
        today = datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%d")
        if today != self.day:
            self.day, self.day_count = today, 0
        if self.day_count >= self.per_day:
            return "The app has reached its daily question limit. It resets at midnight UTC."
        q = self.recent[client_id]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.per_minute:
            return f"You can ask {self.per_minute} questions a minute. Wait a moment and try again."
        q.append(now)
        self.day_count += 1
        return None
