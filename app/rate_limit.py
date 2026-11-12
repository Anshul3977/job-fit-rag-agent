"""In-memory limit for /fit so a free Groq key is harder to burn."""

from __future__ import annotations

import threading
import time
from collections import defaultdict

from app.errors import AppError


class RateLimiter:
    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._hits: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        if self.limit <= 0:
            return
        now = time.monotonic()
        with self._lock:
            recent = [stamp for stamp in self._hits[key] if now - stamp < self.window_seconds]
            if len(recent) >= self.limit:
                self._hits[key] = recent
                raise AppError(
                    429,
                    "RATE_LIMITED",
                    f"Too many fit requests. The limit is {self.limit} per minute.",
                )
            recent.append(now)
            self._hits[key] = recent

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


_limiter: RateLimiter | None = None


def get_rate_limiter(limit: int) -> RateLimiter:
    global _limiter
    if _limiter is None or _limiter.limit != limit:
        _limiter = RateLimiter(limit)
    return _limiter


def reset_rate_limiter() -> None:
    global _limiter
    _limiter = None
