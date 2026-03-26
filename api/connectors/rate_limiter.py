"""Token-bucket rate limiter for connector HTTP calls."""

from __future__ import annotations

import asyncio
import time


class RateLimiter:
    """Simple async token-bucket rate limiter.

    Example::

        limiter = RateLimiter(calls_per_second=0.5)
        async with limiter:
            response = await client.get(...)
    """

    def __init__(self, calls_per_second: float) -> None:
        if calls_per_second <= 0:
            raise ValueError("calls_per_second must be > 0")
        self._min_interval: float = 1.0 / calls_per_second
        self._lock = asyncio.Lock()
        self._last_call_time: float = 0.0

    async def __aenter__(self) -> "RateLimiter":
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call_time
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
            self._last_call_time = time.monotonic()
        return self

    async def __aexit__(self, *_) -> None:
        pass
