"""Rate limiting - protects the API from abuse.

Phase 6, Milestone 1. A fixed-window counter per client key (the client
IP, until real authentication arrives in a later phase). In-memory and
process-local: correct for the single-process uvicorn deployment, and
the RateLimiter.check() interface is the seam a shared store (Redis)
would plug into for a multi-process deployment - callers never change.

WHY FIXED WINDOW AND NOT A SLIDING WINDOW?
A fixed window allows a burst right at a window boundary (e.g. 60
requests in the final second of one window plus the first second of the
next). A token bucket or sliding-log limiter smooths that. This is a
deliberate, documented tradeoff for teaching the simplest correct
solution first - not an oversight.
"""

from __future__ import annotations

import threading
import time


class RateLimiter:
    """Thread-safe, fixed-window rate limiter, in memory.

    Each client key gets a window that starts at its first request.
    Requests within [start, start + window) count toward the limit; when
    the window elapses the counter resets.

    Attributes:
        _limit: Max requests allowed per window.
        _window: Window length in seconds.
        _max_keys: Size guard - stale entries are pruned above this.
    """

    def __init__(self, limit: int, window_seconds: int, max_keys: int = 10_000) -> None:
        self._limit = limit
        self._window = window_seconds
        self._max_keys = max_keys
        self._lock = threading.Lock()
        # key -> (window_start, requests_so_far)
        self._clients: dict[str, tuple[float, int]] = {}

    def check(self, key: str) -> tuple[bool, int]:
        """Record one request for `key` and report whether it may pass.

        Args:
            key: The client identity (an IP address for now).

        Returns:
            (allowed, retry_after_seconds). When allowed is False the
            request is rejected; retry_after is how many seconds to wait
            before trying again (always >= 1).
        """
        with self._lock:
            now = time.monotonic()
            window_start, count = self._clients.get(key, (now, 0))

            # The previous window has elapsed - start a fresh one.
            if now - window_start >= self._window:
                window_start, count = now, 0

            if count >= self._limit:
                retry_after = int(self._window - (now - window_start)) + 1
                return False, retry_after

            self._clients[key] = (window_start, count + 1)
            self._prune_if_large()
            return True, 0

    def _prune_if_large(self) -> None:
        """Drop stale windows once the table grows past the size guard.

        Without this the map grows forever as distinct clients
        accumulate; pruning expired windows bounds it to roughly the
        number of clients active within one window.
        """
        if len(self._clients) <= self._max_keys:
            return
        now = time.monotonic()
        expired = [k for k, (s, _) in self._clients.items() if now - s >= self._window]
        for key in expired:
            del self._clients[key]
