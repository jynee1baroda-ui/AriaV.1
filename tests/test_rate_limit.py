"""Tests for rate limiting (Phase 6, Milestone 1).

Unit tests pin the RateLimiter's fixed-window behavior; the API test
proves the HTTP layer rejects the third request with 429 + Retry-After
when the limit is 2. The limiter is injected via dependency override so
tests never touch the real per-IP counters.
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.api.routes import get_chat_service, get_rate_limiter
from app.db import build_engine, init_db
from app.main import app
from app.persistence.message_repository import MessageRepository
from app.security.rate_limiter import RateLimiter
from app.services.chat_service import ChatService


class TestRateLimiterUnit(unittest.TestCase):
    def test_allows_requests_up_to_limit(self) -> None:
        limiter = RateLimiter(limit=3, window_seconds=60)
        self.assertEqual(limiter.check("ip-a"), (True, 0))
        self.assertEqual(limiter.check("ip-a"), (True, 0))
        self.assertEqual(limiter.check("ip-a"), (True, 0))

    def test_rejects_requests_over_limit_with_retry_after(self) -> None:
        limiter = RateLimiter(limit=2, window_seconds=60)
        limiter.check("ip-a")
        limiter.check("ip-a")
        allowed, retry_after = limiter.check("ip-a")
        self.assertFalse(allowed)
        self.assertGreaterEqual(retry_after, 1)

    def test_clients_are_isolated(self) -> None:
        limiter = RateLimiter(limit=1, window_seconds=60)
        limiter.check("ip-a")
        self.assertEqual(limiter.check("ip-b"), (True, 0))

    def test_window_resets_after_it_elapses(self) -> None:
        limiter = RateLimiter(limit=1, window_seconds=1)
        limiter.check("ip-a")
        self.assertFalse(limiter.check("ip-a")[0])
        time.sleep(1.1)
        self.assertEqual(limiter.check("ip-a"), (True, 0))


class TestRateLimitApi(unittest.TestCase):
    """The HTTP layer must 429 when the injected limiter says no."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._engine = build_engine(
            f"sqlite:///{(Path(cls._tmpdir.name) / 'rl.db').as_posix()}"
        )
        init_db(cls._engine)
        cls.service = ChatService(repository=MessageRepository(cls._engine))
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        app.dependency_overrides.clear()
        cls._engine.dispose()
        cls._tmpdir.cleanup()

    def setUp(self) -> None:
        # A FRESH limiter per test: counters must never leak between
        # test methods (they share the process-wide overrides map).
        # IMPORTANT: capture ONE instance - a lambda that constructs a
        # new RateLimiter per call would reset the counter every request.
        self._limiter = RateLimiter(limit=2, window_seconds=60)
        app.dependency_overrides[get_chat_service] = lambda: self.service
        app.dependency_overrides[get_rate_limiter] = lambda: self._limiter

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_third_request_within_limit_of_two_is_429(self) -> None:
        for _ in range(2):
            response = self.client.post(
                "/api/v1/chat", json={"session_id": "rl", "message": "hi"}
            )
            self.assertEqual(response.status_code, 200)

        response = self.client.post(
            "/api/v1/chat", json={"session_id": "rl", "message": "hi"}
        )
        self.assertEqual(response.status_code, 429)
        # Retry-After tells the client how long to wait; the window is
        # fresh, so it should be positive (roughly the full 60s).
        self.assertGreaterEqual(int(response.headers.get("retry-after", "0")), 1)
        self.assertIn("Too many requests", response.json()["detail"])

    def test_health_probe_is_never_rate_limited(self) -> None:
        self.client.post("/api/v1/chat", json={"session_id": "rl2", "message": "hi"})
        self.client.post("/api/v1/chat", json={"session_id": "rl2", "message": "hi"})
        # Even past the chat limit, the health probe must answer.
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
