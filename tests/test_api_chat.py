"""API tests for chat and history (Phase 2, Milestones 2 + 3).

These tests exercise the REAL HTTP layer (TestClient) but override the
ChatService dependency with one backed by a temporary directory, so
test sessions never pollute data/.
"""

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.api.routes import get_chat_service
from app.main import app
from app.services.chat_service import ChatService


class ChatApiTestCase(unittest.TestCase):
    """Shared setup: a temp-dir-backed service injected via Depends."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        service = ChatService(sessions_dir=Path(cls._tmpdir.name))
        # Dependency override: every request now receives THIS service.
        # This is why Depends() exists - swap an implementation without
        # touching route code.
        app.dependency_overrides[get_chat_service] = lambda: service
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        app.dependency_overrides.clear()
        cls._tmpdir.cleanup()


class TestChatEndpoint(ChatApiTestCase):
    """POST /api/v1/chat must reply and validate."""

    def test_chat_greeting_returns_reply(self) -> None:
        response = self.client.post(
            "/api/v1/chat",
            json={"session_id": "test-greet", "message": "hello"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["session_id"], "test-greet")
        self.assertEqual(payload["reply"], "Hello! I'm Aria. How can I help?")

    def test_chat_echoes_unknown_text(self) -> None:
        response = self.client.post(
            "/api/v1/chat",
            json={"session_id": "test-echo", "message": "tell me a joke"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["reply"], "You said: tell me a joke")

    def test_blank_message_rejected_with_422(self) -> None:
        response = self.client.post(
            "/api/v1/chat",
            json={"session_id": "test-blank", "message": "   "},
        )
        self.assertEqual(response.status_code, 422)

    def test_missing_session_id_rejected_with_422(self) -> None:
        response = self.client.post(
            "/api/v1/chat",
            json={"message": "hello"},
        )
        self.assertEqual(response.status_code, 422)

    def test_oversized_message_rejected_with_422(self) -> None:
        response = self.client.post(
            "/api/v1/chat",
            json={"session_id": "test-big", "message": "x" * 5000},
        )
        self.assertEqual(response.status_code, 422)


class TestSessionIdSecurity(ChatApiTestCase):
    """Path traversal attempts must be rejected at the boundary."""

    def test_slash_is_rejected(self) -> None:
        response = self.client.post(
            "/api/v1/chat",
            json={"session_id": "evil/../id", "message": "hi"},
        )
        self.assertEqual(response.status_code, 422)

    def test_dotdot_is_rejected(self) -> None:
        response = self.client.post(
            "/api/v1/chat",
            json={"session_id": "..", "message": "hi"},
        )
        self.assertEqual(response.status_code, 422)

    def test_whitespace_is_rejected(self) -> None:
        response = self.client.post(
            "/api/v1/chat",
            json={"session_id": "bad id", "message": "hi"},
        )
        self.assertEqual(response.status_code, 422)

    def test_backslash_is_rejected(self) -> None:
        response = self.client.post(
            "/api/v1/chat",
            json={"session_id": "..\\..\\evil", "message": "hi"},
        )
        self.assertEqual(response.status_code, 422)


class TestHistoryEndpoint(ChatApiTestCase):
    """GET /api/v1/history/{session_id} must return the conversation."""

    def test_history_records_both_sides_of_exchange(self) -> None:
        self.client.post(
            "/api/v1/chat",
            json={"session_id": "test-hist", "message": "hello"},
        )
        response = self.client.get("/api/v1/history/test-hist")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["session_id"], "test-hist")
        self.assertEqual(len(payload["messages"]), 2)
        self.assertEqual(payload["messages"][0]["role"], "user")
        self.assertEqual(payload["messages"][0]["content"], "hello")
        self.assertEqual(payload["messages"][1]["role"], "assistant")

    def test_unknown_session_returns_empty_history(self) -> None:
        response = self.client.get("/api/v1/history/never-heard-of-it")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["messages"], [])


class TestSessionIsolation(ChatApiTestCase):
    """Sessions must never leak into each other."""

    def test_two_sessions_stay_separate(self) -> None:
        self.client.post("/api/v1/chat", json={"session_id": "iso-A", "message": "hi"})
        self.client.post("/api/v1/chat", json={"session_id": "iso-B", "message": "hey"})

        hist_a = self.client.get("/api/v1/history/iso-A").json()
        hist_b = self.client.get("/api/v1/history/iso-B").json()

        self.assertEqual(len(hist_a["messages"]), 2)
        self.assertEqual(len(hist_b["messages"]), 2)
        self.assertEqual(hist_a["messages"][0]["content"], "hi")
        self.assertEqual(hist_b["messages"][0]["content"], "hey")


if __name__ == "__main__":
    unittest.main()
