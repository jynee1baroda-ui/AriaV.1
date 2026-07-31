"""API tests for chat and history (Phase 2, Milestone 2)."""

import unittest

from fastapi.testclient import TestClient

from app.main import app


class TestChatEndpoint(unittest.TestCase):
    """POST /api/v1/chat must reply and validate."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

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


class TestHistoryEndpoint(unittest.TestCase):
    """GET /api/v1/history/{session_id} must return the conversation."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

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


class TestSessionIsolation(unittest.TestCase):
    """Sessions must never leak into each other."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_two_sessions_stay_separate(self) -> None:
        self.client.post("/api/v1/chat", json={"session_id": "iso-A", "message": "hi"})
        self.client.post("/api/v1/chat", json={"session_id": "iso-B", "message": "hey"})

        hist_a = self.client.get("/api/v1/history/iso-A").json()
        hist_b = self.client.get("/api/v1/history/iso-B").json()

        self.assertEqual(len(hist_a["messages"]), 2)
        self.assertEqual(len(hist_b["messages"]), 2)
        # Each conversation contains only its own user message
        self.assertEqual(hist_a["messages"][0]["content"], "hi")
        self.assertEqual(hist_b["messages"][0]["content"], "hey")


if __name__ == "__main__":
    unittest.main()
