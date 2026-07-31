"""End-to-end API test for LLM answers (Phase 7, Milestone 1).

Proves the wiring: HTTP request -> ChatService -> Assistant -> LLM ->
reply. The LLM is a stub engine injected via the same dependency
override seam the other API tests use, so nothing here touches the
real model or the network.
"""

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.api.routes import get_chat_service
from app.db import build_engine, init_db
from app.llm.generator import LLMGenerator
from app.main import app
from app.persistence.message_repository import MessageRepository
from app.services.chat_service import ChatService


class StubEngine:
    def __init__(self, text: str) -> None:
        self._text = text

    def generate(self, prompt: str) -> str:
        return self._text


class TestChatWithLlm(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._engine = build_engine(
            f"sqlite:///{(Path(cls._tmpdir.name) / 'llm.db').as_posix()}"
        )
        init_db(cls._engine)
        llm = LLMGenerator(engine=StubEngine(" 4"))
        cls.service = ChatService(
            repository=MessageRepository(cls._engine),
            llm=llm,
        )
        app.dependency_overrides[get_chat_service] = lambda: cls.service
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        app.dependency_overrides.clear()
        cls._engine.dispose()
        cls._tmpdir.cleanup()

    def test_unanswered_question_gets_llm_answer(self):
        response = self.client.post(
            "/api/v1/chat",
            json={"session_id": "llm-q", "message": "what is 2+2"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["reply"], "4")

    def test_greeting_still_rules_based(self):
        response = self.client.post(
            "/api/v1/chat",
            json={"session_id": "llm-greet", "message": "hello"},
        )
        self.assertEqual(response.json()["reply"], "Hello! I'm Aria. How can I help?")


if __name__ == "__main__":
    unittest.main()
