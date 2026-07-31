"""Tests for readiness and connectivity (Phase 4, Milestone 3)."""

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.api.routes import get_chat_service
from app.db import build_engine, init_db
from app.main import app
from app.persistence.message_repository import MessageRepository
from app.services.chat_service import ChatService


class TestReadinessEndpoint(unittest.TestCase):
    """GET /api/v1/ready must reflect real database health."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        engine = build_engine(
            f"sqlite:///{(Path(self._tmpdir.name) / 'ready.db').as_posix()}"
        )
        init_db(engine)
        self._engine = engine
        self._healthy_service = ChatService(repository=MessageRepository(engine))

    def tearDown(self) -> None:
        self._engine.dispose()
        self._tmpdir.cleanup()

    def test_ready_returns_200_when_database_ok(self) -> None:
        app.dependency_overrides[get_chat_service] = lambda: self._healthy_service
        try:
            client = TestClient(app)
            response = client.get("/api/v1/ready")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "ready", "database": "ok"})
        finally:
            app.dependency_overrides.clear()

    def test_ready_returns_503_when_database_down(self) -> None:
        # A repository whose engine points at an impossible location =
        # a database that is down from the service's perspective.
        broken_engine = build_engine("sqlite:///Z:/missing-dir/definitely.db")
        broken_service = ChatService(repository=MessageRepository(broken_engine))
        app.dependency_overrides[get_chat_service] = lambda: broken_service
        try:
            client = TestClient(app)
            response = client.get("/api/v1/ready")
            self.assertEqual(response.status_code, 503)
            self.assertEqual(
                response.json(), {"status": "not_ready", "database": "unavailable"}
            )
        finally:
            app.dependency_overrides.clear()
            broken_engine.dispose()


class TestPing(unittest.TestCase):
    """ChatService.ping must report repository connectivity."""

    def test_ping_true_with_working_db(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = build_engine(
                f"sqlite:///{(Path(tmpdir) / 'p.db').as_posix()}"
            )
            service = ChatService(repository=MessageRepository(engine))
            self.assertTrue(service.ping())
            engine.dispose()

    def test_ping_false_with_broken_db(self) -> None:
        engine = build_engine("sqlite:///Z:/missing-dir/p.db")
        service = ChatService(repository=MessageRepository(engine))
        self.assertFalse(service.ping())
        engine.dispose()


if __name__ == "__main__":
    unittest.main()
