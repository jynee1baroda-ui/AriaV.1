"""Unit tests for ChatService (Phase 2 M3 + Phase 4 M2: persistence).

NOTE the TempDbTestCase pattern: engines are disposed in tearDown
BEFORE the temp directory is removed. Windows refuses to delete a
file while SQLite still holds a handle on it (WinError 32) - POSIX
is lenient, which is why this ordering is Windows-critical.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from app.db import build_engine, init_db
from app.errors import InvalidSessionIdError
from app.persistence.message_repository import MessageRepository
from app.services.chat_service import ChatService


class TempDbTestCase(unittest.TestCase):
    """Shared fixture: fresh temp DB per test, engine disposed first."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self._engines = []

    def tearDown(self) -> None:
        # Order matters on Windows: release the file handles BEFORE
        # deleting the directory they live in.
        for engine in self._engines:
            engine.dispose()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def make_service(self) -> ChatService:
        """A ChatService backed by a repository on a fresh temp DB."""
        engine = build_engine(
            f"sqlite:///{(Path(self._tmpdir) / 'chat.db').as_posix()}"
        )
        init_db(engine)
        self._engines.append(engine)
        return ChatService(repository=MessageRepository(engine))


class TestChatServicePersistence(TempDbTestCase):
    """Conversations must survive service restarts via the database."""

    def test_history_survives_restart(self) -> None:
        # First "server life"
        service_one = self.make_service()
        service_one.process("restart-demo", "hello")

        # Second "server life" - fresh service, same database.
        # A restart forgets RAM; it must remember the DB rows.
        service_two = self.make_service()
        service_two.process("restart-demo", "what is your name")

        history = service_two.get_history("restart-demo")
        self.assertEqual(len(history), 4)
        self.assertEqual(history[0].content, "hello")
        self.assertEqual(history[2].content, "what is your name")

    def test_new_instance_reads_other_sessions(self) -> None:
        service_one = self.make_service()
        service_one.process("solo", "hi")

        service_two = self.make_service()
        history = service_two.get_history("solo")
        self.assertEqual(len(history), 2)


class TestChatServiceValidation(TempDbTestCase):
    """The service's own defense-in-depth must reject bad ids."""

    def test_slash_rejected(self) -> None:
        service = self.make_service()
        with self.assertRaises(InvalidSessionIdError):
            service.process("../../escape", "hi")

    def test_dotdot_rejected_in_get_history(self) -> None:
        service = self.make_service()
        with self.assertRaises(InvalidSessionIdError):
            service.get_history("..")

    def test_valid_id_accepted(self) -> None:
        service = self.make_service()
        reply = service.process("user_1-ok", "hello")
        self.assertEqual(reply, "Hello! I'm Aria. How can I help?")


if __name__ == "__main__":
    unittest.main()
