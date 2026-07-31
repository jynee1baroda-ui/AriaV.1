"""Tests for MessageRepository (Phase 4, Milestone 2)."""

import tempfile
import unittest
from pathlib import Path

from sqlalchemy import func, select

from app.db import build_engine, init_db
from app.errors import StorageError
from app.models import Message
from app.persistence.message_repository import MessageRepository
from app.persistence.sqlalchemy_models import SessionModel


class RepositoryTestCase(unittest.TestCase):
    """Fixture: repository over a temp SQLite database."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.engine = build_engine(
            f"sqlite:///{(Path(self._tmpdir.name) / 'repo.db').as_posix()}"
        )
        init_db(self.engine)
        self.repository = MessageRepository(self.engine)

    def tearDown(self) -> None:
        self.engine.dispose()
        self._tmpdir.cleanup()


class TestGetMessages(RepositoryTestCase):
    """Reading must return ordered domain Messages."""

    def test_unknown_session_returns_empty_list(self) -> None:
        self.assertEqual(self.repository.get_messages("ghost"), [])

    def test_returns_messages_in_insertion_order(self) -> None:
        self.repository.add_messages(
            "s1",
            [Message("user", "a"), Message("assistant", "b")],
        )
        self.repository.add_messages("s1", [Message("user", "c")])

        messages = self.repository.get_messages("s1")
        self.assertEqual([m.content for m in messages], ["a", "b", "c"])
        self.assertEqual(messages[0].role, "user")
        self.assertEqual(messages[1].role, "assistant")


class TestAddMessages(RepositoryTestCase):
    """Writing must upsert the session and insert rows."""

    def test_add_creates_session_and_rows(self) -> None:
        self.repository.add_messages("s2", [Message("user", "hi")])
        messages = self.repository.get_messages("s2")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].content, "hi")

    def test_adding_twice_does_not_duplicate_session(self) -> None:
        self.repository.add_messages("s3", [Message("user", "x")])
        self.repository.add_messages("s3", [Message("assistant", "y")])
        # Exactly one session row exists
        with self.engine.connect() as conn:
            count = conn.scalar(
                select(func.count(SessionModel.id)).where(SessionModel.id == "s3")
            )
        self.assertEqual(count, 1)
        self.assertEqual(len(self.repository.get_messages("s3")), 2)

    def test_failure_raises_storage_error(self) -> None:
        # A path in a directory that cannot exist -> connect fails.
        broken = build_engine("sqlite:///Z:/definitely-missing-dir/db.sqlite")
        repository = MessageRepository(broken)
        with self.assertRaises(StorageError):
            repository.add_messages("s4", [Message("user", "hi")])


if __name__ == "__main__":
    unittest.main()
