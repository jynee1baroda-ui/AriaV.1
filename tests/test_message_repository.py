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


class TestGetMessagesPagination(RepositoryTestCase):
    """limit/offset must page without changing order or missing rows."""

    def setUp(self) -> None:
        super().setUp()
        self.repository.add_messages(
            "paged",
            [
                Message("user", f"m{i}")
                for i in range(5)
            ],
        )

    def test_pages_in_order(self) -> None:
        first = self.repository.get_messages("paged", limit=2, offset=0)
        self.assertEqual([m.content for m in first], ["m0", "m1"])

        second = self.repository.get_messages("paged", limit=2, offset=2)
        self.assertEqual([m.content for m in second], ["m2", "m3"])

    def test_last_page_is_shorter(self) -> None:
        last = self.repository.get_messages("paged", limit=2, offset=4)
        self.assertEqual([m.content for m in last], ["m4"])

    def test_offset_beyond_end_returns_empty(self) -> None:
        self.assertEqual(self.repository.get_messages("paged", limit=2, offset=99), [])

    def test_no_limit_returns_everything(self) -> None:
        self.assertEqual(len(self.repository.get_messages("paged")), 5)

    def test_count_messages(self) -> None:
        self.assertEqual(self.repository.count_messages("paged"), 5)

    def test_count_unknown_session_is_zero(self) -> None:
        self.assertEqual(self.repository.count_messages("ghost"), 0)


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
