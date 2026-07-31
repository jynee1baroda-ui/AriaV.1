"""Tests for MemoryRepository (Phase 5, Milestone 1).

The storage half of long-term memory. Beyond plain CRUD, the central
BEHAVIOR tested here is the ownership policy: a memory belongs to the
ASSISTANT, not to a session. Deleting a session must keep its memories
(only the provenance link is dropped) - the opposite of messages, which
cascade away.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sqlalchemy import delete

from app.db import build_engine, init_db
from app.errors import StorageError
from app.persistence.memory_repository import MemoryRepository
from app.persistence.sqlalchemy_models import SessionModel


class RepositoryTestCase(unittest.TestCase):
    """Fixture: repository over a temp SQLite database."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.engine = build_engine(
            f"sqlite:///{(Path(self._tmpdir.name) / 'mem.db').as_posix()}"
        )
        init_db(self.engine)
        self.repository = MemoryRepository(self.engine)

    def tearDown(self) -> None:
        # Dispose the engine BEFORE removing the tempdir: a live pool
        # still holds the SQLite file open, and Windows refuses to
        # delete an open file (WinError 32) - see storage.py.
        self.engine.dispose()
        self._tmpdir.cleanup()


class TestAddMemory(RepositoryTestCase):
    def test_returns_memory_with_assigned_id(self) -> None:
        memory = self.repository.add_memory("name is Alice")
        self.assertIsInstance(memory.id, int)
        self.assertEqual(memory.content, "name is Alice")
        self.assertEqual(memory.category, "fact")
        self.assertIsNone(memory.session_id)

    def test_stores_category_and_provenance(self) -> None:
        memory = self.repository.add_memory(
            "likes dark mode", category="preference", session_id="s1"
        )
        self.assertEqual(memory.category, "preference")
        self.assertEqual(memory.session_id, "s1")
        self.assertEqual(self.repository.count(), 1)


class TestGetAll(RepositoryTestCase):
    def test_newest_first_ordering(self) -> None:
        self.repository.add_memory("first")
        self.repository.add_memory("second")
        self.repository.add_memory("third")
        contents = [m.content for m in self.repository.get_all()]
        # Stored order was first, second, third -> newest first means third, second, first
        self.assertEqual(contents, ["third", "second", "first"])

    def test_empty_repository_returns_empty_list(self) -> None:
        self.assertEqual(self.repository.get_all(), [])


class TestGetRecent(RepositoryTestCase):
    def test_returns_newest_within_limit(self) -> None:
        for i in range(5):
            self.repository.add_memory(f"fact-{i}")
        recent = self.repository.get_recent(limit=3)
        self.assertEqual(len(recent), 3)
        self.assertEqual(recent[0].content, "fact-4")
        self.assertEqual(recent[2].content, "fact-2")


class TestGetBySession(RepositoryTestCase):
    def test_filters_by_provenance_session(self) -> None:
        self.repository.add_memory("from a", session_id="a")
        self.repository.add_memory("from b", session_id="b")
        self.repository.add_memory("also from a", session_id="a")
        sessions_a = self.repository.get_by_session("a")
        self.assertEqual(len(sessions_a), 2)
        self.assertTrue(all(m.session_id == "a" for m in sessions_a))


class TestDeleteMemory(RepositoryTestCase):
    def test_delete_returns_true_for_existing(self) -> None:
        memory = self.repository.add_memory("doomed")
        self.assertTrue(self.repository.delete_memory(memory.id))
        self.assertEqual(self.repository.count(), 0)

    def test_delete_returns_false_for_missing(self) -> None:
        self.assertFalse(self.repository.delete_memory(9999))


class TestOwnershipPolicy(RepositoryTestCase):
    """The memory-outlives-session behavior that justifies SET NULL."""

    def test_deleting_session_keeps_memory(self) -> None:
        self.repository.add_memory("name is Alice", session_id="s1")
        self.repository.add_memory("likes coffee", session_id="s1")

        # Delete the source session entirely.
        with self.engine.begin() as conn:
            conn.execute(delete(SessionModel).where(SessionModel.id == "s1"))

        remaining = self.repository.get_all()
        self.assertEqual(len(remaining), 2)
        # Provenance is gone but the facts survive (SET NULL).
        self.assertTrue(all(m.session_id is None for m in remaining))

    def test_deleting_session_does_not_delete_other_sessions_memories(self) -> None:
        self.repository.add_memory("from s1", session_id="s1")
        self.repository.add_memory("from s2", session_id="s2")

        with self.engine.begin() as conn:
            conn.execute(delete(SessionModel).where(SessionModel.id == "s1"))

        # BOTH facts survive: s1's as an orphaned memory (SET NULL drops
        # provenance only), s2's with its link intact.
        survivors = self.repository.get_all()
        self.assertEqual(len(survivors), 2)
        provenance = {m.content: m.session_id for m in survivors}
        self.assertIsNone(provenance["from s1"])
        self.assertEqual(provenance["from s2"], "s2")


class TestFailureRaisesStorageError(RepositoryTestCase):
    def test_connect_failure_wraps_in_storage_error(self) -> None:
        broken = build_engine("sqlite:///Z:/definitely-missing-dir/db.sqlite")
        repository = MemoryRepository(broken)
        with self.assertRaises(StorageError):
            repository.add_memory("boom")


if __name__ == "__main__":
    unittest.main()
