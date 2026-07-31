"""Tests for NoteRepository (Phase 11 foundation - tool calling preview).

Mirrors the MemoryRepository test conventions: a throwaway temp
database, ORM rows translated to domain Note objects, and every failure
wrapped in StorageError. Each test gets a FRESH temp database so rows
never leak between tests.
"""

import tempfile
import unittest
from pathlib import Path

from app.db import build_engine, init_db
from app.models import Note
from app.persistence.note_repository import NoteRepository


class TestNoteRepository(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._engine = build_engine(
            f"sqlite:///{(Path(self._tmpdir.name) / 'notes.db').as_posix()}"
        )
        init_db(self._engine)
        self._repo = NoteRepository(self._engine)

    def tearDown(self) -> None:
        self._engine.dispose()
        self._tmpdir.cleanup()

    def test_add_note_returns_complete_domain_object(self):
        note = self._repo.add_note("buy milk", session_id="s1")
        self.assertIsInstance(note, Note)
        self.assertEqual(note.content, "buy milk")
        self.assertEqual(note.session_id, "s1")
        self.assertIsNotNone(note.id)
        self.assertIsNotNone(note.created_at)

    def test_get_by_session_returns_only_that_session(self):
        self._repo.add_note("note a", session_id="s1")
        self._repo.add_note("note b", session_id="s2")
        self.assertEqual([n.content for n in self._repo.get_by_session("s1")], ["note a"])
        self.assertEqual([n.content for n in self._repo.get_by_session("s2")], ["note b"])

    def test_get_by_session_none_for_global_notes(self):
        self._repo.add_note("global note", session_id=None)
        self.assertEqual([n.content for n in self._repo.get_by_session(None)], ["global note"])

    def test_newest_first_ordering(self):
        self._repo.add_note("first", session_id="s3")
        self._repo.add_note("second", session_id="s3")
        self._repo.add_note("third", session_id="s3")
        contents = [n.content for n in self._repo.get_by_session("s3")]
        self.assertEqual(contents, ["third", "second", "first"])

    def test_get_all_returns_everything(self):
        self._repo.add_note("a", session_id="s1")
        self._repo.add_note("b", session_id="s2")
        self.assertEqual(self._repo.count(), 2)

    def test_delete_note(self):
        note = self._repo.add_note("to delete", session_id="s1")
        self.assertTrue(self._repo.delete_note(note.id))
        self.assertEqual(self._repo.get_by_session("s1"), [])

    def test_delete_missing_note_returns_false(self):
        self.assertFalse(self._repo.delete_note(999999))

    def test_session_row_is_created_on_demand(self):
        """A note may reference a session that never stored a message."""
        note = self._repo.add_note("orphan session note", session_id="brand-new")
        self.assertEqual(note.session_id, "brand-new")

    def test_count(self):
        self.assertEqual(self._repo.count(), 0)
        self._repo.add_note("one")
        self.assertEqual(self._repo.count(), 1)


if __name__ == "__main__":
    unittest.main()
