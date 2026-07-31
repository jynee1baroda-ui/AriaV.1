"""Tests for MemoryService (Phase 5, Milestone 2).

Consolidation must extract + dedup + persist; retrieval must pull the
right memories for a question. Both run against a throwaway temp DB.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.db import build_engine, init_db
from app.persistence.memory_repository import MemoryRepository
from app.services.memory_service import MemoryService


class MemoryServiceTestCase(unittest.TestCase):
    """Fixture: service over a temp SQLite database."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.engine = build_engine(
            f"sqlite:///{(Path(self._tmpdir.name) / 'svc.db').as_posix()}"
        )
        init_db(self.engine)
        self.service = MemoryService(MemoryRepository(self.engine))

    def tearDown(self) -> None:
        self.engine.dispose()
        self._tmpdir.cleanup()


class TestConsolidate(MemoryServiceTestCase):
    def test_stores_new_fact(self) -> None:
        added = self.service.consolidate("s1", "my name is Alice")
        self.assertEqual(len(added), 1)
        self.assertEqual(self.service.get_all()[0].content, "the user's name is Alice")

    def test_dedup_repeated_fact(self) -> None:
        self.service.consolidate("s1", "my name is Alice")
        second = self.service.consolidate("s2", "my name is Alice")
        self.assertEqual(second, [])
        self.assertEqual(len(self.service.get_all()), 1)

    def test_irrelevant_message_stores_nothing(self) -> None:
        added = self.service.consolidate("s1", "hello there")
        self.assertEqual(added, [])

    def test_provenance_recorded(self) -> None:
        self.service.consolidate("origin-session", "i like coffee")
        stored = self.service.get_all()[0]
        self.assertEqual(stored.session_id, "origin-session")


class TestRetrieve(MemoryServiceTestCase):
    def test_name_question_finds_name_memory(self) -> None:
        self.service.consolidate("s1", "my name is Alice")
        hits = self.service.retrieve("what is my name?")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].content, "the user's name is Alice")

    def test_favorite_question_finds_favorite_memory(self) -> None:
        self.service.consolidate("s1", "my favorite book is Dune")
        hits = self.service.retrieve("what is my favorite book?")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].content, "the user's favorite book is Dune")

    def test_like_question_finds_like_memory(self) -> None:
        self.service.consolidate("s1", "i like coffee")
        hits = self.service.retrieve("do i like coffee?")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].content, "the user likes coffee")

    def test_irrelevant_query_retrieves_nothing(self) -> None:
        self.service.consolidate("s1", "i like coffee")
        self.assertEqual(self.service.retrieve("tell me a joke"), [])


class TestAddDirect(MemoryServiceTestCase):
    def test_remember_command_style(self) -> None:
        memory = self.service.add("the sky is green", category="fact", session_id="cli")
        self.assertEqual(self.service.get_all()[0].content, "the sky is green")
        self.assertEqual(memory.session_id, "cli")


if __name__ == "__main__":
    unittest.main()
