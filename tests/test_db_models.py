"""Tests for the SQLAlchemy models (Phase 4, Milestone 1).

Every test uses its own temp SQLite file and a fresh engine, so test
runs never touch the real data/assistant.db and cannot interfere
with each other.
"""

import tempfile
import unittest
from pathlib import Path

from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import Base, build_engine, init_db
from app.persistence.sqlalchemy_models import MessageModel, SessionModel


class DbTestCase(unittest.TestCase):
    """Shared fixture: a temp-file SQLite engine + session."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.engine = build_engine(
            f"sqlite:///{(Path(self._tmpdir.name) / 'test.db').as_posix()}"
        )
        init_db(self.engine)
        self.session = Session(self.engine)

    def tearDown(self) -> None:
        self.session.close()
        self.engine.dispose()
        self._tmpdir.cleanup()


class TestSchema(DbTestCase):
    """The schema must contain sessions, messages, and the index."""

    def test_tables_exist(self) -> None:
        inspector = inspect(self.engine)
        self.assertIn("sessions", inspector.get_table_names())
        self.assertIn("messages", inspector.get_table_names())

    def test_composite_index_exists(self) -> None:
        inspector = inspect(self.engine)
        indexes = {ix["name"]: ix["column_names"] for ix in inspector.get_indexes("messages")}
        self.assertIn("ix_messages_session_id_id", indexes)
        self.assertEqual(indexes["ix_messages_session_id_id"], ["session_id", "id"])


class TestSessionAndMessage(DbTestCase):
    """Basic CRUD and ordering through the ORM."""

    def test_create_session_with_messages_and_read_in_order(self) -> None:
        session_model = SessionModel(id="abc")
        session_model.messages = [
            MessageModel(role="user", content="hello"),
            MessageModel(role="assistant", content="Hi!"),
        ]
        self.session.add(session_model)
        self.session.commit()

        stored = self.session.scalar(select(SessionModel).where(SessionModel.id == "abc"))
        self.assertIsNotNone(stored)
        self.assertEqual(len(stored.messages), 2)
        self.assertEqual(stored.messages[0].content, "hello")
        self.assertEqual(stored.messages[1].role, "assistant")

    def test_session_created_at_is_set(self) -> None:
        session_model = SessionModel(id="ts")
        self.session.add(session_model)
        self.session.commit()
        stored = self.session.scalar(select(SessionModel).where(SessionModel.id == "ts"))
        self.assertIsNotNone(stored.created_at)

    def test_message_requires_valid_role(self) -> None:
        # The CHECK constraint must reject an unknown role at the DB.
        session_model = SessionModel(id="roles")
        self.session.add(session_model)
        self.session.commit()
        self.session.add(MessageModel(session_id="roles", role="robot", content="beep"))
        with self.assertRaises(IntegrityError):
            self.session.commit()
        self.session.rollback()

    def test_cascade_delete_removes_messages(self) -> None:
        session_model = SessionModel(id="del")
        session_model.messages = [
            MessageModel(role="user", content="x"),
            MessageModel(role="assistant", content="y"),
        ]
        self.session.add(session_model)
        self.session.commit()

        self.session.delete(session_model)
        self.session.commit()

        remaining = self.session.scalars(select(MessageModel)).all()
        self.assertEqual(remaining, [])

    def test_foreign_key_requires_existing_session(self) -> None:
        # FK constraint: can't attach a message to a missing session.
        self.session.add(MessageModel(session_id="missing", role="user", content="x"))
        with self.assertRaises(IntegrityError):
            self.session.commit()
        self.session.rollback()


if __name__ == "__main__":
    unittest.main()
