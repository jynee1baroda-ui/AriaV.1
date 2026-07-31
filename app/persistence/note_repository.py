"""NoteRepository - the durable store for the notes tool.

Phase 11 foundation (tool calling preview). Mirrors MemoryRepository's
contract exactly:
  * ORM rows in, DOMAIN Note objects out
  * every driver failure wrapped in StorageError (chained, not swallowed)
  * engine injectable so tests use a throwaway temp database

The repository is the notes tool's ONLY database seam - NotesTool talks
to this, never to SQL directly.
"""

from __future__ import annotations

import logging

from sqlalchemy import Engine, delete, func, select
from sqlalchemy.orm import Session

from app import db
from app.errors import StorageError
from app.models import Note
from app.persistence.sqlalchemy_models import NoteModel, SessionModel

logger = logging.getLogger("assistant.storage")


class NoteRepository:
    """Read and write notes in a relational database.

    Attributes:
        _engine: Injectable SQLAlchemy engine (tests use a temp DB).
    """

    def __init__(self, engine: Engine | None = None) -> None:
        self._engine: Engine = engine or db.engine

    @property
    def engine(self) -> Engine:
        """The backing engine, for sharing a test-injected temp DB."""
        return self._engine

    def add_note(self, content: str, session_id: str | None = None) -> Note:
        """Store one note and return it with its new id.

        Like memories, a note OUTLIVES its source session: session_id
        is provenance. When a session_id is given, the session row is
        created if missing so the foreign key never fails on a not-yet
        persisted session.

        Args:
            content: The note text.
            session_id: Which session owns this note (or None).

        Returns:
            The stored Note, including the database-assigned id.

        Raises:
            StorageError: If the insert fails (transaction rolled back).
        """
        try:
            with Session(self._engine) as session, session.begin():
                if session_id is not None and session.get(SessionModel, session_id) is None:
                    session.add(SessionModel(id=session_id))
                row = NoteModel(content=content, session_id=session_id)
                session.add(row)
                session.flush()
                return self._to_note(row)
        except Exception as exc:
            logger.exception("Failed to add note")
            raise StorageError("Failed to add note") from exc

    def get_by_session(self, session_id: str | None) -> list[Note]:
        """Return a session's notes, newest first.

        Args:
            session_id: The session whose notes to load (None = global
                notes not tied to any session).

        Returns:
            That session's notes ordered by recency.

        Raises:
            StorageError: If the query fails.
        """
        try:
            with Session(self._engine) as session:
                stmt = (
                    select(NoteModel)
                    .where(NoteModel.session_id == session_id)
                    .order_by(NoteModel.created_at.desc(), NoteModel.id.desc())
                )
                return [self._to_note(row) for row in session.scalars(stmt).all()]
        except Exception as exc:
            logger.exception("Failed to load notes for session %r", session_id)
            raise StorageError(f"Failed to load notes for session {session_id!r}") from exc

    def get_all(self) -> list[Note]:
        """Return every note, newest first.

        Raises:
            StorageError: If the query fails.
        """
        try:
            with Session(self._engine) as session:
                stmt = select(NoteModel).order_by(
                    NoteModel.created_at.desc(), NoteModel.id.desc()
                )
                return [self._to_note(row) for row in session.scalars(stmt).all()]
        except Exception as exc:
            logger.exception("Failed to load notes")
            raise StorageError("Failed to load notes") from exc

    def delete_note(self, note_id: int) -> bool:
        """Delete one note by its id.

        Args:
            note_id: The id returned by add_note().

        Returns:
            True if a row was deleted, False if no such note existed.

        Raises:
            StorageError: If the delete fails.
        """
        try:
            with Session(self._engine) as session, session.begin():
                result = session.execute(delete(NoteModel).where(NoteModel.id == note_id))
                return result.rowcount > 0
        except Exception as exc:
            logger.exception("Failed to delete note %s", note_id)
            raise StorageError(f"Failed to delete note {note_id}") from exc

    def count(self) -> int:
        """Count all stored notes.

        Raises:
            StorageError: If the query fails.
        """
        try:
            with Session(self._engine) as session:
                return int(session.scalar(select(func.count()).select_from(NoteModel)) or 0)
        except Exception as exc:
            logger.exception("Failed to count notes")
            raise StorageError("Failed to count notes") from exc

    @staticmethod
    def _to_note(row: NoteModel) -> Note:
        """Translate an ORM row into a domain Note object."""
        return Note(
            id=row.id,
            content=row.content,
            session_id=row.session_id,
            created_at=row.created_at,
        )
