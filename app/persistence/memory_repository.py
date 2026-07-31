"""MemoryRepository - the durable long-term memory store.

Phase 5, Milestone 1. Storage layer ONLY: the memories table schema plus
read/write operations. The *semantics* - consolidating conversation
turns into facts and retrieving them into working context - arrive in
M2. Keeping storage separate from meaning is what lets us swap in RAG
(Phase 8) without touching this seam.

Mirrors MessageRepository's contract exactly:
  * ORM rows in, DOMAIN Memory objects out
  * every driver failure wrapped in StorageError (chained, not swallowed)
  * engine injectable so tests use a throwaway temp database
"""

from __future__ import annotations

import logging

from sqlalchemy import Engine, delete, func, select
from sqlalchemy.orm import Session

from app import db
from app.errors import StorageError
from app.models import Memory
from app.persistence.sqlalchemy_models import MemoryModel, SessionModel

logger = logging.getLogger("assistant.storage")


class MemoryRepository:
    """Read and write long-term memories in a relational database.

    Attributes:
        _engine: Injectable SQLAlchemy engine (tests use a temp DB).
    """

    def __init__(self, engine: Engine | None = None) -> None:
        self._engine: Engine = engine or db.engine

    def add_memory(
        self,
        content: str,
        category: str = "fact",
        session_id: str | None = None,
    ) -> Memory:
        """Store one durable memory and return it with its new id.

        A memory OUTLIVES its source session: session_id only records
        where it was learned (provenance). When a session_id is given,
        the session row is created if missing, so the foreign key never
        fails on a not-yet-persisted session.

        Args:
            content: The fact to remember.
            category: Memory kind; default "fact".
            session_id: Provenance; which session taught this fact.

        Returns:
            The stored Memory, including the database-assigned id.

        Raises:
            StorageError: If the insert fails (transaction rolled back).
        """
        try:
            with Session(self._engine) as session, session.begin():
                if session_id is not None and session.get(SessionModel, session_id) is None:
                    session.add(SessionModel(id=session_id))
                row = MemoryModel(content=content, category=category, session_id=session_id)
                session.add(row)
                # Flush assigns row.id and row.created_at so the returned
                # Memory is complete; the begin() block commits on exit.
                session.flush()
                return self._to_memory(row)
        except Exception as exc:
            logger.exception("Failed to add memory")
            raise StorageError("Failed to add memory") from exc

    def get_all(self) -> list[Memory]:
        """Return every memory, newest first.

        Returns:
            All memories ordered by recency (newest first).

        Raises:
            StorageError: If the query fails.
        """
        return self._load()

    def get_recent(self, limit: int = 10) -> list[Memory]:
        """Return the newest memories, capped at `limit`.

        The M2 retrieval entry point: a fresh session is seeded with
        the most recent facts until real relevance-ranking arrives
        with RAG (Phase 8).

        Args:
            limit: How many to return.

        Returns:
            The newest memories, at most `limit`.

        Raises:
            StorageError: If the query fails.
        """
        return self._load(limit=limit)

    def get_by_session(self, session_id: str) -> list[Memory]:
        """Return memories that were learned in a specific session.

        Args:
            session_id: The provenance session to filter on.

        Returns:
            That session's memories, newest first.

        Raises:
            StorageError: If the query fails.
        """
        return self._load(session_id=session_id)

    def delete_memory(self, memory_id: int) -> bool:
        """Delete one memory by its id.

        Args:
            memory_id: The id returned by add_memory().

        Returns:
            True if a row was deleted, False if no such memory existed.

        Raises:
            StorageError: If the delete fails.
        """
        try:
            with Session(self._engine) as session, session.begin():
                result = session.execute(
                    delete(MemoryModel).where(MemoryModel.id == memory_id)
                )
                return result.rowcount > 0
        except Exception as exc:
            logger.exception("Failed to delete memory %s", memory_id)
            raise StorageError(f"Failed to delete memory {memory_id}") from exc

    def count(self) -> int:
        """Count all stored memories.

        Returns:
            The number of memory rows.

        Raises:
            StorageError: If the query fails.
        """
        try:
            with Session(self._engine) as session:
                return int(session.scalar(select(func.count()).select_from(MemoryModel)) or 0)
        except Exception as exc:
            logger.exception("Failed to count memories")
            raise StorageError("Failed to count memories") from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _load(
        self,
        limit: int | None = None,
        session_id: str | None = None,
    ) -> list[Memory]:
        """Shared query: memories, newest first, optionally filtered.

        Newest-first is created_at DESC with id DESC as a tiebreaker
        (two rows stored in the same microsecond still order correctly).
        """
        try:
            with Session(self._engine) as session:
                stmt = select(MemoryModel).order_by(
                    MemoryModel.created_at.desc(), MemoryModel.id.desc()
                )
                if session_id is not None:
                    stmt = stmt.where(MemoryModel.session_id == session_id)
                if limit is not None:
                    stmt = stmt.limit(limit)
                return [self._to_memory(row) for row in session.scalars(stmt).all()]
        except Exception as exc:
            logger.exception("Failed to load memories")
            raise StorageError("Failed to load memories") from exc

    @staticmethod
    def _to_memory(row: MemoryModel) -> Memory:
        """Translate an ORM row into a domain Memory object."""
        return Memory(
            id=row.id,
            content=row.content,
            category=row.category,
            session_id=row.session_id,
            created_at=row.created_at,
        )
