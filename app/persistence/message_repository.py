"""MessageRepository - the database half of the storage swap.

Phase 4, Milestone 2. This is the SEAM of the milestone: everything
that used to touch JSON files (app/storage.py) now routes here. The
rest of the application speaks to this class and nothing else.

Two responsibilities:
  * get_messages  - session rows -> ordered domain Message objects
  * add_messages  - domain Messages -> inserted rows (incremental)

Errors are wrapped in StorageError so callers catch one type. The
original exception is chained with `raise ... from` so debugging
still shows the full stack.
"""

from __future__ import annotations

import logging

from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from app import db
from app.errors import StorageError
from app.models import Message
from app.persistence.sqlalchemy_models import MessageModel, SessionModel

logger = logging.getLogger("assistant.storage")


class MessageRepository:
    """Read and write messages in a relational database.

    Attributes:
        _engine: The SQLAlchemy engine backing this repository.
            Injectable so tests can point at a temp database.
    """

    def __init__(self, engine: Engine | None = None) -> None:
        self._engine: Engine = engine or db.engine

    @property
    def engine(self) -> Engine:
        """The backing engine.

        Public so other services built on the same database (e.g.
        MemoryService) can share a test-injected temp engine instead of
        silently reaching for the real data/assistant.db.
        """
        return self._engine

    def ping(self) -> bool:
        """Verify the database is reachable and responsive.

        Runs the cheapest possible query (SELECT 1). Used by the
        readiness endpoint to distinguish "process is alive" (liveness)
        from "process can actually do its job" (readiness).

        Returns:
            True if the database answered, False otherwise.
        """
        try:
            with Session(self._engine) as session:
                session.execute(select(1))
            return True
        except Exception:
            logger.exception("Database connectivity check failed")
            return False

    def get_messages(
        self,
        session_id: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Message]:
        """Load a session's messages in insertion order.

        Optional pagination: `limit` caps the page size and `offset`
        skips rows. With no limit, the whole conversation is returned
        (used internally and by the CLI). The (session_id, id) index
        makes the ordered lookup a scan of that index, not a sort.

        Args:
            session_id: The session to read.
            limit: Max rows to return; None means all.
            offset: Rows to skip first (for page 2+).

        Returns:
            Domain Message objects in insertion order. Empty list for
            an unknown session (no session is created by reading).

        Raises:
            StorageError: If the query fails.
        """
        try:
            with Session(self._engine) as session:
                stmt = (
                    select(MessageModel)
                    .where(MessageModel.session_id == session_id)
                    .order_by(MessageModel.id)
                )
                if limit is not None:
                    stmt = stmt.limit(limit)
                if offset:
                    stmt = stmt.offset(offset)
                rows = session.scalars(stmt).all()
                return [Message(role=row.role, content=row.content) for row in rows]
        except Exception as exc:
            logger.exception("Failed to load messages for session %r", session_id)
            raise StorageError(f"Failed to load messages for session {session_id!r}") from exc

    def count_messages(self, session_id: str) -> int:
        """Count a session's messages (for pagination's `total`).

        Args:
            session_id: The session to count.

        Returns:
            The number of stored messages.

        Raises:
            StorageError: If the query fails.
        """
        try:
            with Session(self._engine) as session:
                return int(
                    session.scalar(
                        select(func.count(MessageModel.id)).where(
                            MessageModel.session_id == session_id
                        )
                    )
                    or 0
                )
        except Exception as exc:
            logger.exception("Failed to count messages for session %r", session_id)
            raise StorageError(f"Failed to count messages for session {session_id!r}") from exc

    def add_messages(self, session_id: str, messages: list[Message]) -> None:
        """Append messages to a session, creating the session if needed.

        This is INCREMENTAL: only the new messages are inserted, unlike
        the old snapshot-write of the whole conversation. `with ...
        session.begin()` commits on success and rolls back on failure,
        so a partial insert can never be half-committed (atomicity).

        Args:
            session_id: Which session to append to.
            messages: The domain Messages to insert.

        Raises:
            StorageError: If the write fails (session rolled back).
        """
        try:
            with Session(self._engine) as session, session.begin():
                # Upsert the session row: create if missing, else reuse.
                if session.get(SessionModel, session_id) is None:
                    session.add(SessionModel(id=session_id))
                for message in messages:
                    session.add(
                        MessageModel(
                            session_id=session_id,
                            role=message.role,
                            content=message.content,
                        )
                    )
        except Exception as exc:
            logger.exception("Failed to save messages for session %r", session_id)
            raise StorageError(f"Failed to save messages for session {session_id!r}") from exc
