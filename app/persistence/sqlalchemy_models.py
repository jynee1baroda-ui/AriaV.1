"""SQLAlchemy ORM models - the DATABASE schema.

Phase 4, Milestone 1. Contrast these with the domain models in
app/models.py: Message/Conversation are pure Python; these map the
same ideas to tables. The repository (M2) translates between them.

Schema decisions:
  * sessions.id is the session_id string itself - it is already a
    unique, validated natural key; no surrogate id needed.
  * messages references sessions via a foreign key with ON DELETE
    CASCADE: deleting a session removes its messages.
  * memories references sessions with ON DELETE SET NULL: a memory is
    owned by the ASSISTANT, not the session - deleting a session keeps
    the facts it taught (only the provenance link is dropped).
  * The composite index (session_id, id) serves "all messages for a
    session in order" without a sort - index order IS the answer.
  * A CHECK constraint enforces the role whitelist at the data layer,
    mirroring the domain's InvalidRoleError (defense in depth).
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def utcnow() -> datetime:
    """Timezone-aware 'now' - storing naive datetimes is a classic bug."""
    return datetime.now(timezone.utc)


class SessionModel(Base):
    """A conversation session. One row per session_id."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    # ORM-level cascade: deleting a SessionModel deletes its messages.
    # passive_deletes=True lets the database's ON DELETE CASCADE handle
    # it natively (with our SQLite pragma enabled).
    messages: Mapped[list["MessageModel"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return f"<SessionModel id={self.id!r}>"


class MessageModel(Base):
    """One stored message. A row per utterance."""

    __tablename__ = "messages"
    __table_args__ = (
        CheckConstraint(
            "role IN ('user', 'assistant')",
            name="ck_messages_role",
        ),
        Index("ix_messages_session_id_id", "session_id", "id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    session: Mapped["SessionModel"] = relationship(back_populates="messages")

    def __repr__(self) -> str:
        return f"<MessageModel id={self.id} role={self.role!r}>"


class MemoryModel(Base):
    """One durable memory the assistant can recall later.

    Phase 5, Milestone 1. The ownership model differs deliberately from
    messages:

      * messages belong to a session  -> ON DELETE CASCADE
      * memories belong to the ASSISTANT -> ON DELETE SET NULL

    A memory is learned IN a session (session_id is provenance), but
    long-term memory must OUTLIVE it - recall in session B a fact
    learned in session A. Deleting a session therefore keeps its
    memories, dropping only the provenance link.

    category is a plain string with a Python-side whitelist rather than
    a CHECK constraint: unlike roles, the category set is expected to
    grow (Phase 8 adds "entity" for RAG). Every growth would otherwise
    demand a migration.
    """

    __tablename__ = "memories"
    __table_args__ = (
        Index("ix_memories_session_id", "session_id"),
        Index("ix_memories_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("sessions.id", ondelete="SET NULL"), nullable=True
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(16), nullable=False, default="fact")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    # The relationship exists primarily so SQLAlchemy's unit of work
    # INSERTs the parent session row BEFORE this memory row (without it,
    # the FK ordering is unknown and flushes can violate the constraint).
    session: Mapped["SessionModel | None"] = relationship()

    def __repr__(self) -> str:
        return f"<MemoryModel id={self.id} category={self.category!r}>"


class NoteModel(Base):
    """One user note/reminder stored by the notes tool.

    Phase 11 foundation (tool calling preview). Notes mirror memories in
    ownership: they belong to the ASSISTANT (ON DELETE SET NULL), not
    the session - deleting a session keeps its notes but drops the
    provenance link. The (session_id, id) index serves "all notes for a
    session in order" without a sort.
    """

    __tablename__ = "notes"
    __table_args__ = (
        Index("ix_notes_session_id_id", "session_id", "id"),
        Index("ix_notes_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("sessions.id", ondelete="SET NULL"), nullable=True
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    # Same dependency-ordering relationship as MemoryModel: without it
    # the unit of work does not know to INSERT the parent session row
    # before this note row, and the FK flush fails.
    session: Mapped["SessionModel | None"] = relationship()

    def __repr__(self) -> str:
        return f"<NoteModel id={self.id}>"
