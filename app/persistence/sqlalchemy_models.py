"""SQLAlchemy ORM models - the DATABASE schema.

Phase 4, Milestone 1. Contrast these with the domain models in
app/models.py: Message/Conversation are pure Python; these map the
same ideas to tables. The repository (M2) translates between them.

Schema decisions:
  * sessions.id is the session_id string itself - it is already a
    unique, validated natural key; no surrogate id needed.
  * messages references sessions via a foreign key with ON DELETE
    CASCADE: deleting a session removes its messages.
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
