"""Database engine, session factory, and declarative base.

Phase 4, Milestone 1: the ORM infrastructure. Two things to absorb:

1. build_engine() is a FACTORY so tests can point at a temp SQLite
   file while production uses DATABASE_URL - same code, different DB.
2. SQLite quirk: foreign-key enforcement is OFF by default. We turn it
   on per connection with a PRAGMA - otherwise ON DELETE CASCADE and
   FK violations would be silently ignored. PostgreSQL enforces FKs
   natively, so this pragma is a no-op there.
"""

from __future__ import annotations

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app import config


class Base(DeclarativeBase):
    """Base class for every ORM model in the project."""


def build_engine(database_url: str | None = None) -> Engine:
    """Create an engine for the given (or configured) database URL.

    Args:
        database_url: SQLAlchemy URL; defaults to config.DATABASE_URL.

    Returns:
        A configured engine.
    """
    url = database_url or config.DATABASE_URL
    kwargs: dict = {}
    if url.startswith("sqlite"):
        # FastAPI runs handlers on a threadpool; SQLite connections must
        # be usable across threads, so we opt out of its same-thread
        # safety check (our ChatService lock provides the safety).
        kwargs["connect_args"] = {"check_same_thread": False}

        @event.listens_for(Engine, "connect")
        def _enable_sqlite_foreign_keys(dbapi_connection, _record) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return create_engine(url, **kwargs)


# Module-level defaults used by the application and by init_db().
engine: Engine = build_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db(engine: Engine = engine) -> None:
    """Create all tables that don't exist yet.

    Dev-friendly convenience. Real migrations arrive in M4 (Alembic);
    create_all is intentionally NOT a production migration tool.

    Args:
        engine: Which engine to create tables on (tests pass their own).
    """
    # Importing the models registers their tables on Base.metadata.
    from app.persistence import sqlalchemy_models  # noqa: F401

    Base.metadata.create_all(bind=engine)
