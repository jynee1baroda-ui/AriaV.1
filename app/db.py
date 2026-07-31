"""Database engine, session factory, and declarative base.

Phase 4, Milestone 1: the ORM infrastructure. Two things to absorb:

1. build_engine() is a FACTORY so tests can point at a temp SQLite
   file while production uses DATABASE_URL - same code, different DB.
2. SQLite quirk: foreign-key enforcement is OFF by default. We turn it
   on per connection with a PRAGMA - otherwise ON DELETE CASCADE and
   FK violations would be silently ignored. PostgreSQL enforces FKs
   natively, so this pragma is a no-op there.

Phase 4, Milestone 4: Alembic becomes the schema's source of truth.
The application runs `alembic upgrade head` on startup via
run_migrations(); create_all survives only as a fast path for tests.
"""

from __future__ import annotations

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app import config

# Alembic config file location, resolved from the project root so
# run_migrations() works regardless of the process's current directory.
ALEMBIC_INI_PATH = config.PROJECT_ROOT / "alembic.ini"


def alembic_config(database_url: str | None = None) -> Config:
    """Build an Alembic Config pointing at a specific database.

    env.py honors a pre-set sqlalchemy.url (it falls back to the
    app-wide DATABASE_URL only when the ini still holds the default
    placeholder), so tests can migrate a throwaway temp database.

    Args:
        database_url: Which database to migrate; defaults to config.

    Returns:
        A configured Alembic Config object.
    """
    cfg = Config(str(ALEMBIC_INI_PATH))
    cfg.set_main_option("sqlalchemy.url", database_url or config.DATABASE_URL)
    return cfg


def run_migrations(database_url: str | None = None) -> None:
    """Bring the database schema up to the latest migration.

    This is the production-grade replacement for create_all: Alembic
    applies every pending revision in order and records the version in
    the alembic_version table, so any database at any point in history
    can be advanced deterministically.

    Args:
        database_url: Which database to migrate; defaults to config.
    """
    command.upgrade(alembic_config(database_url), "head")


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

    Dev/test convenience ONLY: tests create tables on a throwaway temp
    database faster than running Alembic each time. The application
    itself never calls this - it runs run_migrations() instead, so the
    deployed schema always matches the versioned migration history.

    Args:
        engine: Which engine to create tables on (tests pass their own).
    """
    # Importing the models registers their tables on Base.metadata.
    from app.persistence import sqlalchemy_models  # noqa: F401

    Base.metadata.create_all(bind=engine)
