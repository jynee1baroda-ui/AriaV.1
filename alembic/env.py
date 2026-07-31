"""Alembic environment - connects migrations to OUR models and database.

This file is Alembic's bridge into the project:
  * project root is added to sys.path so `app` is importable (Alembic
    runs this file from the alembic/ directory, not the project root)
  * target_metadata = Base.metadata so `--autogenerate` compares the
    models' schema against the live database
  * sqlalchemy.url is taken from app.config.DATABASE_URL unless the
    caller already provided one (tests do this to migrate a temp DB)
"""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# Make `app` importable regardless of where Alembic is invoked from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import config as app_config  # noqa: E402
from app.db import Base  # noqa: E402
from app.persistence import sqlalchemy_models  # noqa: E402,F401

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Use our DATABASE_URL unless a caller (e.g. a test) pre-set a real one.
# The ini ships with a placeholder URL that is truthy, so compare against it.
_PLACEHOLDER_URL = "driver://user:pass@localhost/dbname"
if config.get_main_option("sqlalchemy.url") == _PLACEHOLDER_URL:
    config.set_main_option("sqlalchemy.url", app_config.DATABASE_URL)

# The models' metadata - the source of truth for autogenerate.
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Emits SQL without a live connection (useful for generating SQL
    scripts to run later, e.g. against production).
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode against a live database."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
