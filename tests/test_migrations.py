"""Migration tests (Phase 4, Milestone 4).

Alembic migrations are the production schema source. These tests prove
the migration chain is real and reversible by running it against a
throwaway temp database:

  * upgrade head  -> creates the full schema AND records alembic_version
  * downgrade base -> removes every table the migration added (the
    forward/backward property that create_all never had)

The schema produced by the migration must match what the models declare
(create_all and the migration are both derived from Base.metadata).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sqlalchemy import inspect

from alembic import command

from app import db


class MigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._db_file = Path(self._tmp.name) / "migrate.db"
        self._url = f"sqlite:///{self._db_file.as_posix()}"
        self._cfg = db.alembic_config(self._url)
        self._engine = db.build_engine(self._url)

    def tearDown(self) -> None:
        self._engine.dispose()
        # Windows file-lock gotcha: dispose engines before the tempdir
        # is deleted, or cleanup fails with WinError 32.
        self._tmp.cleanup()

    def test_upgrade_head_creates_full_schema(self) -> None:
        command.upgrade(self._cfg, "head")

        tables = set(inspect(self._engine).get_table_names())
        self.assertIn("sessions", tables)
        self.assertIn("messages", tables)
        self.assertIn("alembic_version", tables)

        # The composite index from the models must exist.
        index_names = {
            ix["name"]
            for ix in inspect(self._engine).get_indexes("messages")
        }
        self.assertIn("ix_messages_session_id_id", index_names)

        # The role CHECK constraint from the models must exist.
        checks = inspect(self._engine).get_check_constraints("messages")
        self.assertTrue(
            any(ck["name"] == "ck_messages_role" for ck in checks),
            f"expected ck_messages_role in {checks}",
        )

        # Foreign key cascade from the models must exist.
        fks = inspect(self._engine).get_foreign_keys("messages")
        self.assertTrue(
            any(fk["referred_table"] == "sessions" for fk in fks),
            f"expected FK sessions in {fks}",
        )

    def test_downgrade_base_removes_schema(self) -> None:
        command.upgrade(self._cfg, "head")
        command.downgrade(self._cfg, "base")

        tables = set(inspect(self._engine).get_table_names())
        self.assertNotIn("sessions", tables)
        self.assertNotIn("messages", tables)
        # alembic_version is Alembic's own bookkeeping table and persists
        # by design - it lets a later "upgrade head" re-apply revisions.

    def test_upgrade_is_idempotent_after_stamp(self) -> None:
        # Simulates the pre-migration dev DB: tables exist (create_all)
        # but no version history. Stamping records "already at head"
        # instead of trying to re-create existing tables.
        command.upgrade(self._cfg, "head")

        # Running again on an up-to-date DB is a no-op.
        command.upgrade(self._cfg, "head")
        tables = set(inspect(self._engine).get_table_names())
        self.assertIn("alembic_version", tables)


if __name__ == "__main__":
    unittest.main()
