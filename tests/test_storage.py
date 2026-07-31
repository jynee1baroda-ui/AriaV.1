"""Tests for the storage (persistence) module."""

import tempfile
import unittest
from pathlib import Path

from app.models import Message
from app.storage import AtomicFileWriter, load_history, save_history


class TestAtomicFileWriter(unittest.TestCase):
    """The context manager must commit on success, discard on failure."""

    def test_success_commits_and_cleans_temp(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "history.json"
            with AtomicFileWriter(dest) as fh:
                fh.write("complete data")
            self.assertEqual(dest.read_text(encoding="utf-8"), "complete data")
            self.assertFalse(dest.with_suffix(".json.tmp").exists())

    def test_failure_discards_temp_and_leaves_no_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "history.json"
            with self.assertRaises(RuntimeError):
                with AtomicFileWriter(dest) as fh:
                    fh.write("half-written data")
                    raise RuntimeError("boom")
            self.assertFalse(dest.exists())
            self.assertFalse(dest.with_suffix(".json.tmp").exists())

    def test_failure_over_existing_destination_preserves_original(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "history.json"
            dest.write_text("previous good data", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                with AtomicFileWriter(dest) as fh:
                    fh.write("corrupting write")
                    raise RuntimeError("boom")
            # The original file survives untouched
            self.assertEqual(dest.read_text(encoding="utf-8"), "previous good data")


class TestStorageRoundTrip(unittest.TestCase):
    """save_history + load_history must preserve data exactly."""

    def test_save_then_load(self) -> None:
        messages = [
            Message("user", "hello"),
            Message("assistant", "Hi there!"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "history.json"
            save_history(messages, path)
            # The parent directory was created automatically
            self.assertTrue(path.exists())
            loaded = load_history(path)
            self.assertEqual(loaded, messages)

    def test_load_missing_file_returns_empty_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertEqual(load_history(Path(tmpdir) / "missing.json"), [])

    def test_load_corrupt_file_returns_empty_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.json"
            path.write_text("{not valid json", encoding="utf-8")
            self.assertEqual(load_history(path), [])

    def test_load_handles_legacy_dict_entries(self) -> None:
        # Files written before Lesson 4 stored raw dicts - they must
        # still load. Message.from_dict fills missing fields safely.
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy.json"
            path.write_text(
                '[{"role": "user", "message": "hi"}, {"bad": true}]',
                encoding="utf-8",
            )
            loaded = load_history(path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0], Message("user", "hi"))
            # The malformed entry degrades gracefully instead of crashing
            self.assertEqual(loaded[1].role, "unknown")


if __name__ == "__main__":
    unittest.main()
