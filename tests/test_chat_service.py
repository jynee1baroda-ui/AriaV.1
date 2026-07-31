"""Unit tests for ChatService (Phase 2, Milestone 3: persistence)."""

import tempfile
import unittest
from pathlib import Path

from app.errors import InvalidSessionIdError
from app.services.chat_service import ChatService


class TestChatServicePersistence(unittest.TestCase):
    """Conversations must survive service restarts via disk."""

    def test_history_survives_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)

            # First "server life"
            service_one = ChatService(sessions_dir=session_dir)
            service_one.process("restart-demo", "hello")

            # Second "server life" - fresh service, same disk.
            # A restart forgets RAM; it must remember the file.
            service_two = ChatService(sessions_dir=session_dir)
            service_two.process("restart-demo", "what is your name")

            history = service_two.get_history("restart-demo")
            self.assertEqual(len(history), 4)
            self.assertEqual(history[0].content, "hello")
            self.assertEqual(history[2].content, "what is your name")

    def test_new_instance_reads_other_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            service_one = ChatService(sessions_dir=session_dir)
            service_one.process("solo", "hi")

            service_two = ChatService(sessions_dir=session_dir)
            history = service_two.get_history("solo")
            self.assertEqual(len(history), 2)


class TestChatServiceValidation(unittest.TestCase):
    """The service's own defense-in-depth must reject bad ids."""

    def test_slash_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = ChatService(sessions_dir=Path(tmpdir))
            with self.assertRaises(InvalidSessionIdError):
                service.process("../../escape", "hi")

    def test_dotdot_rejected_in_get_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = ChatService(sessions_dir=Path(tmpdir))
            with self.assertRaises(InvalidSessionIdError):
                service.get_history("..")

    def test_valid_id_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = ChatService(sessions_dir=Path(tmpdir))
            reply = service.process("user_1-ok", "hello")
            self.assertEqual(reply, "Hello! I'm Aria. How can I help?")


if __name__ == "__main__":
    unittest.main()
