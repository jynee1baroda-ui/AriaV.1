"""Tests for the Assistant class."""

import unittest

from app.assistant import Assistant
from app.models import Conversation


class TestAssistant(unittest.TestCase):
    """Assistant must greet, answer identity, echo, and run commands."""

    def setUp(self) -> None:
        self.conversation = Conversation()
        self.assistant = Assistant(name="Aria", conversation=self.conversation)

    def test_respond_greeting(self) -> None:
        self.assertEqual(
            self.assistant.respond("hello"),
            "Hello! I'm Aria. How can I help?",
        )

    def test_respond_insensitive_to_case_and_punctuation(self) -> None:
        self.assertIn("Hello!", self.assistant.respond("HELLO!!"))

    def test_respond_identity_question(self) -> None:
        self.assertEqual(
            self.assistant.respond("what is your name"),
            "My name is Aria.",
        )

    def test_respond_fallback_echo(self) -> None:
        self.assertEqual(self.assistant.respond("pancakes"), "You said: pancakes")

    def test_respond_records_both_sides_of_exchange(self) -> None:
        self.assistant.respond("hello")
        messages = self.conversation.messages
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].role, "user")
        self.assertEqual(messages[1].role, "assistant")

    def test_handle_command_help_returns_keepalive_and_text(self) -> None:
        keep_going, output = self.assistant.handle_command("/help")
        self.assertTrue(keep_going)
        self.assertIn("/stats", output)

    def test_handle_command_stats_reports_counts(self) -> None:
        self.assistant.respond("hi")
        keep_going, output = self.assistant.handle_command("/stats")
        self.assertTrue(keep_going)
        self.assertIn("user=1", output)
        self.assertIn("assistant=1", output)

    def test_handle_command_about(self) -> None:
        keep_going, output = self.assistant.handle_command("/about")
        self.assertTrue(keep_going)
        self.assertIn("Aria", output)

    def test_handle_command_exit_ends_session(self) -> None:
        keep_going, _ = self.assistant.handle_command("/exit")
        self.assertFalse(keep_going)

    def test_handle_command_unknown_keeps_session_alive(self) -> None:
        keep_going, output = self.assistant.handle_command("/bogus")
        self.assertTrue(keep_going)
        self.assertIn("Unknown command", output)


if __name__ == "__main__":
    unittest.main()
