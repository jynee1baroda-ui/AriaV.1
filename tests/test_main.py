"""Tests for the Assistant class."""

import tempfile
import unittest
from pathlib import Path

from app.assistant import Assistant
from app.db import build_engine, init_db
from app.models import Conversation, Memory
from app.persistence.memory_repository import MemoryRepository
from app.services.memory_service import MemoryService


class TestAssistant(unittest.TestCase):
    """Assistant must greet, answer identity, echo, and run commands."""

    def setUp(self) -> None:
        self.conversation = Conversation()
        self.assistant = Assistant(
            name="Aria",
            conversation=self.conversation,
            factual_only=True,  # this suite tests factual-mode contracts
        )

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

    def test_respond_fallback_refuses_unverified(self) -> None:
        """Factual mode: an unknown question is REFUSED, never guessed."""
        self.assertEqual(
            self.assistant.respond("pancakes"),
            "I don't have a verified answer for that.",
        )

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


class TestAssistantStatementConfirmation(unittest.TestCase):
    """Statements that teach facts get confirmation replies (M2)."""

    def setUp(self) -> None:
        self.assistant = Assistant(name="Aria", conversation=Conversation())

    def test_name_statement_confirms(self) -> None:
        self.assertIn("Alice", self.assistant.respond("my name is Alice"))

    def test_like_statement_confirms(self) -> None:
        self.assertIn("coffee", self.assistant.respond("i like coffee"))

    def test_remember_statement_confirms(self) -> None:
        self.assertIn("the sky is blue", self.assistant.respond("remember that the sky is blue"))


class TestAssistantMemoryAnswers(unittest.TestCase):
    """Questions answered from working-context memories (M2)."""

    def setUp(self) -> None:
        self.conversation = Conversation()

    def make_assistant(self, memories: list[Memory]) -> Assistant:
        return Assistant(name="Aria", conversation=self.conversation, memories=memories)

    def test_name_question_answered_from_memory(self) -> None:
        assistant = self.make_assistant([Memory(id=1, content="the user's name is Alice")])
        self.assertEqual(assistant.respond("what is my name"), "Your name is Alice.")

    def test_name_question_without_memory_says_so(self) -> None:
        assistant = self.make_assistant([])
        self.assertEqual(
            assistant.respond("what is my name"),
            "I don't know your name yet. Tell me with 'my name is ...'",
        )

    def test_favorite_question_answered_from_memory(self) -> None:
        assistant = self.make_assistant(
            [Memory(id=1, content="the user's favorite book is Dune")]
        )
        self.assertEqual(
            assistant.respond("what is my favorite book"),
            "Your favorite book is Dune.",
        )

    def test_like_question_answered_from_memory(self) -> None:
        assistant = self.make_assistant([Memory(id=1, content="the user likes coffee")])
        self.assertEqual(assistant.respond("do i like coffee"), "You like coffee.")

    def test_dislike_question_answered_from_memory(self) -> None:
        assistant = self.make_assistant([Memory(id=1, content="the user dislikes spiders")])
        self.assertEqual(
            assistant.respond("do i like spiders"),
            "You don't like spiders.",
        )


class TestAssistantMemoryCommands(unittest.TestCase):
    """/memory and /remember need a wired MemoryService (CLI mode)."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.engine = build_engine(
            f"sqlite:///{(Path(self._tmpdir.name) / 'cli.db').as_posix()}"
        )
        init_db(self.engine)
        self.memory_service = MemoryService(MemoryRepository(self.engine))
        self.assistant = Assistant(
            name="Aria",
            conversation=Conversation(),
            session_id="cli",
            memory_service=self.memory_service,
        )

    def tearDown(self) -> None:
        self.engine.dispose()
        self._tmpdir.cleanup()

    def test_remember_command_stores_fact(self) -> None:
        keep_going, output = self.assistant.handle_command("/remember the sky is green")
        self.assertTrue(keep_going)
        self.assertIn("the sky is green", output)
        stored = self.memory_service.get_all()
        self.assertEqual(stored[0].content, "the sky is green")

    def test_memory_command_lists_stored_facts(self) -> None:
        self.assistant.handle_command("/remember the sky is green")
        keep_going, output = self.assistant.handle_command("/memory")
        self.assertTrue(keep_going)
        self.assertIn("the sky is green", output)
        self.assertIn("1 long-term memories", output)

    def test_memory_command_empty_state(self) -> None:
        keep_going, output = self.assistant.handle_command("/memory")
        self.assertIn("I don't remember anything yet", output)

    def test_respond_consolidates_through_wired_service(self) -> None:
        self.assistant.respond("my name is Alice")
        stored = self.memory_service.get_all()
        self.assertEqual(stored[0].content, "the user's name is Alice")

    def test_respond_answers_from_memory_after_consolidation(self) -> None:
        self.assistant.respond("my name is Alice")
        self.assertEqual(self.assistant.respond("what is my name"), "Your name is Alice.")

    def test_forget_command_removes_a_memory(self) -> None:
        """The factual-correction escape hatch: un-teach a wrong fact."""
        self.assistant.respond("my name is Alice")
        stored = self.memory_service.get_all()
        memory_id = stored[0].id
        keep_going, output = self.assistant.handle_command(f"/forget {memory_id}")
        self.assertTrue(keep_going)
        self.assertIn("Forgot", output)
        self.assertEqual(self.memory_service.get_all(), [])
        self.assertIn("I don't know your name yet", self.assistant.respond("what is my name"))

    def test_forget_rejects_non_numeric_id(self) -> None:
        keep_going, output = self.assistant.handle_command("/forget alice")
        self.assertTrue(keep_going)
        self.assertIn("Usage: /forget", output)

    def test_forget_missing_id_is_graceful(self) -> None:
        keep_going, output = self.assistant.handle_command("/forget 999")
        self.assertTrue(keep_going)
        self.assertIn("No memory with id 999", output)


if __name__ == "__main__":
    unittest.main()
