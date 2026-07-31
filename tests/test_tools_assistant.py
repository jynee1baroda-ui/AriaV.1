"""Assistant + tools integration tests (Phase 11 foundation).

The answer chain must place deterministic tools after the verified KB
and BEFORE the LLM/echo:

    rules > taught facts > memory > verified KB > TOOLS > [LLM > echo]

And because tools COMPUTE instead of guessing, they must answer even in
factual mode - "what time is it" is never a hallucination, so it is
never refused.
"""

import tempfile
import unittest
from pathlib import Path

from app.assistant import Assistant
from app.kb.knowledge import KnowledgeBase
from app.models import Conversation
from app.persistence.note_repository import NoteRepository
from app.db import build_engine, init_db
from app.tools import ToolRegistry, default_tools

REFUSAL = "I don't have a verified answer for that."


def make_assistant(**kwargs) -> Assistant:
    # The notes tool is never reached in these tests (clock/calendar/
    # calculator fire first), so a plain object() stands in for the
    # store; the stateful note tests below build a real temp repository.
    return Assistant(
        name="Aria",
        conversation=Conversation(),
        tools=ToolRegistry(default_tools(object())),
        **kwargs,
    )


class TestAssistantWithTools(unittest.TestCase):
    def test_time_answered_by_tool(self):
        assistant = make_assistant()
        reply = assistant.respond("what time is it")
        self.assertTrue(reply.startswith("It's "))
        self.assertTrue(reply.endswith(("AM.", "PM.")))

    def test_date_answered_by_tool(self):
        assistant = make_assistant()
        self.assertTrue(assistant.respond("what is the date").startswith("Today is "))

    def test_arithmetic_answered_by_tool(self):
        assistant = make_assistant()
        self.assertEqual(assistant.respond("what is 5*3"), "5*3 = 15")

    def test_kb_arithmetic_still_wins(self):
        """The verified KB keeps its precedence for its own entries."""
        assistant = Assistant(
            name="Aria",
            conversation=Conversation(),
            knowledge=KnowledgeBase(),
            tools=ToolRegistry(default_tools(object())),
        )
        self.assertEqual(assistant.respond("what is 2+2"), "4.")

    def test_tool_answers_in_factual_mode(self):
        """Tools compute - they are not guesses - so factual mode
        refuses the LLM but NOT the clock."""
        assistant = make_assistant(factual_only=True)
        self.assertEqual(assistant.respond("what time is it")[:4], "It's")
        self.assertEqual(assistant.respond("tell me a joke"), REFUSAL)

    def test_no_tools_falls_through_to_factual_refusal(self):
        assistant = Assistant(
            name="Aria",
            conversation=Conversation(),
            factual_only=True,
        )
        self.assertEqual(assistant.respond("what time is it"), REFUSAL)

    def test_reminder_adds_note_via_session(self):
        tmpdir = tempfile.TemporaryDirectory()
        try:
            engine = build_engine(
                f"sqlite:///{(Path(tmpdir.name) / 't.db').as_posix()}"
            )
            init_db(engine)
            assistant = Assistant(
                name="Aria",
                conversation=Conversation(),
                session_id="session-1",
                tools=ToolRegistry(default_tools(NoteRepository(engine))),
            )
            reply = assistant.respond("remind me to buy milk")
            self.assertEqual(reply, "Reminder saved: buy milk.")
            self.assertEqual(
                [n.content for n in NoteRepository(engine).get_by_session("session-1")],
                ["buy milk"],
            )
            engine.dispose()
        finally:
            tmpdir.cleanup()

    def test_notes_slash_commands(self):
        tmpdir = tempfile.TemporaryDirectory()
        try:
            engine = build_engine(f"sqlite:///{(Path(tmpdir.name) / 't.db').as_posix()}")
            init_db(engine)
            repo = NoteRepository(engine)
            assistant = Assistant(
                name="Aria",
                conversation=Conversation(),
                session_id="session-1",
                tools=ToolRegistry(default_tools(repo)),
            )
            keep, out = assistant.handle_command("/note walk the dog")
            self.assertTrue(keep)
            self.assertEqual(out, "Note saved: walk the dog.")
            keep, out = assistant.handle_command("/notes")
            self.assertIn("walk the dog", out)
            note_id = repo.get_by_session("session-1")[0].id
            keep, out = assistant.handle_command(f"/delete-note {note_id}")
            self.assertEqual(out, f"Deleted note {note_id}.")
            keep, out = assistant.handle_command("/notes")
            self.assertEqual(out, "You don't have any notes yet.")
            engine.dispose()
        finally:
            tmpdir.cleanup()


if __name__ == "__main__":
    unittest.main()
