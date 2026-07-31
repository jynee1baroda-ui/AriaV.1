"""Tests for the tool-calling foundation (Phase 11 preview).

Three contracts under test:
  * Each built-in tool only answers messages it is CERTAIN about and
    returns None otherwise (so unknown text still reaches the LLM).
  * The registry runs tools in order and the first confident match wins.
  * Wired into the Assistant, tools answer BEFORE the LLM and even in
    factual mode (they compute/store - they never guess).

The notes tool is exercised against a throwaway temp database via the
real NoteRepository, exactly like the other storage-backed features.
"""

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from app.db import build_engine, init_db
from app.models import Note
from app.persistence.note_repository import NoteRepository
from app.tools import ToolRegistry
from app.tools.base import Tool
from app.tools.builtin import (
    CalculatorTool,
    CalendarTool,
    ClockTool,
    NotesTool,
    default_tools,
    evaluate_expression,
)

NORMALIZED = "the text the assistant normalizes"


class TestCalculatorTool(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = CalculatorTool()

    def test_what_is_prefix(self):
        self.assertEqual(self.tool.run("what is 5*3"), "5*3 = 15")

    def test_whats_apostrophe(self):
        self.assertEqual(self.tool.run("what's 6*7"), "6*7 = 42")

    def test_whats_no_apostrophe(self):
        self.assertEqual(self.tool.run("whats 6*7"), "6*7 = 42")

    def test_calculate_prefix(self):
        self.assertEqual(self.tool.run("calculate 10/2"), "10/2 = 5")

    def test_how_much_is_prefix(self):
        self.assertEqual(self.tool.run("how much is 9*9"), "9*9 = 81")

    def test_pure_expression_message(self):
        self.assertEqual(self.tool.run("10 / 2"), "10/2 = 5")

    def test_operator_spacing_is_normalized(self):
        self.assertEqual(self.tool.run("what is 5 + 3 * 2"), "5+3*2 = 11")

    def test_parentheses(self):
        self.assertEqual(self.tool.run("what is (2+3)*4"), "(2+3)*4 = 20")

    def test_unary_minus(self):
        self.assertEqual(self.tool.run("what is -5+3"), "-5+3 = -2")

    def test_whole_result_has_no_trailing_decimal(self):
        self.assertEqual(self.tool.run("what is 4/2"), "4/2 = 2")

    def test_division_by_zero_is_honest_refusal(self):
        self.assertEqual(
            self.tool.run("what is 1/0"),
            "I can't divide by zero — division by zero is undefined.",
        )

    def test_non_arithmetic_returns_none(self):
        self.assertIsNone(self.tool.run("what is your name"))
        self.assertIsNone(self.tool.run("pancakes"))
        self.assertIsNone(self.tool.run("tell me a joke"))

    def test_digits_without_operator_returns_none(self):
        """'born in 1984' is prose, not arithmetic."""
        self.assertIsNone(self.tool.run("i was born in 1984"))

    def test_prose_with_dash_returns_none(self):
        """'the score was 3-1' has no arithmetic prefix - not a query."""
        self.assertIsNone(self.tool.run("the score was 3-1"))

    def test_letters_in_expression_returns_none(self):
        self.assertIsNone(self.tool.run("what is q+a"))

    def test_safe_evaluator_rejects_injection(self):
        for exploit in ("__import__('os')", "open('x')", "1+1; print(1)", "'a'+'b'"):
            with self.assertRaises((ValueError, SyntaxError, TypeError)):
                evaluate_expression(exploit)


class TestClockTool(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = ClockTool()

    def test_what_time_is_it(self):
        reply = self.tool.run("what time is it")
        self.assertTrue(reply.startswith("It's "))
        self.assertRegex(reply, r"^(It's \d{1,2}:\d{2} (AM|PM))\.$")

    def test_returns_real_clock_time(self):
        reply = self.tool.run("what's the time")
        self.assertIn(datetime.now().strftime("%M"), reply)

    def test_unknown_returns_none(self):
        self.assertIsNone(self.tool.run("what time is the party"))
        self.assertIsNone(self.tool.run("pancakes"))


class TestCalendarTool(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = CalendarTool()

    def test_what_is_the_date(self):
        reply = self.tool.run("what is the date")
        self.assertTrue(reply.startswith("Today is "))
        self.assertIn(datetime.now().strftime("%Y"), reply)

    def test_what_day_is_it(self):
        reply = self.tool.run("what day is it")
        self.assertIn(datetime.now().strftime("%A"), reply)

    def test_unknown_returns_none(self):
        self.assertIsNone(self.tool.run("what is a date"))
        self.assertIsNone(self.tool.run("pancakes"))


class TestToolRegistry(unittest.TestCase):
    # The notes tool never touches its repository in these dispatch
    # tests (clock/calendar/calculator fire first, or nothing matches),
    # so a plain object() stands in for the store.
    DUMMY_STORE = object()

    def test_first_confident_match_wins(self):
        registry = ToolRegistry(default_tools(self.DUMMY_STORE))
        reply = registry.dispatch("what is 5*3")
        self.assertEqual(reply, "5*3 = 15")

    def test_no_match_returns_none(self):
        registry = ToolRegistry(default_tools(self.DUMMY_STORE))
        self.assertIsNone(registry.dispatch("tell me a joke"))

    def test_broken_tool_degrades_to_next(self):
        class ExplodingTool(Tool):
            name = "explode"
            description = "always crashes"

            def run(self, normalized, session_id=None):
                raise RuntimeError("boom")

        class OkTool(Tool):
            name = "ok"
            description = "answers"

            def run(self, normalized, session_id=None):
                return "fine"

        registry = ToolRegistry([ExplodingTool(), OkTool()])
        self.assertEqual(registry.dispatch("anything"), "fine")

    def test_register_appends(self):
        registry = ToolRegistry()
        registry.register(OkTool())
        self.assertEqual(len(registry), 1)


class OkTool(Tool):
    name = "ok"
    description = "answers"

    def run(self, normalized, session_id=None):
        return "fine"


class TestNotesTool(unittest.TestCase):
    """Notes persist through the real NoteRepository on a temp database."""

    def setUp(self) -> None:
        # Fresh temp DB per test so notes never leak between tests.
        self._tmpdir = tempfile.TemporaryDirectory()
        self._engine = build_engine(
            f"sqlite:///{(Path(self._tmpdir.name) / 'notes.db').as_posix()}"
        )
        init_db(self._engine)
        self._repo = NoteRepository(self._engine)
        self.tool = NotesTool(self._repo)

    def tearDown(self) -> None:
        self._engine.dispose()
        self._tmpdir.cleanup()

    def test_remind_me_to_adds_note(self):
        reply = self.tool.run("remind me to buy milk", session_id="s1")
        self.assertEqual(reply, "Reminder saved: buy milk.")
        notes = self._repo.get_by_session("s1")
        self.assertEqual([n.content for n in notes], ["buy milk"])

    def test_add_a_note(self):
        self.tool.run("add a note: call dad", session_id="s1")
        self.assertIn("call dad", [n.content for n in self._repo.get_by_session("s1")])

    def test_note_colon(self):
        self.tool.run("note: return the book", session_id="s1")
        self.assertIn("return the book", [n.content for n in self._repo.get_by_session("s1")])

    def test_list_notes(self):
        self.tool.run("remind me to walk the dog", session_id="s2")
        reply = self.tool.run("what are my notes", session_id="s2")
        self.assertIn("walk the dog", reply)

    def test_list_empty(self):
        reply = self.tool.run("show my notes", session_id="empty-session")
        self.assertEqual(reply, "You don't have any notes yet.")

    def test_delete_note(self):
        self.tool.run("remind me to pay rent", session_id="s3")
        note_id = self._repo.get_by_session("s3")[0].id
        reply = self.tool.run(f"delete note {note_id}", session_id="s3")
        self.assertEqual(reply, f"Deleted note {note_id}.")
        self.assertEqual(self._repo.get_by_session("s3"), [])

    def test_delete_wrong_session_is_refused(self):
        self.tool.run("remind me to pay rent", session_id="s4")
        note_id = self._repo.get_by_session("s4")[0].id
        reply = self.tool.run(f"delete note {note_id}", session_id="other-session")
        self.assertEqual(reply, f"No note with id {note_id} in this session.")
        self.assertEqual(len(self._repo.get_by_session("s4")), 1)

    def test_notes_are_session_scoped(self):
        self.tool.run("remind me to buy milk", session_id="sa")
        self.tool.run("remind me to buy bread", session_id="sb")
        self.assertEqual(len(self._repo.get_by_session("sa")), 1)
        self.assertEqual(len(self._repo.get_by_session("sb")), 1)

    def test_unknown_returns_none(self):
        self.assertIsNone(self.tool.run("pancakes", session_id="s1"))
        self.assertIsNone(self.tool.run("what is 5*3", session_id="s1"))

    def test_empty_reminder_returns_none(self):
        self.assertIsNone(self.tool.run("remind me to ", session_id="s1"))

    def test_domain_note_roundtrip(self):
        note = self._repo.add_note("hello world", session_id="s5")
        self.assertIsInstance(note, Note)
        self.assertEqual(note.content, "hello world")
        self.assertEqual(note.session_id, "s5")
        self.assertIsNotNone(note.id)


if __name__ == "__main__":
    unittest.main()
