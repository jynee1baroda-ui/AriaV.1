"""Built-in tools for the Phase 11 preview: clock, calendar, calculator,
and notes. Every handler is deterministic and honest:

  * time/date read the REAL system clock — no guessing.
  * the calculator evaluates arithmetic with a restricted AST evaluator
    (never eval()), and honestly refuses division by zero.
  * notes persist to the database (NoteRepository) scoped by session,
    so "remind me to X" survives restarts without overpromising that a
    reminder fires.

Triggers are conservative on purpose: an exact phrase set or a tight
regex. Anything the tool is not sure about returns None and the chain
falls through to the LLM.
"""

from __future__ import annotations

import ast
import re
from datetime import datetime

from app.tools.base import Tool

# ----------------------------------------------------------------------
# Exact-phrase trigger helper
# ----------------------------------------------------------------------

_EXACT_TRIGGERS: set[str] = {
    "what time is it",
    "what's the time",
    "what is the time",
    "whats the time",
    "tell me the time",
    "current time",
    "what time is it now",
}


class ClockTool(Tool):
    """Answer time questions from the real system clock."""

    name = "clock"
    description = "Tells the current local time (e.g. 'what time is it')."

    def run(self, normalized: str, session_id: str | None = None) -> str | None:
        if normalized not in _EXACT_TRIGGERS:
            return None
        now = datetime.now()
        # %-I is a GNU extension; Windows needs the manual leading-zero
        # strip: "%I:%M %p" gives "03:42 PM", which we clean to "3:42 PM".
        time_text = now.strftime("%I:%M %p").lstrip("0")
        return f"It's {time_text}."


class CalendarTool(Tool):
    """Answer date questions from the real system calendar."""

    name = "calendar"
    description = "Tells today's date and weekday (e.g. 'what is the date')."

    _TRIGGERS: set[str] = {
        "what is the date",
        "what's the date",
        "whats the date",
        "what day is it",
        "what day is today",
        "todays date",
        "today's date",
        "current date",
        "what is today's date",
    }

    def run(self, normalized: str, session_id: str | None = None) -> str | None:
        if normalized not in self._TRIGGERS:
            return None
        date_text = datetime.now().strftime("%A, %B %d, %Y")
        return f"Today is {date_text}."


# ----------------------------------------------------------------------
# The safe arithmetic evaluator
# ----------------------------------------------------------------------

_ALLOWED_BINARY_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
_ALLOWED_UNARY_OPS = (ast.UAdd, ast.USub)
# The operator types above are emitted by ast.walk() as BinOp/UnaryOp
# children, so they must be whitelisted alongside the container nodes.
_ALLOWED_AST_NODES: tuple = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    *_ALLOWED_BINARY_OPS,
    *_ALLOWED_UNARY_OPS,
)


def evaluate_expression(expression: str) -> float:
    """Safely evaluate an arithmetic expression (numbers, + - * /, parens).

    Parses to an AST and walks it, rejecting ANY node or operator that
    is not whitelisted — so attribute access, calls, imports, strings
    and other eval() escape hatches are structurally impossible. This is
    the "no eval()" rule made concrete: we compile only what we proved
    safe.

    Args:
        expression: The arithmetic text (digits, + - * /, parentheses).

    Returns:
        The computed value.

    Raises:
        ValueError: If the expression contains anything unsupported.
        ZeroDivisionError: On division by zero.
    """
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise ValueError(f"unsupported expression element: {type(node).__name__}")
        if isinstance(node, ast.BinOp) and type(node.op) not in _ALLOWED_BINARY_OPS:
            raise ValueError("unsupported operator")
        if isinstance(node, ast.UnaryOp) and type(node.op) not in _ALLOWED_UNARY_OPS:
            raise ValueError("unsupported unary operator")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError("non-numeric literal")
    # __builtins__ stripped: the evaluated namespace holds only numbers.
    return float(eval(compile(tree, "<arithmetic>", "eval"), {"__builtins__": {}}, {}))


def _format_result(value: float) -> str:
    """Render a number without the trailing '.0' for whole values."""
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


# A question prefix makes a number-with-operator message a DELIBERATE
# arithmetic question ("what is 5*3") instead of prose that happens to
# contain digits and a dash ("the score was 3-1").
_ARITHMETIC_PREFIX: re.Pattern[str] = re.compile(
    r"^(?:what\s+is\s+|what's\s+|whats\s+|calculate\s+|compute\s+|"
    r"solve\s+|how\s+much\s+is\s+)(.+)$"
)
# A pure-expression message ("5*3", "10 / 2") is ALSO deliberate.
_PURE_EXPRESSION: re.Pattern[str] = re.compile(r"^[\d\s.+\-*/().]+$")

_TOKEN_RE: re.Pattern[str] = re.compile(r"[+\-*/().\d]+")


class CalculatorTool(Tool):
    """Compute arithmetic questions deterministically (no LLM, no eval)."""

    name = "calculator"
    description = "Solves arithmetic like 'what is 5*3' or '10 / 2'."

    def run(self, normalized: str, session_id: str | None = None) -> str | None:
        expression = self._extract(normalized)
        if expression is None:
            return None
        try:
            value = evaluate_expression(expression)
        except ZeroDivisionError:
            return "I can't divide by zero — division by zero is undefined."
        except (ValueError, OverflowError, SyntaxError, TypeError):
            # The message looked like arithmetic but was not computable;
            # defer honestly rather than invent an answer.
            return None
        return f"{expression} = {_format_result(value)}"

    @staticmethod
    def _extract(normalized: str) -> str | None:
        """Pull the arithmetic expression from a message, or None.

        Matches either "<prefix> <expr>" ("what is 5*3") or a message
        that IS an expression ("10 / 2"). The expression must contain
        digits AND an operator - "born in 1984" has no operator and is
        not arithmetic.
        """
        prefix_match = _ARITHMETIC_PREFIX.match(normalized)
        candidate = prefix_match.group(1) if prefix_match else normalized
        if not _PURE_EXPRESSION.match(candidate):
            return None
        expression = re.sub(r"\s+", "", candidate)
        if not re.search(r"[+\-*/]", expression):
            return None
        if not re.search(r"\d", expression):
            return None
        return expression


# ----------------------------------------------------------------------
# Notes (stateful — needs a repository)
# ----------------------------------------------------------------------

_REMIND_PREFIXES: tuple[tuple[str, str], ...] = (
    ("remind me to ", "Reminder saved: "),
    ("remind me: ", "Reminder saved: "),
    ("remind me about ", "Note saved: "),
    ("add a note: ", "Note saved: "),
    ("add a note ", "Note saved: "),
    ("add note ", "Note saved: "),
    ("note: ", "Note saved: "),
    ("make a note ", "Note saved: "),
    ("note to ", "Note saved: "),
)

_LIST_TRIGGERS: set[str] = {
    "what are my notes",
    "show my notes",
    "list my notes",
    "my notes",
    "what are my reminders",
    "show my reminders",
    "list my reminders",
    "my reminders",
}

_DELETE_NOTE: re.Pattern[str] = re.compile(
    r"^(?:delete|remove)\s+(?:note|reminder)\s+(\d+)$"
)


class NotesTool(Tool):
    """Persist and recall per-session notes (stateful, DB-backed)."""

    name = "notes"
    description = "Saves, lists and deletes personal notes/reminders."

    def __init__(self, repository: object) -> None:
        # Duck-typed: add_note(content, session_id), get_by_session(session_id),
        # delete_note(note_id) — a NoteRepository, or a test double.
        self._repository = repository

    def run(self, normalized: str, session_id: str | None = None) -> str | None:
        for prefix, label in _REMIND_PREFIXES:
            if normalized.startswith(prefix):
                content = normalized[len(prefix):].strip()
                if not content:
                    return None
                self._repository.add_note(content, session_id)
                return f"{label}{content}."
        if normalized in _LIST_TRIGGERS:
            return self._list_notes(session_id)
        match = _DELETE_NOTE.match(normalized)
        if match is not None:
            return self._delete_note(int(match.group(1)), session_id)
        return None

    def _list_notes(self, session_id: str | None) -> str:
        notes = self._repository.get_by_session(session_id)
        if not notes:
            return "You don't have any notes yet."
        lines = "\n".join(f"  {note.id}. {note.content}" for note in notes)
        return f"You have {len(notes)} note{'s' if len(notes) != 1 else ''}:\n{lines}"

    def _delete_note(self, note_id: int, session_id: str | None) -> str:
        if not self._note_belongs_to_session(note_id, session_id):
            return f"No note with id {note_id} in this session."
        self._repository.delete_note(note_id)
        return f"Deleted note {note_id}."

    def _note_belongs_to_session(self, note_id: int, session_id: str | None) -> bool:
        return any(note.id == note_id for note in self._repository.get_by_session(session_id))

    # ------------------------------------------------------------------
    # Command conveniences: the /notes, /note and /delete-note REPL
    # commands reach the SAME store through these methods.
    # ------------------------------------------------------------------
    def list_for(self, session_id: str | None = None) -> str:
        """Human-readable note list for the current session."""
        return self._list_notes(session_id)

    def add(self, content: str, session_id: str | None = None):
        """Store one note and return it (duck-typed like the repository)."""
        return self._repository.add_note(content, session_id)

    def delete(self, note_id: int, session_id: str | None = None) -> bool:
        """Delete a note only if it belongs to this session.

        Returns:
            True if deleted, False if the note does not exist here.
        """
        if not self._note_belongs_to_session(note_id, session_id):
            return False
        return self._repository.delete_note(note_id)


def default_tools(repository: object) -> list[Tool]:
    """The standard tool set, in trigger-specificity order.

    Args:
        repository: Duck-typed note store (NoteRepository or double).

    Returns:
        The built-in tools: clock, calendar, calculator, notes.
    """
    return [
        ClockTool(),
        CalendarTool(),
        CalculatorTool(),
        NotesTool(repository),
    ]
