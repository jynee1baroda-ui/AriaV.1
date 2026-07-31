"""Domain models for the assistant.

Phase 1, Lesson 4: our first two classes.

  * Message      - ONE unit of conversation (a dataclass: a class whose
                   whole job is to hold data, so Python writes the
                   boilerplate __init__/__repr__/__eq__ for us).
  * Conversation - an ordered list of Messages, plus the rules about
                   how messages can be added and counted.

Both teach ENCAPSULATION: the data lives inside, and other code can
only interact through the methods we expose.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.errors import InvalidRoleError

# The only roles the system knows. A tuple (not a list) because it is
# immutable - validation rules must never be accidentally mutated.
VALID_ROLES: tuple[str, ...] = ("user", "assistant")


@dataclass
class Message:
    """One utterance in the conversation.

    A dataclass auto-generates __init__ (so Message("user", "hi") works),
    __repr__ (readable printing) and __eq__ (value comparison) from the
    annotated fields. No manual boilerplate needed.

    Attributes:
        role: Who said it - "user" or "assistant".
        content: The text that was said.
    """

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        """Convert to a JSON-friendly dict (for storage).

        Returns:
            {"role": ..., "message": ...} - note storage uses "message"
            as the key, matching our JSON format from Lesson 3.
        """
        return {"role": self.role, "message": self.content}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Message":
        """Recreate a Message from a loaded JSON dict.

        A @classmethod receives the class itself (not an instance), so
        it can construct new Message objects - a standard factory pattern.

        Args:
            data: A dict with "role" and "message" keys.

        Returns:
            A new Message.
        """
        # .get with defaults keeps corrupt/missing fields from crashing
        return cls(
            role=str(data.get("role", "unknown")),
            content=str(data.get("message", "")),
        )


class Conversation:
    """An ordered collection of messages with rules.

    The leading underscore in _messages is the Python convention for
    "private" - outside code should treat it as off-limits and use the
    methods instead. That is encapsulation in action.
    """

    def __init__(self, messages: list[Message] | None = None) -> None:
        # list(messages) copies the input list, so the caller can't mutate
        # our internals through a shared reference.
        self._messages: list[Message] = list(messages) if messages else []

    def add(self, role: str, content: str) -> Message:
        """Append one message and return it.

        RAISE EARLY: validate before doing anything. A bad role is a
        programming error - it should surface loudly at the point of
        failure, not corrupt memory quietly.

        Args:
            role: "user" or "assistant".
            content: The text.

        Returns:
            The newly created Message.

        Raises:
            InvalidRoleError: If role is not one of VALID_ROLES.
        """
        if role not in VALID_ROLES:
            raise InvalidRoleError(
                f"Invalid role {role!r}. Expected one of {VALID_ROLES}."
            )
        message = Message(role=role, content=content)
        self._messages.append(message)
        return message

    def count_by_role(self) -> dict[str, int]:
        """Count messages per role.

        Returns:
            A dict like {"user": 3, "assistant": 2}.
        """
        counts: dict[str, int] = {}
        for message in self._messages:
            counts[message.role] = counts.get(message.role, 0) + 1
        return counts

    @property
    def messages(self) -> list[Message]:
        """Read-only access to the messages.

        A @property turns a method into an attribute (no parentheses).
        It returns a *copy* of the list - the DEFENSIVE COPY pattern.
        Callers can read but cannot corrupt the originals.

        Returns:
            A copy of the message list.
        """
        return list(self._messages)

    # --- Dunder methods: make Conversation behave like a native object --
    def __len__(self) -> int:
        """Support len(conversation)."""
        return len(self._messages)

    def __bool__(self) -> bool:
        """Support `if conversation:` - an empty conversation is falsy."""
        return bool(self._messages)

    def __repr__(self) -> str:
        """Support repr(conversation) - the debugging print."""
        return f"Conversation({len(self)} messages)"
