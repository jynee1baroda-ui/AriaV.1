"""MemoryService - consolidation and retrieval of long-term memory.

Phase 5, Milestone 2. M1 built the STORAGE layer (memories table +
repository). This milestone makes memory *meaningful* with two halves:

  * CONSOLIDATION - MemoryExtractor turns user statements ("my name is
    Alice", "i like coffee", "remember that X") into canonical, durable
    facts. Rule-based because the LLM (which will do this far better)
    arrives in Phase 7; RAG embeddings replace retrieve() in Phase 8.
  * RETRIEVAL - retrieve(query) pulls memories relevant to the current
    message so the Assistant can answer from what it learned earlier,
    even across sessions. Today that is intent-pattern matching; it is
    deliberately simple and honest about being replaced by RAG.

The Assistant (when wired with a MemoryService) calls consolidate()
before answering and retrieve() to populate its working context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from app.models import Memory
from app.persistence.memory_repository import MemoryRepository


@dataclass(frozen=True)
class ExtractedFact:
    """A fact distilled from one user message, ready to store + confirm.

    Attributes:
        content: The canonical memory text ("the user's name is Alice").
        category: fact / preference / knowledge.
        reply: A canned confirmation the Assistant can show the user.
    """

    content: str
    category: str
    reply: str


def _clean(value: str) -> str:
    """Trim a captured value: leading/trailing whitespace and sentence
    punctuation. 'coffee.' becomes 'coffee'."""
    return value.strip().rstrip(".!?").strip()


class MemoryExtractor:
    """Pure statement -> fact extraction. No I/O, fully testable.

    Patterns are anchored to the START of the message so a question
    like "what is my name" never matches "my name is ...". Extraction
    is case-insensitive but the captured value keeps the user's casing
    (the reply shows "Alice", not "alice").
    """

    _PATTERNS: list[tuple[re.Pattern[str], Callable[[re.Match[str]], ExtractedFact]]] = [
        (
            re.compile(
                r"^(?:please\s+)?remember\s*(?::|that\s+)?\s*(.+)$",
                re.IGNORECASE,
            ),
            lambda m: ExtractedFact(
                content=_clean(m.group(1)),
                category="knowledge",
                reply=f"Okay, I'll remember that: {_clean(m.group(1))}.",
            ),
        ),
        (
            re.compile(
                r"^(?:my name is|my name's|call me|i am called|i'm called)\s+(.+)$",
                re.IGNORECASE,
            ),
            lambda m: ExtractedFact(
                content=f"the user's name is {_clean(m.group(1))}",
                category="fact",
                reply=f"Nice to meet you, {_clean(m.group(1))}! I'll remember that.",
            ),
        ),
        (
            re.compile(r"^i\s+(?:really\s+)?(?:like|love|prefer)\s+(.+)$", re.IGNORECASE),
            lambda m: ExtractedFact(
                content=f"the user likes {_clean(m.group(1))}",
                category="preference",
                reply=f"Noted - you like {_clean(m.group(1))}. I'll remember that.",
            ),
        ),
        (
            re.compile(
                r"^i\s+(?:really\s+)?(?:don't like|do not like|dislike|hate)\s+(.+)$",
                re.IGNORECASE,
            ),
            lambda m: ExtractedFact(
                content=f"the user dislikes {_clean(m.group(1))}",
                category="preference",
                reply=f"Noted - you don't like {_clean(m.group(1))}. I'll remember that.",
            ),
        ),
        (
            re.compile(
                r"^my\s+(?:favorite|favourite)\s+(.+?)\s+is\s+(.+)$",
                re.IGNORECASE,
            ),
            lambda m: ExtractedFact(
                content=(
                    f"the user's favorite {_clean(m.group(1))} is {_clean(m.group(2))}"
                ),
                category="preference",
                reply=(
                    f"Noted - your favorite {_clean(m.group(1))} "
                    f"is {_clean(m.group(2))}. I'll remember that."
                ),
            ),
        ),
    ]

    @staticmethod
    def extract(text: str) -> list[ExtractedFact]:
        """Distill every fact pattern present in a message.

        Args:
            text: The raw user message.

        Returns:
            All matching facts (normally zero or one).
        """
        results: list[ExtractedFact] = []
        for pattern, builder in MemoryExtractor._PATTERNS:
            match = pattern.match(text.strip())
            if match is not None:
                results.append(builder(match))
        return results


class MemoryService:
    """The assistant's memory organ: consolidate + retrieve + manage.

    Attributes:
        _repository: The storage backend (injectable for tests).
    """

    def __init__(self, repository: MemoryRepository | None = None) -> None:
        self._repository: MemoryRepository = repository or MemoryRepository()

    def consolidate(self, session_id: str, message: str) -> list[Memory]:
        """Extract facts from one user message and persist the NEW ones.

        Deduplication is by exact normalized content: telling Aria
        twice that your name is Alice stores the fact once. Re-stating
        a fact is a no-op, so repeated reminders cannot bloat memory.

        Args:
            session_id: Provenance for the stored memories.
            message: The user's raw text.

        Returns:
            The newly stored memories (empty when nothing was new).
        """
        existing = {m.content for m in self._repository.get_all()}
        added: list[Memory] = []
        for fact in MemoryExtractor.extract(message):
            if fact.content not in existing:
                added.append(
                    self._repository.add_memory(
                        content=fact.content,
                        category=fact.category,
                        session_id=session_id,
                    )
                )
                existing.add(fact.content)
        return added

    def retrieve(self, query: str) -> list[Memory]:
        """Return memories relevant to the current message.

        Rule-based intent matching for now: name questions pull name
        facts, favorite questions pull favorite facts, and so on. Phase
        8 replaces this with embedding similarity (RAG); the signature
        is designed so the swap is invisible to callers.

        Args:
            query: The user's message.

        Returns:
            The relevant memories (possibly empty).
        """
        memories = self._repository.get_all()
        normalized = query.lower()

        if "my name" in normalized or "your name" in normalized:
            return [m for m in memories if m.content.startswith("the user's name is")]
        if "favorite" in normalized or "favourite" in normalized:
            return [m for m in memories if "favorite" in m.content]
        if "like" in normalized or "dislike" in normalized or "prefer" in normalized:
            return [
                m
                for m in memories
                if any(
                    key in m.content for key in ("likes ", "dislikes ", "prefers ")
                )
            ]
        return []

    def get_all(self) -> list[Memory]:
        """Return every memory, newest first (for /memory and /history).

        Raises:
            StorageError: If the query fails.
        """
        return self._repository.get_all()

    def add(self, content: str, category: str = "fact", session_id: str | None = None) -> Memory:
        """Store a fact directly, bypassing extraction (for /remember).

        Args:
            content: The fact text.
            category: Memory kind; default "fact".
            session_id: Provenance.

        Returns:
            The stored Memory.
        """
        return self._repository.add_memory(content, category, session_id)

    def delete(self, memory_id: int) -> bool:
        """Delete one memory by its id (for /forget).

        The "factual answers" escape hatch: memories are only as true as
        what was taught, so the user must be able to un-teach a wrong
        fact (e.g. a test "my name is Alice" that is not theirs).

        Args:
            memory_id: The id shown by /memory.

        Returns:
            True if a memory was deleted, False if the id did not exist.
        """
        return self._repository.delete_memory(memory_id)
