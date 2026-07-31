"""app.tools - tool calling foundation (Phase 11 preview).

WHAT A TOOL IS: a deterministic action the assistant can perform —
reading the real clock, computing arithmetic, storing a note. Unlike
the LLM, a tool does not *guess*: it computes, queries, or mutates, so
its answer is as true as the underlying system. That is exactly why the
tools run BEFORE the LLM in the answer chain:

    rules > taught facts > long-term memory > verified KB > TOOLS >
    [factual mode: refuse | open mode: LLM > echo]

WHY NOT LLM-DRIVEN TOOL CALLING: state-of-the-art assistants let a big
LLM emit JSON like {"tool": "get_time", "args": {}}. A 135M model
cannot reliably produce that JSON, so a "smart" call would mostly
misparse or hallucinate — the exact failure this project refuses to
ship. Instead, each Tool declares a deterministic trigger (a regex or
an exact-phrase set) and only fires when the user's phrasing clearly
matches. Tools are OPTED IN by construction: no match, no action, and
the message simply falls through to the LLM. This is the Phase 8
"never fake or unverified" rule applied to ACTIONS.

Each tool is a duck-typed class with:

    name: str                 - unique registry key.
    description: str          - what it does (for /tools and docs).
    run(normalized, session_id) -> str | None

run() returns a reply string when this tool is certain it understood
the message, or None to defer. `normalized` is the assistant's
lowercased, punctuation-stripped text; `session_id` scopes stateful
tools (notes) to the conversation that owns them.
"""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger("assistant.tools")


class Tool:
    """One deterministic capability. Subclass and implement run()."""

    name: str = "unnamed_tool"
    description: str = ""

    def run(self, normalized: str, session_id: str | None = None) -> str | None:
        """Handle the message, or return None to defer.

        Args:
            normalized: Lowercased, punctuation-stripped user text.
            session_id: Provenance scope for stateful tools (or None).

        Returns:
            A reply when this tool is confident it understood the
            message; None otherwise (the chain falls through).
        """
        raise NotImplementedError


class ToolRegistry:
    """Ordered list of tools; the first confident match wins.

    Order is part of the contract: more SPECIFIC triggers must precede
    more general ones (exact-phrase tools before the regex calculator),
    exactly like KnowledgeEntry ordering in app/kb.
    """

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: list[Tool] = list(tools) if tools else []

    def register(self, tool: Tool) -> None:
        """Append a tool (order matters — first match wins)."""
        self._tools.append(tool)

    def dispatch(self, normalized: str, session_id: str | None = None) -> str | None:
        """Run every tool until one answers; None means no tool matched.

        Errors never escape: a broken tool must degrade to the next
        tool / the LLM, never crash the chat.

        Args:
            normalized: Lowercased, punctuation-stripped user text.
            session_id: Provenance scope for stateful tools.

        Returns:
            The first tool's reply, or None.
        """
        for tool in self._tools:
            try:
                answer = tool.run(normalized, session_id)
            except Exception:
                logger.exception("Tool %s failed - falling through.", tool.name)
                continue
            if answer is not None:
                return answer
        return None

    def __iter__(self):
        return iter(self._tools)

    def __len__(self) -> int:
        return len(self._tools)


class FunctionTool(Tool):
    """A Tool defined by two callables instead of a subclass.

    Keeps one-liner tools terse while preserving the Tool contract.
    """

    def __init__(
        self,
        name: str,
        description: str,
        handler: Callable[[str, str | None], str | None],
    ) -> None:
        self.name = name
        self.description = description
        self._handler = handler

    def run(self, normalized: str, session_id: str | None = None) -> str | None:
        return self._handler(normalized, session_id)
