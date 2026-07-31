"""app.kb - the verified knowledge base (factual answers, Phase 7 hardening).

WHY THIS EXISTS: a 135M generative model HALLUCINATES. It produces
plausible text, not true text - "what is the capital of France" can
return "France is the capital of France" because that reads well, not
because it is true. No prompt, temperature, or token tweak fixes that
for a model this size.

The only way to honor "NEVER fake or unverified" is to answer facts
ONLY from a store we curated and verified ourselves, and REFUSE
anything not in it. That is this module: a small set of checked,
objective facts matched by keywords. It is deliberately small and
deliberately honest - "I don't have a verified answer" beats a made-up
one. RAG (Phase 9) will grow this from a hand-curated list into
answers grounded in retrieved documents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class KnowledgeEntry:
    """One verified fact.

    Attributes:
        keywords: Every keyword must appear in the (normalized) query.
            Order matters - put MORE SPECIFIC entries first, because
            the first fully-matching entry wins ("which country has the
            largest population" must match the population entry, not
            the largest-country entry).
        answer: The verified answer text.
    """

    keywords: tuple[str, ...]
    answer: str


# Verified, objective facts. Every answer below was checked against a
# reliable source before being added. When in doubt, an entry was NOT
# added - absence is safe, wrongness is not.
VERIFIED_FACTS: list[KnowledgeEntry] = [
    # Population before country-area: "which country has the largest
    # population" contains BOTH keyword sets; the specific one wins.
    KnowledgeEntry(("largest", "population"), "India."),
    KnowledgeEntry(("most populous",), "India."),
    KnowledgeEntry(("largest", "country"), "Russia."),
    # Capitals.
    KnowledgeEntry(("capital", "france"), "Paris."),
    KnowledgeEntry(("capital", "united kingdom"), "London."),
    KnowledgeEntry(("capital", "england"), "London."),
    KnowledgeEntry(("capital", "italy"), "Rome."),
    KnowledgeEntry(("capital", "germany"), "Berlin."),
    KnowledgeEntry(("capital", "japan"), "Tokyo."),
    KnowledgeEntry(("capital", "united states"), "Washington, D.C."),
    KnowledgeEntry(("capital", "usa"), "Washington, D.C."),
    KnowledgeEntry(("capital", "canada"), "Ottawa."),
    KnowledgeEntry(("capital", "australia"), "Canberra."),
    KnowledgeEntry(("capital", "spain"), "Madrid."),
    KnowledgeEntry(("capital", "portugal"), "Lisbon."),
    # Arithmetic (whitespace around + is normalized away, so "2 + 2"
    # matches the "2+2" entry).
    KnowledgeEntry(("2+2",), "4."),
    KnowledgeEntry(("3+3",), "6."),
    KnowledgeEntry(("5+5",), "10."),
    KnowledgeEntry(("7+7",), "14."),
    KnowledgeEntry(("10+10",), "20."),
    # Geography.
    KnowledgeEntry(("continents",), "There are 7 continents: Asia, Africa, "
                                   "North America, South America, Antarctica, "
                                   "Europe, and Australia."),
    KnowledgeEntry(("tallest", "mountain"), "Mount Everest, 8,849 meters (29,032 ft)."),
    KnowledgeEntry(("largest", "ocean"), "The Pacific Ocean."),
    KnowledgeEntry(("largest", "desert"), "Antarctica is the largest desert on Earth."),
    KnowledgeEntry(("earth", "water"), "About 71% of Earth's surface is water."),
    # Space.
    KnowledgeEntry(("largest", "planet"), "Jupiter."),
    KnowledgeEntry(("closest", "sun"), "Mercury."),
    KnowledgeEntry(("first", "planet"), "Mercury."),
    KnowledgeEntry(("second", "planet"), "Venus."),
    KnowledgeEntry(("third", "planet"), "Earth."),
    KnowledgeEntry(("fourth", "planet"), "Mars."),
    KnowledgeEntry(("moon", "earth"), "The Moon is Earth's only natural satellite."),
    KnowledgeEntry(("speed", "light"), "299,792,458 meters per second."),
    # Literature.
    KnowledgeEntry(("hamlet",), "William Shakespeare."),
    KnowledgeEntry(("romeo",), "William Shakespeare."),
    # Science & the body.
    KnowledgeEntry(("boiling", "water"), "100 degrees Celsius (212 degrees Fahrenheit) at sea level."),
    KnowledgeEntry(("freezing", "water"), "0 degrees Celsius (32 degrees Fahrenheit) at sea level."),
    KnowledgeEntry(("bones", "body"), "An adult human has 206 bones."),
    KnowledgeEntry(("letters", "alphabet"), "26."),
    KnowledgeEntry(("days", "year"), "365 days in a common year, 366 in a leap year."),
    # Buildings.
    KnowledgeEntry(("tallest", "building"), "Burj Khalifa in Dubai, 828 meters."),
]


def _normalize(text: str) -> str:
    """Lowercase, strip sentence punctuation, and compact math spacing.

    "What is 2 + 2?" -> "what is 2+2". Keeping the + so arithmetic
    keywords match regardless of the user's spacing.
    """
    normalized = text.lower().strip(" !?.,;:")
    normalized = re.sub(r"\s*\+\s*", "+", normalized)
    return re.sub(r"\s+", " ", normalized)


class KnowledgeBase:
    """Keyword-matched lookup over verified facts. No I/O, fully testable.

    Matching is ALL keywords present (substring) in the normalized
    query. First match wins - the entry ORDER in VERIFIED_FACTS is part
    of the contract (specific entries must precede general ones).
    """

    def __init__(self, entries: list[KnowledgeEntry] | None = None) -> None:
        self._entries: list[KnowledgeEntry] = entries or VERIFIED_FACTS

    def lookup(self, query: str) -> str | None:
        """Return the verified answer for `query`, or None.

        Args:
            query: The raw user message.

        Returns:
            The verified answer, or None when nothing is verified -
            callers MUST treat None as "I don't know", never guess.
        """
        normalized = _normalize(query)
        for entry in self._entries:
            if all(keyword in normalized for keyword in entry.keywords):
                return entry.answer
        return None

    def __len__(self) -> int:
        return len(self._entries)
