"""Tests for MemoryExtractor (Phase 5, Milestone 2).

Pure extraction: statements teach facts, questions do not. The tests
pin down the exact canonical content + category + confirmation reply
for each pattern so consolidation stays deterministic.
"""

from __future__ import annotations

import unittest

from app.services.memory_service import MemoryExtractor


class TestExtract(unittest.TestCase):
    def test_name_statement(self) -> None:
        facts = MemoryExtractor.extract("my name is Alice")
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].content, "the user's name is Alice")
        self.assertEqual(facts[0].category, "fact")
        self.assertIn("Alice", facts[0].reply)

    def test_name_statement_case_and_punctuation(self) -> None:
        facts = MemoryExtractor.extract("My name is Bob.")
        self.assertEqual(facts[0].content, "the user's name is Bob")

    def test_call_me_alias(self) -> None:
        facts = MemoryExtractor.extract("call me Charlie")
        self.assertEqual(facts[0].content, "the user's name is Charlie")

    def test_i_am_called_alias(self) -> None:
        facts = MemoryExtractor.extract("i'm called Dana")
        self.assertEqual(facts[0].content, "the user's name is Dana")

    def test_like_preference(self) -> None:
        facts = MemoryExtractor.extract("i like coffee")
        self.assertEqual(facts[0].content, "the user likes coffee")
        self.assertEqual(facts[0].category, "preference")

    def test_really_like(self) -> None:
        facts = MemoryExtractor.extract("I really like hiking")
        self.assertEqual(facts[0].content, "the user likes hiking")

    def test_dislike_preference(self) -> None:
        facts = MemoryExtractor.extract("i don't like spiders")
        self.assertEqual(facts[0].content, "the user dislikes spiders")

    def test_hate_preference(self) -> None:
        facts = MemoryExtractor.extract("i hate traffic")
        self.assertEqual(facts[0].content, "the user dislikes traffic")

    def test_remember_explicit(self) -> None:
        facts = MemoryExtractor.extract("remember that the sky is blue")
        self.assertEqual(facts[0].content, "the sky is blue")
        self.assertEqual(facts[0].category, "knowledge")

    def test_remember_colon(self) -> None:
        facts = MemoryExtractor.extract("remember: feed the cat")
        self.assertEqual(facts[0].content, "feed the cat")

    def test_favorite(self) -> None:
        facts = MemoryExtractor.extract("my favorite book is Dune")
        self.assertEqual(facts[0].content, "the user's favorite book is Dune")
        self.assertEqual(facts[0].category, "preference")

    def test_favourite_spelling(self) -> None:
        facts = MemoryExtractor.extract("my favourite colour is green")
        self.assertEqual(facts[0].content, "the user's favorite colour is green")

    def test_question_is_not_a_statement(self) -> None:
        self.assertEqual(MemoryExtractor.extract("what is my name"), [])

    def test_irrelevant_text_extracts_nothing(self) -> None:
        self.assertEqual(MemoryExtractor.extract("tell me a joke"), [])
        self.assertEqual(MemoryExtractor.extract("hello"), [])

    def test_i_would_like_is_not_a_like_statement(self) -> None:
        # "i'd like" must not be confused with "i like".
        self.assertEqual(MemoryExtractor.extract("i'd like a coffee"), [])


if __name__ == "__main__":
    unittest.main()
