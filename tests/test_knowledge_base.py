"""Tests for the verified knowledge base (Phase 7 hardening).

The KB contract: an exact, verified answer OR None. None must never be
coerced into a guess - the Assistant turns it into a refusal. These
tests pin the verified answers (so a wrong "fix" is caught) and the
normalization rules.
"""

import unittest

from app.kb.knowledge import VERIFIED_FACTS, KnowledgeBase


class TestKnowledgeBaseLookup(unittest.TestCase):
    def setUp(self) -> None:
        self.kb = KnowledgeBase()

    def test_capital_of_france_is_paris(self):
        self.assertEqual(self.kb.lookup("what is the capital of France?"), "Paris.")

    def test_casing_and_punctuation_are_ignored(self):
        self.assertEqual(self.kb.lookup("WHAT IS THE CAPITAL OF FRANCE!!"), "Paris.")

    def test_math_without_spaces(self):
        self.assertEqual(self.kb.lookup("what is 2+2?"), "4.")

    def test_math_with_spaces(self):
        self.assertEqual(self.kb.lookup("what is 2 + 2?"), "4.")

    def test_hamlet_author_is_verified(self):
        """The model got this wrong in live testing - the KB fixes it."""
        self.assertEqual(self.kb.lookup("who wrote hamlet?"), "William Shakespeare.")

    def test_unknown_question_returns_none(self):
        self.assertIsNone(self.kb.lookup("what is the meaning of life?"))

    def test_specific_entry_beats_general_one(self):
        """'largest population' must match the population entry, not
        the largest-country entry (order is part of the contract)."""
        self.assertEqual(self.kb.lookup("which country has the largest population"), "India.")

    def test_largest_country_is_russia(self):
        self.assertEqual(self.kb.lookup("what is the largest country by area"), "Russia.")

    def test_continents_count(self):
        self.assertIn("7 continents", self.kb.lookup("how many continents are there?"))

    def test_every_verified_fact_is_a_nonempty_string(self):
        for entry in VERIFIED_FACTS:
            self.assertTrue(entry.answer.strip(), f"empty answer for {entry.keywords}")

    def test_no_keyword_is_an_empty_string(self):
        for entry in VERIFIED_FACTS:
            self.assertTrue(all(entry.keywords), f"empty keyword in {entry.keywords}")

    def test_custom_entries_are_used(self):
        from app.kb.knowledge import KnowledgeEntry

        kb = KnowledgeBase([KnowledgeEntry(("special",), "answer.")])
        self.assertEqual(kb.lookup("something special here"), "answer.")
        self.assertIsNone(kb.lookup("unrelated"))


if __name__ == "__main__":
    unittest.main()
