"""Tests for the Phase 4 intent dataset.

Milestone 1, data-quality half. These tests exist because the dataset is
the one thing a better model cannot fix: a duplicate, a typo'd label, or
a starved class silently corrupts every accuracy number the training
script prints.
"""

import unittest

from app.ml.dataset import INTENT_EXAMPLES, Intent, validate_dataset


class TestIntentDataset(unittest.TestCase):
    def test_all_labels_are_intent_members(self):
        """Every label must be a real Intent, not a stray string."""
        for _, intent in INTENT_EXAMPLES:
            self.assertIn(intent, Intent)

    def test_no_duplicate_texts(self):
        """Duplicates inflate a class's apparent learnability."""
        seen = set()
        for text, _ in INTENT_EXAMPLES:
            key = text.strip().lower()
            self.assertNotIn(key, seen, f"duplicate example: {text!r}")
            seen.add(key)

    def test_every_intent_is_represented(self):
        """A class with zero examples is not a class the model can learn."""
        covered = {intent for _, intent in INTENT_EXAMPLES}
        self.assertEqual(covered, set(Intent))

    def test_no_empty_texts(self):
        for text, _ in INTENT_EXAMPLES:
            self.assertTrue(text.strip(), "an empty example teaches nothing")

    def test_dataset_passes_full_validation(self):
        """The shipped dataset must be clean out of the box."""
        self.assertEqual(validate_dataset(INTENT_EXAMPLES), [])

    def test_validation_flags_a_duplicate(self):
        """validate_dataset must catch the exact bugs it documents."""
        dirty = INTENT_EXAMPLES + [("hello", Intent.GREETING)]
        problems = validate_dataset(dirty)
        self.assertTrue(any("duplicate" in p for p in problems))

    def test_validation_flags_an_invalid_label(self):
        dirty = INTENT_EXAMPLES + [("custom message", "not_an_intent")]  # type: ignore[list-item]
        problems = validate_dataset(dirty)
        self.assertTrue(any("not an Intent" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
