"""Tests for the Phase 4 IntentClassifier (Milestone 1).

Strategy: the tests that matter for a classifier are the ones the
training script CANNOT show you, namely
  * behavior on UNSEEN phrasings (generalization),
  * reproducibility (same data -> same model),
  * artifact save/load round-trip (runtime restore),
  * failure modes (untrained predict, missing artifact).

We deliberately do NOT assert a high accuracy threshold on the tiny
labeled dataset - a single threshold would just overfit the tests to
today's examples. Instead the tests pin down guarantees the API must
keep, and data-quality is covered separately in test_intent_dataset.
"""

import tempfile
import unittest
from pathlib import Path

from app.ml.dataset import INTENT_EXAMPLES, Intent
from app.ml.intent_classifier import IntentClassifier


class TestIntentClassifierBehavior(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """One fit for the whole class: training is the slowest step."""
        cls.classifier = IntentClassifier.train(INTENT_EXAMPLES)

    def test_predicts_correct_intent_for_exact_training_texts(self):
        """The model must at least learn what it was shown."""
        for text, intent in INTENT_EXAMPLES:
            with self.subTest(text=text):
                predicted, _ = self.classifier.predict(text)
                self.assertEqual(predicted, intent)

    def test_generalizes_to_unseen_phrasings(self):
        """These messages were never trained on; the model must still
        classify them by the words and word-pairs it learned."""
        cases = [
            ("hello my friend", Intent.GREETING),
            ("hi!", Intent.GREETING),
            ("do you remember my name please", Intent.RECALL_QUESTION),
            ("whats my name?", Intent.RECALL_QUESTION),
            ("tell me your name please", Intent.ASSISTANT_IDENTITY),
            ("what is your name my friend", Intent.ASSISTANT_IDENTITY),
            ("i really love hiking in the mountains", Intent.TEACH_FACT),
            ("please remember to feed the cat", Intent.TEACH_FACT),
            ("can you tell me a funny joke", Intent.OTHER),
            ("the stock market crashed", Intent.OTHER),
        ]
        for text, intent in cases:
            with self.subTest(text=text):
                predicted, _ = self.classifier.predict(text)
                self.assertEqual(predicted, intent)

    def test_distinguishes_teach_from_recall_on_word_order(self):
        """The same words in different order are DIFFERENT intents;
        bigrams exist precisely to capture that."""
        teach, _ = self.classifier.predict("my name is Alice")
        recall, _ = self.classifier.predict("what is my name")
        self.assertEqual(teach, Intent.TEACH_FACT)
        self.assertEqual(recall, Intent.RECALL_QUESTION)

    def test_confidence_is_a_probability_between_zero_and_one(self):
        for text in ("hello", "my name is Alice", "what is my name"):
            _, confidence = self.classifier.predict(text)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)

    def test_predict_top_returns_sorted_candidates(self):
        ranked = self.classifier.predict_top("hello", k=2)
        self.assertEqual(len(ranked), 2)
        self.assertGreaterEqual(ranked[0][1], ranked[1][1])
        self.assertIsInstance(ranked[0][0], Intent)

    def test_train_is_reproducible(self):
        """Same data + same random_state must give the same classifier."""
        again = IntentClassifier.train(INTENT_EXAMPLES)
        first = {text: self.classifier.predict(text)[0] for text, _ in INTENT_EXAMPLES}
        second = {text: again.predict(text)[0] for text, _ in INTENT_EXAMPLES}
        self.assertEqual(first, second)


class TestIntentClassifierLifecycle(unittest.TestCase):
    def test_untrained_classifier_refuses_to_predict(self):
        classifier = IntentClassifier()
        with self.assertRaises(ValueError):
            classifier.predict("hello")

    def test_untrained_classifier_refuses_to_save(self):
        classifier = IntentClassifier()
        with self.assertRaises(ValueError):
            classifier.save()

    def test_save_then_load_round_trip_preserves_behavior(self):
        """The artifact written by train/save must restore identically."""
        classifier = IntentClassifier.train(INTENT_EXAMPLES)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            classifier.save(path)
            restored = IntentClassifier.load(path)
        for text, intent in INTENT_EXAMPLES:
            predicted, _ = restored.predict(text)
            self.assertEqual(predicted, intent)

    def test_load_missing_artifact_raises_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "does_not_exist.joblib"
            with self.assertRaises(FileNotFoundError):
                IntentClassifier.load(missing)

    def test_train_rejects_invalid_dataset(self):
        dirty = INTENT_EXAMPLES + [("hello", Intent.GREETING)]
        with self.assertRaises(ValueError):
            IntentClassifier.train(dirty)


if __name__ == "__main__":
    unittest.main()
