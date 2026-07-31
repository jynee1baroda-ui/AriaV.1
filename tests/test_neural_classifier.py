"""Tests for the Phase 5 NeuralIntentClassifier (Milestone 1).

The neural classifier must be a DROP-IN for the Phase 4 IntentClassifier
behind the router: same public API, same data-quality gate, same
lifecycle guarantees. Tests keep the network tiny (a few epochs) so the
suite stays fast, and assert guarantees rather than accuracy numbers:

  * training runs and predictions are valid Intents,
  * train/predict/save/load round-trip preserves behavior,
  * same seed -> same predictions (reproducibility),
  * validation/early-stopping path works,
  * failure modes (untrained predict/save, missing artifact).
"""

import tempfile
import unittest
from pathlib import Path

from sklearn.model_selection import train_test_split

from app.dl.neural_classifier import NeuralIntentClassifier
from app.ml.dataset import INTENT_EXAMPLES, Intent


class TestNeuralClassifierBehavior(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Small: 30 epochs keeps the suite fast while still training a
        # network that recognizes its own training examples.
        cls.classifier = NeuralIntentClassifier.train(INTENT_EXAMPLES, epochs=30)

    def test_predicts_expected_intent_for_training_texts(self):
        for text, intent in INTENT_EXAMPLES:
            with self.subTest(text=text):
                predicted, _ = self.classifier.predict(text)
                self.assertEqual(predicted, intent)

    def test_confidence_is_a_probability(self):
        for text in ("hello", "my name is Alice", "what is my name"):
            _, confidence = self.classifier.predict(text)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)

    def test_predict_top_returns_sorted_candidates(self):
        ranked = self.classifier.predict_top("hello", k=2)
        self.assertEqual(len(ranked), 2)
        self.assertGreaterEqual(ranked[0][1], ranked[1][1])
        self.assertIsInstance(ranked[0][0], Intent)

    def test_training_with_validation_and_early_stopping(self):
        """The held-out path must run and return a working model.

        A stratified split keeps every class represented in both halves
        (a naive slice would starve some class and fail validation).
        """
        texts = [text for text, _ in INTENT_EXAMPLES]
        labels = [intent.value for _, intent in INTENT_EXAMPLES]
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        train = list(zip(train_texts, [Intent(l) for l in train_labels]))
        held_out = list(zip(val_texts, [Intent(l) for l in val_labels]))
        classifier = NeuralIntentClassifier.train(train, validation=held_out, epochs=30)
        predicted, _ = classifier.predict("hello")
        self.assertIn(predicted, Intent)

    def test_training_is_reproducible_with_same_seed(self):
        first = NeuralIntentClassifier.train(INTENT_EXAMPLES, epochs=30, seed=42)
        second = NeuralIntentClassifier.train(INTENT_EXAMPLES, epochs=30, seed=42)
        first_preds = [first.predict(text)[0] for text, _ in INTENT_EXAMPLES]
        second_preds = [second.predict(text)[0] for text, _ in INTENT_EXAMPLES]
        self.assertEqual(first_preds, second_preds)

    def test_train_rejects_invalid_dataset(self):
        dirty = INTENT_EXAMPLES + [("hello", Intent.GREETING)]
        with self.assertRaises(ValueError):
            NeuralIntentClassifier.train(dirty, epochs=2)


class TestNeuralClassifierLifecycle(unittest.TestCase):
    def test_untrained_classifier_refuses_to_predict(self):
        classifier = NeuralIntentClassifier()
        with self.assertRaises(ValueError):
            classifier.predict("hello")

    def test_untrained_classifier_refuses_to_save(self):
        classifier = NeuralIntentClassifier()
        with self.assertRaises(ValueError):
            classifier.save()

    def test_save_then_load_round_trip_preserves_behavior(self):
        classifier = NeuralIntentClassifier.train(INTENT_EXAMPLES, epochs=30)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "neural.pt"
            classifier.save(path)
            restored = NeuralIntentClassifier.load(path)
        for text, intent in INTENT_EXAMPLES:
            predicted, _ = restored.predict(text)
            self.assertEqual(predicted, intent)

    def test_load_missing_artifact_raises_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "does_not_exist.pt"
            with self.assertRaises(FileNotFoundError):
                NeuralIntentClassifier.load(missing)


if __name__ == "__main__":
    unittest.main()
