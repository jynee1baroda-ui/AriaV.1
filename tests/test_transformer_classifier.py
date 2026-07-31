"""Tests for the Phase 6 TransformerIntentClassifier (Milestone 1).

The transformer classifier must be a DROP-IN for the Phase 4/5
classifiers: same public API, same data-quality gate, same lifecycle
guarantees. Architecture correctness is covered in
test_transformer_architecture; here we test the wrapper's contract.
"""

import tempfile
import unittest
from pathlib import Path

from sklearn.model_selection import train_test_split

from app.ml.dataset import INTENT_EXAMPLES, Intent
from app.transformer.classifier import TransformerIntentClassifier


class TestTransformerClassifierBehavior(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = TransformerIntentClassifier.train(INTENT_EXAMPLES, epochs=60)

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
        texts = [text for text, _ in INTENT_EXAMPLES]
        labels = [intent.value for _, intent in INTENT_EXAMPLES]
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        train = list(zip(train_texts, [Intent(l) for l in train_labels]))
        held_out = list(zip(val_texts, [Intent(l) for l in val_labels]))
        classifier = TransformerIntentClassifier.train(train, validation=held_out, epochs=30)
        predicted, _ = classifier.predict("hello")
        self.assertIn(predicted, Intent)

    def test_training_is_reproducible_with_same_seed(self):
        first = TransformerIntentClassifier.train(INTENT_EXAMPLES, epochs=20, seed=42)
        second = TransformerIntentClassifier.train(INTENT_EXAMPLES, epochs=20, seed=42)
        first_preds = [first.predict(text)[0] for text, _ in INTENT_EXAMPLES]
        second_preds = [second.predict(text)[0] for text, _ in INTENT_EXAMPLES]
        self.assertEqual(first_preds, second_preds)

    def test_train_rejects_invalid_dataset(self):
        dirty = INTENT_EXAMPLES + [("hello", Intent.GREETING)]
        with self.assertRaises(ValueError):
            TransformerIntentClassifier.train(dirty, epochs=2)

    def test_string_labels_are_coerced(self):
        texts = [text for text, _ in INTENT_EXAMPLES]
        labels = [intent.value for _, intent in INTENT_EXAMPLES]
        classifier = TransformerIntentClassifier.train(
            list(zip(texts, labels)), epochs=10
        )
        predicted, _ = classifier.predict("hello")
        self.assertIn(predicted, Intent)


class TestTransformerClassifierLifecycle(unittest.TestCase):
    def test_untrained_classifier_refuses_to_predict(self):
        classifier = TransformerIntentClassifier()
        with self.assertRaises(ValueError):
            classifier.predict("hello")

    def test_untrained_classifier_refuses_to_save(self):
        classifier = TransformerIntentClassifier()
        with self.assertRaises(ValueError):
            classifier.save()

    def test_save_then_load_round_trip_preserves_behavior(self):
        """The artifact must restore EXACTLY the model that saved it:
        loaded predictions must match the original model's predictions
        on every example (not the ground-truth labels - a 20-epoch model
        has not memorized the corpus, and that is fine)."""
        classifier = TransformerIntentClassifier.train(INTENT_EXAMPLES, epochs=20)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "transformer.pt"
            classifier.save(path)
            restored = TransformerIntentClassifier.load(path)
        for text, intent in INTENT_EXAMPLES:
            with self.subTest(text=text):
                original_pred, _ = classifier.predict(text)
                restored_pred, _ = restored.predict(text)
                self.assertEqual(restored_pred, original_pred)

    def test_load_missing_artifact_raises_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "does_not_exist.pt"
            with self.assertRaises(FileNotFoundError):
                TransformerIntentClassifier.load(missing)


if __name__ == "__main__":
    unittest.main()
