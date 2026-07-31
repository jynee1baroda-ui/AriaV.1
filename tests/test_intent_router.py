"""Tests for IntentRouter (Phase 4, Milestone 2).

The router's contract: return a confident intent, or None to let the
rules decide. None must cover every unconfident state - disabled,
missing model, corrupt model, or low confidence. These tests use a
deterministic STUB classifier so the gating logic is tested without
depending on a trained artifact (which is gitignored and may be absent).
"""

import tempfile
import unittest
from pathlib import Path

from app.dl.neural_classifier import NeuralIntentClassifier
from app.ml.dataset import INTENT_EXAMPLES, Intent
from app.services.intent_router import IntentRouter
from app.transformer.classifier import TransformerIntentClassifier


class StubClassifier:
    """Deterministic fake: always predicts the configured result."""

    def __init__(self, intent: Intent, confidence: float) -> None:
        self._intent = intent
        self._confidence = confidence

    def predict(self, text: str) -> tuple[Intent, float]:
        return self._intent, self._confidence


class TestIntentRouterGating(unittest.TestCase):
    def test_confident_prediction_is_returned(self):
        router = IntentRouter(classifier=StubClassifier(Intent.GREETING, 0.95))
        self.assertEqual(router.route("hello"), Intent.GREETING)

    def test_below_threshold_falls_back_to_rules(self):
        router = IntentRouter(classifier=StubClassifier(Intent.GREETING, 0.49))
        self.assertIsNone(router.route("hello"))

    def test_at_threshold_is_confident(self):
        router = IntentRouter(classifier=StubClassifier(Intent.OTHER, 0.5), threshold=0.5)
        self.assertEqual(router.route("anything"), Intent.OTHER)

    def test_custom_threshold_changes_decisions(self):
        strict = IntentRouter(classifier=StubClassifier(Intent.OTHER, 0.6), threshold=0.75)
        self.assertIsNone(strict.route("anything"))
        loose = IntentRouter(classifier=StubClassifier(Intent.OTHER, 0.6), threshold=0.5)
        self.assertEqual(loose.route("anything"), Intent.OTHER)

    def test_disabled_router_never_acts(self):
        router = IntentRouter(
            classifier=StubClassifier(Intent.GREETING, 0.99), enabled=False
        )
        self.assertIsNone(router.route("hello"))

    def test_injected_classifier_needs_no_artifact(self):
        """A supplied classifier must be used without touching disk."""
        router = IntentRouter(
            classifier=StubClassifier(Intent.TEACH_FACT, 0.9),
            artifact_path=Path("definitely/not/a/real/path.joblib"),
        )
        self.assertEqual(router.route("my name is Alice"), Intent.TEACH_FACT)


class TestIntentRouterGracefulDegradation(unittest.TestCase):
    def test_missing_artifact_degrades_to_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "no_model.joblib"
            router = IntentRouter(artifact_path=missing)
            self.assertIsNone(router.route("hello"))
            # And it stays degraded - no repeated disk attempts.
            self.assertIsNone(router.route("hello again"))

    def test_corrupt_artifact_degrades_to_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            corrupt = Path(tmp) / "corrupt.joblib"
            corrupt.write_bytes(b"this is not a joblib file at all")
            router = IntentRouter(artifact_path=corrupt)
            self.assertIsNone(router.route("hello"))

    def test_disabled_router_without_classifier_is_safe(self):
        router = IntentRouter(enabled=False)
        self.assertIsNone(router.route("hello"))


class TestIntentRouterModelSelection(unittest.TestCase):
    """Phase 5 M2: both model backends plug into the same router."""

    def test_injected_neural_classifier_is_used(self):
        """A supplied NeuralIntentClassifier drives the router exactly
        like the linear one - the duck-typed API is the whole point of
        the interface parity. threshold=0 removes confidence gating so
        the test proves the router consults the model at all (a 5-epoch
        net is rarely >50% confident on real messages)."""
        neural = NeuralIntentClassifier.train(INTENT_EXAMPLES, epochs=5)
        router = IntentRouter(classifier=neural, kind="neural", threshold=0.0)
        intent, _ = neural.predict("hello")
        self.assertEqual(router.route("hello"), intent)

    def test_neural_kind_defaults_to_neural_artifact(self):
        router = IntentRouter(kind="neural", enabled=False)
        self.assertIsNone(router.route("hello"))

    def test_missing_neural_artifact_degrades_gracefully(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "no_neural.pt"
            router = IntentRouter(kind="neural", artifact_path=missing)
            self.assertIsNone(router.route("hello"))

    def test_injected_transformer_classifier_is_used(self):
        """Phase 6 M2: the from-scratch transformer drives the router
        exactly like the other two backends."""
        transformer = TransformerIntentClassifier.train(INTENT_EXAMPLES, epochs=5)
        router = IntentRouter(classifier=transformer, kind="transformer", threshold=0.0)
        intent, _ = transformer.predict("hello")
        self.assertEqual(router.route("hello"), intent)

    def test_missing_transformer_artifact_degrades_gracefully(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "no_transformer.pt"
            router = IntentRouter(kind="transformer", artifact_path=missing)
            self.assertIsNone(router.route("hello"))

    def test_unknown_kind_fails_fast(self):
        """A typo in config must explode at construction, not silently
        fall back at the first chat."""
        with self.assertRaises(ValueError):
            IntentRouter(kind="recurrent")


if __name__ == "__main__":
    unittest.main()
