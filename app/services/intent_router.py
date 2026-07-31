"""IntentRouter - the ML half of Phase 4, Milestone 2.

Milestone 1 built the classifier as a MODULE. This milestone wires it
into the assistant as an ADVISORY LAYER: the router asks the model for
the message's intent, and only acts when the model is confident enough.
Everything else falls back to the deterministic rules.

Why advisory instead of "ML everywhere"? Three honest reasons:

  1. CONFIDENCE IS NOT TRUTH. "hello" -> greeting at 58% is a real
     prediction; "hello" at 30% is a guess. Routing on a weak guess
     makes the assistant WORSE than the rules, which never guess.
     The threshold turns probability into a decision boundary.

  2. THE MODEL CAN BE MISSING. The trained artifact is gitignored (a
     build artifact, not source). A fresh checkout has no model file.
     The router must degrade to rules gracefully - never crash a chat
     because a .joblib file is absent - and the APP must work before
     anyone ever runs the training script.

  3. RULES KNOW DETAILS THE MODEL DOES NOT. ML picks the coarse intent
     (teach_fact); MemoryExtractor still parses "my name is Alice"
     into the exact fact. Small-data models should not try to do
     structured extraction - that is a Phase 5+ (deep learning) or
     Phase 7 (LLM) job.

Graceful degradation is a feature, not a fallback: every failure mode
below resolves to "let the rules decide" and logs a single warning.

Phase 5, M2: the router can load either model backend - the Phase 4
logistic regression ("linear", the default) or the Phase 5 PyTorch
MLP ("neural") - selected by config.INTENT_MODEL_KIND. They share the
same predict API, so the gating logic below is backend-agnostic.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from app import config
from app.dl.neural_classifier import DEFAULT_ARTIFACT_PATH as NEURAL_ARTIFACT_PATH
from app.dl.neural_classifier import NeuralIntentClassifier
from app.ml.dataset import Intent
from app.ml.intent_classifier import IntentClassifier, DEFAULT_ARTIFACT_PATH

logger = logging.getLogger("assistant.ml")

# Allowed model backends. "linear" = Phase 4 logistic regression;
# "neural" = Phase 5 PyTorch MLP. Both expose the same predict API, so
# the router treats them identically (duck typing).
_SUPPORTED_KINDS: frozenset[str] = frozenset({"linear", "neural"})


class IntentRouter:
    """Gate the classifier behind a confidence threshold.

    Attributes:
        _classifier: The trained model (linear or neural), or None.
        _threshold:  Minimum confidence to act (default 0.5).
        _enabled:    Master switch (config.INTENT_CLASSIFIER_ENABLED).
        _kind:       Which model backend to load ("linear" | "neural").
        _artifact_path: Where to load the model from.
        _load_lock:  Guards lazy loading across threads (FastAPI runs
            handlers in a threadpool, so the first chat can race).
    """

    def __init__(
        self,
        classifier: IntentClassifier | NeuralIntentClassifier | None = None,
        threshold: float = config.INTENT_CONFIDENCE_THRESHOLD,
        enabled: bool = config.INTENT_CLASSIFIER_ENABLED,
        kind: str = config.INTENT_MODEL_KIND,
        artifact_path: Path | None = None,
    ) -> None:
        if kind not in _SUPPORTED_KINDS:
            raise ValueError(
                f"unknown INTENT_MODEL_KIND {kind!r}; expected one of {sorted(_SUPPORTED_KINDS)}"
            )
        self._classifier = classifier
        self._threshold = threshold
        self._enabled = enabled
        self._kind = kind
        # The artifact default depends on the backend: the two models
        # persist in different formats (joblib vs torch) at different
        # paths. An explicit artifact_path overrides for tests.
        self._artifact_path = artifact_path or (
            DEFAULT_ARTIFACT_PATH if kind == "linear" else NEURAL_ARTIFACT_PATH
        )
        self._load_lock = threading.Lock()

    def route(self, text: str) -> Intent | None:
        """Return the predicted intent, or None when rules should decide.

        The None answer covers every "not confident enough" case: model
        disabled, artifact missing, load failure, or confidence below
        the threshold. None never means "no intent" - it means "the
        rules own this message".

        Args:
            text: The raw user message.

        Returns:
            A confident Intent, or None to fall back to rule dispatch.
        """
        if not self._enabled:
            return None
        if self._classifier is None:
            self._classifier = self._load_or_none()
            if self._classifier is None:
                return None
        intent, confidence = self._classifier.predict(text)
        if confidence < self._threshold:
            return None
        return intent

    def _load_or_none(self) -> IntentClassifier | None:
        """Load the artifact once, thread-safely; None if unavailable.

        Lazy (not constructor) loading keeps app startup and tests fast
        when no model is needed. A missing or corrupt model logs ONE
        warning and permanently disables this router's ML - chat must
        survive infrastructure problems, not amplify them.

        Returns:
            The loaded classifier, or None on any failure.
        """
        if not self._artifact_path.exists():
            logger.warning(
                "ML router: no model at %s - running on rules only. "
                "Train one with `python -m app.ml.train`.",
                self._artifact_path,
            )
            self._enabled = False
            return None
        with self._load_lock:
            if self._classifier is not None:
                return self._classifier
            try:
                loader = (
                    NeuralIntentClassifier.load
                    if self._kind == "neural"
                    else IntentClassifier.load
                )
                return loader(self._artifact_path)
            except Exception:
                logger.exception("ML router: model failed to load - running on rules only.")
                self._enabled = False
                return None
