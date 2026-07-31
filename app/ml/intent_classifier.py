"""IntentClassifier - a supervised text classifier (Phase 4, M1).

The core of Machine Learning milestone 1. It wraps a scikit-learn
Pipeline in a small API the rest of the app can call, and separates the
ML machinery from the training recipe.

Why this model? (the "what" and "how" of Phase 4)

  1. TF-IDF VECTORIZER - turns raw text into numbers. Each message
     becomes a vector; each dimension is a word (or word pair, thanks
     to ngram_range) weighted by how important it is. "the" appears in
     everything, so it scores ~0; "pancakes" appears rarely and only in
     OTHER examples, so it scores high. Word ORDER also matters: "my
     name is Alice" and "what is my name" share almost every word, so
     we also emit bigrams ("name is" vs "is my") to tell them apart.

  2. LOGISTIC REGRESSION - the classifier. It learns a weight for each
     vocabulary dimension per intent class; prediction picks the class
     with the highest score. It is a LINEAR model: fast, tiny,
     interpretable, and for small text datasets it beats deep networks.
     The LLM (Phase 7) will eventually route intents too, but a 120
     example problem does not need one - this is the honest tradeoff.

  3. PIPELINE - bundles vectorizer + classifier so the raw text flows
     through one fitted object. It also guarantees the EXACT same
     preprocessing at train and predict time, eliminating a whole class
     of bugs where the two disagree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.ml.dataset import INTENT_EXAMPLES, Intent

# Where the trained model is persisted. Committed artifacts are not
# stored in git (see .gitignore); run `python -m app.ml.train` to
# regenerate it after changing the dataset.
DEFAULT_ARTIFACT_PATH: Path = Path(__file__).parent / "artifacts" / "intent_pipeline.joblib"


def build_pipeline() -> Pipeline:
    """Create an UNTRAINED (but wired-up) TF-IDF + logistic pipeline.

    ngram_range=(1, 2) emits unigrams AND bigrams. Without bigrams,
    "my name is Alice" vs "what is my name" share almost all words and
    become nearly indistinguishable. random_state makes the result
    reproducible - train twice, get the same classifier.

    Returns:
        An unfitted Pipeline ready for .fit().
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(lowercase=True, ngram_range=(1, 2), strip_accents="unicode"),
            ),
            (
                "classifier",
                # class_weight="balanced" counteracts the class-prior
                # problem: with 40 TEACH examples vs 21 greetings, a
                # plain logistic regression learns "when unsure, guess
                # teach" and every weak signal falls into the big class.
                # Balanced weights re-scale each class's loss so the
                # smaller classes can win on their own terms.
                LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced"),
            ),
        ]
    )


def _texts_and_labels(
    examples: Sequence[tuple[str, Intent]],
) -> tuple[list[str], list[str]]:
    """Split (text, Intent) pairs into the two lists scikit-learn wants.

    X = the input texts, y = the answer labels. scikit-learn only
    accepts arrays of strings for the labels, so we map each Intent
    through .value (StrEnum keeps that string identical to the enum
    member, which is how predict() maps results back to Intents).
    """
    texts = [text for text, _ in examples]
    labels = [intent.value for _, intent in examples]
    return texts, labels


class IntentClassifier:
    """Trainable/predictable wrapper around the sklearn pipeline.

    Usage (Milestone 2 wires this into the Assistant):

        classifier = IntentClassifier.train(INTENT_EXAMPLES)
        classifier.predict("hey there")    # -> (Intent.GREETING, 0.99)
        classifier.save()
        classifier = IntentClassifier.load()
        classifier.predict("hi")

    Attributes:
        _pipeline: The fitted Pipeline, or None before training.
    """

    def __init__(self, pipeline: Pipeline | None = None) -> None:
        self._pipeline = pipeline

    # -------- lifecycle --------
    @classmethod
    def train(cls, examples: Sequence[tuple[str, Intent]] = INTENT_EXAMPLES) -> "IntentClassifier":
        """Fit the pipeline on the labeled examples.

        Labels may be Intent members or their string values; both are
        accepted so callers can hand in data straight from a split
        (strings) or from the dataset module (enums).

        Args:
            examples: (text, Intent) training pairs.

        Returns:
            A new, trained IntentClassifier.

        Raises:
            ValueError: If the dataset fails data-quality validation.
        """
        from app.ml.dataset import validate_dataset

        normalized: list[tuple[str, Intent]] = []
        problems: list[str] = []
        for i, (text, label) in enumerate(examples):
            if isinstance(label, Intent):
                normalized.append((text, label))
            elif isinstance(label, str):
                try:
                    normalized.append((text, Intent(label)))
                except ValueError:
                    problems.append(f"example {i}: label {label!r} is not an Intent")
            else:
                problems.append(f"example {i}: label {label!r} is not an Intent")

        problems += validate_dataset(normalized)
        if problems:
            raise ValueError("dataset is invalid:\n  " + "\n  ".join(problems))

        texts, labels = _texts_and_labels(normalized)
        pipeline = build_pipeline()
        pipeline.fit(texts, labels)
        return cls(pipeline)

    def save(self, path: Path = DEFAULT_ARTIFACT_PATH) -> Path:
        """Persist the fitted pipeline so load() can restore it.

        joblib is sklearn's serialization format: it handles numpy
        arrays (the fitted weights) that plain pickle would mangle.

        Args:
            path: Destination file.

        Returns:
            The path written (for chaining/reporting).

        Raises:
            ValueError: If the classifier was never trained.
        """
        if self._pipeline is None:
            raise ValueError("cannot save an untrained classifier")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, path)
        return path

    @classmethod
    def load(cls, path: Path = DEFAULT_ARTIFACT_PATH) -> "IntentClassifier":
        """Restore a previously saved classifier.

        Args:
            path: The artifact file from save().

        Returns:
            A trained IntentClassifier.

        Raises:
            FileNotFoundError: If the artifact is missing (run
                `python -m app.ml.train` to produce it).
        """
        if not path.exists():
            raise FileNotFoundError(
                f"no trained model at {path} - run `python -m app.ml.train` first"
            )
        return cls(joblib.load(path))

    # -------- prediction --------
    def predict(self, text: str) -> tuple[Intent, float]:
        """Classify one message into its most likely intent.

        Args:
            text: The raw user message.

        Returns:
            (predicted Intent, confidence in [0, 1]). The confidence is
            the soft-max probability the model assigned to this class;
            it is NOT the truth - use it as a routing threshold, not a
            fact about the world.

        Raises:
            ValueError: If the classifier was never trained.
        """
        if self._pipeline is None:
            raise ValueError("cannot predict with an untrained classifier")
        label = self._pipeline.predict([text])[0]
        probabilities = self._pipeline.predict_proba([text])[0]
        confidence = float(max(probabilities))
        return Intent(label), confidence

    def predict_top(
        self, text: str, k: int = 2
    ) -> list[tuple[Intent, float]]:
        """Return the k most likely (intent, confidence) pairs.

        Useful when the top pick is too weak to trust: if GREETING and
        ASSISTANT_IDENTITY are both at 0.4, the message is genuinely
        ambiguous and the caller should fall back to rules.

        Args:
            text: The raw user message.
            k: How many candidates to return.

        Returns:
            (intent, confidence) pairs, highest confidence first.
        """
        if self._pipeline is None:
            raise ValueError("cannot predict with an untrained classifier")
        probabilities = self._pipeline.predict_proba([text])[0]
        classes = self._pipeline.classes_
        ranked = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)
        return [(Intent(label), float(prob)) for label, prob in ranked[:k]]
