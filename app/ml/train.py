"""Training script for the Phase 4 intent classifier.

Run it from the repo root:

    .\\.venv\\Scripts\\python -m app.ml.train

What it does, and why in this order (a script anyone can re-run):

  1. HOLD OUT 25% OF THE DATA. We must never evaluate a model on the
     examples it was trained on - it would just "remember" them and
     report ~100% accuracy that means nothing. The stratified split
     keeps every class's share of train vs test the same.

  2. TRAIN ON THE REMAINING 75% AND SCORE ON THE HELD-OUT 25%. The
     accuracy/confusion matrix you see is an HONEST estimate of how the
     model behaves on messages it has never seen.

  3. REFIT ON 100% OF THE DATA. Evaluation is over, so we now hand the
     model every example we own - more data is always better for the
     model that ships - and save that artifact.

  4. SAVE app/ml/artifacts/intent_pipeline.joblib, which
     IntentClassifier.load() reads at runtime.

The confusion matrix is printed in a compact grid: rows are TRUE intent
(test label), columns are PREDICTED intent. The diagonal (row == column)
is correct. Off-diagonal cells are the model confusing one intent for
another - read those and ask WHY (usually shared vocabulary: e.g.
"name" appears in both TEACH_FACT and RECALL_QUESTION).
"""

from __future__ import annotations

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from app.ml.dataset import INTENT_EXAMPLES, Intent
from app.ml.intent_classifier import IntentClassifier, _texts_and_labels

TEST_SIZE = 0.25
RANDOM_STATE = 42


def _render_confusion_matrix(matrix: list[list[int]], labels: list[str]) -> str:
    """Turn a sklearn matrix into a readable fixed-width grid."""
    width = max(len(label) for label in labels) + 2
    header = " " * width + "".join(f"{label:>{width}}" for label in labels)
    rows = [header]
    for label, row in zip(labels, matrix):
        rows.append(f"{label:>{width}}" + "".join(f"{value:>{width}}" for value in row))
    return "\n".join(rows)


def evaluate(classifier: IntentClassifier, texts: list[str], true_labels: list[str]) -> None:
    """Print accuracy, per-class metrics, and the confusion matrix."""
    predictions = classifier._pipeline.predict(texts)
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Accuracy on held-out test set: {accuracy:.1%}")
    print()
    print(classification_report(
        true_labels,
        predictions,
        labels=[intent.value for intent in Intent],
        zero_division=0,
    ))
    print("Confusion matrix (rows=true intent, columns=predicted intent):")
    matrix = confusion_matrix(
        true_labels,
        predictions,
        labels=[intent.value for intent in Intent],
    ).tolist()
    print(_render_confusion_matrix(matrix, [intent.value for intent in Intent]))


def main() -> None:
    """Run the full train -> evaluate -> refit -> save cycle."""
    print(f"Loading {len(INTENT_EXAMPLES)} labeled examples...")
    texts, labels = _texts_and_labels(INTENT_EXAMPLES)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    print(f"Split: {len(train_texts)} train / {len(test_texts)} test (stratified)\n")

    print("Training on the train split...")
    candidate = IntentClassifier.train(list(zip(train_texts, train_labels)))
    evaluate(candidate, test_texts, test_labels)

    print("\nRefitting on ALL examples and saving...")
    final = IntentClassifier.train(INTENT_EXAMPLES)
    path: Path = final.save()
    print(f"Saved model -> {path}")


if __name__ == "__main__":
    main()
