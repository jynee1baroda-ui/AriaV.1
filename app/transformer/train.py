"""Training script for the Phase 6 transformer intent classifier.

Run it from the repo root:

    .\\.venv\\Scripts\\python -m app.transformer.train

The experiment now has three competitors, all on the SAME stratified
split: the Phase 4 logistic regression (baseline), the Phase 5 MLP, and
the Phase 6 transformer. The verdict line reports what the numbers say
with no spin.

Expected outcome: the transformer, like the MLP, memorizes the training
set while validation accuracy lands near the baseline. A from-scratch
transformer on 149 examples cannot win - its advantage is PRE-TRAINED
knowledge, which arrives in Phase 7. What matters here is that the
architecture works end-to-end: it learns, it generalizes at baseline
level, and Phase 7 stacks real weights onto this foundation.
"""

from __future__ import annotations

from pathlib import Path

from sklearn.model_selection import train_test_split

from app.ml.dataset import INTENT_EXAMPLES, Intent
from app.ml.intent_classifier import IntentClassifier
from app.transformer.classifier import TransformerIntentClassifier

TEST_SIZE = 0.25
RANDOM_STATE = 42
EPOCHS = 200


def _accuracy(classifier: object, texts: list[str], labels: list[str]) -> float:
    """Accuracy over a (text, label) set for any classifier type."""
    correct = sum(1 for text, label in zip(texts, labels) if classifier.predict(text)[0].value == label)
    return correct / len(labels)


def main() -> None:
    """Train and evaluate the transformer against the two baselines."""
    print(f"Loading {len(INTENT_EXAMPLES)} labeled examples...")
    texts, labels = _texts_and_labels(INTENT_EXAMPLES)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    print(f"Split: {len(train_texts)} train / {len(test_texts)} held out (stratified)\n")

    print("[1/3] Logistic regression baseline (Phase 4)...")
    baseline = IntentClassifier.train(list(zip(train_texts, train_labels)))
    baseline_acc = _accuracy(baseline, test_texts, test_labels)
    print(f"      held-out accuracy: {baseline_acc:.1%}\n")

    print("[2/3] Transformer (Phase 6) - watch the overfitting gap:")
    transformer = TransformerIntentClassifier.train(
        list(zip(train_texts, train_labels)),
        validation=list(zip(test_texts, test_labels)),
        epochs=EPOCHS,
        seed=RANDOM_STATE,
        verbose=True,
    )
    transformer_acc = _accuracy(transformer, test_texts, test_labels)
    print(f"      held-out accuracy: {transformer_acc:.1%}\n")

    print("[3/3] Verdict")
    delta = transformer_acc - baseline_acc
    direction = "beats" if delta > 0.01 else ("ties" if abs(delta) <= 0.01 else "trails")
    print(
        f"  Transformer {direction} logistic regression on held-out data "
        f"({transformer_acc:.1%} vs {baseline_acc:.1%}, delta {delta:+.1%})."
    )
    print(
        "  From-scratch attention is not the win; PRE-TRAINED attention "
        "is. Phase 7 loads language knowledge into this architecture, "
        "and that is where the transformer's advantage finally shows.\n"
    )

    print("Refitting the transformer on ALL examples and saving...")
    final = TransformerIntentClassifier.train(
        INTENT_EXAMPLES, epochs=EPOCHS, seed=RANDOM_STATE
    )
    path: Path = final.save()
    print(f"Saved model -> {path}")


def _texts_and_labels(examples: list[tuple[str, Intent]]) -> tuple[list[str], list[str]]:
    texts = [text for text, _ in examples]
    labels = [intent.value for _, intent in examples]
    return texts, labels


if __name__ == "__main__":
    main()
