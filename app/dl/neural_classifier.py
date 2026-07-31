"""app.dl - Phase 5 (Deep Learning), Milestone 1.

Upgrades the intent classifier from a LINEAR model (logistic
regression, Phase 4) to a real neural network while keeping the exact
same API (train / predict / predict_top / save / load), so Milestone 2
can swap one for the other behind the router.

The deep-learning perspective on Phase 4: logistic regression IS a
neural network - a single linear layer feeding a softmax. It has no
hidden layer, so it can only learn LINEAR boundaries in feature space.
This module adds one hidden layer with ReLU, which lets the network
learn nonlinear decision boundaries, plus dropout and weight decay to
tame the extra capacity.

The honest caveat (the whole point of this phase): extra capacity only
helps when there is enough data to shape it. With 149 labeled examples
the network can MEMORIZE the training set (train accuracy -> ~100%)
while generalizing no better than the Phase 4 model. Deep learning's
advantage arrives with scale - pre-training (Phase 6) or lots of data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer

from app.ml.dataset import INTENT_EXAMPLES, Intent

# Canonical class order: index i in the network output = Intent_ORDER[i].
# Explicit and stable (alphabetical order would be arbitrary anyway).
INTENT_ORDER: list[str] = [intent.value for intent in Intent]

DEFAULT_ARTIFACT_PATH: Path = Path(__file__).parent / "artifacts" / "neural_intent.pt"


class IntentMLP(nn.Module):
    """The network: input (TF-IDF vector) -> hidden (ReLU) -> classes.

    A plain PyTorch module. The layers:
      * nn.Linear(input_dim, hidden_dim) - the hidden layer; each of the
        hidden_dim neurons is a weighted sum of the TF-IDF features,
        i.e. a learned "feature detector".
      * nn.ReLU - nonlinearity. Without it, two linear layers collapse
        into one linear layer and the network gains NOTHING. ReLU
        lets later layers build on patterns the earlier layer detected.
      * nn.Dropout - randomly zeroes neurons during training, forcing
        the network not to rely on any single feature. Regularization
        for a model with more capacity than data.
      * nn.Linear(hidden_dim, n_classes) - the same softmax readout
        logistic regression has; nothing new is needed here.

    Attributes:
        net: The Sequential stack; forward() passes x through it.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int = len(INTENT_ORDER),
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (one score per class) for a batch.

        Raw logits (not probabilities) are what CrossEntropyLoss wants,
        and predict() applies softmax itself.
        """
        return self.net(x)


class NeuralIntentClassifier:
    """Drop-in replacement for IntentClassifier backed by a neural net.

    Shares the public API so Milestone 2 can swap it in behind
    IntentRouter. The only internal difference from Phase 4: the model
    is an IntentMLP trained with backpropagation instead of a fitted
    sklearn estimator.

    Attributes:
        _vectorizer: Fitted TF-IDF features (identical recipe to the
            Phase 4 model - the FAIR comparison uses the same inputs).
        _model: The trained IntentMLP, or None before training.
    """

    def __init__(
        self,
        vectorizer: TfidfVectorizer | None = None,
        model: IntentMLP | None = None,
    ) -> None:
        self._vectorizer = vectorizer
        self._model = model

    # -------- lifecycle --------
    @classmethod
    def train(
        cls,
        examples: Sequence[tuple[str, Intent]] = INTENT_EXAMPLES,
        validation: Sequence[tuple[str, Intent]] | None = None,
        epochs: int = 200,
        batch_size: int = 16,
        lr: float = 1e-3,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        seed: int = 42,
        verbose: bool = False,
    ) -> "NeuralIntentClassifier":
        """Train the network with the full backpropagation loop.

        The loop is shown explicitly (not hidden in a fit() call) so the
        pieces of EVERY deep learning training run are visible: forward
        pass, loss, backward pass, optimizer step, per-epoch evaluation,
        and (when validation data is supplied) early stopping.

        Args:
            examples: Training pairs.
            validation: Optional held-out pairs; when given, loss is
                tracked and training stops early on a plateau.
            epochs: Max training epochs.
            batch_size: Examples per optimizer step.
            lr: Learning rate for Adam.
            hidden_dim: Width of the hidden layer.
            dropout: Dropout probability during training.
            seed: Reproducibility seed.
            verbose: Print a per-epoch progress line.

        Returns:
            A trained NeuralIntentClassifier.

        Raises:
            ValueError: If the dataset fails data-quality validation.
        """
        from app.ml.dataset import validate_dataset

        # Labels may be Intent members or their string values (the
        # training script hands over string labels from the split).
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

        torch.manual_seed(seed)
        texts, labels = _texts_and_labels(normalized)

        vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), strip_accents="unicode"
        )
        X = vectorizer.fit_transform(texts)
        y = torch.tensor([INTENT_ORDER.index(label) for label in labels])

        val_tensors: tuple[torch.Tensor, torch.Tensor] | None = None
        if validation:
            v_normalized = [
                (text, Intent(label) if isinstance(label, str) else label)
                for text, label in validation
            ]
            v_texts, v_labels = _texts_and_labels(v_normalized)
            X_val = vectorizer.transform(v_texts)
            val_tensors = (
                torch.tensor(X_val.toarray(), dtype=torch.float32),
                torch.tensor([INTENT_ORDER.index(label) for label in v_labels]),
            )

        model = IntentMLP(
            input_dim=X.shape[1],
            hidden_dim=hidden_dim,
            dropout=dropout,
            n_classes=len(INTENT_ORDER),
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        X_tensor = torch.tensor(X.toarray(), dtype=torch.float32)
        n = X_tensor.shape[0]
        best_state: dict[str, torch.Tensor] | None = None
        best_val_loss = float("inf")
        patience_remaining = 15

        for epoch in range(1, epochs + 1):
            model.train()
            permutation = torch.randperm(n)
            epoch_loss = 0.0
            correct = 0
            for start in range(0, n, batch_size):
                idx = permutation[start : start + batch_size]
                batch_x, batch_y = X_tensor[idx], y[idx]
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * idx.numel()
                correct += (logits.argmax(dim=1) == batch_y).sum().item()
            train_acc = correct / n
            train_loss = epoch_loss / n

            if val_tensors is None:
                if verbose and epoch % 10 == 0:
                    print(f"  epoch {epoch:>3}: train loss={train_loss:.3f} acc={train_acc:.2%}")
                continue

            model.eval()
            with torch.no_grad():
                v_x, v_y = val_tensors
                v_logits = model(v_x)
                val_loss = criterion(v_logits, v_y).item()
                val_acc = (v_logits.argmax(dim=1) == v_y).float().mean().item()
            if verbose:
                print(
                    f"  epoch {epoch:>3}: train loss={train_loss:.3f} acc={train_acc:.2%} | "
                    f"val loss={val_loss:.3f} acc={val_acc:.2%}"
                )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_remaining = 15
            else:
                patience_remaining -= 1
                if patience_remaining == 0:
                    if verbose:
                        print(f"  early stopping at epoch {epoch} (val loss plateaued)")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        return cls(vectorizer=vectorizer, model=model)

    def save(self, path: Path = DEFAULT_ARTIFACT_PATH) -> Path:
        """Persist vectorizer + network weights for load().

        torch.save writes the dict; joblib is NOT used here because the
        payload is a torch state dict (its own format). The fitted
        vectorizer travels inside as a joblib-pickled object.

        Args:
            path: Destination file.

        Returns:
            The path written.

        Raises:
            ValueError: If the classifier was never trained.
        """
        if self._model is None or self._vectorizer is None:
            raise ValueError("cannot save an untrained classifier")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "vectorizer": self._vectorizer,
                "state_dict": self._model.state_dict(),
                "input_dim": self._model.net[0].in_features,
                "hidden_dim": self._model.net[0].out_features,
                "n_classes": len(INTENT_ORDER),
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path = DEFAULT_ARTIFACT_PATH) -> "NeuralIntentClassifier":
        """Restore a previously saved classifier.

        Rebuilds the same architecture (input/hidden/classes dimensions
        are persisted) and loads the trained weights into it.

        Args:
            path: The artifact file from save().

        Returns:
            A trained NeuralIntentClassifier.

        Raises:
            FileNotFoundError: If the artifact is missing.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"no trained model at {path} - run `python -m app.dl.train` first"
            )
        payload = torch.load(path, weights_only=False)
        model = IntentMLP(
            input_dim=payload["input_dim"],
            hidden_dim=payload["hidden_dim"],
            n_classes=payload["n_classes"],
        )
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return cls(vectorizer=payload["vectorizer"], model=model)

    # -------- prediction --------
    def predict(self, text: str) -> tuple[Intent, float]:
        """Classify one message; identical contract to Phase 4."""
        if self._model is None or self._vectorizer is None:
            raise ValueError("cannot predict with an untrained classifier")
        vector = torch.tensor(self._vectorizer.transform([text]).toarray(), dtype=torch.float32)
        self._model.eval()
        with torch.no_grad():
            probs = torch.softmax(self._model(vector), dim=1)[0]
        index = int(probs.argmax(dim=0))
        return Intent(INTENT_ORDER[index]), float(probs[index])

    def predict_top(self, text: str, k: int = 2) -> list[tuple[Intent, float]]:
        """Return the k most likely (intent, confidence) pairs."""
        if self._model is None or self._vectorizer is None:
            raise ValueError("cannot predict with an untrained classifier")
        vector = torch.tensor(self._vectorizer.transform([text]).toarray(), dtype=torch.float32)
        self._model.eval()
        with torch.no_grad():
            probs = torch.softmax(self._model(vector), dim=1)[0].tolist()
        ranked = sorted(
            zip(INTENT_ORDER, probs), key=lambda item: item[1], reverse=True
        )
        return [(Intent(label), float(prob)) for label, prob in ranked[:k]]


def _texts_and_labels(
    examples: Sequence[tuple[str, Intent]],
) -> tuple[list[str], list[str]]:
    """Split (text, Intent) pairs into scikit/torch-friendly lists."""
    texts = [text for text, _ in examples]
    labels = [intent.value for _, intent in examples]
    return texts, labels
