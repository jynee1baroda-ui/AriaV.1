"""TransformerIntentClassifier - the Phase 6 drop-in classifier.

Same public API as Phases 4-5 (train / predict / predict_top / save /
load), backed by a from-scratch transformer. The NEW machinery versus
Phase 5 is tokenization: a transformer consumes a SEQUENCE of token ids,
not a bag-of-words vector, so we build a vocabulary, encode sentences,
and pad them to a fixed length with a mask that keeps padding out of
attention.

The training loop mirrors Phase 5 (forward, loss, backward, Adam,
per-epoch evaluation, early stopping) - the loop is not what changes
between model families.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

from app.ml.dataset import INTENT_EXAMPLES, Intent
from app.transformer.architecture import IntentTransformer

# Shared by all models in this module: canonical class order.
INTENT_ORDER: list[str] = [intent.value for intent in Intent]

DEFAULT_ARTIFACT_PATH: Path = Path(__file__).parent / "artifacts" / "transformer_intent.pt"

TOK_PAD = "<pad>"
TOK_UNK = "<unk>"
MAX_SEQ_LEN = 16

# Tokenizer: lowercase, keep letters/digits/apostrophes, drop everything
# else. Deliberately simple - real LLM tokenizers (BPE, Phase 7) learn
# subword units from data instead.
_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> list[str]:
    """Split one message into word tokens: 'What's UP!!' -> [what's, up]."""
    return _TOKEN_RE.findall(text.lower())


def _build_vocab(texts: Sequence[str]) -> dict[str, int]:
    """Build word->id map from the training corpus.

    The two special tokens own the lowest ids: 0 is padding (its
    embedding is pinned to zero by padding_idx so padding contributes
    nothing), 1 is unknown (for words never seen at train time).
    """
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(_tokenize(text))
    vocab = {TOK_PAD: 0, TOK_UNK: 1}
    for word, _ in counts.most_common():
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab


class TransformerIntentClassifier:
    """Drop-in replacement for IntentClassifier, transformer-backed.

    Attributes:
        _vocab: word -> id map (built at train time, persisted).
        _model: The trained IntentTransformer, or None.
    """

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        model: IntentTransformer | None = None,
    ) -> None:
        self._vocab = vocab
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
        seed: int = 42,
        verbose: bool = False,
    ) -> "TransformerIntentClassifier":
        """Train the transformer with the backpropagation loop.

        Args:
            examples: Training pairs.
            validation: Optional held-out pairs for early stopping.
            epochs: Max training epochs.
            batch_size: Examples per optimizer step.
            lr: Learning rate for Adam.
            seed: Reproducibility seed.
            verbose: Print a per-epoch progress line.

        Returns:
            A trained TransformerIntentClassifier.

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

        torch.manual_seed(seed)
        texts, labels = _texts_and_labels(normalized)
        vocab = _build_vocab(texts)
        X, mask = _encode_batch(texts, vocab, MAX_SEQ_LEN)
        y = torch.tensor([INTENT_ORDER.index(label) for label in labels])

        val_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        if validation:
            v_normalized = [
                (text, Intent(label) if isinstance(label, str) else label)
                for text, label in validation
            ]
            v_texts, v_labels = _texts_and_labels(v_normalized)
            v_x, v_mask = _encode_batch(v_texts, vocab, MAX_SEQ_LEN)
            val_tensors = (
                v_x,
                v_mask,
                torch.tensor([INTENT_ORDER.index(label) for label in v_labels]),
            )

        model = IntentTransformer(vocab_size=len(vocab), n_classes=len(INTENT_ORDER))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        n = X.shape[0]
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
                batch_x, batch_mask, batch_y = X[idx], mask[idx], y[idx]
                optimizer.zero_grad()
                logits = model(batch_x, batch_mask)
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
                v_x, v_mask, v_y = val_tensors
                v_logits = model(v_x, v_mask)
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

        return cls(vocab=vocab, model=model)

    def save(self, path: Path = DEFAULT_ARTIFACT_PATH) -> Path:
        """Persist vocab + architecture config + weights for load().

        The vocab and hyperparameters travel in the payload so load()
        can rebuild the EXACT same model with zero knowledge of how it
        was trained.

        Args:
            path: Destination file.

        Returns:
            The path written.

        Raises:
            ValueError: If the classifier was never trained.
        """
        if self._model is None or self._vocab is None:
            raise ValueError("cannot save an untrained classifier")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "vocab": self._vocab,
                "state_dict": self._model.state_dict(),
                "embed_dim": self._model.token_embed.embedding_dim,
                "n_heads": self._model.blocks[0].attn.n_heads,
                "n_blocks": len(self._model.blocks),
                "ff_dim": self._model.blocks[0].ff[0].out_features,
                "n_classes": len(INTENT_ORDER),
                "max_len": MAX_SEQ_LEN,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path = DEFAULT_ARTIFACT_PATH) -> "TransformerIntentClassifier":
        """Restore a previously saved classifier.

        Args:
            path: The artifact file from save().

        Returns:
            A trained TransformerIntentClassifier.

        Raises:
            FileNotFoundError: If the artifact is missing.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"no trained model at {path} - run `python -m app.transformer.train` first"
            )
        payload = torch.load(path, weights_only=False)
        model = IntentTransformer(
            vocab_size=len(payload["vocab"]),
            embed_dim=payload["embed_dim"],
            n_heads=payload["n_heads"],
            n_blocks=payload["n_blocks"],
            ff_dim=payload["ff_dim"],
            n_classes=payload["n_classes"],
            max_len=payload["max_len"],
        )
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return cls(vocab=payload["vocab"], model=model)

    # -------- prediction --------
    def predict(self, text: str) -> tuple[Intent, float]:
        """Classify one message; identical contract to Phases 4-5."""
        if self._model is None or self._vocab is None:
            raise ValueError("cannot predict with an untrained classifier")
        tokens, token_mask = _encode_batch([text], self._vocab, MAX_SEQ_LEN)
        self._model.eval()
        with torch.no_grad():
            probs = torch.softmax(self._model(tokens, token_mask), dim=1)[0]
        index = int(probs.argmax(dim=0))
        return Intent(INTENT_ORDER[index]), float(probs[index])

    def predict_top(self, text: str, k: int = 2) -> list[tuple[Intent, float]]:
        """Return the k most likely (intent, confidence) pairs."""
        if self._model is None or self._vocab is None:
            raise ValueError("cannot predict with an untrained classifier")
        tokens, token_mask = _encode_batch([text], self._vocab, MAX_SEQ_LEN)
        self._model.eval()
        with torch.no_grad():
            probs = torch.softmax(self._model(tokens, token_mask), dim=1)[0].tolist()
        ranked = sorted(zip(INTENT_ORDER, probs), key=lambda item: item[1], reverse=True)
        return [(Intent(label), float(prob)) for label, prob in ranked[:k]]


def _encode_batch(
    texts: Sequence[str], vocab: dict[str, int], max_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize + pad a batch into (token ids, padding mask).

    Words outside the vocab map to <unk>. Sequences longer than max_len
    are truncated (our messages are short; 16 covers them all). The
    mask is True where a position is a real token - attention must not
    leak padding, and pooling must not average it in.

    Returns:
        (batch, max_len) long tensor and (batch, max_len) bool tensor.
    """
    ids: list[list[int]] = []
    for text in texts:
        encoded = [
            vocab.get(token, vocab[TOK_UNK]) for token in _tokenize(text)[:max_len]
        ]
        ids.append(encoded)

    batch = torch.zeros(len(ids), max_len, dtype=torch.long)
    mask = torch.zeros(len(ids), max_len, dtype=torch.bool)
    for row, encoded in enumerate(ids):
        length = min(len(encoded), max_len)
        batch[row, :length] = torch.tensor(encoded[:length])
        mask[row, :length] = True
    return batch, mask


def _texts_and_labels(
    examples: Sequence[tuple[str, Intent]],
) -> tuple[list[str], list[str]]:
    texts = [text for text, _ in examples]
    labels = [intent.value for _, intent in examples]
    return texts, labels
