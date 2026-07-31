"""Tests for the Phase 6 transformer ARCHITECTURE (Milestone 1).

These tests pin down the mathematical guarantees a transformer must
provide, independent of training quality:
  * output shapes at every stage,
  * padding masking (pad positions must not influence attention),
  * order sensitivity (unlike bag-of-words, sequence order matters),
  * positional encoding shape/additivity.
"""

import unittest

import torch

from app.transformer.architecture import (
    IntentTransformer,
    MultiHeadAttention,
    PositionalEncoding,
)


def make_model() -> IntentTransformer:
    model = IntentTransformer(
        vocab_size=20,
        embed_dim=16,
        n_heads=2,
        n_blocks=1,
        ff_dim=32,
        n_classes=5,
        max_len=8,
        dropout=0.0,  # determinism for the equality tests below
    )
    model.eval()
    return model


class TestPositionalEncoding(unittest.TestCase):
    def test_encoding_shape_matches_sequence(self):
        encoding = PositionalEncoding(embed_dim=16, max_len=8)
        x = torch.zeros(2, 6, 16)
        out = encoding(x)
        self.assertEqual(out.shape, (2, 6, 16))

    def test_encoding_is_additive_and_nonzero(self):
        encoding = PositionalEncoding(embed_dim=16, max_len=8)
        x = torch.zeros(1, 4, 16)
        out = encoding(x)
        self.assertFalse(torch.allclose(out, x))  # positions were added
        self.assertNotEqual(out[0, 0].tolist(), out[0, 1].tolist())  # ...and differ


class TestMultiHeadAttention(unittest.TestCase):
    def test_output_shape(self):
        attention = MultiHeadAttention(embed_dim=16, n_heads=2)
        x = torch.randn(2, 6, 16)
        out = attention(x)
        self.assertEqual(out.shape, (2, 6, 16))

    def test_multi_head_asserts_divisibility(self):
        with self.assertRaises(AssertionError):
            MultiHeadAttention(embed_dim=16, n_heads=3)


class TestIntentTransformer(unittest.TestCase):
    def test_forward_returns_logits(self):
        model = make_model()
        tokens = torch.tensor([[1, 2, 3, 4]])
        mask = torch.tensor([[True, True, True, True]])
        out = model(tokens, mask)
        self.assertEqual(out.shape, (1, 5))

    def test_batched_forward(self):
        model = make_model()
        tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
        mask = torch.tensor([[True, True, True], [True, True, False]])
        out = model(tokens, mask)
        self.assertEqual(out.shape, (2, 5))

    def test_padding_is_excluded_from_attention(self):
        """Position 0 must attend to positions {0,1} only in BOTH
        sequences; position 2 is masked in both. The CONTENT of a masked
        position must therefore be invisible: change token 2 and the
        output must not change. This is the exact property masking
        guarantees."""
        model = make_model()
        seq_a = torch.tensor([[1, 2, 9]])  # position 2 content differs...
        seq_b = torch.tensor([[1, 2, 10]])  # ...but is masked in both
        mask = torch.tensor([[True, True, False]])
        with torch.no_grad():
            hidden_a = model.blocks[0](model.pos(model.token_embed(seq_a)), mask)
            hidden_b = model.blocks[0](model.pos(model.token_embed(seq_b)), mask)
        # The masked position's own output legitimately differs (it
        # processes its own token); the invariant is that it is
        # INVISIBLE to every other position - 0 and 1 must match.
        self.assertTrue(torch.allclose(hidden_a[:, :2], hidden_b[:, :2], atol=1e-5))

    def test_model_is_order_sensitive(self):
        """The same words in different order must give different logits
        (the thing bag-of-words models cannot do)."""
        model = make_model()
        with torch.no_grad():
            out_a = model(torch.tensor([[1, 2]]), torch.tensor([[True, True]]))
            out_b = model(torch.tensor([[2, 1]]), torch.tensor([[True, True]]))
        self.assertFalse(torch.allclose(out_a, out_b))

    def test_unknown_tokens_do_not_crash(self):
        model = make_model()
        tokens = torch.tensor([[0, 0, 0]])  # all padding (degenerate input)
        mask = torch.tensor([[False, False, False]])
        out = model(tokens, mask)
        self.assertEqual(out.shape, (1, 5))  # clamp keeps the pool finite


if __name__ == "__main__":
    unittest.main()
