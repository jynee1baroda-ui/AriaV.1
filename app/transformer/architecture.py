"""The transformer architecture, built from first principles (Phase 6).

Every component of "Attention is All You Need" is here, written so each
piece can be read and tested alone:

  * PositionalEncoding  - inject word order (transformers have none).
  * MultiHeadAttention  - the core: soft attention over all positions.
  * TransformerBlock    - attention + feed-forward with residuals.
  * IntentTransformer   - the whole model: tokens -> intents.

Why attention (vs the Phase 5 bag-of-words MLP): the MLP sees words as
an unordered set (TF-IDF). A transformer processes the SEQUENCE - each
token gets a context-dependent representation built by looking at every
other token. That is what lets a model later handle "the bank by the
river" vs "the bank on Main Street", pronoun resolution, and the rest
of the word-order meaning that bag-of-words throws away.

The design choices here are the ones every modern transformer makes:
  * PRE-NORM blocks (LayerNorm before each sublayer) - the GPT-style
    arrangement, which is dramatically more stable to train than the
    original post-norm.
  * Padding masking - pad tokens must not leak into attention scores.
  * Masked mean pooling for the classifier head (no CLS token needed
    at this scale).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Add position information to token embeddings.

    A transformer is permutation-invariant: attention scores depend only
    on token content, not order. "my name is Alice" and "Alice is my
    name" would produce identical representations. Positional encoding
    breaks the tie by adding a vector that encodes WHERE each token is.

    Uses the sinusoidal scheme from the original paper: position p gets
    sin/cos waves of increasing frequency. Frequencies are chosen so the
    model can express relative offsets (attention can become a linear
    function of position difference), and no training is needed - the
    pattern is deterministic.

    Attributes:
        pe: Registered buffer (B, max_len, embed_dim); buffers are saved
            and moved with the model but have no gradient.
    """

    def __init__(self, embed_dim: int, max_len: int, dropout: float = 0.1) -> None:
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        # Even dimension indices use sin, odd use cos; frequency
        # decreases geometrically. embed_dim must be even for this
        # interleave to fill every dimension.
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional vectors to token embeddings."""
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    """The attention mechanism, run in parallel across multiple heads.

    The three players, per token:
      * QUERY  (q) - "what am I looking for?"
      * KEY    (k) - "what do I offer?"
      * VALUE  (v) - "what do I contribute when attended to?"

    Attention computes, for every pair of tokens (i, j), how compatible
    token i's query is with token j's key (a dot product, scaled down so
    large dimensions don't push softmax into saturation), turns those
    scores into a probability distribution with softmax, then takes the
    weighted average of the values. Token i's output is a blend of every
    token's value, weighted by relevance - the "context" of token i.

    Multi-head means running this H times with DIFFERENT learned
    projections. Different heads learn different relations ("head 1
    tracks what comes right after 'my name is', head 2 watches for
    questions...") - a single head can only specialize on one pattern,
    and the paper found many heads helps.

    Attributes:
        qkv: One linear projecting input -> query, key, value all at
            once (the efficient batched way).
        out_proj: Mixes the concatenated head outputs back to embed_dim.
    """

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must divide evenly into heads"
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Attend over a sequence.

        Args:
            x: (batch, seq_len, embed_dim) token representations.
            mask: (batch, seq_len) bool, True where the position is a
                real token, False where it is padding.

        Returns:
            (batch, seq_len, embed_dim) context-aware representations.
        """
        batch, seq_len, _ = x.shape
        # One projection, then split into q/k/v and into heads.
        q, k, v = self.qkv(x).reshape(batch, seq_len, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # Attention scores: (batch, heads, seq, seq). q @ k^T scaled by
        # 1/sqrt(head_dim) keeps the dot products' variance ~1.
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if mask is not None:
            # Pad positions (mask False) are never allowed to be
            # attended TO; -inf drives their softmax weight to 0.
            padded = mask.unsqueeze(1).unsqueeze(2).logical_not()
            scores = scores.masked_fill(padded, float("-inf"))
        weights = self.attn_dropout(torch.softmax(scores, dim=-1))
        # Weighted blend of values, then merge heads back together.
        context = (weights @ v).transpose(1, 2).reshape(batch, seq_len, self.embed_dim)
        return self.out_proj(context)


class TransformerBlock(nn.Module):
    """One transformer layer: attention + feed-forward, residual-wrapped.

    Residual connections (x + sublayer(x)) give the gradient a direct
    path through deep stacks - without them, deep networks struggle to
    learn. Pre-norm (LayerNorm BEFORE the sublayer, GPT style) keeps
    activations well-scaled, which is far more stable than the original
    paper's post-norm. The feed-forward is two linear layers per token -
    most of the model's parameters live here; attention is the cheap,
    communication-heavy part.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class IntentTransformer(nn.Module):
    """The complete model: token ids in, logits out.

    Layers: token embedding -> positional encoding -> transformer blocks
    -> layer norm -> mean pooling -> classification head.

    Attributes:
        token_embed: Learns a vector per vocabulary word.
        blocks: The transformer stack (a single block is already a real
            transformer; depth is where capacity scales with data).
        head: Softmax readout - same role as in Phases 4-5.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 32,
        n_heads: int = 2,
        n_blocks: int = 1,
        ff_dim: int = 64,
        n_classes: int = 5,
        max_len: int = 16,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos = PositionalEncoding(embed_dim, max_len, dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, n_heads, ff_dim, dropout) for _ in range(n_blocks)]
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(
        self, tokens: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Run token ids through the stack.

        Args:
            tokens: (batch, seq_len) long tensor of vocabulary indices.
            mask: (batch, seq_len) bool; None means all positions valid.

        Returns:
            (batch, n_classes) raw logits.
        """
        x = self.token_embed(tokens)
        x = self.pos(x)
        for block in self.blocks:
            x = block(x, mask)
        x = self.final_norm(x)
        if mask is None:
            pooled = x.mean(dim=1)
        else:
            # Mask-aware mean pooling: average only over real tokens so
            # padding does not dilute the representation.
            pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        return self.head(pooled)
