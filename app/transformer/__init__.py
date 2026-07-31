"""app.transformer - Phase 6 (Transformers), Milestone 1.

Implements the transformer architecture from scratch - the model family
that powers Phase 7's large language models. "Attention is All You
Need" (Vaswani et al., 2017) replaced recurrence with attention; every
modern LLM is a stack of transformer blocks.

Milestone 1 ships the ARCHITECTURE and a classifier trained on our
intent task (same data, same evaluation as Phases 4-5). Milestone 2
plugs it into the router as a third model kind.

The honest framing (identical to Phase 5): a from-scratch transformer
trained on 149 examples cannot beat a well-tuned linear model. Its
power comes from PRE-TRAINING - a transformer pretrained on billions of
tokens already knows language, and Phase 7 will show what that unlocks.
But without understanding attention you cannot understand transformers,
and without transformers you cannot understand Phase 7.
"""
