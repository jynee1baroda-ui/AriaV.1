"""app.llm - Phase 7 (Large Language Models), Milestone 1.

This is the payoff for the whole Phase 4-6 arc: an ACTUAL pre-trained
language model, loaded locally, generating real answers.

How an LLM "answers" a question (the one idea that unlocks everything):
it is a next-token predictor. Given the text so far, it computes a
probability distribution over the next token, samples one, appends it,
and repeats. "What is 2+2?" becomes a *completion* problem: the model
has read enough text during pre-training that it continues "Q: What is
2+2?\nA:" with " 4". It never "looks up" anything - it predicts what a
plausible continuation looks like. That is both why it works and why it
hallucinates (confidently predicting something that reads well but is
false).

The generator wraps a Hugging Face causal LM (SmolLM2-135M by default -
small enough to run on a laptop CPU, modern enough to be useful). All
the Phase 6 machinery is inside it: tokenizer, embeddings, positional
encodings, multi-head self-attention, decoder blocks. We built a tiny
version by hand in Phase 6 so this library call reads as a THOUSAND
Phase 6s, not magic.
"""

from __future__ import annotations

import logging
import threading

from app import config

logger = logging.getLogger("assistant.llm")


def qa_prompt(question: str) -> str:
    """The minimal zero-shot prompt: 'Q: ...\\nA:'.

    A base (non-instruct) model completes this the way its training
    corpus did - as a question/answer pair. This template is the seed
    of Phase 8 (Prompt Engineering); it will grow a system prompt and
    conversation history there.
    """
    return f"Q: {question}\nA:"


class LLMGenerator:
    """Generate answers with a pre-trained causal language model.

    Hard requirements, learned the hard way in Phases 4-6:
      * LAZY LOAD - loading a model takes seconds and megabytes; it
        must never happen at import or server startup. First use wins.
      * GRACEFUL DEGRADATION - no model, no network, a corrupt cache:
        generate() returns None and the caller (the Assistant) falls
        back to rules. Chat must never crash because an LLM is missing.
      * THREAD SAFETY - FastAPI runs handlers in a threadpool; loading
        and generation are guarded by a lock.

    Attributes:
        _engine: Injectable generator for tests (duck-typed with
            .generate(prompt) -> str), or None to use the real model.
        _model_id: Hugging Face model identifier.
        _max_new_tokens: Token budget per answer.
        _temperature: Sampling temperature (0 = greedy).
    """

    def __init__(
        self,
        model_id: str = config.LLM_MODEL_ID,
        max_new_tokens: int = config.LLM_MAX_NEW_TOKENS,
        temperature: float = config.LLM_TEMPERATURE,
        enabled: bool = config.LLM_ENABLED,
        local_files_only: bool = config.LLM_LOCAL_FILES_ONLY,
        engine: object | None = None,
    ) -> None:
        self._model_id = model_id
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._enabled = enabled
        self._local_files_only = local_files_only
        self._engine = engine
        self._tokenizer: object | None = None
        self._model: object | None = None
        self._loaded = engine is not None
        self._unavailable = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self, prompt: str) -> str | None:
        """Generate a completion for the prompt, or None if unavailable.

        Args:
            prompt: The full text to continue (see qa_prompt).

        Returns:
            The generated continuation (whitespace-trimmed, prompt
            re-echo removed), or None when the LLM cannot run.
        """
        if not self._enabled:
            return None
        if not self._ensure_loaded():
            return None
        try:
            if self._engine is not None:
                answer = self._engine.generate(prompt)
            else:
                answer = self._generate_real(prompt)
            cleaned = self._clean(answer)
            # An empty completion is NOT an answer - the caller must
            # fall back to rules. None is the only failure signal.
            return cleaned or None
        except Exception:
            logger.exception("LLM generation failed - falling back to rules.")
            return None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> bool:
        """Load the model once (thread-safe); False if unavailable."""
        if self._loaded or self._unavailable:
            return self._loaded
        with self._lock:
            if self._loaded or self._unavailable:
                return self._loaded
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(
                    self._model_id, local_files_only=self._local_files_only
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    self._model_id, local_files_only=self._local_files_only
                )
                self._model.eval()
                self._loaded = True
                logger.info("Loaded LLM %s", self._model_id)
            except Exception:
                logger.exception("LLM %s unavailable - falling back to rules.", self._model_id)
                self._unavailable = True
            return self._loaded

    # ------------------------------------------------------------------
    # Generation internals
    # ------------------------------------------------------------------
    def _generate_real(self, prompt: str) -> str:
        """Run the real model: tokenize, generate, decode the new tokens."""
        inputs = self._tokenizer(prompt, return_tensors="pt")  # type: ignore[union-attr]
        generated = self._model.generate(  # type: ignore[union-attr]
            **inputs,
            max_new_tokens=self._max_new_tokens,
            do_sample=True,
            temperature=self._temperature,
            top_p=0.9,
            pad_token_id=self._tokenizer.eos_token_id,  # type: ignore[union-attr]
        )
        new_tokens = generated[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)  # type: ignore[union-attr]

    @staticmethod
    def _clean(text: str) -> str:
        """Trim the answer to its useful part.

        A 135M base model has two run-in-circles failure modes:
          * it answers then re-echoes the prompt (" 4\\n\\nQ: ...\\nA:")
          * it answers then drifts into MCQ noise ("4\\nB: 2+2\\nC: ...")
        Both are the SAME failure - the model continuing past the point
        where it actually answered. Cut at the first tell and keep the
        real answer.
        """
        for marker in ("\nB:", "\nQ:", "\nA:", "\n\n\n"):
            index = text.find(marker)
            if index != -1:
                text = text[:index]
                break
        return text.strip()
