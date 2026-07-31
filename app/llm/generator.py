"""app.llm - Phase 7 (Large Language Models) + Phase 8 (Prompt Engineering).

This is the payoff for the whole Phase 4-6 arc: an ACTUAL pre-trained
language model, loaded locally, generating real answers.

How an LLM "answers" a question (the one idea that unlocks everything):
it is a next-token predictor. Given the text so far, it computes a
probability distribution over the next token, samples one, appends it,
and repeats. It never "looks up" anything - it predicts what a
plausible continuation looks like. That is both why it works and why it
hallucinates (confidently predicting something that reads well but is
false).

The generator wraps a Hugging Face causal LM (SmolLM2-135M-Instruct by
default - the INSTRUCT variant, which follows a system/user/assistant
chat template). All the Phase 6 machinery is inside it: tokenizer,
embeddings, positional encodings, multi-head self-attention, decoder
blocks. We built a tiny version by hand in Phase 6 so this library call
reads as a THOUSAND Phase 6s, not magic.

Phase 8 (Prompt Engineering): the model's behavior is steered by Aria's
SYSTEM PROMPT (app/prompts/aria_system.md) plus recent conversation
history, rendered through the tokenizer's chat template.
"""

from __future__ import annotations

import logging
import threading

from app import config

logger = logging.getLogger("assistant.llm")


def load_system_prompt() -> str:
    """Read Aria's system prompt from disk (config.SYSTEM_PROMPT_PATH).

    The prompt is a versioned document - editing it edits Aria's
    personality. Cached at module level so the file is read once.
    """
    if _SYSTEM_PROMPT_CACHE is not None:
        return _SYSTEM_PROMPT_CACHE
    return _read_system_prompt()


def _read_system_prompt() -> str:
    global _SYSTEM_PROMPT_CACHE
    _SYSTEM_PROMPT_CACHE = config.SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    return _SYSTEM_PROMPT_CACHE


_SYSTEM_PROMPT_CACHE: str | None = None


def build_messages(
    system_prompt: str,
    exchanges: list[tuple[str, str]],
    question: str,
) -> list[dict[str, str]]:
    """Assemble the chat-template message list.

    Phase 8: the system prompt sets Aria's behavior, the exchange pairs
    carry the conversation context (Phase 7 M2), and the current message
    is the turn she must answer.

    Args:
        system_prompt: Aria's core-behavior document.
        exchanges: Previous (user_message, assistant_reply) pairs,
            oldest first.
        question: The new user message.

    Returns:
        A list of {"role": ..., "content": ...} dicts suitable for
        apply_chat_template: system, alternating user/assistant, user.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for user, assistant in exchanges:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": question})
    return messages


def qa_prompt(question: str) -> str:
    """The Phase 7 M1 prompt format: 'Q: ...\\nA:'.

    Retained as the non-chat format used with BASE (non-instruct) models
    and by the Phase 7 tests. The Phase 8 default path uses chat
    messages (build_messages) instead.
    """
    return f"Q: {question}\nA:"


def context_prompt(exchanges: list[tuple[str, str]], question: str) -> str:
    """The Phase 7 M2 prompt format: history + question as Q/A pairs.

    Retained for the base-model path and Phase 7 tests. Phase 8 uses
    structured chat messages (build_messages).
    """
    blocks = [f"Q: {user}\nA: {assistant}" for user, assistant in exchanges]
    blocks.append(f"Q: {question}\nA:")
    return "\n\n".join(blocks)


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
            .generate(messages) -> str), or None to use the real model.
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
    def generate(self, messages: list[dict[str, str]]) -> str | None:
        """Generate Aria's reply for a chat message list, or None.

        Args:
            messages: Chat-template roles/content pairs - typically the
                output of build_messages() (system + history + question).

        Returns:
            The generated reply (whitespace-trimmed, trailing special
            tokens stripped), or None when the LLM cannot run.
        """
        if not self._enabled:
            return None
        if not self._ensure_loaded():
            return None
        try:
            if self._engine is not None:
                answer = self._engine.generate(messages)
            else:
                answer = self._generate_real(messages)
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
    def _generate_real(self, messages: list[dict[str, str]]) -> str:
        """Run the real model: template, tokenize, generate, decode.

        Phase 8: the message list is rendered through the tokenizer's
        apply_chat_template so the Instruct model sees its native
        <|user|>/<|assistant|> format - this is what makes system-prompt
        following possible at all. Generation adds only the assistant
        turn (turn_completion=True), then that new chunk is decoded.

        Default is GREEDY decoding (do_sample=False): pick the single
        most likely token each step. Deterministic - the same question
        gets the same answer every time, and a 135M model is too small
        to be worth sampling from. Setting temperature > 0 opts into
        sampling, which trades reproducibility for variety.
        """
        prompt = self._tokenizer.apply_chat_template(  # type: ignore[union-attr]
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt")  # type: ignore[union-attr]
        generate_kwargs: dict = dict(
            max_new_tokens=self._max_new_tokens,
            do_sample=False,
            pad_token_id=self._tokenizer.eos_token_id,  # type: ignore[union-attr]
        )
        if self._temperature > 0:
            generate_kwargs.update(
                do_sample=True,
                temperature=self._temperature,
                top_p=0.9,
            )
        generated = self._model.generate(**inputs, **generate_kwargs)  # type: ignore[union-attr]
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
