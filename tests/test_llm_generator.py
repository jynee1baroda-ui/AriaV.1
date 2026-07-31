"""Tests for LLMGenerator (Phase 7 + Phase 8 Prompt Engineering).

The generator's contract has two halves:
  * When an LLM IS available it returns a clean answer.
  * When it ISN'T (disabled, engine crash, model missing) it returns
    None so the Assistant falls back to rules. None must be the ONLY
    failure mode - chat must never crash because an LLM is missing.

A stub engine (duck-typed .generate(messages)) makes every behavior
deterministic; the real transformers model is deliberately never
touched here (it is exercised manually in the phase demo). The offline
unavailable path is tested with local_files_only=True, which never
touches the network.

Phase 8: generate() now speaks chat MESSAGES (role/content pairs
produced by build_messages) rather than a raw prompt string. The
qa_prompt/context_prompt builders are retained for the base-model path
and covered below.
"""

import unittest

from app.llm.generator import (
    LLMGenerator,
    build_messages,
    context_prompt,
    load_system_prompt,
    qa_prompt,
)

MSG = [{"role": "user", "content": "q"}]


class StubEngine:
    """Deterministic fake LLM: returns a fixed completion."""

    def __init__(self, text: str = "") -> None:
        self._text = text
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return self._text


class FailingEngine:
    """A broken engine: proves a crash degrades, not explodes."""

    def generate(self, messages: list[dict[str, str]]) -> str:
        raise RuntimeError("gpu melted")


class TestSystemPrompt(unittest.TestCase):
    """Phase 8: Aria's personality document is loaded from disk."""

    def test_system_prompt_loads_and_is_nonempty(self):
        prompt = load_system_prompt()
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 100)


class TestQaPrompt(unittest.TestCase):
    def test_formats_question_answer_pair(self):
        self.assertEqual(qa_prompt("What is 2+2?"), "Q: What is 2+2?\nA:")

    def test_keeps_original_casing(self):
        self.assertEqual(qa_prompt("What is the Capital of France?"),
                         "Q: What is the Capital of France?\nA:")


class TestContextPrompt(unittest.TestCase):
    """Phase 7 M2 (base-model path): history written INTO the prompt."""

    def test_empty_history_degrades_to_qa_prompt(self):
        """A two-line chat is identical to a one-shot question."""
        self.assertEqual(context_prompt([], "hi"), qa_prompt("hi"))

    def test_single_exchange_precedes_the_question(self):
        prompt = context_prompt([("what is the capital of france", "Paris.")], "what about london")
        self.assertEqual(
            prompt,
            "Q: what is the capital of france\nA: Paris.\n\nQ: what about london\nA:",
        )

    def test_multiple_exchanges_oldest_first(self):
        prompt = context_prompt(
            [("first", "one"), ("second", "two")],
            "third",
        )
        self.assertEqual(
            prompt,
            "Q: first\nA: one\n\nQ: second\nA: two\n\nQ: third\nA:",
        )

    def test_question_is_always_the_last_unfinished_pair(self):
        prompt = context_prompt([("a", "b")], "new question")
        self.assertTrue(prompt.endswith("Q: new question\nA:"))


class TestBuildMessages(unittest.TestCase):
    """Phase 8: structured chat messages = system + history + question."""

    def test_question_only_has_system_and_user(self):
        messages = build_messages("be nice", [], "hi")
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "be nice"},
                {"role": "user", "content": "hi"},
            ],
        )

    def test_history_alternates_user_assistant(self):
        messages = build_messages(
            "be nice",
            [("what is 2+2", "4"), ("thanks", "you're welcome")],
            "another",
        )
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "be nice"},
                {"role": "user", "content": "what is 2+2"},
                {"role": "assistant", "content": "4"},
                {"role": "user", "content": "thanks"},
                {"role": "assistant", "content": "you're welcome"},
                {"role": "user", "content": "another"},
            ],
        )


class TestGenerateWithEngine(unittest.TestCase):
    def test_delegates_messages_to_engine(self):
        engine = StubEngine(" 4")
        generator = LLMGenerator(engine=engine)
        self.assertEqual(generator.generate(MSG), "4")
        self.assertEqual(engine.calls, [MSG])

    def test_trims_prompt_reecho(self):
        """Small models answer then re-echo 'Q: ...' - cut it off."""
        engine = StubEngine(" 4\n\nQ: What is 2+2?\nA:")
        generator = LLMGenerator(engine=engine)
        self.assertEqual(generator.generate(MSG), "4")

    def test_trims_triple_newline_tail(self):
        engine = StubEngine("Paris.\n\n\nQ: What is the capital of France?\nA:")
        generator = LLMGenerator(engine=engine)
        self.assertEqual(generator.generate(MSG), "Paris.")

    def test_strips_whitespace_only(self):
        engine = StubEngine("  Paris.  ")
        generator = LLMGenerator(engine=engine)
        self.assertEqual(generator.generate(MSG), "Paris.")

    def test_multiline_answer_is_kept(self):
        engine = StubEngine("Because\nit is a question.\n\nQ:")
        generator = LLMGenerator(engine=engine)
        self.assertEqual(generator.generate(MSG), "Because\nit is a question.")


class TestGracefulDegradation(unittest.TestCase):
    def test_disabled_never_consults_engine(self):
        engine = StubEngine("should not be used")
        generator = LLMGenerator(engine=engine, enabled=False)
        self.assertIsNone(generator.generate(MSG))
        self.assertEqual(engine.calls, [])

    def test_engine_crash_returns_none(self):
        generator = LLMGenerator(engine=FailingEngine())
        self.assertIsNone(generator.generate(MSG))

    def test_unknown_model_offline_returns_none(self):
        """local_files_only=True never hits the network; a model not in
        the cache must degrade to None, not hang or crash."""
        generator = LLMGenerator(model_id="nope/not-a-real-model", local_files_only=True)
        self.assertIsNone(generator.generate(MSG))
        # And it stays degraded - no repeated load attempts.
        self.assertIsNone(generator.generate(MSG))

    def test_empty_engine_answer_is_none(self):
        """An empty string is NOT an answer - the caller echoes."""
        generator = LLMGenerator(engine=StubEngine("   "))
        self.assertIsNone(generator.generate(MSG))


if __name__ == "__main__":
    unittest.main()
