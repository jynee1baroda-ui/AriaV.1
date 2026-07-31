"""Assistant + LLM integration tests (Phase 7 M1+M2 + Phase 8 prompts).

The answer priority must never change:
rules (greeting/identity) > taught facts > long-term memory > verified
knowledge base > [factual mode: REFUSE | open mode: LLM > echo].

Phase 8: the LLM is now sent chat MESSAGES (system prompt + history +
question, via build_messages/load_system_prompt) instead of a raw Q/A
string; expected_messages() encodes that contract.

Factual mode (the default) is the "never fake or unverified" guarantee:
a 135M generative model hallucinates, so unknown questions are refused,
never generated or echoed. The LLM/echo tests below pass
factual_only=False explicitly because they exercise the OPEN mode.
"""

import unittest

from app.assistant import Assistant
from app.kb.knowledge import KnowledgeBase
from app.llm.generator import (
    LLMGenerator,
    build_messages,
    load_system_prompt,
)
from app.ml.dataset import Intent
from app.models import Conversation, Memory
from app.services.intent_router import IntentRouter


class RecordingEngine:
    """Stub LLM that records messages and returns a fixed answer."""

    def __init__(self, text: str = "42") -> None:
        self._text = text
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return self._text


class StubClassifier:
    def __init__(self, intent: Intent, confidence: float) -> None:
        self._intent = intent
        self._confidence = confidence

    def predict(self, text: str) -> tuple[Intent, float]:
        return self._intent, self._confidence


def llm_generator(text: str = "42") -> tuple[LLMGenerator, RecordingEngine]:
    engine = RecordingEngine(text)
    return LLMGenerator(engine=engine), engine


def expected_messages(
    exchanges: list[tuple[str, str]], question: str
) -> list[dict[str, str]]:
    """The exact Phase 8 message list the Assistant must send."""
    return build_messages(load_system_prompt(), exchanges, question)


REFUSAL = "I don't have a verified answer for that."


class TestFactualMode(unittest.TestCase):
    """The 'never fake or unverified' contract (Phase 7 hardening)."""

    def test_verified_fact_answered_without_llm(self):
        """The knowledge base answers; the LLM must NOT be consulted."""
        generator, engine = llm_generator("France is the capital of France.")
        assistant = Assistant(
            name="Aria",
            conversation=Conversation(),
            llm=generator,
            knowledge=KnowledgeBase(),
        )
        self.assertEqual(assistant.respond("what is the capital of France?"), "Paris.")
        self.assertEqual(engine.calls, [])

    def test_unknown_question_refused_even_with_llm_wired(self):
        """A wired LLM is NOT allowed to answer in factual mode."""
        generator, engine = llm_generator("definitely a made-up answer")
        assistant = Assistant(
            name="Aria",
            conversation=Conversation(),
            llm=generator,
            knowledge=KnowledgeBase(),
            factual_only=True,
        )
        self.assertEqual(assistant.respond("tell me a joke"), REFUSAL)
        self.assertEqual(engine.calls, [])

    def test_unknown_refused_without_knowledge_base(self):
        generator, engine = llm_generator("made-up")
        assistant = Assistant(
            name="Aria",
            conversation=Conversation(),
            llm=generator,
            factual_only=True,
        )
        self.assertEqual(assistant.respond("pancakes"), REFUSAL)
        self.assertEqual(engine.calls, [])

    def test_math_comes_from_knowledge_base(self):
        assistant = Assistant(
            name="Aria",
            conversation=Conversation(),
            knowledge=KnowledgeBase(),
        )
        self.assertEqual(assistant.respond("what is 2+2?"), "4.")

    def test_math_spacing_is_normalized(self):
        assistant = Assistant(
            name="Aria",
            conversation=Conversation(),
            knowledge=KnowledgeBase(),
        )
        self.assertEqual(assistant.respond("what is 2 + 2?"), "4.")

    def test_memory_still_wins_over_knowledge(self):
        """A taught fact beats a knowledge-base fact if both matched."""
        assistant = Assistant(
            name="Aria",
            conversation=Conversation(),
            knowledge=KnowledgeBase(),
            memories=[Memory(id=1, content="the user's name is Paris")],
        )
        self.assertEqual(assistant.respond("what is my name"), "Your name is Paris.")


class TestOpenMode(unittest.TestCase):
    """Open mode (factual_only=False): the LLM/echo behavior from M1."""

    def test_rules_still_win_without_consulting_llm(self):
        generator, engine = llm_generator()
        assistant = Assistant(
            name="Aria", conversation=Conversation(), llm=generator, factual_only=False
        )
        self.assertEqual(assistant.respond("hello"), "Hello! I'm Aria. How can I help?")
        self.assertEqual(assistant.respond("what is your name"), "My name is Aria.")
        self.assertEqual(engine.calls, [])

    def test_taught_fact_still_wins(self):
        generator, engine = llm_generator()
        assistant = Assistant(
            name="Aria", conversation=Conversation(), llm=generator, factual_only=False
        )
        reply = assistant.respond("my name is Alice")
        self.assertIn("Alice", reply)
        self.assertEqual(engine.calls, [])

    def test_memory_answer_still_wins(self):
        generator, engine = llm_generator()
        assistant = Assistant(
            name="Aria",
            conversation=Conversation(),
            llm=generator,
            factual_only=False,
            memories=[Memory(id=1, content="the user's name is Alice")],
        )
        self.assertEqual(assistant.respond("what is my name"), "Your name is Alice.")
        self.assertEqual(engine.calls, [])

    def test_unanswered_question_goes_to_llm(self):
        generator, engine = llm_generator(" 4\n\nQ: What is 2+2?\nA:")
        assistant = Assistant(
            name="Aria", conversation=Conversation(), llm=generator, factual_only=False
        )
        self.assertEqual(assistant.respond("what is 2+2"), "4")
        self.assertEqual(engine.calls, [expected_messages([], "what is 2+2")])

    def test_other_intent_goes_to_llm(self):
        generator, engine = llm_generator("Paris.")
        router = IntentRouter(classifier=StubClassifier(Intent.OTHER, 0.99))
        assistant = Assistant(
            name="Aria",
            conversation=Conversation(),
            router=router,
            llm=generator,
            factual_only=False,
        )
        self.assertEqual(
            assistant.respond("what is the capital of france"),
            "Paris.",
        )

    def test_llm_parroting_question_is_echo(self):
        """The model repeating the question verbatim is NOT an answer."""
        generator, engine = llm_generator("who wrote hamlet")
        assistant = Assistant(
            name="Aria", conversation=Conversation(), llm=generator, factual_only=False
        )
        self.assertEqual(assistant.respond("who wrote hamlet"), "You said: who wrote hamlet")

    def test_llm_none_falls_back_to_echo(self):
        generator, engine = llm_generator("   ")
        assistant = Assistant(
            name="Aria", conversation=Conversation(), llm=generator, factual_only=False
        )
        self.assertEqual(assistant.respond("pancakes"), "You said: pancakes")

    def test_no_llm_echoes(self):
        assistant = Assistant(name="Aria", conversation=Conversation(), factual_only=False)
        self.assertEqual(assistant.respond("pancakes"), "You said: pancakes")


class TestAssistantContextWindow(unittest.TestCase):
    """Phase 7 M2: past exchanges are written into the LLM prompt."""

    def test_follow_up_question_includes_prior_exchange(self):
        conversation = Conversation()
        conversation.add("user", "what is the capital of france")
        conversation.add("assistant", "Paris.")
        generator, engine = llm_generator("London.")
        assistant = Assistant(
            name="Aria", conversation=conversation, llm=generator, factual_only=False
        )
        self.assertEqual(assistant.respond("what about london"), "London.")
        self.assertEqual(
            engine.calls,
            [
                expected_messages(
                    [("what is the capital of france", "Paris.")], "what about london"
                )
            ],
        )

    def test_window_only_keeps_last_few_exchanges(self):
        """The default 4-turn window must drop older exchanges."""
        conversation = Conversation()
        for user, reply in [("a", "1"), ("b", "2"), ("c", "3"), ("d", "4"), ("e", "5")]:
            conversation.add("user", user)
            conversation.add("assistant", reply)
        generator, engine = llm_generator("ok")
        assistant = Assistant(
            name="Aria", conversation=conversation, llm=generator, factual_only=False
        )
        assistant.respond("f")
        self.assertEqual(
            engine.calls,
            [
                expected_messages(
                    [("b", "2"), ("c", "3"), ("d", "4"), ("e", "5")], "f"
                )
            ],
        )

    def test_orphaned_messages_do_not_break_pairing(self):
        """Odd role sequences must not crash extraction or the LLM."""
        conversation = Conversation()
        conversation.add("user", "orphan")
        generator, engine = llm_generator("hi")
        assistant = Assistant(
            name="Aria", conversation=conversation, llm=generator, factual_only=False
        )
        assistant.respond("what time is it")
        self.assertEqual(engine.calls, [expected_messages([], "what time is it")])


if __name__ == "__main__":
    unittest.main()
