"""Assistant + LLM integration tests (Phase 7, Milestone 1).

The LLM is the LAST resort. The priority order must never change:
rules (greeting/identity) > taught facts > long-term memory > LLM >
echo. These tests prove the LLM only speaks when nothing else can,
and that its failure degrades to the echo instead of a crash.
"""

import unittest

from app.assistant import Assistant
from app.llm.generator import LLMGenerator, qa_prompt
from app.ml.dataset import Intent
from app.models import Conversation, Memory
from app.services.intent_router import IntentRouter


class RecordingEngine:
    """Stub LLM that records prompts and returns a fixed answer."""

    def __init__(self, text: str = "42") -> None:
        self._text = text
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
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


class TestAssistantWithLlm(unittest.TestCase):
    def test_rules_still_win_without_consulting_llm(self):
        generator, engine = llm_generator()
        assistant = Assistant(name="Aria", conversation=Conversation(), llm=generator)
        self.assertEqual(assistant.respond("hello"), "Hello! I'm Aria. How can I help?")
        self.assertEqual(assistant.respond("what is your name"), "My name is Aria.")
        self.assertEqual(engine.prompts, [])

    def test_taught_fact_still_wins(self):
        generator, engine = llm_generator()
        assistant = Assistant(name="Aria", conversation=Conversation(), llm=generator)
        reply = assistant.respond("my name is Alice")
        self.assertIn("Alice", reply)
        self.assertEqual(engine.prompts, [])

    def test_memory_answer_still_wins(self):
        generator, engine = llm_generator()
        assistant = Assistant(
            name="Aria",
            conversation=Conversation(),
            llm=generator,
            memories=[Memory(id=1, content="the user's name is Alice")],
        )
        self.assertEqual(assistant.respond("what is my name"), "Your name is Alice.")
        self.assertEqual(engine.prompts, [])

    def test_unanswered_question_goes_to_llm(self):
        generator, engine = llm_generator(" 4\n\nQ: What is 2+2?\nA:")
        assistant = Assistant(name="Aria", conversation=Conversation(), llm=generator)
        self.assertEqual(assistant.respond("what is 2+2"), "4")
        self.assertEqual(engine.prompts, [qa_prompt("what is 2+2")])

    def test_other_intent_goes_to_llm(self):
        generator, engine = llm_generator("Paris.")
        router = IntentRouter(classifier=StubClassifier(Intent.OTHER, 0.99))
        assistant = Assistant(name="Aria", conversation=Conversation(), router=router, llm=generator)
        self.assertEqual(
            assistant.respond("what is the capital of france"),
            "Paris.",
        )

    def test_llm_parroting_question_is_echo(self):
        """The model repeating the question verbatim is NOT an answer."""
        generator, engine = llm_generator("who wrote hamlet")
        assistant = Assistant(name="Aria", conversation=Conversation(), llm=generator)
        self.assertEqual(assistant.respond("who wrote hamlet"), "You said: who wrote hamlet")

    def test_llm_none_falls_back_to_echo(self):
        generator, engine = llm_generator("   ")
        assistant = Assistant(name="Aria", conversation=Conversation(), llm=generator)
        self.assertEqual(assistant.respond("pancakes"), "You said: pancakes")

    def test_no_llm_echoes(self):
        assistant = Assistant(name="Aria", conversation=Conversation())
        self.assertEqual(assistant.respond("pancakes"), "You said: pancakes")


if __name__ == "__main__":
    unittest.main()
