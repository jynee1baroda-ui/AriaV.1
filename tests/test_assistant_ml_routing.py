"""Tests for ML routing inside the Assistant (Phase 4, Milestone 2).

M2 integration: when a router supplies a CONFIDENT intent, respond()
uses it to pick the branch; when ML declines or cannot fulfill, the
deterministic rules answer. A stub classifier keeps these tests
deterministic and independent of the trained artifact.

The rules-only behavior (router=None) is covered by test_main.py and
must stay green - it proves the refactor did not change default
answers.
"""

import unittest

from app.assistant import Assistant
from app.ml.dataset import Intent
from app.models import Conversation, Memory
from app.services.intent_router import IntentRouter


class StubClassifier:
    """Deterministic fake: always predicts the configured result."""

    def __init__(self, intent: Intent, confidence: float) -> None:
        self._intent = intent
        self._confidence = confidence

    def predict(self, text: str) -> tuple[Intent, float]:
        return self._intent, self._confidence


def make_assistant(router: IntentRouter, **kwargs) -> Assistant:
    return Assistant(
        name="Aria",
        conversation=Conversation(),
        router=router,
        factual_only=True,  # this suite tests factual-mode routing
        **kwargs,
    )


class TestAssistantMlRouting(unittest.TestCase):
    def test_greeting_intent_routes_to_greeting(self):
        router = IntentRouter(classifier=StubClassifier(Intent.GREETING, 0.99))
        assistant = make_assistant(router)
        self.assertEqual(
            assistant.respond("hey friend"),
            "Hello! I'm Aria. How can I help?",
        )

    def test_identity_intent_routes_to_identity(self):
        router = IntentRouter(classifier=StubClassifier(Intent.ASSISTANT_IDENTITY, 0.99))
        assistant = make_assistant(router)
        self.assertEqual(assistant.respond("who are you"), "My name is Aria.")

    def test_teach_intent_still_requires_extraction(self):
        """ML picks the branch, but the extractor confirms the fact."""
        router = IntentRouter(classifier=StubClassifier(Intent.TEACH_FACT, 0.99))
        assistant = make_assistant(router)
        self.assertIn("Alice", assistant.respond("my name is Alice"))

    def test_teach_intent_without_extractable_fact_falls_back_to_rules(self):
        """A confident TEACH label is not enough: no fact, no answer -
        the deterministic chain (refusal in factual mode) takes over."""
        router = IntentRouter(classifier=StubClassifier(Intent.TEACH_FACT, 0.99))
        assistant = make_assistant(router)
        self.assertEqual(
            assistant.respond("just so you know, i have a cat"),
            "I don't have a verified answer for that.",
        )

    def test_recall_intent_answers_from_memory(self):
        router = IntentRouter(classifier=StubClassifier(Intent.RECALL_QUESTION, 0.99))
        assistant = make_assistant(
            router,
            memories=[Memory(id=1, content="the user's name is Alice")],
        )
        self.assertEqual(assistant.respond("do you remember my name"), "Your name is Alice.")

    def test_recall_intent_without_rule_match_falls_back_to_rules(self):
        """The question matcher is authoritative for WHICH memory to
        fetch; a confident RECALL label for an unmatched phrasing does
        not invent an answer - it is refused."""
        router = IntentRouter(classifier=StubClassifier(Intent.RECALL_QUESTION, 0.99))
        assistant = make_assistant(
            router,
            memories=[Memory(id=1, content="the user's name is Alice")],
        )
        self.assertEqual(
            assistant.respond("what was my name"),
            "I don't have a verified answer for that.",
        )

    def test_other_intent_is_refused_in_factual_mode(self):
        router = IntentRouter(classifier=StubClassifier(Intent.OTHER, 0.99))
        assistant = make_assistant(router)
        self.assertEqual(
            assistant.respond("pancakes"),
            "I don't have a verified answer for that.",
        )

    def test_low_confidence_falls_back_to_rules(self):
        """A weak ML signal must not override the rules - here the
        greeting rules still answer even though ML said 20%."""
        router = IntentRouter(classifier=StubClassifier(Intent.GREETING, 0.20))
        assistant = make_assistant(router)
        self.assertEqual(
            assistant.respond("hello"),
            "Hello! I'm Aria. How can I help?",
        )

    def test_no_router_uses_rules_only(self):
        assistant = Assistant(
            name="Aria",
            conversation=Conversation(),
            factual_only=True,
        )
        self.assertEqual(assistant.respond("hello"), "Hello! I'm Aria. How can I help?")
        self.assertEqual(
            assistant.respond("pancakes"),
            "I don't have a verified answer for that.",
        )

    def test_exchange_is_still_recorded(self):
        router = IntentRouter(classifier=StubClassifier(Intent.GREETING, 0.99))
        conversation = Conversation()
        assistant = Assistant(name="Aria", conversation=conversation, router=router)
        assistant.respond("hey")
        self.assertEqual(len(conversation.messages), 2)


if __name__ == "__main__":
    unittest.main()
