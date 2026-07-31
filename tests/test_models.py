"""Tests for the domain models (Message, Conversation)."""

import unittest

from app.errors import AssistantError, ConversationError, InvalidRoleError
from app.models import Conversation, Message


class TestErrorHierarchy(unittest.TestCase):
    """Custom exceptions must form the documented hierarchy."""

    def test_invalid_role_error_is_a_conversation_error(self) -> None:
        self.assertTrue(issubclass(InvalidRoleError, ConversationError))

    def test_conversation_error_is_an_assistant_error(self) -> None:
        self.assertTrue(issubclass(ConversationError, AssistantError))

    def test_hierarchy_is_never_a_base_exception(self) -> None:
        # Our errors must not swallow KeyboardInterrupt / SystemExit
        self.assertFalse(issubclass(AssistantError, BaseException) and not issubclass(AssistantError, Exception))


class TestMessage(unittest.TestCase):
    """Message must hold role/content and serialize to/from dicts."""

    def test_fields(self) -> None:
        message = Message(role="user", content="hi")
        self.assertEqual(message.role, "user")
        self.assertEqual(message.content, "hi")

    def test_to_dict(self) -> None:
        self.assertEqual(
            Message("assistant", "Hello!").to_dict(),
            {"role": "assistant", "message": "Hello!"},
        )

    def test_from_dict_round_trip(self) -> None:
        original = Message("user", "hello")
        restored = Message.from_dict(original.to_dict())
        self.assertEqual(restored, original)

    def test_from_dict_missing_keys_has_defaults(self) -> None:
        restored = Message.from_dict({"nothing": "here"})
        self.assertEqual(restored.role, "unknown")
        self.assertEqual(restored.content, "")


class TestConversation(unittest.TestCase):
    """Conversation must add messages and report counts safely."""

    def setUp(self) -> None:
        self.conversation = Conversation()

    def test_add_returns_message_and_appends(self) -> None:
        message = self.conversation.add("user", "hi")
        self.assertIsInstance(message, Message)
        self.assertEqual(len(self.conversation), 1)

    def test_count_by_role(self) -> None:
        self.conversation.add("user", "a")
        self.conversation.add("assistant", "b")
        self.conversation.add("user", "c")
        self.assertEqual(
            self.conversation.count_by_role(),
            {"user": 2, "assistant": 1},
        )

    def test_len_dunder(self) -> None:
        self.assertEqual(len(self.conversation), 0)
        self.conversation.add("user", "x")
        self.assertEqual(len(self.conversation), 1)

    def test_bool_is_false_when_empty(self) -> None:
        self.assertFalse(self.conversation)
        self.conversation.add("user", "x")
        self.assertTrue(self.conversation)

    def test_messages_property_is_defensive_copy(self) -> None:
        self.conversation.add("user", "x")
        leaked = self.conversation.messages
        leaked.append(Message("assistant", "fake"))
        self.assertEqual(len(self.conversation), 1)

    def test_init_copies_incoming_list(self) -> None:
        external = [Message("user", "x")]
        conversation = Conversation(external)
        external.append(Message("assistant", "y"))
        self.assertEqual(len(conversation), 1)


class TestConversationValidation(unittest.TestCase):
    """Conversation.add must reject unknown roles loudly."""

    def setUp(self) -> None:
        self.conversation = Conversation()

    def test_accepts_valid_roles(self) -> None:
        self.conversation.add("user", "hi")
        self.conversation.add("assistant", "Hello!")
        self.assertEqual(len(self.conversation), 2)

    def test_rejects_unknown_role(self) -> None:
        with self.assertRaises(InvalidRoleError):
            self.conversation.add("robot", "beep")

    def test_rejected_message_is_not_recorded(self) -> None:
        with self.assertRaises(InvalidRoleError):
            self.conversation.add("robot", "beep")
        self.assertEqual(len(self.conversation), 0)

    def test_can_catch_every_app_error_as_assistant_error(self) -> None:
        # The whole point of the hierarchy: one except clause catches all.
        with self.assertRaises(AssistantError):
            self.conversation.add("system", "beep")


if __name__ == "__main__":
    unittest.main()
