"""ChatService - orchestrates conversations across HTTP requests.

Phase 2, Milestone 2: the assistant moves from one REPL loop to many
concurrent web requests. Two hard problems live here:

1. SESSIONS. Each HTTP request is independent; a client identifies its
   conversation with a session_id. We keep a dict of Conversations,
   created lazily on first contact.

2. CONCURRENCY. FastAPI runs sync endpoints in a threadpool, so two
   requests can touch this object at the same time. A get-or-create
   followed by a mutation is a check-then-act race - the classic bug
   that corrupts data. A threading.Lock makes the whole operation
   atomic.

M3 replaces the in-memory dict with per-session file persistence.
Phase 3 replaces it with a database. Nothing outside this class
changes - that is the Service Layer pattern.
"""

from __future__ import annotations

import threading
from typing import NoReturn

from app import config
from app.assistant import Assistant
from app.models import Conversation, Message


class ChatService:
    """Thread-safe store of per-session conversations."""

    def __init__(self) -> None:
        self._sessions: dict[str, Conversation] = {}
        # One lock guarding ALL session access. Correct and simple.
        # (Improvement for Phase 7: per-session locks, so a slow chat
        # in session A never blocks session B.)
        self._lock = threading.Lock()

    def process(self, session_id: str, message: str) -> str:
        """Run one user message through the assistant.

        Args:
            session_id: Which conversation to continue (created if new).
            message: The validated user text.

        Returns:
            The assistant's reply.
        """
        with self._lock:
            conversation = self._get_or_create(session_id)
            # A fresh Assistant is cheap and stateless (name + logic);
            # respond() records both sides into the shared conversation.
            assistant = Assistant(name=config.ASSISTANT_NAME, conversation=conversation)
            return assistant.respond(message)

    def get_history(self, session_id: str) -> list[Message]:
        """Return a copy of a session's messages.

        Args:
            session_id: The conversation to inspect.

        Returns:
            The messages so far; an empty list for an unknown session.
        """
        with self._lock:
            conversation = self._sessions.get(session_id)
            # conversation.messages already returns a defensive copy -
            # even if we didn't hold the lock, the caller couldn't
            # mutate our internals through this reference.
            return list(conversation.messages) if conversation else []

    def _get_or_create(self, session_id: str) -> Conversation:
        """Return the session's conversation, creating it if missing.

        PRIVATE: must only be called while holding self._lock.
        """
        conversation = self._sessions.get(session_id)
        if conversation is None:
            conversation = Conversation()
            self._sessions[session_id] = conversation
        return conversation
