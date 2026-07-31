"""ChatService - orchestrates conversations across HTTP requests.

Phase 2, Milestone 3 -> Phase 4, Milestone 2: storage moved from JSON
files to the MessageRepository (relational database). The service's
public API - process(), get_history() - is UNCHANGED, which is exactly
the point of the repository seam.

Three responsibilities:
  1. SESSIONS. A dict of Conversations, keyed by session_id.
  2. CONCURRENCY. A threading.Lock makes get-or-create + mutation
     atomic (FastAPI runs sync endpoints in a threadpool).
  3. PERSISTENCE. Conversations are loaded from the repository and new
     messages inserted after every turn (incremental, not snapshots).

SECURITY: session_id is validated here AGAIN (defense in depth) even
though the API schema already rejects bad ids - never trust one layer.
"""

from __future__ import annotations

import re
import threading

from app import config
from app.assistant import Assistant
from app.errors import InvalidSessionIdError
from app.models import Conversation, Message
from app.persistence.memory_repository import MemoryRepository
from app.persistence.message_repository import MessageRepository
from app.services.intent_router import IntentRouter
from app.services.memory_service import MemoryService


class ChatService:
    """Thread-safe, persistent store of per-session conversations."""

    def __init__(
        self,
        repository: MessageRepository | None = None,
        memory_service: MemoryService | None = None,
        router: IntentRouter | None = None,
    ) -> None:
        # Injection point: tests build a repository over a temp database
        # so test sessions never touch the real data/assistant.db. The
        # memory service shares the SAME engine, keeping memory and
        # messages isolated to whichever database the test chose.
        self._repository = repository or MessageRepository()
        self._memory = memory_service or MemoryService(
            MemoryRepository(self._repository.engine)
        )
        # Phase 4 M2: the shared ML intent router (or None = rules only).
        # One router per service, reused across turns and threads - it
        # lazily loads the model once and gates on confidence.
        self._router = router
        self._sessions: dict[str, Conversation] = {}
        # One lock guarding ALL session access. Correct and simple.
        # (Improvement for Phase 7: per-session locks, so a slow chat
        # in session A never blocks session B.)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(self, session_id: str, message: str) -> str:
        """Run one user message through the assistant and persist.

        Args:
            session_id: Which conversation to continue (loaded or created).
            message: The validated user text.

        Returns:
            The assistant's reply.

        Raises:
            InvalidSessionIdError: If session_id fails the whitelist.
        """
        self._validate_session_id(session_id)
        with self._lock:
            conversation = self._load_or_create(session_id)
            before: int = len(conversation)
            # A fresh Assistant is cheap and stateless (name + logic);
            # respond() records both sides into the shared conversation
            # AND consolidates/retrieves long-term memory (the memory
            # hook is wired because we pass a MemoryService).
            assistant = Assistant(
                name=config.ASSISTANT_NAME,
                conversation=conversation,
                session_id=session_id,
                memory_service=self._memory,
                router=self._router,
            )
            reply = assistant.respond(message)
            # Persist ONLY the messages appended by this turn
            # (incremental insert - the relational best practice).
            added: list[Message] = conversation.messages[before:]
            self._repository.add_messages(session_id, added)
            return reply

    def get_history(
        self,
        session_id: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Message]:
        """Return a session's messages (loading them if necessary).

        Reads from memory when the session is live, otherwise from the
        database. A GET never creates a session - unknown sessions
        yield []. When limit/offset are given, only that page of the
        live conversation's messages is returned.

        Args:
            session_id: The conversation to inspect.
            limit: Max messages; None returns the whole conversation.
            offset: How many messages to skip first.

        Returns:
            The ordered messages so far.

        Raises:
            InvalidSessionIdError: If session_id fails the whitelist.
        """
        self._validate_session_id(session_id)
        with self._lock:
            conversation = self._sessions.get(session_id)
            if conversation is not None:
                # Live session: page the in-memory list.
                messages = conversation.messages
                if limit is None:
                    return list(messages[offset:])
                return list(messages[offset: offset + limit])
            # Cold session: let the database do the pagination.
            return list(self._repository.get_messages(session_id, limit=limit, offset=offset))

    def count_history(self, session_id: str) -> int:
        """Return the total number of stored messages in a session.

        Used to build the pagination metadata (`total`) independently
        of the current page. The count comes from the database, so it
        is accurate even for sessions never loaded into memory.

        Args:
            session_id: The conversation to count.

        Returns:
            The message count.

        Raises:
            InvalidSessionIdError: If session_id fails the whitelist.
        """
        self._validate_session_id(session_id)
        return self._repository.count_messages(session_id)

    def ping(self) -> bool:
        """Report whether the persistence backend is reachable.

        Delegates to the repository; the service layer exposes it so
        routes never talk to the repository directly (single seam).

        Returns:
            True if the database is responsive.
        """
        return self._repository.ping()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _validate_session_id(self, session_id: str) -> None:
        """Enforce the session_id whitelist (defense in depth).

        Mirrors the schema validator so every entry point - HTTP or
        programmatic - obeys the same identifier contract.
        """
        if not re.fullmatch(config.SESSION_ID_PATTERN, session_id):
            raise InvalidSessionIdError(
                f"Invalid session_id {session_id!r}; must match "
                f"{config.SESSION_ID_PATTERN}"
            )

    def _load_or_create(self, session_id: str) -> Conversation:
        """Return the live conversation, loading from the DB if needed.

        PRIVATE: must only be called while holding self._lock.
        """
        conversation = self._sessions.get(session_id)
        if conversation is None:
            conversation = Conversation(self._repository.get_messages(session_id))
            self._sessions[session_id] = conversation
        return conversation
