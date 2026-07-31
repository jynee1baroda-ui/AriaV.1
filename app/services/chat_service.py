"""ChatService - orchestrates conversations across HTTP requests.

Phase 2, Milestone 3: sessions are now DURABLE. Each conversation is
lazily loaded from its file on first contact and saved after every
turn, so a server restart loses nothing.

Three responsibilities:
  1. SESSIONS. A dict of Conversations, keyed by session_id.
  2. CONCURRENCY. A threading.Lock makes get-or-create + mutation
     atomic (FastAPI runs sync endpoints in a threadpool).
  3. PERSISTENCE. Load on demand, save after every turn, via the
     repository layer (storage). Phase 3 swaps files for a database
     and nothing outside this class changes.

SECURITY: session_id is validated here AGAIN (defense in depth) even
though the API schema already rejects bad ids - because the service is
the layer that actually builds file paths. Never trust a single layer.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path

from app import config, storage
from app.assistant import Assistant
from app.errors import InvalidSessionIdError
from app.models import Conversation, Message


class ChatService:
    """Thread-safe, persistent store of per-session conversations."""

    def __init__(self, sessions_dir: Path | None = None) -> None:
        # Injection point: tests pass a temp dir so they never touch the
        # real data/ folder. Defaults to the repository's session dir.
        self._sessions_dir = sessions_dir or storage.SESSION_DIR
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
            # A fresh Assistant is cheap and stateless (name + logic);
            # respond() records both sides into the shared conversation.
            assistant = Assistant(name=config.ASSISTANT_NAME, conversation=conversation)
            reply = assistant.respond(message)
            # Save under the lock so the on-disk snapshot can never
            # overtake the in-memory truth.
            storage.save_history(conversation.messages, self._path_for(session_id))
            return reply

    def get_history(self, session_id: str) -> list[Message]:
        """Return a session's messages (loading them if necessary).

        Reads from memory when the session is live, otherwise from disk.
        A GET never creates a session - unknown sessions yield [].

        Args:
            session_id: The conversation to inspect.

        Returns:
            The ordered messages so far.

        Raises:
            InvalidSessionIdError: If session_id fails the whitelist.
        """
        self._validate_session_id(session_id)
        with self._lock:
            conversation = self._sessions.get(session_id)
            if conversation is not None:
                return list(conversation.messages)
            return list(storage.load_history(self._path_for(session_id)))

    # ------------------------------------------------------------------
    # Private helpers - call while holding self._lock (except _validate)
    # ------------------------------------------------------------------
    def _validate_session_id(self, session_id: str) -> None:
        """Reject ids that could escape the sessions directory.

        This mirrors the schema validator as defense in depth. Cheap -
        a single regex - so there is no reason to skip it.
        """
        if not re.fullmatch(config.SESSION_ID_PATTERN, session_id):
            raise InvalidSessionIdError(
                f"Invalid session_id {session_id!r}; must match "
                f"{config.SESSION_ID_PATTERN}"
            )

    def _path_for(self, session_id: str) -> Path:
        """Storage path for a session (ids are pre-validated)."""
        return self._sessions_dir / f"{session_id}.json"

    def _load_or_create(self, session_id: str) -> Conversation:
        """Return the live conversation, loading from disk if needed.

        PRIVATE: must only be called while holding self._lock.
        """
        conversation = self._sessions.get(session_id)
        if conversation is None:
            # load_history fails soft: a missing or corrupt file simply
            # yields a fresh conversation (log only, never crash).
            conversation = Conversation(storage.load_history(self._path_for(session_id)))
            self._sessions[session_id] = conversation
        return conversation
