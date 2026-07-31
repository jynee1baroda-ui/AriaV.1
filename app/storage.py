"""Conversation persistence - the REPOSITORY layer.

Phase 1, Lesson 5: hardening with exceptions and context managers.

  * AtomicFileWriter - a context manager that commits files atomically
    (write to .tmp, rename over the destination only on success), so a
    crash mid-write can never corrupt the memory file.
  * load_history      - uses try/except/else: the else block runs only
    on success, keeping "validated good data" separate from error paths.
  * save_history      - swallows write failures ON PURPOSE, but logs the
    full traceback (logger.exception) so nothing fails silently.

It is a REPOSITORY in the layered-architecture sense: the only module
that knows HOW conversations are stored. Today that is a JSON file;
in Phase 3 it becomes PostgreSQL, and nothing outside this file changes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import TracebackType
from typing import TextIO

from app.models import Message

logger = logging.getLogger("assistant.storage")

# Path(Path(__file__).parent.parent) = project root, regardless of where
# the program is launched from. "data/history.json" lives there.
DEFAULT_HISTORY_FILE: Path = (
    Path(__file__).resolve().parent.parent / "data" / "history.json"
)

# Per-session files for the web server (M3): data/sessions/<id>.json
SESSION_DIR: Path = (
    Path(__file__).resolve().parent.parent / "data" / "sessions"
)


def session_path(session_id: str) -> Path:
    """Build the storage path for one session.

    NOTE: this function is intentionally DUMB - it does no validation.
    Callers (ChatService) must ensure session_id already matches the
    whitelist before this is reached, so the id can never escape the
    sessions directory. Single responsibility: storage builds paths,
    services enforce safety.

    Args:
        session_id: A whitelisted session identifier.

    Returns:
        data/sessions/<session_id>.json
    """
    return SESSION_DIR / f"{session_id}.json"


class AtomicFileWriter:
    """Context manager that commits a file atomically.

    HOW: all writes go to "<name>.tmp". __exit__ renames that temp file
    over the real destination ONLY if the with-block completed without
    an exception. Path.replace is an OS-level atomic rename, so readers
    can never observe a half-written file.

    Usage:
        with AtomicFileWriter(path) as fh:
            fh.write("important data")
    """

    def __init__(self, destination: Path) -> None:
        self.destination = destination
        self.temp_path = destination.with_suffix(destination.suffix + ".tmp")

    def __enter__(self) -> TextIO:
        """Set-up phase of the protocol: open the temp file for writing.

        We KEEP the handle (self._handle) so __exit__ can close it -
        this is load-bearing on Windows, see __exit__.
        """
        self._handle = self.temp_path.open("w", encoding="utf-8")
        return self._handle

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Clean-up phase. exc_type is None iff the suite succeeded.

        Close the handle BEFORE renaming/deleting: Windows refuses to
        rename or unlink a file with an open handle (WinError 32).
        POSIX allows it, which is why this bug hides on Linux and only
        bites Windows - a classic cross-platform portability trap.

        Returns False always: we never want to SUPPRESS an exception -
        the caller's except-block must still see it.

        Args:
            exc_type: The exception class, or None on success.
            exc_value: The exception instance, or None.
            traceback: The traceback object, or None.
        """
        self._handle.close()
        if exc_type is None:
            # Success path: atomically replace the destination.
            self.temp_path.replace(self.destination)
        else:
            # Failure path: the data may be half-written. Throw it away
            # and let the exception propagate to the caller.
            self.temp_path.unlink(missing_ok=True)
        return False


def load_history(path: Path = DEFAULT_HISTORY_FILE) -> list[Message]:
    """Load previously saved conversation history from disk.

    Fails soft: a missing or corrupt file is not a crash - we just
    start fresh and log a warning. A real assistant must never refuse
    to start because its memory was damaged.

    Args:
        path: Where to read from. Defaults to data/history.json.

    Returns:
        A list of Message objects (possibly empty).
    """
    if not path.exists():
        logger.info("No history file yet at %s - starting fresh.", path)
        return []

    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not load history from %s: %s. Starting fresh.", path, exc)
        return []
    else:
        # The else block runs ONLY when the try succeeded - so reaching
        # this line means `data` is guaranteed valid JSON. This keeps
        # "post-success validation" out of the except clause.
        if not isinstance(data, list):
            logger.warning("History file %s is not a JSON list. Starting fresh.", path)
            return []
        return [Message.from_dict(entry) for entry in data]


def save_history(messages: list[Message], path: Path = DEFAULT_HISTORY_FILE) -> None:
    """Write the full conversation to disk as JSON (atomically).

    On failure we log the full traceback and swallow: at REPL exit there
    is no caller left to inform, so the memory of this session is lost -
    but the operator sees EXACTLY what broke (logger.exception prints
    the whole stack). Swallowing + loud logging is a deliberate,
    documented tradeoff - never silent swallowing.

    Args:
        messages: The Message objects to persist.
        path: Where to write. Defaults to data/history.json.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with AtomicFileWriter(path) as fh:
            # A list comprehension converts each Message to a dict.
            json.dump([m.to_dict() for m in messages], fh, ensure_ascii=False, indent=2)
        logger.info("Saved %d messages to %s", len(messages), path)
    except OSError as exc:
        logger.exception("Could not save history to %s: %s", path, exc)
