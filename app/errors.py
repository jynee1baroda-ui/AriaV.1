"""Application-specific exception hierarchy.

Phase 1, Lesson 5: custom exceptions.

WHY A HIERARCHY? So callers can catch at any level of detail:
    except InvalidRoleError:   # handle this specific bug
    except ConversationError:  # handle all conversation problems
    except AssistantError:     # handle ANY error this app raised

The base class also lets us distinguish OUR errors from library
errors (OSError, ValueError, ...) with a single except clause.
"""

from __future__ import annotations


class AssistantError(Exception):
    """Base class for every error raised by the assistant.

    Inheriting from Exception (not BaseException) keeps our errors
    catchable via `except Exception` while never swallowing
    KeyboardInterrupt/SystemExit by accident.
    """


class ConversationError(AssistantError):
    """Base for errors raised while managing a conversation."""


class InvalidRoleError(ConversationError):
    """Raised when a message is added with an unknown role.

    Example:
        conversation.add("robot", "beep")  # -> InvalidRoleError
    """


class InvalidSessionIdError(AssistantError):
    """Raised when a session id fails the whitelist.

    Session ids become file names; allowing arbitrary characters would
    enable path traversal (e.g. "../evil"). The API layer rejects these
    with 422; this error is the service layer's defense in depth.
    """


class StorageError(AssistantError):
    """Raised when a persistence operation fails.

    Wraps whatever the underlying driver raised, so callers can catch
    ONE storage exception type regardless of whether the backend is
    SQLite, PostgreSQL, or (earlier in the project) JSON files. The
    original exception stays attached via `raise ... from exc`.
    """
