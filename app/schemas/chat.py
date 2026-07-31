"""HTTP schemas for the chat and history endpoints.

Pydantic schemas are the VALIDATION BOUNDARY: anything that reaches
our services has already been type-checked, length-checked, and
stripped. Invalid requests never touch business logic.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

from app import config

# Cap message size so one request can't carry gigabytes of text.
MAX_MESSAGE_LENGTH: int = 4000


class ChatRequest(BaseModel):
    """Body of POST /api/v1/chat.

    Attributes:
        session_id: Identifies the conversation. New ids start fresh.
        message: The user's text.
    """

    session_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Conversation identifier; a new id starts a new conversation.",
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
        description="The user's message.",
    )

    @field_validator("session_id", "message")
    @classmethod
    def strip_whitespace(cls, value: str) -> str:
        """Trim surrounding whitespace at the boundary.

        A message of only spaces ("   ") has length >= 1 and passes
        min_length - so we strip first, then reject anything empty.
        """
        return value.strip()

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        """Enforce the session_id whitelist.

        Session ids become file names in M3, so this is a security
        control: rejecting dots and slashes stops path traversal. The
        max_length=64 Field plus this regex keep ids bounded and safe.
        """
        value = value.strip()
        if not re.fullmatch(config.SESSION_ID_PATTERN, value):
            raise ValueError(
                "session_id must be 1-64 characters: letters, digits, "
                "underscore, or hyphen"
            )
        return value

    @field_validator("message")
    @classmethod
    def reject_blank_message(cls, value: str) -> str:
        """A message that is only whitespace is invalid."""
        if not value:
            raise ValueError("message must not be blank")
        return value


class ChatResponse(BaseModel):
    """Body of a successful POST /api/v1/chat response.

    Attributes:
        session_id: Echoed back so clients always know their session.
        reply: The assistant's reply text.
    """

    session_id: str
    reply: str


class MessageSchema(BaseModel):
    """One stored message, as returned by the history endpoint."""

    role: str
    content: str


class HistoryResponse(BaseModel):
    """Body of GET /api/v1/history/{session_id}.

    Attributes:
        session_id: Which session this history belongs to.
        messages: The full ordered conversation so far.
    """

    session_id: str
    messages: list[MessageSchema]
