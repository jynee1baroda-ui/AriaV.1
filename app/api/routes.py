"""HTTP routes (Phase 2, Milestone 2).

The HTTP boundary: parse → validate (Pydantic) → call the service →
serialize. No business logic lives here - ChatService owns that.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app import config
from app.schemas.chat import ChatRequest, ChatResponse, HistoryResponse, MessageSchema
from app.services.chat_service import ChatService

router = APIRouter(prefix="/api/v1")


def get_chat_service(request: Request) -> ChatService:
    """FastAPI dependency: hand the caller the shared ChatService.

    Injected via Depends() so tests can override it with a fake by
    calling app.dependency_overrides[get_chat_service].
    """
    return request.app.state.chat_service


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness probe.

    Load balancers and orchestrators (and you, for testing) hit this
    to confirm the service is up and learn which assistant it is.

    Returns:
        A small JSON identity card.
    """
    return {
        "status": "ok",
        "assistant": config.ASSISTANT_NAME,
        "version": config.ASSISTANT_VERSION,
    }


@router.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """Send one message to the assistant.

    Args:
        payload: The validated request (session_id + message).
        service: The shared ChatService (injected).

    Returns:
        The assistant's reply for that session.
    """
    reply = service.process(payload.session_id, payload.message)
    return ChatResponse(session_id=payload.session_id, reply=reply)


@router.get("/history/{session_id}", response_model=HistoryResponse)
def history(
    session_id: str,
    service: ChatService = Depends(get_chat_service),
) -> HistoryResponse:
    """Fetch the full conversation for a session.

    Args:
        session_id: The conversation to read.
        service: The shared ChatService (injected).

    Returns:
        The ordered messages, or an empty list for an unknown session.
    """
    messages = service.get_history(session_id)
    return HistoryResponse(
        session_id=session_id,
        messages=[MessageSchema(role=m.role, content=m.content) for m in messages],
    )
