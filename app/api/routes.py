"""HTTP routes (Phase 2, Milestone 2).

The HTTP boundary: parse → validate (Pydantic) → call the service →
serialize. No business logic lives here - ChatService owns that.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from app import config
from app.schemas.chat import ChatRequest, ChatResponse, HistoryResponse, MessageSchema
from app.security.rate_limiter import RateLimiter
from app.services.chat_service import ChatService

router = APIRouter(prefix="/api/v1")


def get_chat_service(request: Request) -> ChatService:
    """FastAPI dependency: hand the caller the shared ChatService.

    Injected via Depends() so tests can override it with a fake by
    calling app.dependency_overrides[get_chat_service].
    """
    return request.app.state.chat_service


def get_rate_limiter(request: Request) -> RateLimiter:
    """FastAPI dependency: the shared, process-wide RateLimiter."""
    return request.app.state.rate_limiter


def enforce_rate_limit(
    request: Request,
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> None:
    """Dependency that rejects abusive clients with HTTP 429.

    Applied to the WRITE path only: POST /chat is the surface that
    costs us work (DB writes, memory consolidation). Reads, health,
    and readiness probes are exempt so infrastructure monitoring is
    never throttled.

    Identity is the client IP until authentication exists (a later
    phase); Retry-After tells the client when it may try again.
    """
    if not config.RATE_LIMIT_ENABLED:
        return
    key = request.client.host if request.client else "unknown"
    allowed, retry_after = limiter.check(key)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait and try again.",
            headers={"Retry-After": str(retry_after)},
        )


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness probe.

    Answers "is the process alive?" - it should succeed even if the
    database is down, so an orchestrator knows whether to restart us.
    Load balancers and orchestrators (and you, for testing) hit this.

    Returns:
        A small JSON identity card.
    """
    return {
        "status": "ok",
        "assistant": config.ASSISTANT_NAME,
        "version": config.ASSISTANT_VERSION,
    }


@router.get("/ready")
def ready(service: ChatService = Depends(get_chat_service)) -> JSONResponse:
    """Readiness probe.

    Answers "can this process actually do its job?" - which, for a
    chat service, means the database is reachable. Kubernetes sends
    traffic only to ready pods; a broken DB returns 503 and the pod
    is taken out of rotation.

    Args:
        service: The shared ChatService (injected).

    Returns:
        200 with database status, or 503 when the DB is unreachable.
    """
    if service.ping():
        return JSONResponse(content={"status": "ready", "database": "ok"})
    return JSONResponse(
        status_code=503,
        content={"status": "not_ready", "database": "unavailable"},
    )


@router.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    service: ChatService = Depends(get_chat_service),
    _rate_limited: None = Depends(enforce_rate_limit),
) -> ChatResponse:
    """Send one message to the assistant.

    Args:
        payload: The validated request (session_id + message).
        service: The shared ChatService (injected).
        _rate_limited: Dependency only - enforces the rate limit.

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
