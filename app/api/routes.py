"""HTTP routes (Phase 2, Milestone 1).

This router is the HTTP boundary: it parses requests, calls services,
and returns JSON. It contains NO business logic - that lives in the
service layer from M2 onward.
"""

from __future__ import annotations

from fastapi import APIRouter

from app import config

router = APIRouter(prefix="/api/v1")


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
