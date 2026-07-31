"""AI Assistant - FastAPI application (Phase 2, Milestone 1).

This module is the web server entry point. uvicorn targets it by name:

    .venv\\Scripts\\python -m uvicorn app.main:app --reload --port 8000

It is also an APP FACTORY: create_app() builds a fresh FastAPI app with
all routers wired in. Tests use TestClient(create_app()) - and when we
add dependency injection in M2, the factory is where overrides plug in.
"""

from __future__ import annotations

from fastapi import FastAPI

from app import config
from app.api.routes import router
from app.services.chat_service import ChatService


def create_app() -> FastAPI:
    """Build and wire the FastAPI application.

    Returns:
        A configured FastAPI instance ready to serve.
    """
    app = FastAPI(
        title=f"AI Assistant {config.ASSISTANT_NAME}",
        version=config.ASSISTANT_VERSION,
    )
    # The composition root owns the service's lifecycle. Routes access
    # it via Depends(), which keeps them testable (an override can swap
    # a fake service in without touching route code).
    app.state.chat_service = ChatService()
    app.include_router(router)
    return app


# Module-level instance: uvicorn's standard "app.main:app" target.
app = create_app()
