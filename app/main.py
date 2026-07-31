"""AI Assistant - FastAPI application (Phase 2, Milestone 1).

This module is the web server entry point. uvicorn targets it by name:

    .venv\\Scripts\\python -m uvicorn app.main:app --reload --port 8000

It is also an APP FACTORY: create_app() builds a fresh FastAPI app with
all routers wired in. Tests use TestClient(create_app()) - and when we
add dependency injection in M2, the factory is where overrides plug in.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app import config, db
from app.api.routes import router
from app.errors import InvalidSessionIdError
from app.services.chat_service import ChatService


def create_app() -> FastAPI:
    """Build and wire the FastAPI application.

    Returns:
        A configured FastAPI instance ready to serve.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Startup: migrate schema. Shutdown: close the DB pool.

        The lifespan runs when the server actually starts (uvicorn or a
        TestClient used as a context manager) - the correct place for
        infrastructure setup, not module import time.

        Schema is brought up to date with Alembic (M4), the versioned,
        reversible migration tool - never create_all in production.
        """
        db.run_migrations()
        yield
        db.engine.dispose()

    app = FastAPI(
        title=f"AI Assistant {config.ASSISTANT_NAME}",
        version=config.ASSISTANT_VERSION,
        lifespan=lifespan,
    )
    # The composition root owns the service's lifecycle. Routes access
    # it via Depends(), which keeps them testable (an override can swap
    # a fake service in without touching route code).
    app.state.chat_service = ChatService()

    # If the service's own defense-in-depth validation fires (a bad id
    # that slipped past the schema), surface it as a clean 400 instead
    # of a 500. Never let a validation failure look like a server bug.
    @app.exception_handler(InvalidSessionIdError)
    async def invalid_session_handler(request: Request, exc: InvalidSessionIdError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    app.include_router(router)
    return app


# Module-level instance: uvicorn's standard "app.main:app" target.
app = create_app()
