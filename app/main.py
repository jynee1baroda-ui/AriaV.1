"""AI Assistant - FastAPI application (Phase 2, Milestone 1).

This module is the web server entry point. uvicorn targets it by name:

    .venv\\Scripts\\python -m uvicorn app.main:app --reload --port 8000

It is also an APP FACTORY: create_app() builds a fresh FastAPI app with
all routers wired in. Tests use TestClient(create_app()) - and when we
add dependency injection in M2, the factory is where overrides plug in.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app import config, db
from app.api.routes import router
from app.errors import InvalidSessionIdError
from app.llm.generator import LLMGenerator
from app.security.rate_limiter import RateLimiter
from app.services.chat_service import ChatService
from app.services.intent_router import IntentRouter

# huggingface_hub overrides its own log level at import time (lazily,
# on the first LLM load), so the supported knob is HF_HUB_VERBOSITY, read
# at its import. 'error' silences the "unauthenticated requests" warning
# the HF server sends as an HTTP response header on every download.
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

# httpx logs every LLM-download HTTP request at INFO; raise only its
# level so uvicorn's console stays clean - the app's own logs are
# unaffected.
for _noisy_logger in ("httpx", "httpcore"):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)


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
    # a fake service in without touching route code). The ML intent
    # router (Phase 4 M2) is shared across requests and degrades to
    # rules if the model artifact is missing. The LLM answer engine
    # (Phase 7 M1) is shared too and degrades to the echo if the model
    # cannot be loaded. Its load is LAZY, so startup stays fast and the
    # model only downloads on the first unanswered question.
    app.state.chat_service = ChatService(
        router=IntentRouter() if config.INTENT_CLASSIFIER_ENABLED else None,
        llm=LLMGenerator() if config.LLM_ENABLED else None,
    )

    # The abuse guard: one limiter shared by every request. Routes
    # enforce it via Depends(); tests can override with a tiny limit.
    app.state.rate_limiter = RateLimiter(
        limit=config.RATE_LIMIT_MAX_REQUESTS,
        window_seconds=config.RATE_LIMIT_WINDOW_SECONDS,
    )

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
