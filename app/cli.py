"""AI Assistant - CLI entry point (Phase 1 REPL, preserved).

Moved here in Phase 2 so app/main.py can become the FastAPI server.
The REPL remains fully functional and tested - it is now a secondary
way to drive the assistant.

Run with:
    python -m app.cli
    python app\cli.py

Responsibilities:
  * build the object graph (Conversation, Assistant)
  * own the lifecycle - the try/finally that GUARANTEES memory is
    saved even on crash, /exit, or Ctrl+C.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import NoReturn

# --- Bootstrap: make "app" importable both ways -------------------------
# Running "python app\cli.py" puts app\ (the script's folder) on the
# import path, NOT the project root - so "from app import ..." would
# fail. This prepends the project root so the package resolves. Running
# "python -m app.cli" works without it (CWD is added automatically).
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import config, storage
from app.assistant import Assistant
from app.models import Conversation
from app.services.memory_service import MemoryService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("assistant")


def main() -> NoReturn:
    """Composition root: build the object graph, then run, then save."""
    # 1. Load long-term memory from disk (Lesson 3).
    conversation = Conversation(storage.load_history())

    # 2. Build the assistant around that memory (composition). The
    #    MemoryService (Phase 5 M2) gives it long-term memory that
    #    survives across sessions via the database.
    memory_service = MemoryService()
    assistant = Assistant(
        name=config.ASSISTANT_NAME,
        conversation=conversation,
        session_id="cli",
        memory_service=memory_service,
    )

    # 3. Run, and GUARANTEE persistence on every exit path.
    #    finally runs no matter HOW this block ends: normal return,
    #    /exit, SystemExit from Ctrl+C - even an unhandled exception.
    #    This is why we removed save from /exit: one owner, one rule.
    try:
        assistant.run()
    finally:
        storage.save_history(conversation.messages)
        logger.info("Session ended. Memory persisted.")


if __name__ == "__main__":
    main()
