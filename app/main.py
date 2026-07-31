"""AI Assistant - Phase 1: entry point (composition root).

Lesson 4: main.py became TINY on purpose. This is the "composition
root" pattern: the one place that wires objects together. Every real
framework works this way - Django's manage.py, uvicorn's entry point.

Responsibilities that REMAIN here:
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
# Running "python app\main.py" puts app\ (the script's folder) on the
# import path, NOT the project root - so "from app import ..." would
# fail. This prepends the project root so the package resolves. Running
# "python -m app.main" works without it (CWD is added automatically).
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import config, storage
from app.assistant import Assistant
from app.models import Conversation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("assistant")


def main() -> NoReturn:
    """Composition root: build the object graph, then run, then save."""
    # 1. Load long-term memory from disk (Lesson 3).
    conversation = Conversation(storage.load_history())

    # 2. Build the assistant around that memory (composition).
    assistant = Assistant(name=config.ASSISTANT_NAME, conversation=conversation)

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
