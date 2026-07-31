"""Central configuration for the assistant.

Phase 1, Lesson 2: this module teaches two ideas at once:

1. **Modules** - a file of reusable data/functions you import elsewhere.
2. **Data types** - strings, sets, dicts, and how to choose the right one.

Keeping the assistant's *identity* here, separate from its *behavior*
in main.py, is the Single Responsibility principle (SOLID).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env if present (it is gitignored; never commit real values).
# The .env.example file documents every variable we read.
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
# Public alias so other modules (e.g. db.py's Alembic wiring) can locate
# project files without re-deriving the path.
PROJECT_ROOT: Path = _PROJECT_ROOT
load_dotenv(_PROJECT_ROOT / ".env")

# str - immutable text
ASSISTANT_NAME: str = "Aria"

# Version is stored as a string (not a float!) because "1.10" would
# otherwise become "1.1" and lose the zero. Versions are text labels.
ASSISTANT_VERSION: str = "0.1.0"

# dict - key -> value lookup table of named information.
# A dict is the right tool because we look things up BY NAME.
ASSISTANT_INFO: dict[str, str] = {
    "name": ASSISTANT_NAME,
    "version": ASSISTANT_VERSION,
    "phase": "Phase 4 - Machine Learning",
    "creator": "You",
}

# set - unordered collection of unique items with fast "is this in it?"
# lookups. A set (not a list) is right here because:
#   1. We only check membership ("is 'hello' a greeting?").
#   2. Duplicates would be meaningless anyway.
GREETING_WORDS: set[str] = {"hello", "hi", "hey", "yo", "hiya", "hola"}

# tuple - an immutable ordered sequence. Coordinates/settings that must
# never change once defined belong in a tuple, not a list.
# REPL_PROMPT is the text shown before each input.
REPL_PROMPT: str = "You> "

# Whitelist for session identifiers. Sessions are database keys, but the
# pattern stays a security control: ids appear in URLs, and only letters,
# digits, underscore, hyphen are allowed - never dots or slashes that
# could smuggle path separators or break URL routing.
SESSION_ID_PATTERN: str = r"^[A-Za-z0-9_-]{1,64}$"

# Welcome banner (f-string interpolation built at import time)
WELCOME_MESSAGE: str = (
    f"AI Assistant {ASSISTANT_NAME} v{ASSISTANT_VERSION}. "
    "Type /help for commands."
)


def _default_database_url() -> str:
    """Derive a sensible local SQLite URL when DATABASE_URL is unset.

    Uses forward slashes (as_posix) because SQLAlchemy treats the
    sqlite URL path as a file path and backslashes break it. In the
    deploy phase, DATABASE_URL will point at PostgreSQL and this
    function becomes irrelevant.
    """
    db_path: Path = _PROJECT_ROOT / "data" / "assistant.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path.as_posix()}"


# Database connection string. Read from environment (or .env) so the
# SAME code runs against SQLite locally and PostgreSQL in production.
DATABASE_URL: str = os.getenv("DATABASE_URL", _default_database_url())

# Rate limiting (Phase 6, M1): a per-IP cap on the write path. Limits
# are generous for a dev API; production tuning depends on auth. The
# infrastructure probes /health and /ready are deliberately exempt.
RATE_LIMIT_ENABLED: bool = True
RATE_LIMIT_MAX_REQUESTS: int = 30
RATE_LIMIT_WINDOW_SECONDS: int = 60

# History pagination (Phase 6, M2): a page size cap so one request can
# never load a session's entire unbounded history into memory.
HISTORY_DEFAULT_LIMIT: int = 100
HISTORY_MAX_LIMIT: int = 500

# ML intent routing (Phase 4, M2): the classifier steers respond() only
# when it is confident enough, and the deterministic rules remain the
# safety net. Flip ENABLED off to return to pure rule-based dispatch.
INTENT_CLASSIFIER_ENABLED: bool = True
# Confidence must beat this to let ML override the rules. 0.5 means "the
# model must favor one intent over every other"; lower lets ML act on
# weaker signals (more aggressive), higher keeps rules in charge more
# often (safer). Tune on real user messages, not the training set.
INTENT_CONFIDENCE_THRESHOLD: float = 0.5
# Which trained model the router loads (Phase 5 M2 / Phase 6 M2):
#   "linear"      - Phase 4 logistic regression (default: fastest,
#                   simplest, interpretable; matches the neural net's
#                   71.1% held-out accuracy),
#   "neural"      - Phase 5 PyTorch MLP (ties the linear model),
#   "transformer" - Phase 6 from-scratch transformer (currently trails
#                   at 60.5% without pre-training; the architecture is
#                   the Phase 7 foundation).
INTENT_MODEL_KIND: str = "linear"
