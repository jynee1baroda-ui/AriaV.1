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

# LLM question answering (Phase 7, M1): a pre-trained causal language
# model generates answers for anything the deterministic rules cannot
# handle. The model is loaded lazily on the first unanswered question
# and only from Hugging Face when LLM_LOCAL_FILES_ONLY is False (set it
# True to run strictly offline against the local cache). The token
# budget is small on purpose: a 135M base model drifts into noise after
# a sentence or two - short answers are GOOD. Temperature: 0 (default)
# = GREEDY decoding, deterministic and reliable for a tiny model;
# set >0 to enable sampling (more varied, occasionally echoes the
# question instead of answering).
LLM_ENABLED: bool = True
# Phase 8 (Prompt Engineering): the INSTRUCT variant follows the Aria
# system prompt (chat template); the base variant only completes text.
LLM_MODEL_ID: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
LLM_MAX_NEW_TOKENS: int = 64
LLM_TEMPERATURE: float = 0.0
LLM_LOCAL_FILES_ONLY: bool = False
# How many past (user, assistant) exchange pairs are written into the
# LLM prompt (Phase 7, M2 - the context window). The window is small on
# purpose: every pair costs prompt tokens and a 135M model drifts on
# long prompts. 4 turns is a comfortable short-memory size.
LLM_CONTEXT_TURNS: int = 4

# Phase 8 (Prompt Engineering): Aria's system prompt - the file that
# defines her personality and behavior. Kept as a markdown document so
# the prompt itself is versioned like code.
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "app" / "prompts" / "aria_system.md"

# Factual integrity: exact facts are ALWAYS answered from the verified
# knowledge base first (never generated). FACTUAL_ONLY=True (strict)
# refuses every unverified question; False (the default since the
# conversational system prompt became core behavior) lets the LLM speak
# naturally - the LLM is a plausible-text generator and CAN be wrong.
FACTUAL_ONLY: bool = False
# Consult the verified knowledge base (app/kb) for factual answers.
KNOWLEDGE_ENABLED: bool = True

# Phase 11 foundation (tool calling preview): deterministic tools
# (clock, calendar, calculator, notes) answer BEFORE the LLM. Tools
# compute or store - they never guess - so they answer in factual mode
# too. Flip off to return to pure conversational answering.
TOOLS_ENABLED: bool = True
