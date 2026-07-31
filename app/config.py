"""Central configuration for the assistant.

Phase 1, Lesson 2: this module teaches two ideas at once:

1. **Modules** - a file of reusable data/functions you import elsewhere.
2. **Data types** - strings, sets, dicts, and how to choose the right one.

Keeping the assistant's *identity* here, separate from its *behavior*
in main.py, is the Single Responsibility principle (SOLID).
"""

from __future__ import annotations

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
    "phase": "Phase 1 - Programming Fundamentals",
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

# Whitelist for session identifiers. Sessions become FILE NAMES in M3,
# so this pattern is a security control (path traversal defense), not
# just a style rule. Only letters, digits, underscore, hyphen - never
# dots or slashes that could escape the sessions directory.
SESSION_ID_PATTERN: str = r"^[A-Za-z0-9_-]{1,64}$"

# Welcome banner (f-string interpolation built at import time)
WELCOME_MESSAGE: str = (
    f"AI Assistant {ASSISTANT_NAME} v{ASSISTANT_VERSION}. "
    "Type /help for commands."
)
