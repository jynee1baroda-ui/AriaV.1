"""The Assistant - the brain of the operation.

Phase 1, Lesson 4: an OOP refactor. Everything the assistant IS
(name, conversation) and DOES (respond, commands, run the REPL) now
lives in one cohesive class. This is the Single Responsibility
principle at class scale.

The pattern is COMPOSITION: Assistant HAS-A Conversation. It is the
modern default over inheritance (Assistant IS-A ... ) and it keeps
the pieces independently testable.
"""

from __future__ import annotations

import logging
import sys
from typing import NoReturn

from app import config
from app.models import Conversation

logger = logging.getLogger("assistant")


class Assistant:
    """A conversational agent with a name and a memory.

    Attributes:
        name: The assistant's display name.
        conversation: The Conversation object holding all messages.
    """

    def __init__(self, name: str, conversation: Conversation) -> None:
        self.name = name
        self.conversation = conversation

    # ------------------------------------------------------------------
    # Core behavior
    # ------------------------------------------------------------------
    def respond(self, user_input: str) -> str:
        """Produce a reply AND record the exchange in memory.

        NOTE THE SIDE EFFECT: calling respond() also mutates
        self.conversation (adds user + assistant messages). This is
        intentional - a chatbot should remember everything it processes.

        Args:
            user_input: The text the user typed.

        Returns:
            The reply text.
        """
        # Normalise: lowercase + strip punctuation/spaces so "HELLO!!"
        # matches "hello" - the foundation of all NLP preprocessing.
        normalized: str = user_input.lower().strip(" !?.,;:")

        # Guard: input of only punctuation has nothing left to split.
        first_word: str = normalized.split()[0] if normalized else ""

        if first_word in config.GREETING_WORDS:
            reply = f"Hello! I'm {self.name}. How can I help?"
        elif normalized in {"what is your name", "what's your name", "who are you"}:
            reply = f"My name is {self.name}."
        else:
            reply = f"You said: {user_input}"

        self.conversation.add("user", user_input)
        self.conversation.add("assistant", reply)
        return reply

    def handle_command(self, command: str) -> tuple[bool, str]:
        """Process a slash-command WITHOUT printing (testability).

        Returning (keep_going, output) instead of printing directly is
        the right design: the caller decides where output goes, and
        tests can assert on it.

        Args:
            command: e.g. "/help".

        Returns:
            (should_the_session_continue, text_to_display).
        """
        if command == "/help":
            return True, "Commands: /help  /stats  /about  /exit"
        if command == "/stats":
            counts: dict[str, int] = self.conversation.count_by_role()
            return (
                True,
                f"I remember {len(self.conversation)} messages "
                f"(user={counts.get('user', 0)}, "
                f"assistant={counts.get('assistant', 0)}) across my sessions.",
            )
        if command == "/about":
            return (
                True,
                f"I am {config.ASSISTANT_INFO.get('name', 'Unknown')}, "
                f"version {config.ASSISTANT_INFO.get('version', '?')}. "
                f"Currently: {config.ASSISTANT_INFO.get('phase', '?')}.",
            )
        if command == "/exit":
            logger.info("User requested exit.")
            return False, "Goodbye."
        return True, f"Unknown command: {command}. Try /help"

    # ------------------------------------------------------------------
    # The REPL session
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Run the interactive loop until the user exits."""
        logger.info("Assistant %s started.", self.name)
        print(config.WELCOME_MESSAGE)

        if self.conversation:
            print(f"(Loaded {len(self.conversation)} past messages from memory.)")

        while True:
            try:
                user_input = input(config.REPL_PROMPT).strip()
            except (EOFError, KeyboardInterrupt):
                # Ctrl+D / Ctrl+C: quit gracefully. SystemExit still runs
                # the try/finally in main(), so memory is saved.
                print("\nGoodbye.")
                raise SystemExit(0)

            if not user_input:
                continue  # ignore blank lines

            # CATCH LATE, AT THE BOUNDARY: this try/except is the outer
            # wall of the program. If ANY unexpected error happens while
            # processing one exchange, we log the full traceback and
            # keep the session alive instead of dying. A production
            # assistant with a million turns cannot afford to crash on
            # one. (logger.exception logs WITH the stack trace.)
            try:
                if user_input.startswith("/"):
                    keep_going, output = self.handle_command(user_input)
                    if output:
                        print(output)
                    if not keep_going:
                        return  # normal exit - main()'s finally saves memory
                else:
                    print(self.respond(user_input))
            except Exception:
                logger.exception("Unexpected error processing: %r", user_input)
                print("Sorry, something went wrong. Please try again.")
