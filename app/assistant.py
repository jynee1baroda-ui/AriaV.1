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
from typing import TYPE_CHECKING

from app import config
from app.llm.generator import build_messages, load_system_prompt
from app.ml.dataset import Intent
from app.models import Conversation, Memory
from app.services.memory_service import MemoryExtractor, MemoryService

if TYPE_CHECKING:
    from app.kb.knowledge import KnowledgeBase
    from app.llm.generator import LLMGenerator
    from app.services.intent_router import IntentRouter

logger = logging.getLogger("assistant")

# Canonical prefix constants - the exact strings MemoryExtractor writes
# for the facts the Assistant answers from long-term memory.
_NAME_MEMORY_PREFIX = "the user's name is "
_USER_PREFIX = "the user's "
_LIKES_PREFIX = "the user likes "
_DISLIKES_PREFIX = "the user dislikes "
_PREFERS_PREFIX = "the user prefers "


class Assistant:
    """A conversational agent with a name and a memory.

    Attributes:
        name: The assistant's display name.
        conversation: The Conversation object holding all messages.
        memories: Working-context memories retrieved for the current
            exchange (used to answer from long-term memory).
        session_id: Identifies this conversation to the memory store.
        memory_service: Optional memory organ. When wired, respond()
            consolidates new facts and refreshes memories each turn.
    """

    def __init__(
        self,
        name: str,
        conversation: Conversation,
        memories: list[Memory] | None = None,
        session_id: str | None = None,
        memory_service: MemoryService | None = None,
        router: IntentRouter | None = None,
        llm: LLMGenerator | None = None,
        knowledge: KnowledgeBase | None = None,
        factual_only: bool | None = None,
    ) -> None:
        self.name = name
        self.conversation = conversation
        self.memories: list[Memory] = list(memories) if memories else []
        self.session_id = session_id
        self.memory_service = memory_service
        # Phase 4 M2: the ML intent router. When present it gets first
        # shot at every message; when it declines (low confidence or
        # model unavailable) the deterministic rules answer instead.
        self.router = router
        # Phase 7 M1: the LLM answer engine. It is the last resort for
        # anything the rules cannot handle (i.e. the old echo). When
        # absent - or when it fails - the echo remains the floor.
        self.llm = llm
        # Phase 7 hardening: the verified knowledge base. Facts are
        # answered ONLY from here (never free-generated) in factual mode.
        self.knowledge = knowledge
        # When True (the default), Aria refuses to answer anything that
        # is not verified - no LLM generation, no echo pretending to be
        # an answer. Honesty over coverage.
        self.factual_only = config.FACTUAL_ONLY if factual_only is None else factual_only

    # ------------------------------------------------------------------
    # Core behavior
    # ------------------------------------------------------------------
    def respond(self, user_input: str) -> str:
        """Produce a reply AND record the exchange in memory.

        NOTE THE SIDE EFFECT: calling respond() also mutates
        self.conversation (adds user + assistant messages). This is
        intentional - a chatbot should remember everything it processes.

        When a memory_service is wired (Phase 5 M2), respond() ALSO
        consolidates facts from this turn and refreshes the working
        context BEFORE answering - so a statement like "my name is
        Alice" is stored, and a question like "what is my name" is
        answered from long-term memory, even across sessions.

        Phase 4 M2 routing: the intent router (ML) is consulted first,
        but ONLY as an advisor. If it returns a confident intent we try
        that branch; the branch must still satisfy its own rules (the
        extractor must find a fact, memory must hold an answer), and if
        it cannot - or if ML declines - we fall back to the
        deterministic dispatch below. ML suggests, rules confirm.

        Args:
            user_input: The text the user typed.

        Returns:
            The reply text.
        """
        if self.memory_service is not None and self.session_id is not None:
            self.memory_service.consolidate(self.session_id, user_input)
            self.memories = self.memory_service.retrieve(user_input)

        # Normalise: lowercase + strip punctuation/spaces so "HELLO!!"
        # matches "hello" - the foundation of all NLP preprocessing.
        normalized: str = user_input.lower().strip(" !?.,;:")

        reply = self._route_reply(user_input, normalized) or self._rule_reply(user_input, normalized)

        self.conversation.add("user", user_input)
        self.conversation.add("assistant", reply)
        return reply

    def _route_reply(self, user_input: str, normalized: str) -> str | None:
        """Phase 4 M2: let a confident ML intent pick the reply branch.

        Returns None to defer to the rules. The coarse intent only
        chooses the BRANCH; the details still come from the rules
        (MemoryExtractor for teach statements, MemoryService for recall
        questions), because rules extract structured facts the model
        does not.

        Args:
            user_input: Raw text (the extractor wants original casing).
            normalized: Lowercased/punctuation-stripped text (the
                memory question matcher expects it this way).

        Returns:
            A reply, or None when ML cannot answer confidently.
        """
        if self.router is None:
            return None
        intent = self.router.route(user_input)
        if intent is None:
            return None

        if intent is Intent.GREETING:
            return f"Hello! I'm {self.name}. How can I help?"
        if intent is Intent.ASSISTANT_IDENTITY:
            return f"My name is {self.name}."
        if intent is Intent.TEACH_FACT:
            extracted = MemoryExtractor.extract(user_input)
            return extracted[0].reply if extracted else None
        if intent is Intent.RECALL_QUESTION:
            return self._answer_from_memory(normalized)
        # Intent.OTHER: still give verified facts a chance before
        # refusing (or, in open mode, before the LLM/echo).
        return self._knowledge_answer(normalized) or self._fallback_answer(user_input, normalized)

    def _rule_reply(self, user_input: str, normalized: str) -> str:
        """Deterministic fallback - the original Phase 1-3 dispatch.

        Ran whenever ML is absent, declines, or cannot fulfill its
        intent branch. Order matters (greeting beats extraction beats
        echo); changing it changes real answers.

        Args:
            user_input: Raw user text.
            normalized: Lowercased/punctuation-stripped user text.

        Returns:
            The reply text.
        """
        # Guard: input of only punctuation has nothing left to split.
        first_word: str = normalized.split()[0] if normalized else ""

        if first_word in config.GREETING_WORDS:
            return f"Hello! I'm {self.name}. How can I help?"
        if normalized in {"what is your name", "what's your name", "who are you"}:
            return f"My name is {self.name}."
        # STATEMENTS that teach a fact are answered from the text
        # itself; the extractor also supplies the confirmation text.
        extracted = MemoryExtractor.extract(user_input)
        if extracted:
            return extracted[0].reply
        # QUESTIONS that consult long-term memory come next, then the
        # verified knowledge base, then the last resort.
        return (
            self._answer_from_memory(normalized)
            or self._knowledge_answer(normalized)
            or self._fallback_answer(user_input, normalized)
        )

    def _knowledge_answer(self, normalized: str) -> str | None:
        """Answer a factual question from the VERIFIED knowledge base.

        This is the "never fake" guarantee: facts come only from the
        curated, checked store. None means no verified fact matches -
        the caller must refuse rather than guess.

        Args:
            normalized: The normalized user message.

        Returns:
            The verified answer, or None.
        """
        if self.knowledge is None:
            return None
        return self.knowledge.lookup(normalized)

    def _fallback_answer(self, user_input: str, normalized: str) -> str:
        """The last resort: refuse, generate, or echo.

        Phase 7 M1/M2: the deterministic rules win on greetings,
        identity, taught facts, stored memories and verified facts;
        everything else used to go to the LLM (with context).

        Phase 7 hardening - FACTUAL MODE: when factual_only is True
        (the default), NOTHING unverified is said. The LLM is a
        plausible-text generator, not a truth oracle; a 135M base model
        hallucinates confidently. So the honest answer is a refusal:
        "I don't have a verified answer for that." No echo, no LLM -
        silence about the unknown is the whole point.

        Only when factual_only is False (open mode) does the LLM
        generate, with the echo as its own failure floor.

        Args:
            user_input: The raw text.
            normalized: The normalized text.

        Returns:
            A refusal, an LLM answer, or the echo.
        """
        if self.factual_only:
            return "I don't have a verified answer for that."
        if self.llm is not None:
            exchanges = self._recent_exchanges(config.LLM_CONTEXT_TURNS)
            messages = build_messages(load_system_prompt(), exchanges, user_input)
            answer = self.llm.generate(messages)
            # Guard: a tiny base model sometimes answers by REPEATING
            # the question. Parroting the input is not answering - fall
            # through to the echo, which is at least honest.
            if answer and answer.strip().lower() != user_input.strip().lower():
                return answer
        return f"You said: {user_input}"

    def _recent_exchanges(self, turns: int) -> list[tuple[str, str]]:
        """Return the last `turns` (user, assistant) pairs, oldest first.

        The conversation alternates user/assistant by construction, so
        pairs are read backwards from the end and then reversed.

        Args:
            turns: How many pairs to keep (the context window).

        Returns:
            Up to `turns` pairs as (user_message, assistant_reply).
        """
        messages = self.conversation.messages
        pairs: list[tuple[str, str]] = []
        index = len(messages) - 1
        while index >= 1 and len(pairs) < turns:
            assistant, user = messages[index], messages[index - 1]
            if user.role == "user" and assistant.role == "assistant":
                pairs.append((user.content, assistant.content))
            index -= 2
        return list(reversed(pairs))

    def _answer_from_memory(self, normalized: str) -> str | None:
        """Answer a memory-backed question, or None if not a question.

        Returns:
            The memory answer, or None to fall through to the echo.
        """
        if normalized in {
            "what is my name",
            "what's my name",
            "whats my name",
            "do you remember my name",
            "do you know my name",
        }:
            return self._answer_name()
        if "favorite" in normalized or "favourite" in normalized:
            return self._answer_favorite()
        if "like" in normalized or "dislike" in normalized or "prefer" in normalized:
            return self._answer_like_dislike()
        return None

    def _answer_name(self) -> str:
        """Answer the name question from working-context memories."""
        for memory in self.memories:
            if memory.content.startswith(_NAME_MEMORY_PREFIX):
                value = memory.content[len(_NAME_MEMORY_PREFIX):]
                return f"Your name is {value}."
        return "I don't know your name yet. Tell me with 'my name is ...'"

    def _answer_favorite(self) -> str:
        """Answer a favorite question from working-context memories."""
        favorites = [m for m in self.memories if "favorite" in m.content]
        if favorites:
            return f"Your {favorites[0].content[len(_USER_PREFIX):]}."
        return "I don't know your favorites yet. Tell me with 'my favorite <thing> is ...'"

    def _answer_like_dislike(self) -> str:
        """Answer a like/dislike question from working-context memories."""
        for memory in self.memories:
            if memory.content.startswith(_LIKES_PREFIX):
                return f"You like {memory.content[len(_LIKES_PREFIX):]}."
            if memory.content.startswith(_DISLIKES_PREFIX):
                return f"You don't like {memory.content[len(_DISLIKES_PREFIX):]}."
            if memory.content.startswith(_PREFERS_PREFIX):
                return f"You prefer {memory.content[len(_PREFERS_PREFIX):]}."
        return "I don't remember what you like yet."

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
            return (
                True,
                "Commands: /help  /stats  /about  /memory  /remember <fact>  "
                "/forget <id>  /exit",
            )
        if command == "/stats":
            counts: dict[str, int] = self.conversation.count_by_role()
            return (
                True,
                f"I remember {len(self.conversation)} messages "
                f"(user={counts.get('user', 0)}, "
                f"assistant={counts.get('assistant', 0)}) across my sessions.",
            )
        if command == "/memory":
            if self.memory_service is None:
                return True, "Long-term memory is not available in this mode."
            memories = self.memory_service.get_all()
            if not memories:
                return True, "I don't remember anything yet."
            lines = "\n".join(f"  - {m.content}" for m in memories)
            return True, f"I remember {len(memories)} long-term memories:\n{lines}"
        if command.startswith("/remember "):
            if self.memory_service is None:
                return True, "Long-term memory is not available in this mode."
            content = command[len("/remember "):].strip()
            if not content:
                return True, "Usage: /remember <something to remember>"
            self.memory_service.add(content, category="fact", session_id=self.session_id)
            return True, f"Okay, I'll remember that: {content}"
        if command.startswith("/forget "):
            if self.memory_service is None:
                return True, "Long-term memory is not available in this mode."
            memory_id = command[len("/forget "):].strip()
            if not memory_id.isdigit():
                return True, "Usage: /forget <memory id> (find ids with /memory)"
            deleted = self.memory_service.delete(int(memory_id))
            if deleted:
                return True, f"Forgot memory {memory_id}."
            return True, f"No memory with id {memory_id}. Check /memory."
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
