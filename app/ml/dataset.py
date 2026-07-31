"""Labeled intent dataset for the Phase 4 intent classifier.

An intent is the CLASS a message belongs to - what the user is trying
to DO, not the words they used. The five classes mirror the Assistant's
current dispatch branches (app/assistant.py) so Milestone 2 can route on
them without inventing new behavior:

  * GREETING          - "hello", "hey there"         -> greeting reply.
  * ASSISTANT_IDENTITY - "what is your name"          -> identity reply.
  * TEACH_FACT        - "my name is X", "i like Y",
                        "remember that Z"             -> MemoryExtractor.
  * RECALL_QUESTION   - "what is my name", "do i
                        like Y"                       -> MemoryService.
  * OTHER             - anything else                 -> echo fallback.

Supervised learning needs LABELED examples: the human writes text AND
picks the correct class for each one. The classifier then learns to
generalize - it can classify sentences it has never seen, which is
exactly what a rule-based match cannot do (it must have seen the words
in advance).

Data quality rules we enforce (see validate_dataset):
  * No duplicate examples - duplicates quietly inflate accuracy.
  * Every class has enough examples - a class with 2 examples can't
    be learned, and a 120-example dataset dominated by one class
    teaches the model to guess that class forever.
  * Labels come only from the Intent enum - a typo in a label string
    silently becomes a mystery class.
"""

from __future__ import annotations

from enum import StrEnum


class Intent(StrEnum):
    """The finite set of message intents the classifier can predict.

    Inherits from StrEnum (Python 3.11+) so each member IS its string
    value: Intent.GREETING == "greeting". scikit-learn needs plain
    string labels for classification, and StrEnum gives that for free.
    """

    GREETING = "greeting"
    ASSISTANT_IDENTITY = "assistant_identity"
    TEACH_FACT = "teach_fact"
    RECALL_QUESTION = "recall_question"
    OTHER = "other"


# Every labeled example is a (text, Intent) pair. The text is the raw
# user message (mixed casing on purpose - the vectorizer lowercases),
# and the label is the intent a well-behaved assistant should act on.
INTENT_EXAMPLES: list[tuple[str, Intent]] = [
    # ---------------- GREETING ----------------
    ("hello", Intent.GREETING),
    ("hi", Intent.GREETING),
    ("hey", Intent.GREETING),
    ("hiya", Intent.GREETING),
    ("howdy", Intent.GREETING),
    ("greetings", Intent.GREETING),
    ("hello there", Intent.GREETING),
    ("hi there", Intent.GREETING),
    ("hey there", Intent.GREETING),
    ("good morning", Intent.GREETING),
    ("good afternoon", Intent.GREETING),
    ("good evening", Intent.GREETING),
    ("good morning!", Intent.GREETING),
    ("good day", Intent.GREETING),
    ("morning!", Intent.GREETING),
    ("greetings friend", Intent.GREETING),
    ("top of the morning", Intent.GREETING),
    ("hi, how are you", Intent.GREETING),
    ("hey, what's new", Intent.GREETING),
    ("HELLO!", Intent.GREETING),
    ("hello friend", Intent.GREETING),
    ("what's up", Intent.GREETING),
    ("sup", Intent.GREETING),
    ("yo", Intent.GREETING),
    ("hi buddy", Intent.GREETING),
    # ---------------- ASSISTANT_IDENTITY ----------------
    ("what is your name", Intent.ASSISTANT_IDENTITY),
    ("what's your name", Intent.ASSISTANT_IDENTITY),
    ("whats your name", Intent.ASSISTANT_IDENTITY),
    ("who are you", Intent.ASSISTANT_IDENTITY),
    ("tell me your name", Intent.ASSISTANT_IDENTITY),
    ("what are you called", Intent.ASSISTANT_IDENTITY),
    ("your name please", Intent.ASSISTANT_IDENTITY),
    ("what is your name?", Intent.ASSISTANT_IDENTITY),
    ("what's your name please", Intent.ASSISTANT_IDENTITY),
    ("do you have a name", Intent.ASSISTANT_IDENTITY),
    ("introduce yourself", Intent.ASSISTANT_IDENTITY),
    ("what should i call you", Intent.ASSISTANT_IDENTITY),
    ("are you aria", Intent.ASSISTANT_IDENTITY),
    ("is your name aria", Intent.ASSISTANT_IDENTITY),
    ("what are you", Intent.ASSISTANT_IDENTITY),
    ("are you a robot", Intent.ASSISTANT_IDENTITY),
    ("are you human", Intent.ASSISTANT_IDENTITY),
    ("what do they call you", Intent.ASSISTANT_IDENTITY),
    ("your name is what", Intent.ASSISTANT_IDENTITY),
    ("what is the assistant's name", Intent.ASSISTANT_IDENTITY),
    ("who are you really", Intent.ASSISTANT_IDENTITY),
    ("tell me who you are", Intent.ASSISTANT_IDENTITY),
    ("do you have a name buddy", Intent.ASSISTANT_IDENTITY),
    ("how are you called", Intent.ASSISTANT_IDENTITY),
    ("who am i talking to", Intent.ASSISTANT_IDENTITY),
    # ---------------- TEACH_FACT ----------------
    ("my name is Alice", Intent.TEACH_FACT),
    ("my name's Bob", Intent.TEACH_FACT),
    ("call me Charlie", Intent.TEACH_FACT),
    ("i am called Dana", Intent.TEACH_FACT),
    ("i'm called Erin", Intent.TEACH_FACT),
    ("my name is frank", Intent.TEACH_FACT),
    ("my name's Grace", Intent.TEACH_FACT),
    ("please call me Henry", Intent.TEACH_FACT),
    ("i like coffee", Intent.TEACH_FACT),
    ("i love hiking", Intent.TEACH_FACT),
    ("i prefer tea", Intent.TEACH_FACT),
    ("i really like books", Intent.TEACH_FACT),
    ("i love dogs", Intent.TEACH_FACT),
    ("i prefer mornings", Intent.TEACH_FACT),
    ("i like the rain", Intent.TEACH_FACT),
    ("i love coding", Intent.TEACH_FACT),
    ("i really love jazz", Intent.TEACH_FACT),
    ("i don't like spiders", Intent.TEACH_FACT),
    ("i hate traffic", Intent.TEACH_FACT),
    ("i dislike crowds", Intent.TEACH_FACT),
    ("i don't like loud music", Intent.TEACH_FACT),
    ("i hate early mornings", Intent.TEACH_FACT),
    ("i do not like waiting", Intent.TEACH_FACT),
    ("my favorite book is Dune", Intent.TEACH_FACT),
    ("my favorite color is green", Intent.TEACH_FACT),
    ("my favourite food is pizza", Intent.TEACH_FACT),
    ("my favorite movie is Interstellar", Intent.TEACH_FACT),
    ("my favourite season is autumn", Intent.TEACH_FACT),
    ("my favorite team is the eagles", Intent.TEACH_FACT),
    ("remember that the sky is blue", Intent.TEACH_FACT),
    ("remember: feed the cat", Intent.TEACH_FACT),
    ("remember the code is 1234", Intent.TEACH_FACT),
    ("please remember that i like jazz", Intent.TEACH_FACT),
    ("remember to water the plants", Intent.TEACH_FACT),
    ("remember that i am a doctor", Intent.TEACH_FACT),
    ("note that i live in berlin", Intent.TEACH_FACT),
    ("just so you know, i have a cat", Intent.TEACH_FACT),
    ("keep this in mind: i am allergic to nuts", Intent.TEACH_FACT),
    ("i'm from canada", Intent.TEACH_FACT),
    ("i work as an engineer", Intent.TEACH_FACT),
    # ---------------- RECALL_QUESTION ----------------
    ("what is my name", Intent.RECALL_QUESTION),
    ("what's my name", Intent.RECALL_QUESTION),
    ("whats my name", Intent.RECALL_QUESTION),
    ("do you remember my name", Intent.RECALL_QUESTION),
    ("do you know my name", Intent.RECALL_QUESTION),
    ("what did i tell you my name was", Intent.RECALL_QUESTION),
    ("what is my favorite book", Intent.RECALL_QUESTION),
    ("what's my favorite color", Intent.RECALL_QUESTION),
    ("what is my favourite food", Intent.RECALL_QUESTION),
    ("do i like coffee", Intent.RECALL_QUESTION),
    ("do i like hiking", Intent.RECALL_QUESTION),
    ("what do i like", Intent.RECALL_QUESTION),
    ("what do i dislike", Intent.RECALL_QUESTION),
    ("do you remember what i like", Intent.RECALL_QUESTION),
    ("what's my favorite movie", Intent.RECALL_QUESTION),
    ("what did i say i liked", Intent.RECALL_QUESTION),
    ("what do i prefer", Intent.RECALL_QUESTION),
    ("do you remember me", Intent.RECALL_QUESTION),
    ("what do you know about me", Intent.RECALL_QUESTION),
    ("tell me what i told you", Intent.RECALL_QUESTION),
    ("what was my name", Intent.RECALL_QUESTION),
    ("do you remember what i told you", Intent.RECALL_QUESTION),
    ("what do you remember about me", Intent.RECALL_QUESTION),
    # ---------------- OTHER ----------------
    ("tell me a joke", Intent.OTHER),
    ("pancakes", Intent.OTHER),
    ("the weather is nice today", Intent.OTHER),
    ("what time is it", Intent.OTHER),
    ("how do i cook pasta", Intent.OTHER),
    ("hello world", Intent.OTHER),
    ("lorem ipsum", Intent.OTHER),
    ("42", Intent.OTHER),
    ("tell me a story", Intent.OTHER),
    ("sing me a song", Intent.OTHER),
    ("who won the game yesterday", Intent.OTHER),
    ("what is the meaning of life", Intent.OTHER),
    ("show me the news", Intent.OTHER),
    ("i have a question", Intent.OTHER),
    ("nothing", Intent.OTHER),
    ("yes", Intent.OTHER),
    ("no", Intent.OTHER),
    ("maybe", Intent.OTHER),
    ("thanks", Intent.OTHER),
    ("bye", Intent.OTHER),
    ("see you later", Intent.OTHER),
    ("good night", Intent.GREETING),
    ("ok", Intent.OTHER),
    ("random gibberish xyz", Intent.OTHER),
    ("the eagle has landed", Intent.OTHER),
    ("the stock market crashed", Intent.OTHER),
    ("it is raining outside", Intent.OTHER),
    ("i bought a new car", Intent.OTHER),
    ("the game starts at eight", Intent.OTHER),
    ("the flight was delayed", Intent.OTHER),
    ("python is fun", Intent.OTHER),
    ("that restaurant was great", Intent.OTHER),
    ("the cat is sleeping", Intent.OTHER),
    ("traffic was terrible today", Intent.OTHER),
    ("we are going to the beach", Intent.OTHER),
    ("it costs twenty dollars", Intent.OTHER),
]


def validate_dataset(examples: list[tuple[str, Intent]]) -> list[str]:
    """Return every data-quality problem found in the dataset (or []).

    A dataset is the ONE thing you cannot fix with a better model. These
    checks catch the failures that silently poison accuracy:
      * a label that is not an Intent member (typo -> mystery class);
      * duplicate texts (a repeated example inflates its class);
      * a class with fewer than MIN_PER_CLASS examples (unlearnable);
      * any empty text (a blank example teaches nothing).

    Args:
        examples: (text, Intent) pairs, normally INTENT_EXAMPLES.

    Returns:
        A list of human-readable problems; empty means the dataset
        passed every check.
    """
    MIN_PER_CLASS = 10
    problems: list[str] = []
    seen: set[str] = set()
    counts: dict[str, int] = {}

    for i, (text, intent) in enumerate(examples):
        if not isinstance(text, str) or not text.strip():
            problems.append(f"example {i}: empty text")
        key = text.strip().lower()
        if key in seen:
            problems.append(f"duplicate example {i}: {text!r}")
        seen.add(key)
        if not isinstance(intent, Intent):
            problems.append(f"example {i}: label {intent!r} is not an Intent")
            continue
        counts[intent.value] = counts.get(intent.value, 0) + 1

    for intent in Intent:
        if counts.get(intent.value, 0) < MIN_PER_CLASS:
            problems.append(
                f"{intent.value!r} has {counts.get(intent.value, 0)} examples "
                f"(need >= {MIN_PER_CLASS})"
            )
    return problems
