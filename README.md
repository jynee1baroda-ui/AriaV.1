# AI Assistant

A production-grade AI assistant built from scratch, following a 20-phase
roadmap. See the current phase below.

## Status

- Phase 1 (Programming Fundamentals): **Lesson 5 complete** — exception
  hierarchy (`AssistantError` → ...), atomic file writes via the
  `AtomicFileWriter` context manager, crash-proof REPL, Windows handle-
  locking gotcha fixed. 34 passing tests.

## Requirements

- Python 3.12+ (verified with 3.14)

## Run

Both commands work (see the sys.path bootstrap in `app/main.py`):

```powershell
python app\main.py
# or, the module way real projects use:
python -m app.main
```

## Test

```powershell
python -m unittest discover -s tests -v
```

## Usage

```
You> HELLO!!
Hello! I'm Aria. How can I help?
You> what is your name
My name is Aria.
You> /stats
I remember 4 messages (user=2, assistant=2) across my sessions.
You> /about
I am Aria, version 0.1.0. Currently: Phase 1 - Programming Fundamentals.
You> /help
Commands: /help  /stats  /about  /exit
You> /exit
```

Conversations are saved to `data/history.json` and reloaded on the next
start — the seed of long-term memory.
