# Project Roadmap — AI Assistant

The authoritative, 20-phase roadmap. **Do not skip phases.** Every
completed phase leaves a working project behind.

## The 20 phases

| Phase | Topic | Status |
|-------|-------|--------|
| 1 | Programming fundamentals | ✅ complete |
| 2 | Backend development | ✅ complete |
| 3 | Databases | ✅ complete |
| 4 | Machine Learning | ✅ complete |
| 5 | Deep Learning | ✅ complete |
| 6 | Transformers | ✅ complete |
| 7 | Large Language Models | 🔶 M1 complete (real LLM answers) |
| 8 | Prompt Engineering | ⬜ |
| 9 | RAG | ⬜ |
| 10 | Memory Systems | 🔶 foundation built (preview) |
| 11 | Tool Calling | ⬜ |
| 12 | Agents | ⬜ |
| 13 | Vision | ⬜ |
| 14 | Voice | ⬜ |
| 15 | Deployment | ⬜ |
| 16 | Scaling | ⬜ |
| 17 | Monitoring | ⬜ |
| 18 | Security | ⬜ |
| 19 | Fine Tuning | ⬜ |
| 20 | Production | ⬜ |

## Notes

- **Phase 7 (Large Language Models)** M1 complete: a pre-trained
  causal LM (`HuggingFaceTB/SmolLM2-135M`, ~270 MB, runs on CPU) is
  wired into the echo fallback via `app/llm/generator.py`. Lazy load,
  graceful degradation to the echo, stub-injectable for tests. M2 adds
  a conversation context window + honest capability assessment.
- **Phase 10 (Memory Systems)** is the only phase with work done ahead
  of schedule: `memories` table, `MemoryRepository`, `MemoryService`
  (consolidation + retrieval). It is the storage foundation that RAG
  (Phase 9) and contextual memory (Phase 10) will build on. It is NOT
  the full phase — vector/contextual memory arrives in place.
- **Engineering hardening** (not a roadmap phase): rate limiting on
  `POST /chat`, pagination on `GET /history`, Alembic versioned
  migrations, readiness probe. Rate limiting/pagination are revisited
  formally in Phase 18 (Security).

## Legacy milestone labels in docstrings

Source docstrings written before this file existed use an old internal
numbering. Mapping to the real phases:

- old "Phase 4" (Database Integration) → **Phase 3**
- old "Phase 5" (Memory System) → **Phase 10 foundation** (preview)
- old "Phase 6" (API hardening) → **engineering hardening** (Phase 18 preview)

New work uses only the real phase numbers above.

## Definition of done for a phase

- Every milestone committed with passing tests
- Live verification against the real server
- README status updated to match this table
- No fabricated results — anything uncertain is stated as uncertain
