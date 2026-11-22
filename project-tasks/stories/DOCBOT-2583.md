# DOCBOT-2583 NEW: Demo chat hangs when Groq rate-limits
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-26 Pre-seed | 0 | - | P0 | 3 | senior-dev | 2026-11-27 | In progress (2026-11-20) |

## User story
As the founder demoing DocBot on 2026-12-01, I want the demo hybrid chat to always finish or show an error, so that a Groq rate limit never leaves an investor staring at an open stream.

## Why now
Prod logs (2026-11-20): a second demo chat within a minute gets a Groq 429, falls back to the Gemini stream, and the SSE stream then never emits another event or `done` (150 s+). Prod freeze is 11-27.

## Scope
In:
- Bound the Gemini stream fallback (deadline per chunk, no SDK retry) and drop Groq SDK retries on the streaming path.
- Hybrid chat emits `done` after its synthesis `error` event.
- Bind `df` in the DB chart/analysis sandbox run (NameError wasted an E2B run + LLM call per chat).
- Add `llm_calls.prompt_version` via idempotent `ADD COLUMN IF NOT EXISTS` (trace rows were being lost).

Out:
- Raising the Groq tier or TPM limit; reducing the ~4 Groq calls per chat.

## Acceptance criteria
- [ ] Given a stalled Gemini stream, when no chunk arrives within GEMINI_TIMEOUT_S, then a TimeoutError is raised.
- [ ] Given both providers fail, when hybrid chat streams, then it emits an `error` event followed by `done`.
- [ ] Given the DB analysis step, when code runs in the sandbox, then `df` is defined.
- [ ] Given init_db on a DB lacking `llm_calls.prompt_version`, when it runs twice, then the column exists and no error occurs.

## Deliverable and approach
Fix in `api/utils/llm_provider.py`, `api/hybrid_service.py`, `api/sandbox_service.py`, `api/db_service.py`, `api/usage_service.py`.

## Test / verification
`pytest tests/unit/test_demo_chat_hang.py`; `pytest tests/ -m "not external and not postgres"`.

## Dependencies
Blocked by: none.
Blocks: none.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] Plane issue updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Gemini hang root cause is inferred, not reproduced against the live API.
