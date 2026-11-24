# DOCBOT-2632 BUG: Demo session breaks after a backend restart
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-26 Pre-seed | 0 | - | P0 | 3 | senior-dev | 2026-11-25 | In progress (found 2026-11-23) |

## User story
As the founder demoing DocBot on 2026-12-01, I want an open demo tab to keep answering after the backend restarts, so that a redeploy during the call never shows "I encountered an error with the hybrid query".

## Why now
Prod redeployed 14 times on 2026-11-23 and every restart broke open demo tabs. Two causes: startup cleanup misread `demo-<uuid>` Chroma collection names and deleted them as orphaned, and the demo SQLite lives in /tmp, which a new container does not have.

## Scope
In:
- Fix `list_stored_sessions()` so `demo-` ids round-trip (no more orphan deletes).
- `ensure_demo_session()`: after the ownership check, rebuild a lost demo vector store and SQLite file under the same ids and path. Only for ids with a `sessions` row where `source='demo'`; it never creates rows.
- Frontend: a 404 "Session not found" / "Connection not found" on a demo restarts the same dataset and re-sends the question once; other sessions get "This session expired. Re-upload your file or reconnect your database."

Out:
- Moving demo SQLite to the Chroma volume (rehydrate makes it unnecessary).
- Recreating deleted session rows.

## Acceptance criteria
- [ ] Given an open demo, when the backend restarts and /tmp is wiped, then the next question is answered with the same session and connection ids.
- [ ] Given a demo whose Chroma collection is gone, when the owner asks a question, then the store is rebuilt and the answer has doc citations.
- [ ] Given a non-demo id, or a `demo-` id with no server row, when a request arrives, then nothing is created and the route still returns 404.
- [ ] Given a 404 on a demo session in the browser, when the user asks, then a fresh demo starts and the question is re-sent once.

## Deliverable and approach
`api/demo_service.py` (`ensure_demo_session`), called from `_require_session` / `_require_connection` in `api/index.py`; `api/utils/vector_store.py` (`list_stored_sessions`); `src/lib/httpGate.ts`, `src/hooks/useChatSubmit.ts`, `src/app/chat/page.tsx`.

## Test / verification
`tests/unit/test_demo_rehydrate.py`, `tests/unit/test_vector_store.py`. Local repro: init demo, stop server, delete SQLite and Chroma, restart, ask again (answers with citations).

## Dependencies
Blocked by: none.
Blocks: 2603.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] OpenProject ticket updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- none
