# DOCBOT-2636 NEW: Recover a hybrid answer whose stream dropped in the browser
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-26 Pre-seed | 0 | S0 | P0 | 3 | senior-dev | 2026-11-25 | Testing (2026-11-24) |

## User story
As a fractional CFO running a hybrid tie-out, I want an answer whose connection dropped mid-stream to come back in full, so that a network blip never costs me the answer the backend already produced.

## Why now
DOCBOT-2635 (PR #163) keeps the partial result and names the cause, but the full answer is still lost. On 2026-11-24 the backend had sent the whole 66 KB answer while the browser showed only SQL and an error. Investor demo is 2026-12-01.

## Scope
In:
- `/api/hybrid/chat`: `run_id` in the first `metadata` event; the turn runs in a detached task, so a client disconnect no longer cancels it before the answer is persisted.
- `GET /api/hybrid/answer/{run_id}` (`api/routes/answer_routes.py`, `api/answer_models.py`, Zod `src/lib/schemas/answer.ts`): persisted answer (agent_traces `final_answer` + citations) and lineage; `pending` while the turn still runs; 404 when missing or not the caller's (same owner check as `GET /api/lineage/{run_id}`).
- `agent_traces.final_answer` cap 4000 -> 20000 chars so the whole answer is stored.
- Client: on a dropped stream with a known run id, toast "Connection dropped, recovering answer…", poll every 2s up to 60s, swap in the answer + lineage; otherwise fall back to the 2635 partial + note.

Out:
- Docs, DB and autopilot chat paths.
- Multi-worker recovery (registry and detached tasks are per process; one uvicorn worker today).

## Acceptance criteria
- [ ] Given a finished hybrid run, when its owner calls the route, then the full answer, citations and lineage come back.
- [ ] Given another caller, when they call the route for that run (finished or running), then 404.
- [ ] Given a run still in progress, when the owner calls the route, then `status: "pending"`.
- [ ] Given the client stops reading mid-stream, then the backend still finishes the turn and persists the answer.
- [ ] Given the browser stream rejects after `metadata`, then the bubble shows the recovered full answer and the Inspector shows its lineage; when recovery fails, the 2635 partial + note is shown.

## Deliverable and approach
`api/answer_recovery_service.py` (`detach`, `mark_running`/`mark_done`, `recover_answer`), `api/resource_scope_service.py` (`ids_accessible`), `api/trace_service.py` (`wait_pending_writes`, cap), `api/hybrid_service.py`, `api/index.py` (wiring + detach), `src/lib/chatStream.ts` (`pollRecoveredAnswer`, `applyRecoveredAnswer`), `src/hooks/useChatSubmit.ts`.

## Test / verification
`tests/integration/test_answer_recovery.py` (owner ready, foreign 404, pending owner-only, unknown 404, route mapping, auth, detach keeps running after a drop); `tests/unit/test_answer_recovery_frontend_2636.py` (poll, timeout, apply via node). Full CI suite, `npm run lint`, `npx tsc --noEmit`.

## Dependencies
Blocked by: DOCBOT-2635 (merged).
Blocks: investor demo 2026-12-01.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] OpenProject ticket updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- A turn now finishes (and spends LLM credits) after the user leaves the page. Acceptable for one hybrid turn; revisit if abandoned turns show up in cost.
