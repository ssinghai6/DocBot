# DOCBOT-2597 NEW: Sandbox sessions for test API keys (demo data, daily test allowance)
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-39 API | 1 | S1 | P1 | 5 | senior-dev | - | New (2026-11-23) |

## User story
As a developer with a test key, I want to create a sandbox session on demo data and query it, so that I can try the API without uploading client data or spending my org's credits.

## Why now
Playing with the API needs safe data. `/v1` cannot reach demo sessions today by design, and test keys must never touch live org data or the org balance.

## Scope
In:
- `POST /v1/sandbox/sessions` wraps `init_demo_session()` (`api/demo_service.py`) with owner = key user + org and `sessions.is_sandbox=true`.
- Test keys get 404 on any non-sandbox session or connection (`require_sandbox_resource()`).
- Per-owner create window (5/hour) and daily test allowance (default 300 credits via `cost_to_credits`).
- Purge of sandbox sessions after 24h.

Out:
- Document upload with test keys (DOCBOT-2612 returns 403 for them).
- Portal UI (DOCBOT-3905).

## Acceptance criteria
- [ ] Given a test key, when it calls `POST /v1/sandbox/sessions` with dataset `quickbite`, then it gets a `session_id` and `connection_id` it can query.
- [ ] Given a test key, when it references a non-sandbox session or connection, then the API returns 404.
- [ ] Given a test key whose daily allowance is used up, when it makes a cost-bearing call, then it gets 429 `test_allowance_exhausted` and the org balance is unchanged.
- [ ] Given a sixth sandbox create inside one hour for the same owner, when it is sent, then it is rejected with 429.
- [ ] Given a sandbox session older than 24h, when the purge runs, then the session, vector store and SQLite file are gone.

## Deliverable and approach
Route in `api/routes/v1_routes.py`, models in `api/v1_models.py`, logic in a small `api/sandbox_session_service.py`. Reuse the existing counter store for the window and allowance; `copy_ctx_run()` for executor calls. Repo rules from `CLAUDE.md` apply (F-4 routes, models + Zod mirror, auth guard, tests in `tests/`). Plan: `docs/wiki/api/04-implementation-backlog.md` (stories), `docs/wiki/api/02-public-api-plan.md` (design), `docs/wiki/api/03-public-api-reference.md` (contract).

## Test / verification
`tests/unit/test_v1_sandbox.py`: isolation, allowance exhaustion 429, org balance untouched, purge.

## Dependencies
Blocked by: 2907, 3901.
Blocks: 3905.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] OpenProject ticket updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Is 300 credits/day the right default for the founder's own testing?
