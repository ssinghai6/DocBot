# DOCBOT-2599 NEW: /v1 session read/delete + catalog endpoints
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-39 API | 1 | S2 | P2 | 3 | senior-dev | - | New (2026-11-23) |

## User story
As an API client, I want to list, read and delete my sessions and see connections, schemas and personas, so that I can build on DocBot data without the web app.

## Why now
After the playground, real clients need to find what they can query. No `/v1` session or catalog route exists today.

## Scope
In:
- `GET /v1/sessions`, `GET/DELETE /v1/sessions/{id}`, `GET /v1/db/connections`, `GET /v1/db/connections/{id}/schema`, `GET /v1/personas`.
- Cursor pagination.
- Scoped by key org, and to sandbox resources for test keys.

Out:
- Session creation by upload (DOCBOT-2612).

## Acceptance criteria
- [ ] Given a live key of org A, when it lists sessions, then only org A sessions are returned, paged by cursor.
- [ ] Given a key of org A, when it reads or deletes a session of org B, then the API returns 404.
- [ ] Given a test key, when it lists sessions or connections, then only its sandbox resources are returned.
- [ ] Given a connection the key can see, when it requests the schema, then table and column names are returned and no credentials are.

## Deliverable and approach
Handlers in `api/routes/v1_routes.py` over `resource_scope_service`; models in `api/v1_models.py` + `src/lib/schemas/v1.ts`. Repo rules from `CLAUDE.md` apply (F-4 routes, models + Zod mirror, auth guard, tests in `tests/`). Plan: `docs/wiki/api/04-implementation-backlog.md` (stories), `docs/wiki/api/02-public-api-plan.md` (design), `docs/wiki/api/03-public-api-reference.md` (contract).

## Test / verification
Unit tests for cross-org and cross-mode access and pagination.

## Dependencies
Blocked by: 3901, 2907.
Blocks: 2611, 2612.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] OpenProject ticket updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Soft delete or hard delete for `DELETE /v1/sessions/{id}`?
