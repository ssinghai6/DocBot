# DOCBOT-2612 NEW: /v1/documents PDF upload (live keys)
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-39 API | 1 | S2 | P2 | 3 | senior-dev | - | New (2026-11-23) |

## User story
As an API client with a live key, I want to upload a PDF and get a session back, so that I can run the whole flow from my own system.

## Why now
Without upload, a live key can only query sessions created in the web app.

## Scope
In:
- Multipart PDF upload reusing `/api/upload` validation and `upload_limits`.
- New session stamped with org and `created_by` so `session_in_org` accepts it.

Out:
- CSV and SQLite uploads.
- Upload with test keys.

## Acceptance criteria
- [ ] Given a live key, when it uploads a valid PDF, then it gets a `session_id` it can query.
- [ ] Given a test key, when it calls the route, then it gets 403 `test_key_not_allowed`.
- [ ] Given a file over the size limit, when it is uploaded, then the API returns 413.
- [ ] Given the new session, when a key of another org reads it, then the API returns 404.

## Deliverable and approach
Call the existing upload service; no copy of its validation. Repo rules from `CLAUDE.md` apply (F-4 routes, models + Zod mirror, auth guard, tests in `tests/`). Plan: `docs/wiki/api/04-implementation-backlog.md` (stories), `docs/wiki/api/02-public-api-plan.md` (design), `docs/wiki/api/03-public-api-reference.md` (contract).

## Test / verification
Unit tests for 403, 413 and org stamping.

## Dependencies
Blocked by: 2599.
Blocks: none.

## Definition of done
- [ ] AC met
- [ ] tests/CI or evidence
- [ ] OpenProject ticket updated
- [ ] tracker row updated
- [ ] no secrets / no banned claims

## Open questions
- Should CSV follow in the same route later?
